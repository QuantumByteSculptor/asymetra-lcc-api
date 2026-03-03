# V3 API Contract — Asymetra Signal Model

**schema_version**: `3.1`
**Generated from**: `models/v3/v3_meta.json` (2026-03-03)
**Canonical dataset**: `data/training/train_v3_all.jsonl`

---

## 1. Source of Truth

| Artifact | Path | Lines | SHA-256 | Date range |
|---|---|---|---|---|
| Full dataset | `data/training/train_v3_all.jsonl` | 54,963 | `6562a84a…` | 2010-12-31 → 2025-11-28 |
| Val fold (fold_5) | `data/training/v3/fold_5/val.jsonl` | 47,161 | `bb2822e8…` | 2015-06-17 → 2025-12-08 |
| Model dir | `models/v3/` | — | — | Trained 2026-03-03 |

Model artifacts:

| File | Purpose |
|---|---|
| `v3_xgb_model.joblib` | Fitted XGBoost classifier (base) |
| `v3_lr_model.joblib` | Fitted Logistic Regression (reference) |
| `v3_calibrator.joblib` | Isotonic calibrator fitted on fold_5/val |
| `v3_feature_names.joblib` | Exact ordered list of 50 feature names |
| `v3_meta.json` | Schema version, feature list, medians, thresholds |
| `v3_thresholds.json` | t_lo / t_hi + provenance |
| `v3_metrics.json` | CV ROC-AUC, Brier, ECE per fold |

---

## 2. Input Contract

### 2.1 Feature Matrix

The model expects a **float64 matrix X of shape (n, 50)** where columns are **exactly** the features listed in `v3_feature_names.joblib`, in the order given. Any deviation in order, naming, or count is an error.

```python
import joblib
feat_names = joblib.load("models/v3/v3_feature_names.joblib")  # list[str], len=50
```

Feature list (alphabetical, as stored):

```
autocorr_1       autocorr_5       bb_distance      dd_duration      dd_duration_per_n
dd_to_var99      downside_dev     downside_div_vol es95             es95_var95
es99             es99_es95        es99_var99       hill_tail_index  jump_indicator
kurtosis_excess  log_n_used       macd_hist        max_dd           max_drawdown
missing_pct      n_used           rsi              rsi_centered     semivariance
skew             sma_slope_20     sma_slope_60     stress_multiplier stress_var99
tail_obs_99      tuw_pct          var95            var99            var99_var95
vol120_vol_ann   vol20_vol60      vol20_vol_ann    vol60_vol_ann    vol_120d
vol_20d          vol_60d          vol_ann          vol_ewma_ann     vol_of_vol
vol_to_var95     worst_10d_ret    worst_20d_ret    worst_5d_ret     worst_5d_vs_var99
```

### 2.2 Sentinel Values

The following features use **sentinel -1.0** instead of NaN when their condition is undefined. The model was trained with this convention; do not replace sentinel with NaN.

| Feature | Sentinel | Meaning when sentinel |
|---|---|---|
| `recovery_days` | `-1.0` | Drawdown has not yet recovered |
| `recovery_per_dd` | `-1.0` | Drawdown has not yet recovered |
| `dd_duration` | `-1.0` | No qualifying drawdown in window |

Companion binary flag:

| Feature | Type | Values |
|---|---|---|
| `recovery_defined` | float | `1.0` = recovery occurred, `0.0` = not recovered |

> **Note**: `recovery_days` and `recovery_per_dd` were dropped from the current model due to >30% NaN rate in the training split (nan_drop_threshold=0.30). They are **not** in the 50 active features. `recovery_defined` IS included.

### 2.3 Dropped Features (excluded from X)

13 features were dropped at training time (nan_rate > 0.30):

```
abs_corr_mkt  beta_market   corr_spy      corr_vix
credit_spread_hy  rate_10y  rate_2y       recovery_days
recovery_per_dd   term_spread  vix_level  vix_pct_60d
vol_regime
```

Do not include these in X.

### 2.4 Metadata Fields (excluded from X)

These fields are present in JSONL records but excluded from the feature matrix:

```
asset_type  market  ticker  market_proxy
```

### 2.5 Preprocessing

Before inference, apply **median imputation** using the training-set medians stored in `v3_meta.json`:

```python
import json, numpy as np

meta   = json.load(open("models/v3/v3_meta.json"))
medians = meta["medians"]  # dict: feature_name -> float

# Impute: replace NaN with training median
for j, feat in enumerate(feat_names):
    col = X[:, j]
    nan_mask = ~np.isfinite(col)
    if nan_mask.any():
        X[nan_mask, j] = medians.get(feat, 0.0)
```

Do not use a live-fit imputer in production — use the frozen medians from v3_meta.json.

---

## 3. Output Contract

### 3.1 Prediction Pipeline

```python
import joblib, json, numpy as np

xgb_model  = joblib.load("models/v3/v3_xgb_model.joblib")
calibrator  = joblib.load("models/v3/v3_calibrator.joblib")
thresholds  = json.load(open("models/v3/v3_thresholds.json"))

t_lo = thresholds["t_lo"]  # 0.4863 — warn threshold
t_hi = thresholds["t_hi"]  # 0.6654 — block threshold

# Get calibrated probability of non-ok event
proba_raw      = xgb_model.predict_proba(X)[:, 1]
proba_non_ok   = calibrator.predict(proba_raw)        # shape (n,), float64

# Assign label
labels = np.where(proba_non_ok >= t_hi, "block",
         np.where(proba_non_ok >= t_lo, "warn", "ok"))
```

### 3.2 Response Fields

| Field | Type | Description |
|---|---|---|
| `proba_non_ok` | `float` ∈ [0, 1] | Calibrated probability of a non-ok outcome |
| `label` | `str` ∈ {`ok`, `warn`, `block`} | Signal label derived from thresholds |
| `t_lo` | `float` | Warn threshold (current: 0.4863) |
| `t_hi` | `float` | Block threshold (current: 0.6654) |
| `schema_version` | `str` | `"3.1"` |
| `model_version` | `str` | `"v3_xgb_calibrated"` |

### 3.3 Thresholds

| Condition | Label |
|---|---|
| `proba_non_ok < t_lo` | `ok` |
| `t_lo ≤ proba_non_ok < t_hi` | `warn` |
| `proba_non_ok ≥ t_hi` | `block` |

Current values (fitted on fold_5/val, 2026-03-03):
- `t_lo = 0.4863` (target FPR 10%)
- `t_hi = 0.6654` (target FPR 25%)
- Model: `xgb_calibrated`
- Fitted on: `last_fold_val` (fold_5, n=47,161)

---

## 4. Invariants

All invariants must hold before shipping a model update. They are checked by `preflight_v3_api.py`.

| # | Invariant | Check |
|---|---|---|
| I-1 | X has no NaN or inf after imputation | `np.all(np.isfinite(X))` |
| I-2 | proba_non_ok ∈ [0, 1] for all records | `np.all((p >= 0) & (p <= 1))` |
| I-3 | Label is consistent with thresholds | label == threshold rule for each record |
| I-4 | Threshold monotonicity: `t_lo < t_hi` | enforced at load time |
| I-5 | Feature count matches model | `len(feat_names) == 50` |
| I-6 | Feature order matches model | exact list equality |
| I-7 | Sentinel features not NaN before imputation | recovery sentinels must be -1.0, not NaN |
| I-8 | recovery_defined is 0.0 or 1.0, never NaN | explicit check |

---

## 5. CV Performance (fold_5/val, n=47,161)

| Model | ROC-AUC | Brier | ECE |
|---|---|---|---|
| XGB (raw) | 0.7823 | 0.193 | 0.056 |
| XGB + isotonic calibration | 0.7830 | 0.188 | 0.000 |
| Logistic Regression | 0.7857 | 0.190 | 0.055 |

Backtest (fold_5/val, 5 bps, horizon=20d):
- Signal Sharpe: **+0.872** (vs always_ok +0.308, random +0.263)
- Skip rate (block): 22.4%
- Net 5bps Sharpe: **+0.867**

---

## 6. Feature Descriptions

| Feature | Category | Notes |
|---|---|---|
| `vol_ann` | Volatility | Annualised vol (full window) |
| `vol_20d` | Volatility | 20-day rolling vol |
| `vol_60d` | Volatility | 60-day rolling vol |
| `vol_120d` | Volatility | 120-day rolling vol |
| `vol_ewma_ann` | Volatility | EWMA annualised vol |
| `vol_of_vol` | Volatility | Volatility of volatility |
| `vol20_vol_ann` | Vol ratio | vol_20d / vol_ann |
| `vol60_vol_ann` | Vol ratio | vol_60d / vol_ann |
| `vol120_vol_ann` | Vol ratio | vol_120d / vol_ann |
| `vol20_vol60` | Vol ratio | vol_20d / vol_60d |
| `var95` | Risk | 95th percentile daily VaR |
| `var99` | Risk | 99th percentile daily VaR |
| `var99_var95` | Risk ratio | var99 / var95 |
| `es95` | Risk | Expected Shortfall at 95% |
| `es99` | Risk | Expected Shortfall at 99% |
| `es95_var95` | Risk ratio | es95 / var95 |
| `es99_es95` | Risk ratio | es99 / es95 |
| `es99_var99` | Risk ratio | es99 / var99 |
| `vol_to_var95` | Risk ratio | vol / var95 |
| `stress_var99` | Risk | Stress-period VaR99 |
| `stress_multiplier` | Risk | Stress VaR / normal VaR |
| `tail_obs_99` | Tail | Observations above 99th percentile |
| `hill_tail_index` | Tail | Hill estimator of tail index |
| `jump_indicator` | Tail | Fraction of jump days |
| `kurtosis_excess` | Distribution | Excess kurtosis |
| `skew` | Distribution | Return distribution skew |
| `downside_dev` | Downside | Downside deviation |
| `semivariance` | Downside | Lower semi-variance |
| `downside_div_vol` | Downside | Downside dev / vol_ann |
| `max_dd` | Drawdown | Maximum drawdown (same as max_drawdown) |
| `max_drawdown` | Drawdown | Maximum drawdown in window |
| `dd_duration` | Drawdown | Drawdown duration in days (-1 if no DD) |
| `dd_duration_per_n` | Drawdown | dd_duration / n_used |
| `dd_to_var99` | Drawdown | max_drawdown / var99 |
| `worst_5d_ret` | Return | Worst 5-day rolling return |
| `worst_10d_ret` | Return | Worst 10-day rolling return |
| `worst_20d_ret` | Return | Worst 20-day rolling return |
| `worst_5d_vs_var99` | Return ratio | abs(worst_5d) / var99 |
| `autocorr_1` | Return | 1-lag autocorrelation |
| `autocorr_5` | Return | 5-lag autocorrelation |
| `log_n_used` | Meta | log(n observations used) |
| `n_used` | Meta | Observations used in window |
| `missing_pct` | Meta | Fraction of missing observations |
| `sma_slope_20` | Technical | 20-day SMA slope (normalised) |
| `sma_slope_60` | Technical | 60-day SMA slope (normalised) |
| `macd_hist` | Technical | MACD histogram |
| `rsi` | Technical | RSI (14-day) |
| `rsi_centered` | Technical | RSI − 50 |
| `bb_distance` | Technical | Distance from Bollinger Band midline |
| `tuw_pct` | Path | Time-underwater percentage |

---

## 7. Versioning & Update Protocol

1. Retrain triggers: drift PSI > 0.20 on ≥ 3 features, or model ROC-AUC < 0.70 on recent data
2. On retrain: regenerate `v3_meta.json`, `v3_thresholds.json`, `v3_feature_names.joblib`
3. Run `python scripts/ml/validation/preflight_v3_api.py` — must exit 0
4. Bump `schema_version` if feature list or sentinel convention changes
5. Update this document's date and hash entries in Section 1
