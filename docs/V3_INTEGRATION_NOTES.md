# V3 Integration Notes — API Team

> Schema version: **3.1**
> Branch: `feat/v3-quant-pipeline`
> Proposed tag: `v3.1.0`

---

## 1. Files to Deploy

Deploy exactly these **4 files** from `models/v3/` to the API container:

| File | Purpose |
|------|---------|
| `v3_xgb_model.joblib` | XGBoost sklearn Pipeline (SimpleImputer + XGBClassifier) |
| `v3_calibrator.joblib` | IsotonicRegression calibrator — applied after `predict_proba` |
| `v3_meta.json` | Feature list, medians, dropped features, schema_version |
| `v3_thresholds.json` | `t_lo` (warn) and `t_hi` (block) — FPR-optimised |

**Do NOT deploy:**

- `v3_lr_model.joblib` — LR baseline only, not used in production scoring
- `v3_feature_names.joblib` — redundant; `feature_cols` is already in `v3_meta.json`
- `v3_metrics.json` — evaluation artefact; not needed at inference time

---

## 2. Loading Snippet

```python
import joblib
import json
from pathlib import Path

MODELS_DIR = Path("models/v3")

# Load once at startup
xgb_pipe   = joblib.load(MODELS_DIR / "v3_xgb_model.joblib")
calibrator = joblib.load(MODELS_DIR / "v3_calibrator.joblib")
meta       = json.loads((MODELS_DIR / "v3_meta.json").read_text())
thresholds = json.loads((MODELS_DIR / "v3_thresholds.json").read_text())

# Sanity-check on load (fail fast in case of stale artefacts)
assert meta["schema_version"] == "3.1", (
    f"Stale artefact — expected schema_version '3.1', "
    f"got {meta['schema_version']!r}"
)

FEATURE_COLS = meta["feature_cols"]   # ordered list of 64 feature names
T_LO         = thresholds["t_lo"]    # 0.5203 — warn threshold
T_HI         = thresholds["t_hi"]    # 0.6667 — block threshold
```

---

## 3. Inference Signature

```python
import numpy as np

def score_v3(input_features: dict[str, float]) -> dict:
    """
    input_features : flat dict of raw feature values (any subset of FEATURE_COLS).
                     Missing features are imputed inside the XGB pipeline.
    Returns        : {"score": float, "label": str, "t_lo": float, "t_hi": float}
    """
    # 1. Build feature row in correct column order (missing → NaN for imputer)
    X = np.array(
        [input_features.get(col, np.nan) for col in FEATURE_COLS],
        dtype=np.float64,
    ).reshape(1, -1)

    # 2. XGB pipeline: SimpleImputer handles NaN internally
    proba_raw = xgb_pipe.predict_proba(X)[:, 1]   # shape (1,)

    # 3. Calibrate (IsotonicRegression — separate step, not in pipeline)
    score = float(calibrator.predict(proba_raw)[0])

    # 4. Apply thresholds
    if score >= T_HI:
        label = "block"
    elif score >= T_LO:
        label = "warn"
    else:
        label = "ok"

    return {"score": score, "label": label, "t_lo": T_LO, "t_hi": T_HI}
```

---

## 4. Known Pitfalls

### Feature order — always use `feature_cols` from `v3_meta.json`
**NEVER hardcode the feature list.** The retained features are determined
dynamically by a NaN-rate filter applied at training time. Use:
```python
FEATURE_COLS = meta["feature_cols"]
```

### Dropped features (current model: 0 dropped, 64 kept)
After the dataset rebuild (SPY timezone fix), all cross-asset features are populated.
The current model has **0 dropped features** — all 64 features are in `FEATURE_COLS`.
`dropped_features` in `v3_meta.json` will be an empty list `[]`.
If you pass features not in `FEATURE_COLS`, they are silently skipped by the dict lookup.

### SimpleImputer is INSIDE the XGB pipeline — do NOT pre-impute
The `v3_xgb_model.joblib` pipeline includes `SimpleImputer(strategy="median")`.
Call `predict_proba(X)` with `NaN` values for missing features; the imputer
will fill them with training-set medians automatically.

> The `medians` dict in `v3_meta.json` is for **manual / fallback inference only**
> (e.g. constructing feature rows outside sklearn). Do NOT apply it before calling
> `xgb_pipe.predict_proba()` — you would double-impute.

### Calibrator is a separate step — call it explicitly
The `v3_calibrator.joblib` (`IsotonicRegression`) is **not** part of the XGB
pipeline. The two-step sequence is mandatory:
```python
proba_raw = xgb_pipe.predict_proba(X)[:, 1]
score     = float(calibrator.predict(proba_raw)[0])
```
Skipping calibration produces poorly-calibrated probabilities (ECE ≈ 0.12 vs
0.00 with calibration) and invalidates the FPR-based thresholds.

### Thresholds are FPR-optimised, not business-tuned
`t_lo=0.4863` and `t_hi=0.6654` were chosen to hit target FPR rates (10% / 25%)
on the last fold's validation set. Review against your risk policy before
deploying to production — they are **not** Sharpe-optimised or cost-adjusted.

### Schema version check — assert `"3.1"` on load
Always assert `meta["schema_version"] == "3.1"` at startup. This guards against
accidentally loading a v2 bundle or a future incompatible schema. See the
loading snippet above.

### XGBoost >= 2.0 required
The pipeline was saved with XGBoost 2.x (`use_label_encoder` parameter removed).
Loading with XGBoost 1.x will raise a `ValueError`. Verify:
```bash
python3 -c "import xgboost; print(xgboost.__version__)"  # must be >= 2.0
```

---

## 5. Anti-Patterns

| Anti-pattern | Why it breaks |
|---|---|
| Hardcoding 50 (or 63) feature names | NaN filter is dynamic; future retrains may drop different features |
| Pre-filling NaN with medians before `predict_proba` | Double-imputation corrupts values; use raw NaN |
| Calling `calibrator.predict()` with `predict_proba` output reshaped to 2D | Calibrator expects 1D array |
| Using `v3_lr_model.joblib` for production scoring | LR is a baseline; lower AUC (0.765 vs 0.783) |
| Ignoring the `schema_version` field | Silent incompatibility with future artifact sets |
| Deploying `v3_metrics.json` to the container | Not needed; adds ~50 KB of JSON with no runtime benefit |

---

## 6. Verification Command

Before any deployment, run the release gate:

```bash
bash scripts/ml/validation/check_v3_release.sh
```

Expected output on success:
```
════════════════════════════════════
  ✅  V3 RELEASE — GO
  artifacts : 7/7
  models    : 3 loaded
  tests     : passed
════════════════════════════════════
```

Exit code `0` → GO. Exit code `1` → NO-GO; review output above the banner.
