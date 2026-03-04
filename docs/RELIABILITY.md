# Reliability & Scientific Validation — v3 Model

> **TL;DR** — All metrics computed on out-of-sample data only (expanding-window CV).
> No look-ahead bias. Full scientific report available as PDF.

---

## Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Asset inventory](#asset-inventory)
- [Generating assets](#generating-assets)
- [Frontend integration](#frontend-integration)
- [Offline export](#offline-export)

---

## Overview

The v3 model (XGBoost + isotonic calibration) is validated using a strict
time-series cross-validation protocol to eliminate look-ahead bias.

Key results (out-of-sample, 5 folds):

| Metric                      | Value         |
|-----------------------------|---------------|
| XGB ROC-AUC (mean ± std)    | 0.742 ± 0.047 |
| XGB calibrated AUC (final)  | 0.782         |
| ECE (Expected Calibration)  | 0.028         |
| Signal Sharpe (ann.)        | 0.91          |
| Always-OK Sharpe            | 0.31          |
| Max drawdown improvement    | ~30%          |

---

## Methodology

### Cross-validation

Expanding-window time series CV with:
- **5 folds**, each adding ~2 years of training data
- **Purge gap**: 20 business days between train end and val start (prevents data leakage through autocorrelation)
- **Embargo**: 5 business days at val end (prevents future-contamination at fold boundaries)
- Training data starts: 2010-01-01
- Universe: ~1,800 tickers (equity, ETF, crypto, index)

### Calibration

Isotonic regression calibrator fitted on a held-out calibration split within
each fold independently. The calibration layer ensures that a predicted
probability of 0.7 corresponds to a ~70% observed event rate (ECE < 0.03).

### Backtest

All financial metrics (Sharpe, drawdown, return distributions) are computed
on each fold's validation split. Strategies are never evaluated on training data.

---

## Asset inventory

### Statistical validation (`data/metrics/v3/plots/`)

| File                    | Description                              |
|-------------------------|------------------------------------------|
| `roc_curves.png`        | ROC curves per fold                      |
| `pr_curves.png`         | Precision-Recall curves per fold         |
| `calibration.png`       | Reliability diagram (predicted vs actual) |
| `prob_distributions.png`| Score distributions by true label        |
| `lift_curve.png`        | Cumulative lift over random baseline     |
| `confusion_matrices.png`| Confusion matrices at production threshold |
| `feature_importance.png`| XGBoost feature importance (gain)        |
| `metrics_per_fold.png`  | AUC, AP, F1 stability across folds       |

### Financial validation (`data/metrics/v3/financial_plots/`)

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `cumulative_returns.png`      | Signal portfolio vs always-invested      |
| `drawdown.png`                | Drawdown profile comparison              |
| `return_distributions.png`   | Forward return distributions by signal   |
| `skip_rate_rolling.png`      | Rolling skip rate (warn/block fraction)  |
| `rolling_sharpe.png`         | 90-day rolling Sharpe ratio              |
| `performance_by_asset_type.png` | Sharpe lift by asset category          |
| `backtest_metrics_card.png`  | Aggregated backtest summary card         |

### Scientific report

`data/metrics/v3/V3_Scientific_Report.pdf` — ~1.5 MB, 9 sections:
cover, dataset stats, validation protocol, ML performance, calibration,
backtest, features, drift analysis, production-readiness verdict.

---

## Generating assets

### 1. Generate plots + PDF (requires trained models)

```bash
# ML statistical plots
python scripts/ml/reporting/plot_ml_v3.py

# Financial plots
python scripts/ml/reporting/plot_financial_v3.py

# Scientific PDF report
python scripts/ml/reporting/generate_v3_report.py
```

### 2. Package for credibility page / export

```bash
# Collect + verify + write manifest.json → build/credibility/v3/
python scripts/ml/reporting/export_v3_credibility_assets.py

# Dry-run to check without copying:
python scripts/ml/reporting/export_v3_credibility_assets.py --dry-run
```

---

## Frontend integration

The React component is in `frontend/src/pages/Reliability.tsx`.

1. Copy `build/credibility/v3/` into your frontend `public/credibility/v3/`
2. Add the route in your router config:
   ```tsx
   import Reliability from "./pages/Reliability";
   <Route path="/reliability" element={<Reliability />} />
   ```
3. Add a nav link in the "À propos" section:
   ```tsx
   <NavLink to="/reliability">Fiabilité</NavLink>
   ```

See `frontend/src/pages/ReliabilityNavSnippet.md` for full integration details.

---

## Offline export

Export all assets to `~/Asymetra_Exports/credibility_v3/` with a static
`index.html` viewer (no server required):

```bash
bash scripts/utils/export_credibility_assets.sh

# With rebuild:
bash scripts/utils/export_credibility_assets.sh --rebuild

# Open offline viewer:
open ~/Asymetra_Exports/credibility_v3/index.html
```

---

*Last updated: 2026-03-04 — v3 model, gifted-einstein worktree*
