# Reliability & Scientific Validation — v3/v4 Model

> **TL;DR** — All metrics computed on out-of-sample data only (expanding-window CV).
> No look-ahead bias. Full scientific report available as PDF.
>
> **v4** (current) adds bootstrap CI + statistical significance tests to the v3 baseline.

---

## Contents

- [Overview](#overview)
- [Version history](#version-history)
- [Methodology](#methodology)
- [Asset inventory](#asset-inventory)
- [Generating assets](#generating-assets)
- [Frontend integration](#frontend-integration)
- [Offline export (v3 and v4)](#offline-export)
- [Finder paths](#finder-paths)

---

## Overview

The v3 model (XGBoost + isotonic calibration) is validated using a strict
time-series cross-validation protocol to eliminate look-ahead bias.

Key results (out-of-sample, 5 folds):

| Metric                           | Value                     |
|----------------------------------|---------------------------|
| Fold 5 ROC-AUC (bootstrap mean)  | 0.787 [95% CI: 0.780, 0.795] |
| Fold 5 PR-AUC (bootstrap mean)   | 0.769 [95% CI: 0.759, 0.779] |
| XGB calibrated AUC (final)       | 0.782                     |
| ECE (Expected Calibration)       | 0.028                     |
| Signal Sharpe (ann.)             | 0.91                      |
| Always-OK Sharpe                 | 0.31                      |
| Max drawdown improvement         | ~30%                      |

---

## Version history

| Version | Date       | Content                                   | Script                              |
|---------|------------|-------------------------------------------|-------------------------------------|
| **v3**  | 2026-03-04 | 8 ML + 7 financial plots + PDF            | `export_credibility_assets.sh`      |
| **v4**  | 2026-03-04 | v3 + 4 robustness plots (bootstrap, fold 5) | `export_credibility_v4.sh`        |

**v4 additions** (new plots):
- `recent_fold_table.png` — Fold 5 detailed performance (most recent regime)
- `auc_bootstrap_hist.png` — 1,000 bootstrap CI on ROC-AUC + PR-AUC
- `sharpe_bootstrap_hist.png` — Bootstrap Sharpe significance test (signal vs baseline)
- `confusion_metrics_per_fold.png` — Recall / Precision / FPR / F1 per fold

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

### Bootstrap (v4)

1,000 bootstrap resamples with replacement on Fold 5 (12,291 out-of-sample observations).
95% CI = [2.5th, 97.5th] percentile of resample distribution.

---

## Asset inventory

### Statistical validation — 12 plots in v4 (`data/metrics/v3/plots/`)

| File                              | v3 | v4 | Description                                      |
|-----------------------------------|----|-----|--------------------------------------------------|
| `roc_curves.png`                  | ✓  | ✓   | ROC curves per fold + mean                       |
| `pr_curves.png`                   | ✓  | ✓   | Precision-Recall curves per fold                 |
| `calibration.png`                 | ✓  | ✓   | Reliability diagram (predicted vs actual)        |
| `prob_distributions.png`          | ✓  | ✓   | Score distributions by true label                |
| `lift_curve.png`                  | ✓  | ✓   | Cumulative lift over random baseline             |
| `confusion_matrices.png`          | ✓  | ✓   | Confusion matrices at production threshold       |
| `feature_importance.png`          | ✓  | ✓   | XGBoost feature importance (gain)                |
| `metrics_per_fold.png`            | ✓  | ✓   | AUC, AP, F1 stability across folds               |
| `recent_fold_table.png`           |    | ✓   | **NEW** Fold 5 detailed metrics table            |
| `auc_bootstrap_hist.png`          |    | ✓   | **NEW** Bootstrap CI: ROC-AUC + PR-AUC           |
| `sharpe_bootstrap_hist.png`       |    | ✓   | **NEW** Bootstrap Sharpe significance test       |
| `confusion_metrics_per_fold.png`  |    | ✓   | **NEW** Recall / Precision / FPR / F1 per fold   |

### Financial validation — 7 plots (`data/metrics/v3/financial_plots/`)

| File                            | Description                              |
|---------------------------------|------------------------------------------|
| `cumulative_returns.png`        | Signal portfolio vs always-invested      |
| `drawdown.png`                  | Drawdown profile comparison              |
| `return_distributions.png`      | Forward return distributions by signal   |
| `skip_rate_rolling.png`         | Rolling skip rate (warn/block fraction)  |
| `rolling_sharpe.png`            | 90-day rolling Sharpe ratio              |
| `performance_by_asset_type.png` | Sharpe lift by asset category            |
| `backtest_metrics_card.png`     | Aggregated backtest summary card         |

### Scientific report

`data/metrics/v3/V3_Scientific_Report.pdf` — 1.90 MB, 11 sections:
cover, dataset stats, validation protocol, ML performance, calibration,
backtest, features, drift analysis, **most-recent fold**, **statistical significance**,
production-readiness verdict.

---

## Generating assets

### 1. Generate all plots + PDF (requires trained models in `models/v3/`)

```bash
# Step 1a — ML/statistical plots
python scripts/ml/reporting/plot_ml_v3.py

# Step 1b — Robustness plots (bootstrap CI, confusion metrics, fold 5 table)
python scripts/ml/reporting/plot_robustness_v3.py

# Step 1c — Financial plots
python scripts/ml/reporting/plot_financial_v3.py

# Step 1d — PDF report (11 sections, ~1.9 MB)
python scripts/ml/reporting/generate_v3_report.py --no_plots
```

### 2. Package for credibility page

```bash
# Export as v3 (8 stat + 7 finance = 16 assets)
python scripts/ml/reporting/export_v3_credibility_assets.py

# Export as v4 (12 stat + 7 finance = 20 assets) — RECOMMENDED
python scripts/ml/reporting/export_v3_credibility_assets.py --version v4

# Dry-run to verify sources before copying:
python scripts/ml/reporting/export_v3_credibility_assets.py --version v4 --dry-run
```

---

## Frontend integration

The React component is in `frontend/src/pages/Reliability.tsx`.

```bash
# Copy the versioned assets into your frontend public folder
cp -r build/credibility/v4/ frontend/public/credibility/v4/

# (or v3 for the original)
cp -r build/credibility/v3/ frontend/public/credibility/v3/
```

Then in your router config:
```tsx
import Reliability from "./pages/Reliability";
<Route path="/reliability" element={<Reliability />} />
```

Nav link in the "À propos" section:
```tsx
<NavLink to="/reliability">Fiabilité</NavLink>
```

See `frontend/src/pages/ReliabilityNavSnippet.md` for full integration details.

---

## Offline export

### v3 export (original — 16 assets)

```bash
bash scripts/utils/export_credibility_assets.sh            # use existing build
bash scripts/utils/export_credibility_assets.sh --rebuild  # rebuild then export
open ~/Asymetra_Exports/credibility_v3/index.html
```

### v4 export (robustness upgrade — 20 assets) — RECOMMENDED

```bash
bash scripts/utils/export_credibility_v4.sh            # use existing build/credibility/v4/
bash scripts/utils/export_credibility_v4.sh --rebuild  # rebuild v4 then export
open ~/Asymetra_Exports/credibility_v4/index.html
```

### Single command to regenerate v4 from scratch

```bash
cd "/Users/paul.nytts/Asymetra code/.claude/worktrees/gifted-einstein" && \
  /Users/paul.nytts/Asymetra\ code/.venv/bin/python3 \
    scripts/ml/reporting/plot_robustness_v3.py && \
  /Users/paul.nytts/Asymetra\ code/.venv/bin/python3 \
    scripts/ml/reporting/generate_v3_report.py --no_plots && \
  bash scripts/utils/export_credibility_v4.sh --rebuild
```

---

## Finder paths

| What                   | Path                                                            |
|------------------------|-----------------------------------------------------------------|
| Source plots (ML)      | `data/metrics/v3/plots/`                                       |
| Source plots (finance) | `data/metrics/v3/financial_plots/`                              |
| Source PDF             | `data/metrics/v3/V3_Scientific_Report.pdf`                     |
| Build v3               | `build/credibility/v3/`                                         |
| Build v4               | `build/credibility/v4/`                                         |
| Export v3 (Finder)     | `~/Asymetra_Exports/credibility_v3/`                            |
| Export v4 (Finder)     | `~/Asymetra_Exports/credibility_v4/`                            |
| React component        | `frontend/src/pages/Reliability.tsx`                            |
| Export script v3       | `scripts/utils/export_credibility_assets.sh`                    |
| Export script v4       | `scripts/utils/export_credibility_v4.sh`                        |
| Python packager        | `scripts/ml/reporting/export_v3_credibility_assets.py`          |

> **Finder shortcut:** `open ~/Asymetra_Exports/credibility_v4` opens the v4 folder in Finder.
> `open ~/Asymetra_Exports/credibility_v4/index.html` opens the static viewer in your browser.

---

*Last updated: 2026-03-04 — v4 robustness upgrade, gifted-einstein worktree*
