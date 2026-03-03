# Release Checklist — v3.1.0

> Proposed tag: **`v3.1.0`** (mirrors schema_version 3.1)
> Branch: `feat/v3-quant-pipeline`

---

## Changelog

1. **Full v3 pipeline**: build → QA → split → train → backtest → drift monitoring
2. **Dataset rebuild**: corr_spy/beta_market/corr_vix timezone fix — 0 features dropped (64 kept, vs 50 before)
3. **XGB calibrated** (IsotonicRegression): ROC-AUC=0.787, Brier=0.187, ECE=0.000
4. **Thresholds FPR-based**: `t_lo=0.5203` (warn, FPR≤10%), `t_hi=0.6667` (block, FPR≤25%)
5. **Backtest**: signal Sharpe=0.90 vs benchmark 0.31 (+5bps cost)

---

## Pre-Release Checklist

### Automated gate (required)
- [ ] `bash scripts/ml/validation/check_v3_release.sh` exits **0**
  - [ ] Step 1: 7/7 artifacts present in `models/v3/`
  - [ ] Step 2: smoke load passes (3 models, schema_version=3.1, 0 < t_lo < t_hi < 1)
  - [ ] Step 3: `test_train_v3_smoke`, `test_qa_v3`, `test_split_v3`, `test_nan_preprocessing` — all green

### Manual review (required before prod)
- [ ] `v3_meta.json` — `schema_version == "3.1"`, `n_features == 50`, `dropped_features` list correct
- [ ] `v3_thresholds.json` — `t_lo` / `t_hi` reviewed against risk policy (currently FPR-optimised, not business-tuned)
- [ ] XGBoost version in container/venv `>= 2.0`
- [ ] API team has read `docs/V3_INTEGRATION_NOTES.md` (especially §4 Known Pitfalls)

### Deployment (API team)
- [ ] Copy 4 files to API container: `v3_xgb_model.joblib`, `v3_calibrator.joblib`, `v3_meta.json`, `v3_thresholds.json`
- [ ] Assert `schema_version == "3.1"` at API startup (fail fast)
- [ ] Smoke test endpoint with a known fixture and verify score/label plausible

### Post-deploy monitoring
- [ ] Log `score`, `label`, `t_lo`, `t_hi`, `schema_version` in shadow logs
- [ ] Run drift check within 7 days of prod traffic: `scripts/ml/monitoring/drift_v3.py`
- [ ] Schedule threshold review after 30 days of prod data

---

## Artifacts NOT deployed to API

| File | Reason |
|------|--------|
| `v3_lr_model.joblib` | LR baseline only — not used in prod scoring |
| `v3_feature_names.joblib` | Redundant — `feature_cols` already in `v3_meta.json` |
| `v3_metrics.json` | Evaluation artifact — no runtime use |

---

## Rollback

If GO-LIVE fails:

1. Revert API to previous bundle (`models/unsup_bundle.joblib` + `models/sup_bundle.joblib`)
2. Do NOT delete `models/v3/` — regenerate from `scripts/ml/train/train_v3.py` if needed
3. File issue with CI/CD output and `check_v3_release.sh` banner
