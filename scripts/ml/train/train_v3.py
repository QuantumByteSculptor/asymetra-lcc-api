"""
scripts/ml/train/train_v3.py
============================
Phase 3 — Training pipeline for v3 binary classifier (target_non_ok).

Trains:
  1. Logistic Regression baseline (calibrated with sigmoid)
  2. XGBoost classifier (calibrated with sigmoid/isotonic)
  (Optional ensemble stacking if XGB AUC > LR AUC by ≥ 1%)

Cross-validation:
  - Reads splits_manifest.json (output of split_v3_time.py)
  - Trains on each fold's train.jsonl, evaluates on val.jsonl
  - Aggregates metrics across folds

Metrics per fold:
  - ROC-AUC, PR-AUC, Brier score
  - ECE (Expected Calibration Error)
  - FPR @ TPR=0.80 constraint
  - Recall, Precision, F1 at threshold 0.5

Artifacts (in models/v3/):
  v3_lr_model.joblib        — LR pipeline (scaler + clf)
  v3_xgb_model.joblib       — XGBoost model
  v3_calibrator.joblib      — Calibrator fitted on full last-fold train
  v3_thresholds.json        — t_lo (warn) / t_hi (block) + metadata
  v3_metrics.json           — Fold metrics + aggregate

Usage:
    python scripts/ml/train/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3

No API / prod impact. Models saved in models/v3/ (parallel to prod).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("train_v3")


# ---------------------------------------------------------------------------
# Feature columns (numeric only, excludes meta and always-null)
# ---------------------------------------------------------------------------

# NOTE: corr_spy / beta_market / abs_corr_mkt are all-null in the *current* fold files
# (SPY download failed during the original build).  After rebuilding with the fixed
# download_spy_returns() in build_dataset_v3.py, remove these from _ALWAYS_NULL so
# the training pipeline picks them up automatically.
_ALWAYS_NULL = {"corr_spy", "beta_market", "abs_corr_mkt"}
_META_COLS   = {"asset_type", "market", "ticker", "market_proxy"}   # string fields → excluded from X

# Ordered feature list (determined at first fold, reused across all folds)
_FEAT_COLS: List[str] = []

# Known macro features that may be partially null → imputed with median
_MACRO_FEATS = {
    "vix_level", "vix_pct_60d", "rate_10y", "rate_2y", "term_spread",
    "credit_spread_hy", "credit_spread_ig", "vol_regime",
    "corr_vix",
}

# Recovery / drawdown features use sentinel -1.0 when undefined (price didn't recover).
# dd_duration also uses -1 sentinel.  These are NOT imputed — -1 is a valid signal value.
# recovery_defined (0/1 flag) is always numeric, never NaN.
_SENTINEL_FEATS = {"recovery_days", "recovery_per_dd", "dd_duration", "recovery_defined"}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_jsonl_fold(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a fold JSONL file.
    Returns (X, y, feat_names) where X is float32.
    """
    global _FEAT_COLS

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No records loaded from {path}")

    # Determine feature columns on first call
    if not _FEAT_COLS:
        sample_feat = records[0].get("features", {})
        _FEAT_COLS = sorted(
            k for k in sample_feat
            if k not in _META_COLS and k not in _ALWAYS_NULL
        )
        log.info("Feature columns: %d features", len(_FEAT_COLS))

    n = len(records)
    X = np.zeros((n, len(_FEAT_COLS)), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    for i, rec in enumerate(records):
        feats = rec.get("features", {})
        for j, col in enumerate(_FEAT_COLS):
            v = feats.get(col)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                X[i, j] = np.nan
            else:
                X[i, j] = float(v)
        y[i] = int(rec.get("target_non_ok", 0))

    return X, y, _FEAT_COLS


# ---------------------------------------------------------------------------
# Preprocessing (imputation + scaling)
# ---------------------------------------------------------------------------

def _build_preprocessor():
    """SimpleImputer (median) + StandardScaler pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_lr(class_weight="balanced"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def build_xgb(scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> Dict[str, Any]:
    """Compute classification metrics for binary target."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        confusion_matrix,
    )

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"label": label, "n": n, "warning": "single class"}

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc  = float(average_precision_score(y_true, y_prob))
    brier   = float(brier_score_loss(y_true, y_prob))

    # ECE (10 bins)
    ece = _ece(y_true, y_prob, n_bins=10)

    # FPR @ TPR >= 0.80
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_true, y_prob)
    fpr_at_80 = float(np.interp(0.80, tprs, fprs))

    # Metrics at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall    = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "label":         label,
        "n":             int(n),
        "n_pos":         int(n_pos),
        "n_neg":         int(n_neg),
        "pos_rate":      round(n_pos / n, 4),
        "roc_auc":       round(roc_auc, 4),
        "pr_auc":        round(pr_auc, 4),
        "brier":         round(brier, 4),
        "ece":           round(ece, 4),
        "fpr_at_tpr80":  round(fpr_at_80, 4),
        "recall_t05":    round(float(recall), 4),
        "precision_t05": round(float(precision), 4),
        "f1_t05":        round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        avg_pred = y_prob[mask].mean()
        ece += abs(avg_pred - frac_pos) * mask.sum() / n
    return float(ece)


def _aggregate_metrics(fold_metrics: List[Dict]) -> Dict[str, Any]:
    """Average numeric metrics across folds."""
    keys = ["roc_auc", "pr_auc", "brier", "ece", "fpr_at_tpr80",
            "recall_t05", "precision_t05", "f1_t05"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m and isinstance(m[k], float)]
        if vals:
            agg[k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[k + "_std"]  = round(float(np.std(vals)), 4)
    return agg


# ---------------------------------------------------------------------------
# Threshold optimisation (find t_lo / t_hi)
# ---------------------------------------------------------------------------

def optimise_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray,
    target_fpr_lo: float = 0.10,   # warn starts here
    target_fpr_hi: float = 0.25,   # block starts here
) -> Dict[str, float]:
    """
    Find t_lo (warn) and t_hi (block) thresholds by FPR targeting.
    - t_lo: lowest threshold where FPR ≤ target_fpr_lo
    - t_hi: lowest threshold where FPR ≤ target_fpr_hi
    These are heuristic; fine-tune per deployment.
    """
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_true, y_prob)
    # Find threshold where FPR just crosses targets
    t_lo = t_hi = 0.5  # fallback

    for fpr, tpr, t in zip(fprs, tprs, thresholds):
        if fpr <= target_fpr_lo:
            t_lo = float(t)
        if fpr <= target_fpr_hi:
            t_hi = float(t)

    # Swap if inverted (shouldn't happen but guard)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo

    return {
        "t_lo": round(t_lo, 4),
        "t_hi": round(t_hi, 4),
        "target_fpr_lo": target_fpr_lo,
        "target_fpr_hi": target_fpr_hi,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    manifest_path: Path,
    out_dir: Path,
    train_both: bool = True,
) -> Dict[str, Any]:
    """
    Main training loop. Consumes splits_manifest.json, loops folds,
    aggregates metrics, saves final model on last fold.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    log.info("Loaded manifest: %d folds from %s", len(splits), manifest_path)

    lr_fold_metrics: List[Dict] = []
    xgb_fold_metrics: List[Dict] = []

    last_train_path: Optional[Path] = None
    last_val_path:   Optional[Path] = None

    for fold in splits:
        fk = fold["fold"]
        train_path = Path(fold["train_jsonl"])
        val_path   = Path(fold["val_jsonl"])

        if not train_path.exists() or not val_path.exists():
            log.warning("Fold %d: files missing — skipping", fk)
            continue

        log.info("─── Fold %d: loading...", fk)
        t0 = time.perf_counter()
        X_train, y_train, feat_names = load_jsonl_fold(train_path)
        X_val,   y_val,   _          = load_jsonl_fold(val_path)
        log.info("  train=%d val=%d  pos_rate_train=%.2f%%  pos_rate_val=%.2f%%",
                 len(y_train), len(y_val),
                 100 * y_train.mean(), 100 * y_val.mean())

        scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

        # ---- LR ----
        if train_both:
            log.info("  Training LR...")
            lr = build_lr()
            lr.fit(X_train, y_train)
            lr_prob = lr.predict_proba(X_val)[:, 1]
            m_lr = compute_metrics(y_val, lr_prob, label=f"lr_fold{fk}")
            lr_fold_metrics.append(m_lr)
            log.info("  LR  → ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
                     m_lr["roc_auc"], m_lr["pr_auc"], m_lr["brier"])

        # ---- XGB ----
        log.info("  Training XGB...")
        xgb = build_xgb(scale_pos_weight=scale_pos)
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        m_xgb = compute_metrics(y_val, xgb_prob, label=f"xgb_fold{fk}")
        xgb_fold_metrics.append(m_xgb)
        log.info("  XGB → ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
                 m_xgb["roc_auc"], m_xgb["pr_auc"], m_xgb["brier"])

        last_train_path = train_path
        last_val_path   = val_path
        log.info("  Fold %d done in %.1fs", fk, time.perf_counter() - t0)

    if not xgb_fold_metrics:
        raise RuntimeError("No folds completed successfully")

    # ----- Aggregate metrics -----
    lr_agg  = _aggregate_metrics(lr_fold_metrics)
    xgb_agg = _aggregate_metrics(xgb_fold_metrics)

    log.info("=== AGGREGATE METRICS ===")
    log.info("LR  ROC-AUC: %.4f ± %.4f", lr_agg.get("roc_auc_mean", 0), lr_agg.get("roc_auc_std", 0))
    log.info("XGB ROC-AUC: %.4f ± %.4f", xgb_agg.get("roc_auc_mean", 0), xgb_agg.get("roc_auc_std", 0))

    # ----- Retrain final model on last fold's train (largest set) -----
    log.info("Retraining final model on last fold train set...")
    X_final, y_final, feat_names = load_jsonl_fold(last_train_path)
    X_val_f, y_val_f, _          = load_jsonl_fold(last_val_path)

    scale_pos_final = float((y_final == 0).sum()) / max(float((y_final == 1).sum()), 1)

    # Final LR
    lr_final = build_lr()
    lr_final.fit(X_final, y_final)

    # Final XGB
    xgb_final = build_xgb(scale_pos_weight=scale_pos_final)
    xgb_final.fit(X_final, y_final)

    # Calibration on val
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline as SKPipeline

    # Use XGB's raw probabilities for calibration
    xgb_raw_prob = xgb_final.predict_proba(X_val_f)[:, 1]

    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(xgb_raw_prob, y_val_f)
    cal_prob = calibrator.predict(xgb_raw_prob)

    m_cal = compute_metrics(y_val_f, cal_prob, label="xgb_calibrated_final")
    log.info("Final XGB calibrated → ROC-AUC=%.4f  ECE=%.4f", m_cal["roc_auc"], m_cal["ece"])

    # Thresholds
    thresholds_info = optimise_thresholds(y_val_f, cal_prob)
    thresholds_info["fitted_on"] = "last_fold_val"
    thresholds_info["model"]     = "xgb_calibrated"
    log.info("Thresholds: t_lo=%.4f t_hi=%.4f", thresholds_info["t_lo"], thresholds_info["t_hi"])

    # Feature importance (XGB)
    xgb_clf = xgb_final["clf"]
    feat_imp = dict(zip(feat_names, xgb_clf.feature_importances_.tolist()))
    top20_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1])[:20])

    # ----- Save artifacts -----
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_final,  out_dir / "v3_lr_model.joblib",  compress=3)
    joblib.dump(xgb_final, out_dir / "v3_xgb_model.joblib", compress=3)
    joblib.dump(calibrator, out_dir / "v3_calibrator.joblib", compress=3)
    joblib.dump(feat_names, out_dir / "v3_feature_names.joblib", compress=1)

    thresholds_out = {
        "generated_at": datetime.now().isoformat(),
        **thresholds_info,
    }
    (out_dir / "v3_thresholds.json").write_text(
        json.dumps(thresholds_out, indent=2), encoding="utf-8"
    )

    metrics_out = {
        "generated_at":    datetime.now().isoformat(),
        "manifest":        str(manifest_path),
        "n_folds":         len(xgb_fold_metrics),
        "n_features":      len(feat_names),
        "lr": {
            "fold_metrics": lr_fold_metrics,
            "aggregate":    lr_agg,
        },
        "xgb": {
            "fold_metrics":  xgb_fold_metrics,
            "aggregate":     xgb_agg,
            "final_calibrated": m_cal,
        },
        "feature_importance_top20": top20_imp,
        "thresholds": thresholds_out,
    }
    (out_dir / "v3_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Artifacts saved to %s", out_dir)
    return metrics_out


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(metrics: Dict[str, Any]) -> None:
    lr_agg  = metrics["lr"]["aggregate"]
    xgb_agg = metrics["xgb"]["aggregate"]
    m_cal   = metrics["xgb"]["final_calibrated"]
    thr     = metrics["thresholds"]

    print(f"\n{'='*62}")
    print(f"  TRAIN V3 — RESULTS SUMMARY")
    print(f"{'='*62}")
    print(f"  Folds trained       : {metrics['n_folds']}")
    print(f"  Features            : {metrics['n_features']}")
    print(f"")
    print(f"  {'Model':<28} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7}")
    print(f"  {'─'*54}")
    print(f"  {'LR (mean ± std)':<28} "
          f"{lr_agg.get('roc_auc_mean', 0):>7.4f}  "
          f"{lr_agg.get('pr_auc_mean', 0):>7.4f}  "
          f"{lr_agg.get('brier_mean', 0):>6.4f}")
    print(f"  {'XGB (mean ± std)':<28} "
          f"{xgb_agg.get('roc_auc_mean', 0):>7.4f}  "
          f"{xgb_agg.get('pr_auc_mean', 0):>7.4f}  "
          f"{xgb_agg.get('brier_mean', 0):>6.4f}")
    print(f"  {'XGB+Calibrated (final)':<28} "
          f"{m_cal.get('roc_auc', 0):>7.4f}  "
          f"{m_cal.get('pr_auc', 0):>7.4f}  "
          f"{m_cal.get('brier', 0):>6.4f}")
    print(f"")
    print(f"  Thresholds: t_lo={thr['t_lo']} (warn)  t_hi={thr['t_hi']} (block)")

    top5 = list(metrics["feature_importance_top20"].items())[:5]
    print(f"  Top-5 features: {top5}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Train v3 binary classifier (target_non_ok)")
    ap.add_argument("--manifest", required=True,
                    help="Path to splits_manifest.json (from split_v3_time.py)")
    ap.add_argument("--out_dir", default="models/v3",
                    help="Output directory for artifacts (default: models/v3)")
    ap.add_argument("--no_lr", action="store_true",
                    help="Skip LR baseline (train XGB only)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir       = Path(args.out_dir)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    metrics = train(
        manifest_path=manifest_path,
        out_dir=out_dir,
        train_both=not args.no_lr,
    )

    print_summary(metrics)
    log.info("Done. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
