"""
scripts/ml/train/train_v3.py
============================
Phase 3 — Training pipeline for v3 binary classifier (target_non_ok).

Single entrypoint for all v3 training. Reads a splits_manifest.json
produced by split_v3_time.py — no implicit splits, fully reproducible.

Trains:
  1. Logistic Regression baseline (SimpleImputer + StandardScaler + LR)
  2. XGBoost classifier (SimpleImputer + XGBClassifier)
  (Optional --no_lr to skip LR and train XGB only)

NaN handling (finance-grade):
  - Features with NaN rate > --nan_drop_threshold (default 0.30) are DROPPED
    and logged explicitly with their NaN %. This replaces the old _ALWAYS_NULL
    hardcode and correctly handles ALL sparse features dynamically.
  - Remaining NaN → SimpleImputer(strategy="median") in the sklearn Pipeline.
  - Sentinel-valued features (recovery_days=-1 when undefined) pass through as-is.

Cross-validation:
  - Reads splits_manifest.json (output of split_v3_time.py)
  - Trains on each fold's train.jsonl, evaluates on val.jsonl
  - Aggregates metrics across folds
  - Per-fold signal backtest (proba_non_ok < threshold → invest)

Metrics per fold:
  - ROC-AUC, PR-AUC, Brier score
  - ECE (Expected Calibration Error)
  - FPR @ TPR=0.80 constraint
  - Recall, Precision, F1 at threshold 0.5
  - Signal backtest: skip_rate, sharpe_proxy (annualised 20d periods)

Artifacts (in --out_dir, default models/v3/):
  v3_lr_model.joblib        — LR pipeline (imputer + scaler + clf)
  v3_xgb_model.joblib       — XGBoost pipeline (imputer + clf)
  v3_calibrator.joblib      — IsotonicRegression calibrator (fitted on last-fold val)
  v3_feature_names.joblib   — Ordered feature list used in training
  v3_thresholds.json        — t_lo (warn) / t_hi (block) + metadata
  v3_meta.json              — Consolidated: feature_cols, medians, thresholds, schema
  v3_metrics.json           — Fold metrics + aggregate (legacy, backward-compat)

Canonical metrics output:
  data/metrics/train_v3_report.json — same content as v3_metrics.json, new canonical path

Usage:
    # Full run (all folds)
    python scripts/ml/train/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3

    # Smoke run (first 2000 rows per fold, logistic only)
    python scripts/ml/train/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3 \\
        --max_rows 2000 \\
        --no_lr

    # Skip LR (XGB only, faster)
    python scripts/ml/train/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3 --no_lr

    # Aggressive NaN dropping (drop features with >10% NaN)
    python scripts/ml/train/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3 --nan_drop_threshold 0.10

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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger("train_v3")

SCHEMA_VERSION = "3.1"

# String fields — excluded from feature matrix X
_META_COLS = {"asset_type", "market", "ticker", "market_proxy"}

# Non-feature target/forward-return columns
_TARGET_COLS = {
    "target_non_ok", "label", "forward_return_20d", "forward_return_5d",
    "forward_return_10d", "forward_return_60d", "future_dd_20d", "future_vol_ratio",
}

# Annualised Sharpe factor for 20-day holding periods
_PERIODS_PER_YEAR = 252 / 20


# ─────────────────────────────────────────────────────────────────────────────
# Feature resolution — dynamic NaN detection (replaces _ALWAYS_NULL hardcode)
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_feat_cols(features_sample: Dict[str, Any]) -> List[str]:
    """
    Extract all numeric feature keys from one sample features dict,
    excluding meta/string columns.
    """
    return sorted(
        k for k, v in features_sample.items()
        if k not in _META_COLS
        and not isinstance(v, str)
    )


def filter_high_nan_cols(
    feat_cols: List[str],
    nan_rates: Dict[str, float],
    threshold: float = 0.30,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Return (kept_cols, dropped_cols_with_rates).
    Drops any feature with NaN rate > threshold and logs it clearly.
    """
    kept: List[str] = []
    dropped: List[Tuple[str, float]] = []

    for col in feat_cols:
        rate = nan_rates.get(col, 0.0)
        if rate > threshold:
            dropped.append((col, rate))
        else:
            kept.append(col)

    if dropped:
        log.warning(
            "NaN filter (threshold=%.0f%%): dropping %d/%d features:",
            threshold * 100, len(dropped), len(feat_cols),
        )
        for col, rate in sorted(dropped, key=lambda x: -x[1]):
            log.warning("  DROPPED %-35s  %.1f%% NaN", col, rate * 100)
    else:
        log.info("NaN filter: all %d features below %.0f%% threshold — none dropped",
                 len(feat_cols), threshold * 100)

    log.info("Features after NaN filter: %d / %d kept", len(kept), len(feat_cols))
    return kept, dropped


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_fold(
    path: Path,
    feat_cols: List[str],
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a fold JSONL file into (X, y) given explicit feat_cols.
    max_rows limits lines read (smoke / dev mode).
    Returns X as float32 with NaN where values are missing/non-finite.
    """
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No records in {path}")

    n = len(records)
    X = np.full((n, len(feat_cols)), np.nan, dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    for i, rec in enumerate(records):
        feats = rec.get("features", {})
        for j, col in enumerate(feat_cols):
            v = feats.get(col)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                X[i, j] = float(v)
            # else: leave as np.nan (includes None, NaN, Inf)
        y[i] = int(rec.get("target_non_ok", 0))

    return X, y


def probe_nan_rates(
    path: Path,
    feat_cols: List[str],
    max_rows: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute NaN fraction per feature column from a JSONL file.
    Used to determine which columns to drop before training.
    """
    X, _ = load_jsonl_fold(path, feat_cols, max_rows=max_rows)
    n = X.shape[0]
    if n == 0:
        return {col: 1.0 for col in feat_cols}
    nan_counts = np.isnan(X).sum(axis=0)
    return {col: float(nan_counts[j] / n) for j, col in enumerate(feat_cols)}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_preprocessor():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def build_lr(seed: int = 42) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=seed,
        )),
    ])


def build_xgb(scale_pos_weight: float = 1.0, seed: int = 42, n_estimators: int = 400) -> Any:
    from sklearn.pipeline import Pipeline
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.warning("xgboost not available — falling back to HistGradientBoostingClassifier")
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline as _P
        return _P([
            ("pre", _build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=n_estimators, max_depth=4, learning_rate=0.05,
                min_samples_leaf=10, class_weight="balanced", random_state=seed,
            )),
        ])

    # Handle use_label_encoder removal in XGBoost >= 2.0
    xgb_params: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
    )
    try:
        import xgboost as _x
        if int(_x.__version__.split(".")[0]) < 2:
            xgb_params["use_label_encoder"] = False
    except Exception:
        pass

    return Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", XGBClassifier(**xgb_params)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> Dict[str, Any]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        confusion_matrix, roc_curve,
    )

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"label": label, "n": n, "warning": "single class in val"}

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc  = float(average_precision_score(y_true, y_prob))
    brier   = float(brier_score_loss(y_true, y_prob))
    ece     = _ece(y_true, y_prob)

    fprs, tprs, _ = roc_curve(y_true, y_prob)
    fpr_at_80 = float(np.interp(0.80, tprs, fprs))

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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _aggregate_metrics(fold_metrics: List[Dict]) -> Dict[str, Any]:
    keys = ["roc_auc", "pr_auc", "brier", "ece", "fpr_at_tpr80",
            "recall_t05", "precision_t05", "f1_t05"]
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if isinstance(m.get(k), float)]
        if vals:
            agg[k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[k + "_std"]  = round(float(np.std(vals)), 4)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation (warn / block)
# ─────────────────────────────────────────────────────────────────────────────

def optimise_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr_lo: float = 0.10,
    target_fpr_hi: float = 0.25,
) -> Dict[str, float]:
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_true, y_prob)
    t_lo = t_hi = 0.5

    for fpr, tpr, t in zip(fprs, tprs, thresholds):
        if fpr <= target_fpr_lo:
            t_lo = float(t)
        if fpr <= target_fpr_hi:
            t_hi = float(t)

    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo

    return {
        "t_lo":           round(t_lo, 4),
        "t_hi":           round(t_hi, 4),
        "target_fpr_lo":  target_fpr_lo,
        "target_fpr_hi":  target_fpr_hi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold signal backtest
# ─────────────────────────────────────────────────────────────────────────────

def _fold_signal_backtest(
    y_prob: np.ndarray,
    val_path: Path,
    threshold: float = 0.5,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quick signal backtest: risk_on if proba_non_ok < threshold.
    Loads forward_return_20d from val JSONL and computes:
      skip_rate, strategy_mean_return, baseline_mean_return, sharpe_proxy.
    """
    fwd: List[Optional[float]] = []
    with val_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fwd.append(rec.get("forward_return_20d"))
            except json.JSONDecodeError:
                fwd.append(None)

    if len(fwd) != len(y_prob):
        return {"error": f"length mismatch fwd={len(fwd)} prob={len(y_prob)}"}

    fwd_arr = np.array(
        [x if (x is not None and math.isfinite(x)) else float("nan")
         for x in fwd],
        dtype=float,
    )

    valid   = np.isfinite(fwd_arr)
    risk_on = y_prob < threshold

    n_total   = int(valid.sum())
    n_invested = int((risk_on & valid).sum())
    skip_rate = round(float((~risk_on)[valid].mean()), 4) if n_total > 0 else None

    inv_rets = fwd_arr[risk_on & valid]
    all_rets = fwd_arr[valid]

    strat_mean   = round(float(inv_rets.mean()), 6) if len(inv_rets) > 0 else None
    baseline_mean = round(float(all_rets.mean()), 6) if len(all_rets) > 0 else None

    sharpe = None
    if len(inv_rets) > 1 and np.std(inv_rets) > 1e-12:
        sharpe = round(
            float(np.mean(inv_rets) / np.std(inv_rets, ddof=1))
            * math.sqrt(_PERIODS_PER_YEAR),
            4,
        )

    # FP rate: invested despite truly non_ok
    # We can compute from y_prob vs the actual labels (not val_path)
    # This is a proxy — exact FP requires the actual val labels

    return {
        "threshold":             threshold,
        "n_invested":            n_invested,
        "n_skipped":             n_total - n_invested,
        "skip_rate":             skip_rate,
        "strategy_mean_return":  strat_mean,
        "baseline_mean_return":  baseline_mean,
        "sharpe_proxy":          sharpe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    manifest_path: Path,
    out_dir: Path,
    nan_drop_threshold: float = 0.30,
    max_rows: Optional[int] = None,
    train_both: bool = True,
    seed: int = 42,
    n_estimators: int = 400,
) -> Dict[str, Any]:
    """
    Main training loop.
    Reads splits_manifest.json, loops folds, aggregates metrics, saves artifacts.

    NaN drop threshold:
      Features where NaN rate (on first fold's train) > nan_drop_threshold are
      excluded from training. Logged at WARNING level with their NaN %.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    log.info("Manifest loaded: %d folds from %s", len(splits), manifest_path)
    if max_rows:
        log.info("Smoke mode: --max_rows=%d (rows per fold file)", max_rows)

    # ── Step 1: Determine feature columns from first available fold ────────────
    first_fold = splits[0]
    first_train_path = Path(first_fold["train_jsonl"])
    if not first_train_path.exists():
        raise FileNotFoundError(f"First fold train not found: {first_train_path}")

    # Get candidate feature names from first record
    with first_train_path.open("r", encoding="utf-8") as f:
        sample_rec = json.loads(f.readline().strip())
    candidate_cols = _candidate_feat_cols(sample_rec.get("features", {}))
    log.info("Candidate feature columns: %d", len(candidate_cols))

    # Probe NaN rates on first fold (use max_rows if smoke mode)
    log.info("Probing NaN rates on first fold train: %s ...", first_train_path.name)
    nan_rates = probe_nan_rates(first_train_path, candidate_cols, max_rows=max_rows)

    feat_cols, dropped_features = filter_high_nan_cols(
        candidate_cols, nan_rates, threshold=nan_drop_threshold,
    )

    if not feat_cols:
        raise RuntimeError(
            f"All {len(candidate_cols)} candidate features dropped "
            f"(nan_drop_threshold={nan_drop_threshold:.0%}). "
            "Lower the threshold or fix the dataset."
        )

    # ── Step 2: Training loop ─────────────────────────────────────────────────
    lr_fold_metrics:  List[Dict] = []
    xgb_fold_metrics: List[Dict] = []

    last_train_path: Optional[Path] = None
    last_val_path:   Optional[Path] = None

    for fold in splits:
        fk         = fold["fold"]
        train_path = Path(fold["train_jsonl"])
        val_path   = Path(fold["val_jsonl"])

        if not train_path.exists() or not val_path.exists():
            log.warning("Fold %d: files missing — skipping", fk)
            continue

        log.info("─── Fold %d: loading...", fk)
        t0 = time.perf_counter()

        X_train, y_train = load_jsonl_fold(train_path, feat_cols, max_rows=max_rows)
        X_val,   y_val   = load_jsonl_fold(val_path,   feat_cols, max_rows=max_rows)

        log.info(
            "  train=%d val=%d  pos_rate_train=%.2f%%  pos_rate_val=%.2f%%",
            len(y_train), len(y_val),
            100 * y_train.mean(), 100 * y_val.mean(),
        )

        scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

        # ── LR ──────────────────────────────────────────────────────────────
        if train_both:
            log.info("  Training LR...")
            lr = build_lr(seed=seed)
            lr.fit(X_train, y_train)
            lr_prob = lr.predict_proba(X_val)[:, 1]
            m_lr = compute_metrics(y_val, lr_prob, label=f"lr_fold{fk}")
            m_lr["backtest"] = _fold_signal_backtest(
                lr_prob, val_path, threshold=0.5, max_rows=max_rows,
            )
            lr_fold_metrics.append(m_lr)
            log.info("  LR  → ROC=%.4f PR=%.4f Brier=%.4f skip=%.1f%%",
                     m_lr.get("roc_auc", 0), m_lr.get("pr_auc", 0),
                     m_lr.get("brier", 0),
                     100 * (m_lr["backtest"].get("skip_rate") or 0))

        # ── XGB ─────────────────────────────────────────────────────────────
        log.info("  Training XGB...")
        xgb_model = build_xgb(scale_pos_weight=scale_pos, seed=seed,
                               n_estimators=n_estimators)
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
        m_xgb = compute_metrics(y_val, xgb_prob, label=f"xgb_fold{fk}")
        m_xgb["backtest"] = _fold_signal_backtest(
            xgb_prob, val_path, threshold=0.5, max_rows=max_rows,
        )
        xgb_fold_metrics.append(m_xgb)
        log.info("  XGB → ROC=%.4f PR=%.4f Brier=%.4f skip=%.1f%%",
                 m_xgb.get("roc_auc", 0), m_xgb.get("pr_auc", 0),
                 m_xgb.get("brier", 0),
                 100 * (m_xgb["backtest"].get("skip_rate") or 0))

        last_train_path = train_path
        last_val_path   = val_path
        log.info("  Fold %d done in %.1fs", fk, time.perf_counter() - t0)

    if not xgb_fold_metrics:
        raise RuntimeError("No folds completed — check manifest paths")

    # ── Step 3: Aggregate ─────────────────────────────────────────────────────
    lr_agg  = _aggregate_metrics(lr_fold_metrics)
    xgb_agg = _aggregate_metrics(xgb_fold_metrics)

    log.info("=== AGGREGATE METRICS ===")
    if train_both:
        log.info("LR  ROC-AUC: %.4f ± %.4f",
                 lr_agg.get("roc_auc_mean", 0), lr_agg.get("roc_auc_std", 0))
    log.info("XGB ROC-AUC: %.4f ± %.4f",
             xgb_agg.get("roc_auc_mean", 0), xgb_agg.get("roc_auc_std", 0))

    # ── Step 4: Final refit on last fold's train ──────────────────────────────
    log.info("Retraining final model on last fold: %s", last_train_path.name)
    X_final, y_final = load_jsonl_fold(last_train_path, feat_cols, max_rows=max_rows)
    X_val_f, y_val_f = load_jsonl_fold(last_val_path,   feat_cols, max_rows=max_rows)

    # Compute medians on final train set (for v3_meta.json)
    col_medians: Dict[str, Optional[float]] = {}
    for j, col in enumerate(feat_cols):
        col_data = X_final[:, j]
        valid = col_data[~np.isnan(col_data)]
        col_medians[col] = round(float(np.median(valid)), 6) if len(valid) > 0 else None

    scale_pos_final = float((y_final == 0).sum()) / max(float((y_final == 1).sum()), 1)

    lr_final = build_lr(seed=seed)
    lr_final.fit(X_final, y_final)

    xgb_final = build_xgb(scale_pos_weight=scale_pos_final, seed=seed,
                           n_estimators=n_estimators)
    xgb_final.fit(X_final, y_final)

    # Calibration (isotonic) on last fold val
    from sklearn.isotonic import IsotonicRegression

    xgb_raw_prob = xgb_final.predict_proba(X_val_f)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(xgb_raw_prob, y_val_f)
    cal_prob = calibrator.predict(xgb_raw_prob)

    m_cal = compute_metrics(y_val_f, cal_prob, label="xgb_calibrated_final")
    log.info("Final XGB calibrated → ROC=%.4f  ECE=%.4f", m_cal["roc_auc"], m_cal["ece"])

    # Thresholds
    thresholds_info = optimise_thresholds(y_val_f, cal_prob)
    thresholds_info["fitted_on"] = "last_fold_val"
    thresholds_info["model"]     = "xgb_calibrated"
    log.info("Thresholds: t_lo=%.4f t_hi=%.4f",
             thresholds_info["t_lo"], thresholds_info["t_hi"])

    # Feature importance
    feat_imp_list: Dict[str, float] = {}
    try:
        xgb_clf = xgb_final[-1]   # last step of Pipeline
        feat_imp_list = dict(zip(feat_cols, xgb_clf.feature_importances_.tolist()))
    except Exception:
        pass
    top20_imp = dict(sorted(feat_imp_list.items(), key=lambda x: -x[1])[:20])

    # ── Step 5: Save artifacts ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_final,   out_dir / "v3_lr_model.joblib",  compress=3)
    joblib.dump(xgb_final,  out_dir / "v3_xgb_model.joblib", compress=3)
    joblib.dump(calibrator, out_dir / "v3_calibrator.joblib", compress=3)
    joblib.dump(feat_cols,  out_dir / "v3_feature_names.joblib", compress=1)

    thresholds_out = {"generated_at": datetime.now().isoformat(), **thresholds_info}
    (out_dir / "v3_thresholds.json").write_text(
        json.dumps(thresholds_out, indent=2), encoding="utf-8"
    )

    # v3_meta.json — new consolidated metadata for inference
    dropped_feat_names = [col for col, _ in dropped_features]
    meta_out = {
        "schema_version":       SCHEMA_VERSION,
        "generated_at":         datetime.now().isoformat(),
        "manifest":             str(manifest_path),
        "nan_drop_threshold":   nan_drop_threshold,
        "n_features":           len(feat_cols),
        "feature_cols":         feat_cols,
        "medians":              col_medians,
        "n_dropped_features":   len(dropped_features),
        "dropped_features":     [
            {"col": col, "nan_rate": round(rate, 4)}
            for col, rate in sorted(dropped_features, key=lambda x: -x[1])
        ],
        "thresholds":           thresholds_out,
        "calibration":          "isotonic",
    }
    (out_dir / "v3_meta.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # v3_metrics.json — backward compat
    metrics_out = {
        "generated_at":    datetime.now().isoformat(),
        "manifest":        str(manifest_path),
        "n_folds":         len(xgb_fold_metrics),
        "n_features":      len(feat_cols),
        "n_dropped":       len(dropped_features),
        "dropped_features": dropped_feat_names,
        "lr": {
            "fold_metrics": lr_fold_metrics,
            "aggregate":    lr_agg,
        },
        "xgb": {
            "fold_metrics":     xgb_fold_metrics,
            "aggregate":        xgb_agg,
            "final_calibrated": m_cal,
        },
        "feature_importance_top20": top20_imp,
        "thresholds": thresholds_out,
    }
    (out_dir / "v3_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # train_v3_report.json — new canonical metrics path
    metrics_report_path = _REPO_ROOT / "data" / "metrics" / "train_v3_report.json"
    metrics_report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_report_path.write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Artifacts saved: %s", out_dir)
    log.info("Canonical metrics: %s", metrics_report_path)

    return metrics_out


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: Dict[str, Any]) -> None:
    lr_agg  = metrics["lr"]["aggregate"]
    xgb_agg = metrics["xgb"]["aggregate"]
    m_cal   = metrics["xgb"]["final_calibrated"]
    thr     = metrics["thresholds"]

    print(f"\n{'='*66}")
    print(f"  TRAIN V3 — RESULTS SUMMARY")
    print(f"{'='*66}")
    print(f"  Folds trained : {metrics['n_folds']}")
    print(f"  Features      : {metrics['n_features']} kept / "
          f"{metrics.get('n_dropped', 0)} dropped (NaN filter)")
    print(f"")
    print(f"  {'Model':<30} {'ROC-AUC':>9} {'PR-AUC':>8} {'Brier':>7} {'ECE':>7}")
    print(f"  {'─'*62}")
    if lr_agg:
        print(f"  {'LR (mean ± std)':<30} "
              f"{lr_agg.get('roc_auc_mean', 0):>7.4f}  "
              f"{lr_agg.get('pr_auc_mean', 0):>7.4f}  "
              f"{lr_agg.get('brier_mean', 0):>6.4f}  "
              f"{lr_agg.get('ece_mean', 0):>6.4f}")
    print(f"  {'XGB (mean ± std)':<30} "
          f"{xgb_agg.get('roc_auc_mean', 0):>7.4f}  "
          f"{xgb_agg.get('pr_auc_mean', 0):>7.4f}  "
          f"{xgb_agg.get('brier_mean', 0):>6.4f}  "
          f"{xgb_agg.get('ece_mean', 0):>6.4f}")
    print(f"  {'XGB + Calibrated (final)':<30} "
          f"{m_cal.get('roc_auc', 0):>7.4f}  "
          f"{m_cal.get('pr_auc', 0):>7.4f}  "
          f"{m_cal.get('brier', 0):>6.4f}  "
          f"{m_cal.get('ece', 0):>6.4f}")
    print(f"")
    print(f"  Thresholds: t_lo={thr['t_lo']} (warn)  t_hi={thr['t_hi']} (block)")

    top5 = list(metrics["feature_importance_top20"].items())[:5]
    print(f"  Top-5 features: {top5}")
    print(f"{'='*66}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(
        description="Train v3 binary classifier (target_non_ok) — manifest-based"
    )
    ap.add_argument("--manifest", required=True,
                    help="Path to splits_manifest.json (from split_v3_time.py)")
    ap.add_argument("--out_dir",  default="models/v3",
                    help="Output directory for artifacts (default: models/v3)")
    ap.add_argument("--no_lr",   action="store_true",
                    help="Skip LR baseline (train XGB only)")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Limit rows per fold file — smoke/dev mode (default: all)")
    ap.add_argument("--nan_drop_threshold", type=float, default=0.30,
                    help="Drop features with NaN rate > this fraction (default: 0.30)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--n_estimators",type=int, default=400,
                    help="XGBoost n_estimators (default: 400)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir       = Path(args.out_dir)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    metrics = train(
        manifest_path=manifest_path,
        out_dir=out_dir,
        nan_drop_threshold=args.nan_drop_threshold,
        max_rows=args.max_rows,
        train_both=not args.no_lr,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )

    print_summary(metrics)
    log.info("Done. Artifacts: %s", out_dir)


if __name__ == "__main__":
    main()
