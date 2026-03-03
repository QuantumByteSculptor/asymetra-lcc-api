"""
scripts/ml/train/train_experts_v3.py
=====================================
*** DEPRECATED — use ml/train_v3.py instead ***

This script was superseded during the convergence refactor (Agent 4, 2026-03-03).
The single authoritative training entrypoint is now:

    python ml/train_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --out_dir  models/v3

Reasons for deprecation:
  - Used implicit split generation (not manifest-based → not reproducible).
  - Outputs diverged from pipeline standard (models/v3/v3_*_final.joblib vs
    v3_lr_model.joblib / v3_xgb_model.joblib used by API).
  - No calibration or threshold optimisation.
  - Smoke mode now available in ml/train_v3.py via --max_rows.

This file is kept for reference only. Do not use in production or CI.
See: ml/train_v3.py, docs/HOWTO_V3_PIPELINE.md

─────────────────────────────────────────────────────────────────────────────
ORIGINAL DOCSTRING (preserved for reference):
─────────────────────────────────────────────────────────────────────────────
Trainer v3 — expanding-window CV, model export, per-fold signal backtest.

Modèles disponibles : logistic, xgb (→ hgb si xgboost absent), hgb, lgbm

CV : expanding-window avec embargo via scripts.ml.validation.time_series_split_v3

Export :
  {models_dir}/v3_{model}_final.joblib   — modèle refitté sur tout le dataset
  {models_dir}/v3_{model}_meta.json      — feature_cols + medians (pour inférence)

Rapport :
  {out_dir}/metrics_report_v3.json

CLI:
  python scripts/ml/train/train_experts_v3.py \\
      --input        data/training/train_v3_all.jsonl \\
      --out_dir      data/metrics/v3 \\
      --models_dir   models/v3 \\
      --n_splits     5 \\
      --embargo_days 20 \\
      --seed         42 \\
      --models       xgb,logistic

  # Smoke run (limite à 5000 lignes)
  python scripts/ml/train/train_experts_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --out_dir data/metrics/v3 --models_dir models/v3 \\
      --max_rows 5000 --models logistic --n_splits 3

No API / prod impact.
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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Repo root: scripts/ml/train/ → scripts/ml/ → scripts/ → root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ml.validation.time_series_split_v3 import (  # noqa: E402
    generate_expanding_splits,
    verify_no_overlap,
)

log = logging.getLogger("train_experts_v3")

SEED = 42
_PERIODS_PER_YEAR = 252 / 20  # ≈ 12.6 twenty-day periods per year

# Columns that are NOT features
_META_COLS = {
    "asset_type", "market", "ticker", "window_end_date",
    "label_start_date", "label_end_date", "window_start_date",
    "label_end_date_60d", "tuw_pct", "source",
}
_TARGET_COLS = {
    "target_non_ok", "label", "forward_return_20d", "forward_return_5d",
    "forward_return_10d", "forward_return_60d", "future_dd_20d", "future_vol_ratio",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_streaming(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Stream JSONL line by line; flatten record["features"] dict into columns.
    Only numeric feature values are kept (strings skipped).
    max_rows limits lines read from disk (useful for smoke runs).
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row: Dict[str, Any] = {
                "window_end_date":    rec.get("window_end_date"),
                "label":              rec.get("label"),
                "target_non_ok":      rec.get("target_non_ok"),
                "forward_return_20d": rec.get("forward_return_20d"),
            }
            for k, v in rec.get("features", {}).items():
                if isinstance(v, (int, float)) or v is None:
                    row[k] = v
                # skip string/bool features (asset_type, ticker, ...)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"], errors="coerce")
    df = df.dropna(subset=["window_end_date", "target_non_ok"]).copy()
    df = df.sort_values("window_end_date").reset_index(drop=True)

    log.info(
        "Loaded %d rows  %s → %s",
        len(df),
        df["window_end_date"].min().date(),
        df["window_end_date"].max().date(),
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric columns that are not meta/target fields."""
    return [
        c for c in df.columns
        if c not in _META_COLS
        and c not in _TARGET_COLS
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
    ]


def apply_macro_strategy(
    df: pd.DataFrame,
    feat_cols: List[str],
    drop_macro: bool = False,
    nan_threshold: float = 0.50,
) -> List[str]:
    """
    drop_macro=True  → drop features with > nan_threshold fraction of NaN values
                       (targets macro/alt-data columns that are often sparse)
    drop_macro=False → keep all columns; NaN handled by median imputation
    """
    if not drop_macro:
        return feat_cols
    nan_rate = df[feat_cols].isna().mean()
    kept = [c for c in feat_cols if nan_rate[c] <= nan_threshold]
    n_dropped = len(feat_cols) - len(kept)
    if n_dropped:
        log.info("drop_macro: removed %d/%d features (NaN > %.0f%%)",
                 n_dropped, len(feat_cols), nan_threshold * 100)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Imputation (train stats applied to val — no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def impute(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Median imputation: medians computed on train, applied to both.
    Returns (X_train_np, X_val_np, medians Series).
    """
    medians = X_train.median()

    def _clean(df: pd.DataFrame) -> np.ndarray:
        return (
            df.fillna(medians)
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .to_numpy(dtype=float)
        )

    return _clean(X_train), _clean(X_val), medians


# ─────────────────────────────────────────────────────────────────────────────
# ECE (Expected Calibration Error)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    n = len(y_true)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(
            float(y_true[mask].mean()) - float(y_prob[mask].mean())
        )
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold signal backtest
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_under_fp_constraint(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_max: float = 0.15,
) -> float:
    """
    Find threshold that maximises skip_rate (conservatism) subject to:
        FP rate = P(risk_on | truly non_ok) <= fp_max

    risk_on = proba_non_ok < threshold  (we invest when model says "probably ok")
    FP      = invested in an asset that is truly non_ok (warn/block)
    """
    best_t, best_skip = 0.5, -1.0
    non_ok = y_true == 1
    if non_ok.sum() == 0:
        return 0.5
    for t in np.arange(0.30, 0.85, 0.05):
        risk_on = y_prob < t
        fp_rate = float(risk_on[non_ok].mean())
        skip_rate = float((~risk_on).mean())
        if fp_rate <= fp_max and skip_rate > best_skip:
            best_skip, best_t = skip_rate, float(t)
    return best_t


def _signal_stats(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    fwd_returns: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute risk-on/off metrics for a given decision threshold."""
    risk_on = y_prob < threshold
    valid = np.isfinite(fwd_returns)

    n_total = int(valid.sum())
    n_inv = int((risk_on & valid).sum())
    skip_rate = round(float((~risk_on)[valid].mean()), 4) if n_total > 0 else None

    invested_rets = fwd_returns[risk_on & valid]
    all_rets = fwd_returns[valid]

    baseline_mean = round(float(all_rets.mean()), 6) if len(all_rets) > 0 else None
    strat_mean = round(float(invested_rets.mean()), 6) if len(invested_rets) > 0 else None

    # Sharpe proxy (annualised over 20-day periods)
    if len(invested_rets) > 1 and np.std(invested_rets) > 1e-12:
        sharpe = round(
            float(np.mean(invested_rets) / np.std(invested_rets, ddof=1))
            * math.sqrt(_PERIODS_PER_YEAR),
            4,
        )
    else:
        sharpe = None

    # MaxDD proxy: cumulative return on full val sequence (skip → 0 return)
    max_dd = None
    if n_total > 0:
        seq = np.where(risk_on & valid, fwd_returns, 0.0)[valid]
        seq = np.clip(seq, -0.99, None)
        cum = np.cumprod(1.0 + seq)
        roll_max = np.maximum.accumulate(cum)
        dd = cum / (roll_max + 1e-12) - 1.0
        max_dd = round(float(dd.min()), 4)

    # FP rate: among truly non_ok, how many did we invest in?
    non_ok = y_true == 1
    fp_rate = round(float(risk_on[non_ok].mean()), 4) if non_ok.sum() > 0 else None

    return {
        "threshold":              round(float(threshold), 3),
        "n_invested":             n_inv,
        "n_skipped":              n_total - n_inv,
        "skip_rate":              skip_rate,
        "strategy_mean_return":   strat_mean,
        "baseline_mean_return":   baseline_mean,
        "sharpe_proxy":           sharpe,
        "max_drawdown_proxy":     max_dd,
        "fp_rate":                fp_rate,
    }


def backtest_fold(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    fwd_returns: np.ndarray,
    fp_constraint: float = 0.15,
) -> Dict[str, Any]:
    """
    Run two signal strategies on the validation fold:
      - "simple"      : threshold = 0.5
      - "constrained" : threshold optimised s.t. FP rate <= fp_constraint

    Returns a dict with per-strategy stats.
    """
    t_simple = 0.5
    t_constr = _threshold_under_fp_constraint(y_true, y_prob, fp_max=fp_constraint)

    return {
        "threshold_constrained": round(t_constr, 3),
        "simple":      _signal_stats(y_prob, y_true, fwd_returns, t_simple),
        "constrained": _signal_stats(y_prob, y_true, fwd_returns, t_constr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_ctor_params(seed: int, n_estimators: int, early_stop: bool = True) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=2.5,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
    )
    if early_stop:
        params["early_stopping_rounds"] = 30
    # use_label_encoder removed in XGBoost >= 2.0
    try:
        import xgboost as _x
        if int(_x.__version__.split(".")[0]) < 2:
            params["use_label_encoder"] = False
    except Exception:
        pass
    return params


def build_model(name: str, seed: int = SEED, n_estimators: int = 300) -> Optional[Any]:
    """
    Return an untrained sklearn-compatible estimator.
    Returns None if the requested backend is unavailable.
    """
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=500, class_weight="balanced",
                random_state=seed, solver="lbfgs",
            )),
        ])

    if name == "xgb":
        if not HAS_XGB:
            log.warning("xgboost not available — use 'hgb' instead")
            return None
        return xgb.XGBClassifier(**_xgb_ctor_params(seed, n_estimators, early_stop=True))

    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
        )

    if name == "lgbm":
        if not HAS_LGB:
            log.warning("lightgbm not available — skip")
            return None
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            scale_pos_weight=2.5,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )

    raise ValueError(f"Unknown model name: '{name}'. Choices: logistic, xgb, hgb, lgbm")


def _build_final_model(name: str, seed: int, n_estimators: int) -> Optional[Any]:
    """Build a model variant without early stopping for final refit."""
    if name == "xgb" and HAS_XGB:
        return xgb.XGBClassifier(**_xgb_ctor_params(seed, n_estimators, early_stop=False))
    return build_model(name, seed=seed, n_estimators=n_estimators)


# ─────────────────────────────────────────────────────────────────────────────
# Fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_fold(
    name: str,
    model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
    fold_idx: int,
    fwd_returns_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit model on train, evaluate on val, return metrics dict (with backtest)."""
    t0 = time.time()

    if name == "xgb" and HAS_XGB:
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    else:
        model.fit(X_tr, y_tr)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_vl)[:, 1]
    else:
        y_prob = 1.0 / (1.0 + np.exp(-model.decision_function(X_vl)))

    y_pred = (y_prob >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "fold":        fold_idx,
        "model":       name,
        "n_val":       int(len(y_vl)),
        "n_pos_val":   int(y_vl.sum()),
        "n_neg_val":   int((y_vl == 0).sum()),
        "elapsed_sec": round(time.time() - t0, 2),
    }

    if metrics["n_pos_val"] == 0 or metrics["n_neg_val"] == 0:
        log.warning("Fold %d %s: single class in val — AUC skipped", fold_idx, name)
        return metrics

    try:
        metrics["roc_auc"]            = round(float(roc_auc_score(y_vl, y_prob)), 4)
        metrics["pr_auc"]             = round(float(average_precision_score(y_vl, y_prob)), 4)
        metrics["brier"]              = round(float(brier_score_loss(y_vl, y_prob)), 4)
        metrics["ece"]                = round(compute_ece(y_vl, y_prob), 4)
        metrics["f1"]                 = round(float(f1_score(y_vl, y_pred, zero_division=0)), 4)
        metrics["accuracy"]           = round(float((y_pred == y_vl).mean()), 4)

        k = max(1, int(0.20 * len(y_vl)))
        top_k = np.argsort(y_prob)[::-1][:k]
        metrics["precision_top20pct"] = round(float(y_vl[top_k].mean()), 4)
        base_rate = float(y_vl.mean())
        metrics["lift_top20pct"]      = round(
            metrics["precision_top20pct"] / (base_rate + 1e-12), 2
        )
    except Exception as e:
        log.warning("Fold %d %s metric error: %s", fold_idx, name, e)

    # Per-fold backtest signal
    if fwd_returns_val is not None:
        try:
            metrics["backtest"] = backtest_fold(y_prob, y_vl, fwd_returns_val)
        except Exception as e:
            log.warning("Fold %d backtest error: %s", fold_idx, e)

    log.info(
        "  Fold %d %-8s | ROC=%.3f  PR=%.3f  Brier=%.3f  ECE=%.3f  F1=%.3f",
        fold_idx, name,
        metrics.get("roc_auc") or 0,
        metrics.get("pr_auc")  or 0,
        metrics.get("brier")   or 0,
        metrics.get("ece")     or 0,
        metrics.get("f1")      or 0,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_folds(fold_results: List[Dict]) -> Dict[str, Any]:
    valid = [f for f in fold_results if isinstance(f.get("roc_auc"), float)]
    if not valid:
        return {"n_valid_folds": 0}

    keys = [
        "roc_auc", "pr_auc", "brier", "ece", "f1", "accuracy",
        "precision_top20pct", "lift_top20pct",
    ]
    agg: Dict[str, Any] = {"n_valid_folds": len(valid)}
    for k in keys:
        vals = [f[k] for f in valid if isinstance(f.get(k), float)]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
            agg[f"{k}_min"]  = round(float(np.min(vals)),  4)
            agg[f"{k}_max"]  = round(float(np.max(vals)),  4)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(
        description="Train v3 experts — expanding-window CV + per-fold signal backtest"
    )
    ap.add_argument("--input",         required=True,            help="JSONL dataset path")
    ap.add_argument("--out_dir",       default="data/metrics/v3",help="Metrics output directory")
    ap.add_argument("--models_dir",    default="models/v3",      help="Model export directory")
    ap.add_argument("--n_splits",      type=int, default=5)
    ap.add_argument("--embargo_days",  type=int, default=20,
                    help="Days between last train date and first val date (default 20)")
    ap.add_argument("--max_rows",      type=int, default=None,
                    help="Limit JSONL rows loaded (smoke / dev run)")
    ap.add_argument("--seed",          type=int, default=SEED)
    ap.add_argument("--n_estimators",  type=int, default=300,
                    help="n_estimators for tree models (default 300)")
    ap.add_argument("--models",        default="xgb,logistic",
                    help="Comma-separated model list: xgb, hgb, logistic, lgbm")
    ap.add_argument("--drop_macro",    action="store_true",
                    help="Drop features with >--nan_threshold fraction of NaN")
    ap.add_argument("--nan_threshold", type=float, default=0.50,
                    help="NaN fraction threshold for --drop_macro (default 0.50)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_jsonl_streaming(Path(args.input), max_rows=args.max_rows)
    if args.max_rows:
        log.info("Smoke mode: loaded %d rows (--max_rows %d)", len(df), args.max_rows)

    feat_cols = get_feature_cols(df)
    feat_cols = apply_macro_strategy(
        df, feat_cols,
        drop_macro=args.drop_macro,
        nan_threshold=args.nan_threshold,
    )
    log.info("Feature count: %d", len(feat_cols))

    # ── CV splits ─────────────────────────────────────────────────────────────
    splits = generate_expanding_splits(
        df, n_splits=args.n_splits, embargo_days=args.embargo_days,
    )
    if not splits:
        log.error("No valid splits generated — check dataset size and date range")
        sys.exit(1)

    try:
        verify_no_overlap(splits, df)
    except ValueError as e:
        log.error("CV integrity check FAILED: %s", e)
        sys.exit(1)

    log.info("Generated %d valid expanding folds (embargo=%dd)", len(splits), args.embargo_days)

    # ── Model resolution (xgb → hgb fallback) ────────────────────────────────
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    effective: List[str] = []
    for mn in requested:
        if mn == "xgb" and not HAS_XGB:
            log.warning("xgboost not found — substituting 'hgb' for 'xgb'")
            effective.append("hgb")
        else:
            effective.append(mn)

    # ── Training loop ─────────────────────────────────────────────────────────
    all_results: List[Dict] = []

    for split in splits:
        fi     = split["fold"]
        tr_idx = split["train_indices"]
        vl_idx = split["val_indices"]

        X_tr_raw = df.iloc[tr_idx][feat_cols]
        X_vl_raw = df.iloc[vl_idx][feat_cols]
        y_tr = df.iloc[tr_idx]["target_non_ok"].to_numpy(dtype=int)
        y_vl = df.iloc[vl_idx]["target_non_ok"].to_numpy(dtype=int)
        fwd_val = df.iloc[vl_idx]["forward_return_20d"].to_numpy(dtype=float)

        X_tr, X_vl, _ = impute(X_tr_raw, X_vl_raw)

        log.info(
            "Fold %d  train=%d  val=%d  pos_train=%.1f%%  pos_val=%.1f%%",
            fi, len(y_tr), len(y_vl),
            100.0 * y_tr.mean(), 100.0 * y_vl.mean(),
        )

        for mname in effective:
            model = build_model(mname, seed=args.seed, n_estimators=args.n_estimators)
            if model is None:
                continue
            result = eval_fold(mname, model, X_tr, y_tr, X_vl, y_vl, fi, fwd_val)
            all_results.append(result)

    # ── Aggregate per model ───────────────────────────────────────────────────
    aggregated: Dict[str, Any] = {}
    for mname in effective:
        aggregated[mname] = aggregate_folds(
            [r for r in all_results if r.get("model") == mname]
        )

    # ── Final refit on full dataset + export ──────────────────────────────────
    out_dir    = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Impute on full dataset (medians from full train)
    all_medians = df[feat_cols].median()
    X_all = (
        df[feat_cols]
        .fillna(all_medians)
        .fillna(0.0)
        .replace([np.inf, -np.inf], 0.0)
        .to_numpy(dtype=float)
    )
    y_all = df["target_non_ok"].to_numpy(dtype=int)

    saved_models: List[str] = []
    for mname in effective:
        model_final = _build_final_model(mname, seed=args.seed, n_estimators=args.n_estimators)
        if model_final is None:
            continue
        log.info("Refit final %s on %d samples...", mname, len(y_all))
        model_final.fit(X_all, y_all)

        model_path = models_dir / f"v3_{mname}_final.joblib"
        joblib.dump(model_final, model_path)
        saved_models.append(str(model_path))
        log.info("Saved model: %s", model_path)

        # Metadata for inference: feature names + median fill values
        medians_serialisable = {
            k: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
            for k, v in all_medians.items()
        }
        meta = {
            "model":        mname,
            "feature_cols": feat_cols,
            "medians":      medians_serialisable,
            "n_features":   len(feat_cols),
            "n_samples":    int(len(y_all)),
            "seed":         args.seed,
            "trained_at":   datetime.utcnow().isoformat() + "Z",
        }
        meta_path = models_dir / f"v3_{mname}_meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Write report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    report = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "input_file":    str(Path(args.input)),
        "n_samples":     int(len(df)),
        "n_features":    len(feat_cols),
        "feature_cols":  feat_cols,
        "n_splits":      len(splits),
        "embargo_days":  args.embargo_days,
        "seed":          args.seed,
        "max_rows":      args.max_rows,
        "models":        effective,
        "elapsed_sec":   round(elapsed, 1),
        "fold_results":  all_results,
        "aggregated":    aggregated,
        "models_saved":  saved_models,
    }

    report_path = out_dir / "metrics_report_v3.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Report written: %s", report_path)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"TRAIN V3 — {len(splits)} folds | {len(df):,} samples | {len(feat_cols)} features")
    print(f"{'─'*68}")
    print(f"{'Model':<12} {'ROC-AUC':>14}  {'PR-AUC':>9}  {'Brier':>8}  {'ECE':>7}  {'F1':>7}")
    print(f"{'─'*68}")
    for mname, agg in aggregated.items():
        if "roc_auc_mean" in agg:
            roc_str = f"{agg['roc_auc_mean']:.4f}±{agg['roc_auc_std']:.4f}"
        else:
            roc_str = "N/A"
        pr    = f"{agg['pr_auc_mean']:.4f}"    if "pr_auc_mean"  in agg else "N/A"
        brier = f"{agg['brier_mean']:.4f}"     if "brier_mean"   in agg else "N/A"
        ece   = f"{agg['ece_mean']:.4f}"       if "ece_mean"     in agg else "N/A"
        f1    = f"{agg['f1_mean']:.4f}"        if "f1_mean"      in agg else "N/A"
        print(f"{mname:<12} {roc_str:>14}  {pr:>9}  {brier:>8}  {ece:>7}  {f1:>7}")
    print(f"{'='*68}")
    print(f"Report : {report_path}")
    print(f"Models : {models_dir}/")
    print(f"Elapsed: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
