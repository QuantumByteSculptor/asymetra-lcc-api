"""
scripts/ml/train_v3_expanding.py
=================================
Phase 3 — Entraînement baseline avec expanding-window CV.

Modèles:
  - XGBoost (gradient boosting avec early stopping)
  - Logistic Regression (baseline linéaire, calibré)

Métriques par fold:
  - ROC-AUC
  - PR-AUC (average precision)
  - Brier score
  - ECE (Expected Calibration Error)
  - Accuracy, F1 (threshold=0.5)

Rapport: data/metrics/train_v3_cv_report.json

Usage:
  python scripts/ml/train_v3_expanding.py \\
      --input data/training/train_v3_all.jsonl \\
      --out data/metrics/train_v3_cv_report.json \\
      --n_splits 5 --seed 42

Contraintes:
  - Seed fixe pour reproductibilité
  - Pas de grid search lourd (RandomizedSearch léger max 20 iter)
  - Early stopping XGBoost
  - Aucune fuite temporelle (splits strictement expanding)
  - No API / prod impact
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
    logging.warning("xgboost not available — XGB model will be skipped")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ml.validation.time_series_split_v3 import (  # noqa: E402
    load_jsonl_as_df,
    generate_expanding_splits,
)

log = logging.getLogger("train_v3_expanding")

SEED = 42

# ---------------------------------------------------------------------------
# Feature columns (excludes identity and label fields)
# ---------------------------------------------------------------------------
_META_COLS = {
    "asset_type", "market", "ticker", "tuw_pct",
    "window_end_date", "label_start_date", "label_end_date",
    "window_start_date", "label_end_date_60d",
}

_ORDINAL_COLS = {"vol_regime"}   # treat as numeric (0/1/2)


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns from the features dict, excluding meta."""
    feat_cols = [
        c for c in df.columns
        if c not in _META_COLS
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        and c not in ("target_non_ok", "label", "forward_return_20d",
                      "forward_return_5d", "forward_return_10d", "forward_return_60d",
                      "future_dd_20d", "future_vol_ratio")
    ]
    return feat_cols


# ---------------------------------------------------------------------------
# Data loader (full features)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                row: Dict[str, Any] = {}
                # Top-level fields
                row["window_end_date"]  = rec.get("window_end_date")
                row["label"]            = rec.get("label")
                row["target_non_ok"]    = rec.get("target_non_ok")
                row["forward_return_20d"] = rec.get("forward_return_20d")
                row["asset_type"]       = rec.get("features", {}).get("asset_type")
                row["ticker"]           = rec.get("features", {}).get("ticker")
                # Feature dict (flatten)
                for k, v in rec.get("features", {}).items():
                    if isinstance(v, (int, float)) or v is None:
                        row[k] = v
                    elif isinstance(v, str):
                        pass   # skip non-numeric strings
                records.append(row)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"], errors="coerce")
    df = df.dropna(subset=["window_end_date", "target_non_ok"]).copy()
    df = df.sort_values("window_end_date").reset_index(drop=True)
    log.info("Dataset loaded: %d samples, date range %s → %s",
             len(df), df["window_end_date"].min().date(),
             df["window_end_date"].max().date())
    return df


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_xgb(seed: int = SEED, n_estimators: int = 400) -> Any:
    if not HAS_XGB:
        return None
    return xgb.XGBClassifier(
        n_estimators       = n_estimators,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        scale_pos_weight   = 2.5,   # handles class imbalance (ok >> non_ok)
        use_label_encoder  = False,
        eval_metric        = "logloss",
        early_stopping_rounds = 30,
        random_state       = seed,
        verbosity          = 0,
        n_jobs             = 1,
    )


def build_logistic(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.1, max_iter=500, class_weight="balanced",
            random_state=seed, solver="lbfgs",
        )),
    ])


# ---------------------------------------------------------------------------
# Imputer — median imputation on train stats, applied to val
# ---------------------------------------------------------------------------

def impute(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Median imputation computed on train, applied to val."""
    medians = X_train.median()
    X_tr = X_train.fillna(medians).fillna(0.0)
    X_vl = X_val.fillna(medians).fillna(0.0)
    # Replace inf
    X_tr = X_tr.replace([np.inf, -np.inf], 0.0)
    X_vl = X_vl.replace([np.inf, -np.inf], 0.0)
    return X_tr.to_numpy(dtype=float), X_vl.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Evaluate one model on one fold
# ---------------------------------------------------------------------------

def evaluate_fold(
    model_name: str,
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
    fold_idx: int,
) -> Dict[str, Any]:

    t0 = time.time()

    # Fit
    if model_name == "xgb" and HAS_XGB:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr)

    # Predict
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_vl)[:, 1]
    else:
        y_prob = model.decision_function(X_vl)
        y_prob = 1 / (1 + np.exp(-y_prob))   # sigmoid

    y_pred = (y_prob >= 0.5).astype(int)
    elapsed = time.time() - t0

    # Metrics
    n_pos = int(y_vl.sum())
    n_neg = int((y_vl == 0).sum())

    metrics: Dict[str, Any] = {
        "fold":        fold_idx,
        "model":       model_name,
        "n_val":       int(len(y_vl)),
        "n_pos_val":   n_pos,
        "n_neg_val":   n_neg,
        "elapsed_sec": round(elapsed, 2),
    }

    if n_pos == 0 or n_neg == 0:
        log.warning("Fold %d %s: only one class in val — skipping AUC", fold_idx, model_name)
        metrics.update({"roc_auc": None, "pr_auc": None, "brier": None, "ece": None})
        return metrics

    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_vl, y_prob)), 4)
        metrics["pr_auc"]  = round(float(average_precision_score(y_vl, y_prob)), 4)
        metrics["brier"]   = round(float(brier_score_loss(y_vl, y_prob)), 4)
        metrics["ece"]     = round(compute_ece(y_vl, y_prob), 4)
        metrics["f1"]      = round(float(f1_score(y_vl, y_pred, zero_division=0)), 4)
        metrics["accuracy"]= round(float((y_pred == y_vl).mean()), 4)

        # Top-k precision (flag only top 20% most risky)
        k = max(1, int(0.20 * len(y_vl)))
        top_k_idx = np.argsort(y_prob)[::-1][:k]
        metrics["precision_top20pct"] = round(float(y_vl[top_k_idx].mean()), 4)

        # Positive rate in top-k vs base rate
        base_rate = float(y_vl.mean())
        metrics["lift_top20pct"] = round(
            metrics["precision_top20pct"] / (base_rate + 1e-12), 2
        )

    except Exception as e:
        log.warning("Fold %d %s metric error: %s", fold_idx, model_name, e)
        metrics.update({"roc_auc": None, "pr_auc": None, "brier": None, "ece": None})

    log.info(
        "  Fold %d %-8s | ROC=%.3f PR=%.3f Brier=%.3f ECE=%.3f F1=%.3f",
        fold_idx, model_name,
        metrics.get("roc_auc") or 0,
        metrics.get("pr_auc")  or 0,
        metrics.get("brier")   or 0,
        metrics.get("ece")     or 0,
        metrics.get("f1")      or 0,
    )
    return metrics


# ---------------------------------------------------------------------------
# Aggregate metrics across folds
# ---------------------------------------------------------------------------

def aggregate_metrics(fold_results: List[Dict]) -> Dict[str, Any]:
    valid = [f for f in fold_results if f.get("roc_auc") is not None]
    if not valid:
        return {"n_valid_folds": 0}

    keys = ["roc_auc", "pr_auc", "brier", "ece", "f1", "accuracy",
            "precision_top20pct", "lift_top20pct"]
    agg: Dict[str, Any] = {"n_valid_folds": len(valid)}

    for k in keys:
        vals = [f[k] for f in valid if f.get(k) is not None]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"]  = round(float(np.std(vals)), 4)
            agg[f"{k}_min"]  = round(float(np.min(vals)), 4)
            agg[f"{k}_max"]  = round(float(np.max(vals)), 4)

    return agg


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

    ap = argparse.ArgumentParser(description="Train v3 — expanding CV baseline")
    ap.add_argument("--input",        required=True, help="v3 JSONL dataset")
    ap.add_argument("--out",          default="data/metrics/train_v3_cv_report.json")
    ap.add_argument("--n_splits",     type=int, default=5)
    ap.add_argument("--embargo_days", type=int, default=20)
    ap.add_argument("--seed",         type=int, default=SEED)
    ap.add_argument("--n_estimators", type=int, default=400,
                    help="XGBoost n_estimators (default 400, early_stop at 30)")
    ap.add_argument("--models",       default="xgb,logistic",
                    help="Comma-separated model list (xgb,logistic)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()

    # Load data
    df = load_dataset(Path(args.input))
    feat_cols = _get_feature_cols(df)
    log.info("Feature columns: %d", len(feat_cols))

    # Generate splits
    splits = generate_expanding_splits(
        df,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
    )
    log.info("Generated %d expanding folds", len(splits))

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    # ---- Training loop
    all_fold_results: List[Dict] = []

    for split in splits:
        fi      = split["fold"]
        tr_idx  = split["train_indices"]
        vl_idx  = split["val_indices"]

        X_tr_raw = df.iloc[tr_idx][feat_cols]
        X_vl_raw = df.iloc[vl_idx][feat_cols]
        y_tr     = df.iloc[tr_idx]["target_non_ok"].to_numpy(dtype=int)
        y_vl     = df.iloc[vl_idx]["target_non_ok"].to_numpy(dtype=int)

        X_tr, X_vl = impute(X_tr_raw, X_vl_raw)

        log.info("Fold %d  train=%d  val=%d  pos_train=%.1f%%  pos_val=%.1f%%",
                 fi, len(y_tr), len(y_vl),
                 100 * y_tr.mean(), 100 * y_vl.mean())

        for mname in model_names:
            if mname == "xgb":
                model = build_xgb(seed=args.seed, n_estimators=args.n_estimators)
                if model is None:
                    continue
            elif mname == "logistic":
                model = build_logistic(seed=args.seed)
            else:
                log.warning("Unknown model %s — skip", mname)
                continue

            result = evaluate_fold(mname, model, X_tr, y_tr, X_vl, y_vl, fi)
            all_fold_results.append(result)

    # ---- Aggregate per model
    aggregated: Dict[str, Any] = {}
    for mname in model_names:
        fold_res = [r for r in all_fold_results if r["model"] == mname]
        aggregated[mname] = aggregate_metrics(fold_res)

    elapsed = time.time() - t_start

    # ---- Report
    report = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "input_file":    str(Path(args.input)),
        "n_samples":     int(len(df)),
        "n_features":    len(feat_cols),
        "n_splits":      args.n_splits,
        "embargo_days":  args.embargo_days,
        "seed":          args.seed,
        "elapsed_sec":   round(elapsed, 1),
        "feature_cols":  feat_cols,
        "fold_results":  all_fold_results,
        "aggregated":    aggregated,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Report written: %s", out_path)

    # ---- Console summary
    print(f"\n{'='*60}")
    print(f"TRAIN V3 — EXPANDING CV RESULTS")
    print(f"Samples: {len(df):,}  |  Features: {len(feat_cols)}  |  Folds: {len(splits)}")
    print(f"{'─'*60}")
    print(f"{'Model':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'Brier':>8} {'ECE':>8} {'F1':>8}")
    print(f"{'─'*60}")
    for mname, agg in aggregated.items():
        print(
            f"{mname:<12} "
            f"{agg.get('roc_auc_mean', 'N/A'):>10} "
            f"±{agg.get('roc_auc_std', ''):>6}  "
            f"{agg.get('pr_auc_mean', 'N/A'):>8}  "
            f"{agg.get('brier_mean', 'N/A'):>8}  "
            f"{agg.get('ece_mean', 'N/A'):>8}  "
            f"{agg.get('f1_mean', 'N/A'):>8}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
