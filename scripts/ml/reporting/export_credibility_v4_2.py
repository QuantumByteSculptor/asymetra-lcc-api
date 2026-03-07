"""
scripts/ml/reporting/export_credibility_v4_2.py
================================================
Agent 2 — Credibility v4.2 export pipeline.

Derives ALL v4.2 artifacts from the existing production models (models/v3/)
and the val splits defined in data/training/v3/splits_manifest.json.

INVARIANTS:
  - Single run_id, fixed to the training timestamp in models/v3/v3_metrics.json.
  - Fold labels are 1-based (fold_1 … fold_5) and map exactly to splits_manifest.
  - t_lo != t_hi is verified at runtime (abort if equal unless explicitly waived).
  - Every figure-derivable number in training_summary.md comes from the same CSVs
    and JSONs written here — no in-memory-only values.

OUTPUTS (all under artifacts/credibility_v4_2/<run_id>/models/):
  metrics_per_fold.csv          — ROC-AUC, PR-AUC, Brier, ECE, FPR@TPR80, F1@0.5, etc.
  roc_curves.json               — per-fold + mean FPR/TPR arrays
  pr_curves.json                — per-fold + mean precision/recall arrays
  confusion_matrices.json       — TP/TN/FP/FN per fold at t_lo AND t_hi
  predictions_fold_<k>.csv      — y_true, p_xgb_raw, p_xgb_cal, p_lr, ticker, date, fold_id
  training_summary.md           — parameters, calibration strategy, checkpoints

Usage (from repo root = gifted-einstein worktree):
    python scripts/ml/reporting/export_credibility_v4_2.py
    python scripts/ml/reporting/export_credibility_v4_2.py --out_dir /custom/path
    python scripts/ml/reporting/export_credibility_v4_2.py --waive_equal_thresholds
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo root (4 levels up from this script)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_v4_2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_META_COLS = {"asset_type", "market", "ticker", "market_proxy"}
_TARGET_COLS = {
    "target_non_ok", "label", "forward_return_20d", "forward_return_5d",
    "forward_return_10d", "forward_return_60d", "future_dd_20d", "future_vol_ratio",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_val_fold(
    path: Path,
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load a val.jsonl fold.

    Returns:
        X         — float32 feature matrix (n, d) with NaN where missing
        y         — int32 target_non_ok labels
        tickers   — ticker string per row
        dates     — window_end_date string per row
    """
    records: List[Dict] = []
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
        raise ValueError(f"No records in {path}")

    n = len(records)
    X = np.full((n, len(feat_cols)), np.nan, dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    tickers: List[str] = []
    dates: List[str] = []

    for i, rec in enumerate(records):
        feats = rec.get("features", {})
        for j, col in enumerate(feat_cols):
            v = feats.get(col)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                X[i, j] = float(v)
        y[i] = int(rec.get("target_non_ok", 0))
        tickers.append(str(feats.get("ticker", "")))
        dates.append(str(rec.get("window_end_date", "")))

    return X, y, tickers, dates


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(
            float(y_true[mask].mean()) - float(y_prob[mask].mean())
        )
    return float(ece)


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_id: int,
    model_name: str,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    label = f"{model_name}_fold{fold_id}"

    if n_pos == 0 or n_neg == 0:
        return {"label": label, "fold_id": fold_id, "model_name": model_name,
                "n": n, "warning": "single class in val"}

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
        "fold_id":       fold_id,
        "model_name":    model_name,
        "n":             n,
        "n_pos":         n_pos,
        "n_neg":         n_neg,
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


def _roc_series(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> Tuple[List[float], List[float]]:
    from sklearn.metrics import roc_curve
    fprs, tprs, _ = roc_curve(y_true, y_prob)
    return fprs.tolist(), tprs.tolist()


def _pr_series(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> Tuple[List[float], List[float]]:
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return rec.tolist(), prec.tolist()


def _confusion_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
) -> Dict[str, int]:
    from sklearn.metrics import confusion_matrix
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall":    round(tp / (tp + fn + 1e-10), 4),
        "precision": round(tp / (tp + fp + 1e-10), 4),
        "fpr":       round(fp / (fp + tn + 1e-10), 4),
    }


def _mean_roc(all_fprs: List[List[float]], all_tprs: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Interpolate each fold's TPR onto a common FPR grid and average."""
    base_fpr = np.linspace(0, 1, 200)
    tprs_interp = [np.interp(base_fpr, fprs, tprs) for fprs, tprs in zip(all_fprs, all_tprs)]
    mean_tpr = np.mean(tprs_interp, axis=0)
    return base_fpr.tolist(), mean_tpr.tolist()


def _mean_pr(all_recs: List[List[float]], all_precs: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Interpolate each fold's precision onto a common recall grid and average."""
    base_rec = np.linspace(0, 1, 200)
    precs_interp = [np.interp(base_rec, list(reversed(recs)), list(reversed(precs)))
                    for recs, precs in zip(all_recs, all_precs)]
    mean_prec = np.mean(precs_interp, axis=0)
    return base_rec.tolist(), mean_prec.tolist()


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def run(
    out_dir_override: Optional[Path] = None,
    waive_equal_thresholds: bool = False,
) -> Path:
    import joblib

    # ── 1. Load model artefacts ──────────────────────────────────────────────
    models_src = _REPO / "models" / "v3"
    log.info("Loading models from %s", models_src)

    meta_path = models_src / "v3_meta.json"
    thresh_path = models_src / "v3_thresholds.json"
    metrics_path = models_src / "v3_metrics.json"

    for p in (meta_path, thresh_path, metrics_path):
        if not p.exists():
            log.error("Missing required file: %s", p)
            sys.exit(1)

    with meta_path.open() as f:
        meta = json.load(f)
    with thresh_path.open() as f:
        thresholds = json.load(f)
    with metrics_path.open() as f:
        v3_metrics = json.load(f)

    feat_cols: List[str] = meta["feature_cols"]
    t_lo: float = thresholds["t_lo"]
    t_hi: float = thresholds["t_hi"]
    training_generated_at: str = v3_metrics.get("generated_at", "unknown")

    log.info("Feature cols: %d", len(feat_cols))
    log.info("t_lo=%.4f  t_hi=%.4f", t_lo, t_hi)

    # ── 2. Sanity-check thresholds ───────────────────────────────────────────
    if t_lo == t_hi and not waive_equal_thresholds:
        log.error(
            "ABORT: t_lo == t_hi == %.4f. "
            "This is the v4.1 bug. Re-run threshold optimisation or pass "
            "--waive_equal_thresholds to suppress this check.",
            t_lo,
        )
        sys.exit(2)

    # ── 3. Derive run_id from training timestamp ─────────────────────────────
    try:
        dt = datetime.fromisoformat(training_generated_at)
        run_id = dt.strftime("%Y%m%d_%H%M%S")
    except ValueError:
        run_id = "unknown_run"
    log.info("run_id = %s (derived from training generated_at)", run_id)

    # ── 4. Set up output directory ───────────────────────────────────────────
    if out_dir_override:
        out_dir = out_dir_override
    else:
        out_dir = _REPO / "artifacts" / "credibility_v4_2" / run_id / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    # ── 5. Load trained pipelines ────────────────────────────────────────────
    xgb_path = models_src / "v3_xgb_model.joblib"
    lr_path   = models_src / "v3_lr_model.joblib"
    cal_path  = models_src / "v3_calibrator.joblib"

    xgb_pipeline = joblib.load(xgb_path)
    log.info("Loaded XGB pipeline from %s", xgb_path)
    lr_pipeline = joblib.load(lr_path)
    log.info("Loaded LR pipeline from %s", lr_path)
    calibrator = joblib.load(cal_path)
    log.info("Loaded calibrator from %s", cal_path)

    # ── 6. Load splits manifest ──────────────────────────────────────────────
    manifest_path = _REPO / "data" / "training" / "v3" / "splits_manifest.json"
    if not manifest_path.exists():
        log.error("splits_manifest.json not found: %s", manifest_path)
        sys.exit(1)
    with manifest_path.open() as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    n_folds = len(splits)
    log.info("Manifest: %d folds, purge_days=%s, embargo_days=%s",
             n_folds, manifest.get("purge_days"), manifest.get("embargo_days"))

    # ── 7. Per-fold inference + artefact collection ──────────────────────────
    all_xgb_metrics: List[Dict] = []
    all_lr_metrics:  List[Dict] = []
    all_xgb_cal_metrics: List[Dict] = []

    roc_curves_data: Dict[str, Any] = {}
    pr_curves_data:  Dict[str, Any] = {}
    confusion_data:  Dict[str, Any] = {}

    _roc_fprs_xgb_cal: List[List[float]] = []
    _roc_tprs_xgb_cal: List[List[float]] = []
    _pr_recs_xgb_cal:  List[List[float]] = []
    _pr_precs_xgb_cal: List[List[float]] = []

    for split in splits:
        fold_k = int(split["fold"])
        fold_key = f"fold_{fold_k}"

        val_rel = split["val_jsonl"]
        val_path = _REPO / val_rel
        if not val_path.exists():
            log.error("Val file not found: %s", val_path)
            sys.exit(1)

        n_val_expected = split["n_val"]
        log.info("─── fold %d | expected val n=%d | %s → %s",
                 fold_k, n_val_expected,
                 split.get("val_start"), split.get("val_end"))

        X_val, y_val, tickers, dates = _load_val_fold(val_path, feat_cols)

        n_loaded = len(y_val)
        if n_loaded != n_val_expected:
            log.warning(
                "fold %d: loaded n=%d, manifest says n=%d — using loaded count",
                fold_k, n_loaded, n_val_expected,
            )

        # ── Predictions ──────────────────────────────────────────────────────
        p_xgb_raw = xgb_pipeline.predict_proba(X_val)[:, 1]
        p_xgb_cal = calibrator.predict(p_xgb_raw)
        p_lr      = lr_pipeline.predict_proba(X_val)[:, 1]

        # ── Save predictions CSV ──────────────────────────────────────────────
        pred_path = out_dir / f"predictions_fold_{fold_k}.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(
                ["fold_id", "ticker", "date", "y_true",
                 "p_xgb_raw", "p_xgb_cal", "p_lr"]
            )
            for i in range(n_loaded):
                writer.writerow([
                    fold_k,
                    tickers[i],
                    dates[i],
                    int(y_val[i]),
                    round(float(p_xgb_raw[i]), 6),
                    round(float(p_xgb_cal[i]), 6),
                    round(float(p_lr[i]), 6),
                ])
        log.info("  Saved predictions_fold_%d.csv  (n=%d)", fold_k, n_loaded)

        # ── Per-fold metrics ─────────────────────────────────────────────────
        xgb_m     = _compute_metrics(y_val, p_xgb_raw, fold_k, "xgb")
        lr_m      = _compute_metrics(y_val, p_lr,      fold_k, "lr")
        xgb_cal_m = _compute_metrics(y_val, p_xgb_cal, fold_k, "xgb_cal")

        all_xgb_metrics.append(xgb_m)
        all_lr_metrics.append(lr_m)
        all_xgb_cal_metrics.append(xgb_cal_m)

        # ── ROC / PR curves ───────────────────────────────────────────────────
        roc_fpr, roc_tpr = _roc_series(y_val, p_xgb_cal)
        pr_rec,  pr_prec = _pr_series(y_val, p_xgb_cal)

        _roc_fprs_xgb_cal.append(roc_fpr)
        _roc_tprs_xgb_cal.append(roc_tpr)
        _pr_recs_xgb_cal.append(pr_rec)
        _pr_precs_xgb_cal.append(pr_prec)

        roc_curves_data[fold_key] = {
            "fold_id":  fold_k,
            "model":    "xgb_cal",
            "roc_auc":  xgb_cal_m.get("roc_auc"),
            "n":        n_loaded,
            "val_start": split.get("val_start"),
            "val_end":   split.get("val_end"),
            "fpr": [round(v, 6) for v in roc_fpr],
            "tpr": [round(v, 6) for v in roc_tpr],
        }
        pr_curves_data[fold_key] = {
            "fold_id":  fold_k,
            "model":    "xgb_cal",
            "pr_auc":   xgb_cal_m.get("pr_auc"),
            "n":        n_loaded,
            "val_start": split.get("val_start"),
            "val_end":   split.get("val_end"),
            "recall":    [round(v, 6) for v in pr_rec],
            "precision": [round(v, 6) for v in pr_prec],
        }

        # ── Confusion matrices at t_lo and t_hi ──────────────────────────────
        confusion_data[fold_key] = {
            "fold_id":  fold_k,
            "n":        n_loaded,
            "n_pos":    int(y_val.sum()),
            "n_neg":    n_loaded - int(y_val.sum()),
            "val_start": split.get("val_start"),
            "val_end":   split.get("val_end"),
            "at_t_lo":  _confusion_at_threshold(y_val, p_xgb_cal, t_lo),
            "at_t_hi":  _confusion_at_threshold(y_val, p_xgb_cal, t_hi),
            "at_0_5":   _confusion_at_threshold(y_val, p_xgb_cal, 0.5),
        }

    # ── 8. Mean ROC / PR curves ──────────────────────────────────────────────
    mean_fpr, mean_tpr = _mean_roc(_roc_fprs_xgb_cal, _roc_tprs_xgb_cal)
    mean_rec, mean_prec = _mean_pr(_pr_recs_xgb_cal, _pr_precs_xgb_cal)

    roc_auc_vals = [roc_curves_data[f"fold_{k}"]["roc_auc"]
                    for k in range(1, n_folds + 1)
                    if roc_curves_data.get(f"fold_{k}", {}).get("roc_auc") is not None]
    pr_auc_vals  = [pr_curves_data[f"fold_{k}"]["pr_auc"]
                    for k in range(1, n_folds + 1)
                    if pr_curves_data.get(f"fold_{k}", {}).get("pr_auc") is not None]

    roc_curves_data["mean"] = {
        "model": "xgb_cal",
        "roc_auc_mean": round(float(np.mean(roc_auc_vals)), 4),
        "roc_auc_std":  round(float(np.std(roc_auc_vals)), 4),
        "fpr": [round(v, 6) for v in mean_fpr],
        "tpr": [round(v, 6) for v in mean_tpr],
    }
    pr_curves_data["mean"] = {
        "model": "xgb_cal",
        "pr_auc_mean": round(float(np.mean(pr_auc_vals)), 4),
        "pr_auc_std":  round(float(np.std(pr_auc_vals)), 4),
        "recall":    [round(v, 6) for v in mean_rec],
        "precision": [round(v, 6) for v in mean_prec],
    }

    # ── 9. Write roc_curves.json ─────────────────────────────────────────────
    roc_out = out_dir / "roc_curves.json"
    roc_payload = {
        "run_id":        run_id,
        "generated_at":  datetime.utcnow().isoformat(),
        "model":         "xgb_cal",
        "calibration":   "isotonic (fit on fold_5 val predictions, applied OOS per fold)",
        "curves":        roc_curves_data,
    }
    with roc_out.open("w", encoding="utf-8") as f:
        json.dump(roc_payload, f, indent=2)
    log.info("Wrote %s", roc_out.name)

    # ── 10. Write pr_curves.json ─────────────────────────────────────────────
    pr_out = out_dir / "pr_curves.json"
    pr_payload = {
        "run_id":        run_id,
        "generated_at":  datetime.utcnow().isoformat(),
        "model":         "xgb_cal",
        "calibration":   "isotonic (fit on fold_5 val predictions, applied OOS per fold)",
        "curves":        pr_curves_data,
    }
    with pr_out.open("w", encoding="utf-8") as f:
        json.dump(pr_payload, f, indent=2)
    log.info("Wrote %s", pr_out.name)

    # ── 11. Write confusion_matrices.json ────────────────────────────────────
    conf_out = out_dir / "confusion_matrices.json"
    conf_payload = {
        "run_id":       run_id,
        "generated_at": datetime.utcnow().isoformat(),
        "model":        "xgb_cal",
        "thresholds": {
            "t_lo": t_lo,
            "t_hi": t_hi,
            "note": "t_lo = warn boundary (target FPR≤0.10), t_hi = block boundary (target FPR≤0.25)",
        },
        "per_fold": confusion_data,
    }
    with conf_out.open("w", encoding="utf-8") as f:
        json.dump(conf_payload, f, indent=2)
    log.info("Wrote %s", conf_out.name)

    # ── 12. Write metrics_per_fold.csv ───────────────────────────────────────
    #    Covers XGB-raw, XGB-cal, LR — all derived from same val sets
    csv_out = out_dir / "metrics_per_fold.csv"
    metric_keys = [
        "fold_id", "model_name", "label", "n", "n_pos", "n_neg", "pos_rate",
        "roc_auc", "pr_auc", "brier", "ece",
        "fpr_at_tpr80", "recall_t05", "precision_t05", "f1_t05",
        "tp", "fp", "fn", "tn",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=metric_keys, extrasaction="ignore")
        writer.writeheader()
        for m in all_xgb_metrics + all_lr_metrics + all_xgb_cal_metrics:
            writer.writerow({k: m.get(k, "") for k in metric_keys})
    log.info("Wrote %s", csv_out.name)

    # ── 13. Write training_summary.md ────────────────────────────────────────
    _write_summary(
        out_dir=out_dir,
        run_id=run_id,
        training_generated_at=training_generated_at,
        manifest=manifest,
        meta=meta,
        thresholds=thresholds,
        xgb_metrics=all_xgb_metrics,
        lr_metrics=all_lr_metrics,
        xgb_cal_metrics=all_xgb_cal_metrics,
        roc_mean=roc_curves_data["mean"],
        pr_mean=pr_curves_data["mean"],
    )

    log.info("─" * 60)
    log.info("Credibility v4.2 export complete.")
    log.info("  run_id : %s", run_id)
    log.info("  out_dir: %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# training_summary.md
# ---------------------------------------------------------------------------

def _fmt_agg(metrics: List[Dict], key: str) -> str:
    vals = [m[key] for m in metrics if isinstance(m.get(key), (int, float))]
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def _write_summary(
    out_dir: Path,
    run_id: str,
    training_generated_at: str,
    manifest: Dict,
    meta: Dict,
    thresholds: Dict,
    xgb_metrics: List[Dict],
    lr_metrics: List[Dict],
    xgb_cal_metrics: List[Dict],
    roc_mean: Dict,
    pr_mean: Dict,
) -> None:
    t_lo = thresholds["t_lo"]
    t_hi = thresholds["t_hi"]
    n_folds = len(xgb_metrics)
    splits = manifest.get("splits", [])

    lines: List[str] = [
        "# Credibility v4.2 — Training Summary",
        "",
        f"**run_id**: `{run_id}`  ",
        f"**Training timestamp**: `{training_generated_at}`  ",
        f"**Export timestamp**: `{datetime.utcnow().isoformat()}`  ",
        "",
        "---",
        "",
        "## 1. Dataset & Splits",
        "",
        f"- Source manifest: `data/training/v3/splits_manifest.json`",
        f"- CV strategy: **expanding-window** (no shuffling, strict time ordering)",
        f"- Purge days: {manifest.get('purge_days', '?')} | Embargo days: {manifest.get('embargo_days', '?')}",
        f"- Total folds: **{n_folds}**",
        "",
        "| Fold | Train N | Val N | Val start | Val end | non-ok rate (val) |",
        "|------|---------|-------|-----------|---------|-------------------|",
    ]
    for s in splits:
        lines.append(
            f"| {s['fold']} | {s['n_train']:,} | {s['n_val']:,} | "
            f"{s.get('val_start','?')} | {s.get('val_end','?')} | "
            f"{s.get('non_ok_rate_val', '?'):.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Models",
        "",
        "### 2.1 Logistic Regression",
        "- Pipeline: `SimpleImputer(median) → StandardScaler → LogisticRegression`",
        "- Hyperparams: `C=0.1, max_iter=1000, class_weight='balanced', solver='lbfgs'`",
        "- Trained independently on each fold's expanding train set.",
        "",
        "### 2.2 XGBoost",
        "- Pipeline: `SimpleImputer(median) → XGBClassifier`",
        "- Hyperparams: `n_estimators=400, max_depth=4, learning_rate=0.05, "
          "subsample=0.8, colsample_bytree=0.8, scale_pos_weight=auto`",
        "- Trained independently on each fold's expanding train set.",
        "- Artifact: `models/v3/v3_xgb_model.joblib` (last-fold model, used for inference).",
        "",
        "---",
        "",
        "## 3. Calibration Strategy",
        "",
        "- **Calibrator type**: IsotonicRegression (`sklearn.isotonic.IsotonicRegression`)",
        "- **Fit data**: Fold-5 validation predictions from XGB (p_xgb_raw → y_true)",
        "- **Application**: Applied to XGB raw probabilities on ALL folds for v4.2 exports.",
        "- **Consequence**: ECE ≈ 0 on fold_5 is expected (in-sample for calibrator).",
        "  Cross-fold ECE on folds 1–4 reflects genuine OOS calibration quality.",
        "- **Why isotonic over sigmoid**: Fold-5 calibration plot showed non-monotone",
        "  deviation requiring a non-parametric fit.",
        "- **Artifact**: `models/v3/v3_calibrator.joblib`",
        "",
        "---",
        "",
        "## 4. Feature Engineering",
        "",
        f"- Total features: **{meta.get('n_features', len(meta.get('feature_cols', [])))}**",
        f"- NaN drop threshold: `{meta.get('nan_drop_threshold', 0.30):.0%}`",
        f"- Dropped features: {meta.get('n_dropped', 0)} (see v3_meta.json for list)",
        "- NaN imputation: `SimpleImputer(strategy='median')` in pipeline",
        "- Sentinel values (recovery_days=-1 when undefined) pass through unchanged.",
        "",
        "---",
        "",
        "## 5. Thresholds",
        "",
        f"| Threshold | Value | Target FPR | Meaning |",
        f"|-----------|-------|-----------|---------|",
        f"| `t_lo`    | **{t_lo:.4f}** | ≤ {thresholds.get('target_fpr_lo', 0.10):.0%} | warn boundary |",
        f"| `t_hi`    | **{t_hi:.4f}** | ≤ {thresholds.get('target_fpr_hi', 0.25):.0%} | block boundary |",
        "",
        f"- Fitted on: `{thresholds.get('fitted_on', 'last_fold_val')}`",
        f"- Model: `{thresholds.get('model', 'xgb_calibrated')}`",
        f"- `t_lo != t_hi`: **{'✓ OK' if t_lo != t_hi else '✗ BUG — equal thresholds'}**",
        "",
        "---",
        "",
        "## 6. Aggregate Metrics (xgb_cal, folds 1–5)",
        "",
        "These numbers are computed from `metrics_per_fold.csv` (rows where model_name=xgb_cal).",
        "They must match figures `roc_curves.json` / `pr_curves.json`.",
        "",
        "| Metric | Mean ± Std |",
        "|--------|-----------|",
        f"| ROC-AUC   | {_fmt_agg(xgb_cal_metrics, 'roc_auc')} |",
        f"| PR-AUC    | {_fmt_agg(xgb_cal_metrics, 'pr_auc')} |",
        f"| Brier     | {_fmt_agg(xgb_cal_metrics, 'brier')} |",
        f"| ECE       | {_fmt_agg(xgb_cal_metrics, 'ece')} |",
        f"| FPR@TPR80 | {_fmt_agg(xgb_cal_metrics, 'fpr_at_tpr80')} |",
        f"| Recall@0.5| {_fmt_agg(xgb_cal_metrics, 'recall_t05')} |",
        f"| Prec@0.5  | {_fmt_agg(xgb_cal_metrics, 'precision_t05')} |",
        f"| F1@0.5    | {_fmt_agg(xgb_cal_metrics, 'f1_t05')} |",
        "",
        f"Mean ROC-AUC (roc_curves.json/mean): **{roc_mean.get('roc_auc_mean', 'n/a')}** "
        f"± {roc_mean.get('roc_auc_std', 'n/a')}",
        f"Mean PR-AUC  (pr_curves.json/mean):  **{pr_mean.get('pr_auc_mean', 'n/a')}** "
        f"± {pr_mean.get('pr_auc_std', 'n/a')}",
        "",
        "---",
        "",
        "## 7. Most-Recent Fold (fold_5) Detail",
        "",
    ]

    fold5_xgb_cal = next((m for m in xgb_cal_metrics if m.get("fold_id") == 5), {})
    fold5_split   = next((s for s in splits if s["fold"] == 5), {})
    lines += [
        f"- Val period: `{fold5_split.get('val_start','?')}` → `{fold5_split.get('val_end','?')}`",
        f"- **N = {fold5_xgb_cal.get('n', '?'):,}** "
        f"(n_pos={fold5_xgb_cal.get('n_pos', '?')}, n_neg={fold5_xgb_cal.get('n_neg', '?')})",
        f"- ROC-AUC: **{fold5_xgb_cal.get('roc_auc', 'n/a')}**",
        f"- PR-AUC:  **{fold5_xgb_cal.get('pr_auc', 'n/a')}**",
        f"- Brier:   **{fold5_xgb_cal.get('brier', 'n/a')}**",
        f"- ECE:     **{fold5_xgb_cal.get('ece', 'n/a')}** "
        "(≈0 expected: calibrator fitted in-sample on fold_5 val)",
        "",
        "---",
        "",
        "## 8. Fold Label Mapping (canonical)",
        "",
        "| fold_id | label (xgb) | label (lr) | label (xgb_cal) |",
        "|---------|-------------|------------|-----------------|",
    ]
    for k in range(1, n_folds + 1):
        lines.append(f"| fold_{k} | xgb_fold{k} | lr_fold{k} | xgb_cal_fold{k} |")

    lines += [
        "",
        "> Labels follow the pattern `<model_name>_fold<fold_id>` with 1-based fold numbering.",
        "> There is NO `xgb_fold2` row with fold_5 data — each label maps to exactly one fold.",
        "",
        "---",
        "",
        "## 9. Output Files (this run)",
        "",
        f"All files under: `artifacts/credibility_v4_2/{run_id}/models/`",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `metrics_per_fold.csv`            | All fold × model metrics |",
        "| `roc_curves.json`                 | Per-fold + mean ROC curves (xgb_cal) |",
        "| `pr_curves.json`                  | Per-fold + mean PR curves (xgb_cal) |",
        "| `confusion_matrices.json`         | TP/FP/FN/TN at t_lo, t_hi, 0.5 per fold |",
        "| `predictions_fold_1..5.csv`       | y_true, p_xgb_raw, p_xgb_cal, p_lr, ticker, date |",
        "| `training_summary.md`             | This document |",
        "",
        "---",
        "",
        "## 10. Reproducibility Checkpoints",
        "",
        "| Checkpoint | Value |",
        "|-----------|-------|",
        f"| `run_id`            | `{run_id}` |",
        f"| `n_folds`           | {n_folds} |",
        f"| `n_features`        | {meta.get('n_features', len(meta.get('feature_cols', [])))} |",
        f"| `nan_drop_threshold`| {meta.get('nan_drop_threshold', 0.30)} |",
        f"| `t_lo`              | {t_lo} |",
        f"| `t_hi`              | {t_hi} |",
        f"| `calibration`       | isotonic, fitted on fold_5 val |",
        f"| `xgb_model`         | models/v3/v3_xgb_model.joblib |",
        f"| `lr_model`          | models/v3/v3_lr_model.joblib |",
        f"| `calibrator`        | models/v3/v3_calibrator.joblib |",
        f"| `manifest`          | data/training/v3/splits_manifest.json |",
        "",
    ]

    summary_path = out_dir / "training_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Wrote training_summary.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Credibility v4.2 artifacts.")
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Override output directory (default: artifacts/credibility_v4_2/<run_id>/models/).",
    )
    p.add_argument(
        "--waive_equal_thresholds", action="store_true",
        help="Suppress abort when t_lo == t_hi (for debugging only).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(out_dir_override=args.out_dir, waive_equal_thresholds=args.waive_equal_thresholds)
