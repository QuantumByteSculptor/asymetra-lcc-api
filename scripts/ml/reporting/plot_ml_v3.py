"""
scripts/ml/reporting/plot_ml_v3.py
===================================
Phase 1 — ML visualizations for the v3 pipeline.

Generates (for each fold + mean):
  1. ROC Curve
  2. Precision-Recall Curve
  3. Calibration curve (reliability diagram)
  4. Probability distribution (ok vs non_ok)
  5. Lift curve (top-decile focus)
  6. Confusion matrix heatmap
  7. Feature importance (XGB gain + permutation importance)
  8. SHAP summary (if shap installed)

All PNGs → data/metrics/v3/plots/

Usage:
    python scripts/ml/reporting/plot_ml_v3.py \\
        --manifest data/training/v3/splits_manifest.json \\
        --models   models/v3 \\
        --out      data/metrics/v3/plots

    # With permutation importance (slower)
    python scripts/ml/reporting/plot_ml_v3.py ... --perm_importance

No network required. Gracefully skips if model or data files are missing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

log = logging.getLogger("plot_ml_v3")

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
DPI = 150


# ── Data loading ───────────────────────────────────────────────────────────────

def load_fold_predictions(
    manifest_path: Path,
    model_dir: Path,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    For each fold in the manifest, load val.jsonl + apply calibrated XGB.
    Returns {fold_k: (y_true, y_prob_xgb_cal, y_prob_lr)} for available folds.
    """
    import joblib

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]

    # Load artifacts
    xgb_path  = model_dir / "v3_xgb_model.joblib"
    lr_path   = model_dir / "v3_lr_model.joblib"
    cal_path  = model_dir / "v3_calibrator.joblib"
    feat_path = model_dir / "v3_feature_names.joblib"

    if not xgb_path.exists():
        log.warning("XGB model not found at %s — skipping prediction-based plots", xgb_path)
        return {}

    xgb_model  = joblib.load(xgb_path)
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    lr_model   = joblib.load(lr_path) if lr_path.exists() else None
    feat_cols  = joblib.load(feat_path) if feat_path.exists() else None

    if feat_cols is None:
        log.warning("Feature names not found — cannot build prediction matrix")
        return {}

    results: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for fold in splits:
        fk       = fold["fold"]
        val_path = Path(fold["val_jsonl"])
        if not val_path.exists():
            log.warning("Fold %d val not found: %s", fk, val_path)
            continue

        log.info("Loading fold %d predictions (%s)...", fk, val_path.name)
        X, y = _load_xy(val_path, feat_cols)

        if len(y) == 0:
            continue

        xgb_raw = xgb_model.predict_proba(X)[:, 1]
        xgb_cal = calibrator.predict(xgb_raw) if calibrator is not None else xgb_raw
        lr_prob  = lr_model.predict_proba(X)[:, 1] if lr_model is not None else np.full_like(xgb_cal, np.nan)

        results[fk] = (y.astype(int), xgb_cal, lr_prob)
        log.info("  fold %d: n=%d pos=%.1f%%", fk, len(y), 100 * y.mean())

    return results


def _load_xy(
    path: Path,
    feat_cols: List[str],
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            feats = rec.get("features", {})
            row = []
            for col in feat_cols:
                v = feats.get(col)
                if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                    row.append(float(v))
                else:
                    row.append(np.nan)
            X_rows.append(row)
            y_rows.append(int(rec.get("target_non_ok", 0)))

    if not X_rows:
        return np.empty((0, len(feat_cols)), dtype=np.float32), np.array([], dtype=int)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=int)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int = DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", path)


# ── 1. ROC Curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> List[Path]:
    from sklearn.metrics import roc_curve, auc

    if not folds_data:
        log.warning("No fold data for ROC — skipping")
        return []

    paths = []
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50)")

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []

    for i, (fk, (y, y_xgb, _)) in enumerate(sorted(folds_data.items())):
        fprs, tprs, _ = roc_curve(y, y_xgb)
        fold_auc = auc(fprs, tprs)
        color = FOLD_COLORS[i % len(FOLD_COLORS)]
        ax.plot(fprs, tprs, color=color, alpha=0.55, lw=1.5,
                label=f"Fold {fk} (AUC={fold_auc:.3f})")
        interp_tprs.append(np.interp(mean_fpr, fprs, tprs))

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(interp_tprs, axis=0)

    ax.plot(mean_fpr, mean_tpr, "navy", lw=2.5,
            label=f"Mean (AUC={mean_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="navy", alpha=0.12, label="±1 std")

    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — v3 XGB Calibrated (per fold + mean)",
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.05])
    ax.legend(loc="lower right", fontsize=9)
    p = out_dir / "roc_curves.png"
    _savefig(fig, p)
    paths.append(p)
    return paths


# ── 2. Precision-Recall Curves ────────────────────────────────────────────────

def plot_pr_curves(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> List[Path]:
    from sklearn.metrics import precision_recall_curve, auc

    if not folds_data:
        return []

    fig, ax = plt.subplots(figsize=(7, 6))
    recall_pts = np.linspace(0, 1, 200)
    interp_precs = []
    baselines = []

    for i, (fk, (y, y_xgb, _)) in enumerate(sorted(folds_data.items())):
        prec, rec, _ = precision_recall_curve(y, y_xgb)
        fold_auc = auc(rec, prec)
        baselines.append(y.mean())
        color = FOLD_COLORS[i % len(FOLD_COLORS)]
        ax.plot(rec, prec, color=color, alpha=0.55, lw=1.5,
                label=f"Fold {fk} (AP={fold_auc:.3f})")
        interp_precs.append(np.interp(recall_pts, rec[::-1], prec[::-1]))

    mean_prec = np.mean(interp_precs, axis=0)
    std_prec  = np.std(interp_precs, axis=0)
    mean_ap   = auc(recall_pts, mean_prec)
    baseline  = np.mean(baselines)

    ax.plot(recall_pts, mean_prec, "navy", lw=2.5,
            label=f"Mean (AP={mean_ap:.3f})")
    ax.fill_between(recall_pts, mean_prec - std_prec, mean_prec + std_prec,
                    color="navy", alpha=0.12, label="±1 std")
    ax.axhline(baseline, ls=":", color="gray", alpha=0.8,
               label=f"No-skill baseline ({baseline:.2f})")

    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve — v3 XGB Calibrated",
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.05])
    ax.legend(loc="upper right", fontsize=9)
    p = out_dir / "pr_curves.png"
    _savefig(fig, p)
    return [p]


# ── 3. Calibration Curve ──────────────────────────────────────────────────────

def plot_calibration(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> List[Path]:
    from sklearn.calibration import calibration_curve

    if not folds_data:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_cal, ax_hist = axes

    # Perfect calibration line
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect calibration")

    for i, (fk, (y, y_xgb, y_lr)) in enumerate(sorted(folds_data.items())):
        color = FOLD_COLORS[i % len(FOLD_COLORS)]
        try:
            frac_pos, mean_pred = calibration_curve(y, y_xgb, n_bins=10, strategy="quantile")
            ax_cal.plot(mean_pred, frac_pos, "o-", color=color, alpha=0.75,
                        ms=5, lw=1.8, label=f"XGB Fold {fk}")
        except Exception:
            pass

    # Histogram of probabilities on last fold
    last_fk = max(folds_data)
    y_last, yp_last, _ = folds_data[last_fk]
    ax_hist.hist(yp_last[y_last == 0], bins=40, alpha=0.6, color="#4C72B0",
                 density=True, label="OK (target=0)", edgecolor="white", lw=0.3)
    ax_hist.hist(yp_last[y_last == 1], bins=40, alpha=0.6, color="#DD8452",
                 density=True, label="Non-OK (target=1)", edgecolor="white", lw=0.3)
    ax_hist.set(xlabel="Predicted probability", ylabel="Density",
                title=f"Score Distribution — Fold {last_fk}")
    ax_hist.legend(fontsize=9)

    ax_cal.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
               title="Reliability Diagram (Calibration)", xlim=[-0.02, 1.02], ylim=[-0.02, 1.05])
    ax_cal.legend(loc="upper left", fontsize=9)

    fig.suptitle("Calibration Analysis — v3 XGB", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = out_dir / "calibration.png"
    _savefig(fig, p)
    return [p]


# ── 4. Probability Distributions ─────────────────────────────────────────────

def plot_prob_distributions(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    thresholds: Optional[Dict] = None,
) -> List[Path]:
    if not folds_data:
        return []

    n_folds = len(folds_data)
    ncols = min(n_folds, 2)
    nrows = math.ceil(n_folds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                             squeeze=False)

    t_lo = thresholds.get("t_lo", 0.5) if thresholds else 0.5
    t_hi = thresholds.get("t_hi", 0.65) if thresholds else 0.65

    for idx, (fk, (y, y_xgb, _)) in enumerate(sorted(folds_data.items())):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.hist(y_xgb[y == 0], bins=50, alpha=0.65, color="#4C72B0",
                density=True, label="OK", edgecolor="white", lw=0.2)
        ax.hist(y_xgb[y == 1], bins=50, alpha=0.65, color="#DD8452",
                density=True, label="Non-OK (warn+block)", edgecolor="white", lw=0.2)
        ax.axvline(t_lo, color="#2ca02c", ls="--", lw=1.5, label=f"t_lo={t_lo:.3f}")
        ax.axvline(t_hi, color="#d62728", ls="--", lw=1.5, label=f"t_hi={t_hi:.3f}")
        ax.set(title=f"Fold {fk} (n={len(y):,})", xlabel="P(non_ok)",
               ylabel="Density")
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_folds, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Predicted Probability Distributions — v3 XGB Calibrated",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "prob_distributions.png"
    _savefig(fig, p)
    return [p]


# ── 5. Lift Curve ─────────────────────────────────────────────────────────────

def plot_lift_curve(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> List[Path]:
    if not folds_data:
        return []

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="Random baseline")

    x_pct = np.linspace(0.01, 1.0, 100)

    for i, (fk, (y, y_xgb, _)) in enumerate(sorted(folds_data.items())):
        base_rate = y.mean()
        if base_rate < 1e-10:
            continue
        order = np.argsort(y_xgb)[::-1]
        y_sorted = y[order]
        n = len(y_sorted)
        cumsum_pos = np.cumsum(y_sorted) / (base_rate * n)
        x_pts = np.arange(1, n + 1) / n
        # Interpolate to common x grid
        lift = np.interp(x_pct, x_pts, cumsum_pos)
        color = FOLD_COLORS[i % len(FOLD_COLORS)]
        ax.plot(x_pct * 100, lift, color=color, alpha=0.75, lw=2,
                label=f"Fold {fk}")

    ax.set(xlabel="Population selected (%)", ylabel="Lift (× baseline rate)",
           title="Lift Curve — v3 XGB Calibrated (sorted by P(non_ok))",
           xlim=[0, 100], ylim=[0, None])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=9)
    p = out_dir / "lift_curve.png"
    _savefig(fig, p)
    return [p]


# ── 6. Confusion Matrix Heatmap ───────────────────────────────────────────────

def plot_confusion_matrix(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    thresholds: Optional[Dict] = None,
) -> List[Path]:
    from sklearn.metrics import confusion_matrix

    if not folds_data:
        return []

    t_lo = thresholds.get("t_lo", 0.5) if thresholds else 0.5
    n_folds = len(folds_data)
    ncols = min(n_folds, 2)
    nrows = math.ceil(n_folds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (fk, (y, y_xgb, _)) in enumerate(sorted(folds_data.items())):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        y_pred = (y_xgb >= t_lo).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        cm_pct = cm.astype(float) / cm.sum() * 100
        labels = np.array([
            [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ])
        sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues",
                    xticklabels=["Pred OK", "Pred Non-OK"],
                    yticklabels=["True OK", "True Non-OK"],
                    ax=ax, cbar=False, linewidths=0.5)
        n = len(y)
        acc = (cm[0, 0] + cm[1, 1]) / n
        ax.set_title(f"Fold {fk}  |  t_lo={t_lo:.3f}  |  Acc={acc:.3f}",
                     fontsize=10)

    for idx in range(n_folds, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Confusion Matrices — v3 XGB Calibrated", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    p = out_dir / "confusion_matrices.png"
    _savefig(fig, p)
    return [p]


# ── 7. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(
    model_dir: Path,
    out_dir: Path,
    top_n: int = 25,
) -> List[Path]:
    import joblib

    feat_path  = model_dir / "v3_feature_names.joblib"
    xgb_path   = model_dir / "v3_xgb_model.joblib"

    if not xgb_path.exists():
        log.warning("XGB model missing — skipping feature importance")
        return []

    xgb_model = joblib.load(xgb_path)
    feat_cols  = joblib.load(feat_path) if feat_path.exists() else None

    try:
        importances = xgb_model[-1].feature_importances_
    except Exception as e:
        log.warning("Could not extract feature importances: %s", e)
        return []

    if feat_cols is None or len(feat_cols) != len(importances):
        feat_cols = [f"feat_{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feat_cols[i] for i in idx]
    top_imps  = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_feats)))[::-1]
    bars = ax.barh(range(len(top_feats)), top_imps[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set(xlabel="XGB Gain Importance",
           title=f"Feature Importance — Top {top_n} (XGB Gain)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    for bar, val in zip(bars, top_imps[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    p = out_dir / "feature_importance.png"
    _savefig(fig, p)
    return [p]


# ── 7b. Permutation Importance ────────────────────────────────────────────────

def plot_permutation_importance(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_dir: Path,
    out_dir: Path,
    top_n: int = 20,
) -> List[Path]:
    from sklearn.inspection import permutation_importance
    import joblib

    if not folds_data:
        return []

    feat_path = model_dir / "v3_feature_names.joblib"
    xgb_path  = model_dir / "v3_xgb_model.joblib"
    if not xgb_path.exists() or not feat_path.exists():
        return []

    xgb_model = joblib.load(xgb_path)
    feat_cols  = joblib.load(feat_path)

    # Use last fold
    last_fk = max(folds_data)
    y, _, _ = folds_data[last_fk]

    # Reconstruct X for last fold
    manifest_p = model_dir.parent / "data" / "training" / "v3" / "splits_manifest.json"
    # Try to find manifest
    for candidate in [
        model_dir.parent / "data" / "training" / "v3" / "splits_manifest.json",
        Path("data/training/v3/splits_manifest.json"),
    ]:
        if candidate.exists():
            splits = json.loads(candidate.read_text())["splits"]
            val_path = None
            for s in splits:
                if s["fold"] == last_fk:
                    val_path = Path(s["val_jsonl"])
                    break
            if val_path and val_path.exists():
                X, y_check = _load_xy(val_path, feat_cols)
                break
    else:
        log.warning("Cannot find val file for permutation importance — skipping")
        return []

    log.info("Computing permutation importance on fold %d (n=%d)...", last_fk, len(y_check))
    result = permutation_importance(
        xgb_model, X, y_check,
        n_repeats=5, random_state=42, n_jobs=1, scoring="roc_auc",
    )

    idx = np.argsort(result.importances_mean)[::-1][:top_n]
    top_feats = [feat_cols[i] for i in idx]
    top_means = result.importances_mean[idx]
    top_stds  = result.importances_std[idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    ax.barh(range(len(top_feats)), top_means[::-1],
            xerr=top_stds[::-1], color="#55A868", alpha=0.8,
            ecolor="gray", capsize=3)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set(xlabel="Mean decrease in ROC-AUC (±std)",
           title=f"Permutation Importance — Top {top_n} (Fold {last_fk}, 5 repeats)")
    ax.axvline(0, color="gray", ls="--", lw=1)
    fig.tight_layout()
    p = out_dir / "permutation_importance.png"
    _savefig(fig, p)
    return [p]


# ── 8. SHAP Summary ───────────────────────────────────────────────────────────

def plot_shap(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_dir: Path,
    out_dir: Path,
    max_display: int = 20,
) -> List[Path]:
    try:
        import shap
    except ImportError:
        log.info("shap not installed — skipping SHAP plot")
        return []

    import joblib

    xgb_path  = model_dir / "v3_xgb_model.joblib"
    feat_path = model_dir / "v3_feature_names.joblib"
    if not xgb_path.exists():
        return []

    xgb_model = joblib.load(xgb_path)
    feat_cols  = joblib.load(feat_path) if feat_path.exists() else None

    if not folds_data:
        return []

    last_fk = max(folds_data)
    # Find val file for last fold
    for candidate in [Path("data/training/v3/splits_manifest.json")]:
        if not candidate.exists():
            continue
        splits = json.loads(candidate.read_text())["splits"]
        val_path = None
        for s in splits:
            if s["fold"] == last_fk:
                val_path = Path(s["val_jsonl"])
                break
        if val_path and val_path.exists():
            X, _ = _load_xy(val_path, feat_cols, max_rows=2000)
            break
    else:
        return []

    log.info("Computing SHAP values (n=%d)...", len(X))
    try:
        # Get the xgboost classifier from pipeline
        xgb_clf = xgb_model[-1]
        X_imputed = xgb_model[:-1].transform(X)  # Apply preprocessor
        explainer = shap.TreeExplainer(xgb_clf)
        shap_values = explainer.shap_values(X_imputed)

        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.35)))
        shap.summary_plot(
            shap_values, X_imputed,
            feature_names=feat_cols if feat_cols else None,
            max_display=max_display,
            show=False, plot_type="dot",
        )
        p = out_dir / "shap_summary.png"
        plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        log.info("Saved: %s", p)
        return [p]
    except Exception as e:
        log.warning("SHAP plot failed: %s", e)
        return []


# ── Per-fold metrics table ────────────────────────────────────────────────────

def plot_metrics_summary(
    metrics_path: Path,
    out_dir: Path,
) -> List[Path]:
    if not metrics_path.exists():
        return []

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    fold_mets = data.get("xgb", {}).get("fold_metrics", [])
    if not fold_mets:
        return []

    keys = ["roc_auc", "pr_auc", "brier", "ece", "fpr_at_tpr80", "f1_t05"]
    labels = ["ROC-AUC", "PR-AUC", "Brier↓", "ECE↓", "FPR@TPR80↓", "F1@0.5"]
    fold_ids = [m["label"] for m in fold_mets]
    n_folds = len(fold_mets)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax_i, (k, lbl) in enumerate(zip(keys, labels)):
        ax = axes[ax_i]
        vals = [m.get(k, np.nan) for m in fold_mets]
        colors = [FOLD_COLORS[i % len(FOLD_COLORS)] for i in range(n_folds)]
        bars = ax.bar(fold_ids, vals, color=colors, alpha=0.85, edgecolor="white")
        mean_val = np.nanmean(vals)
        ax.axhline(mean_val, color="navy", ls="--", lw=1.5,
                   label=f"Mean={mean_val:.3f}")
        ax.set(title=lbl, ylabel=lbl)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Per-Fold Metrics — v3 XGB Calibrated", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    p = out_dir / "metrics_per_fold.png"
    _savefig(fig, p)
    return [p]


# ── Main entry ─────────────────────────────────────────────────────────────────

def generate_all(
    manifest_path: Path,
    model_dir: Path,
    metrics_path: Path,
    out_dir: Path,
    perm_importance: bool = False,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    # Load thresholds
    thresholds = {}
    thr_path = model_dir / "v3_thresholds.json"
    if thr_path.exists():
        thresholds = json.loads(thr_path.read_text())

    # Load fold predictions
    folds_data = load_fold_predictions(manifest_path, model_dir)

    if folds_data:
        for p in plot_roc_curves(folds_data, out_dir):
            generated["roc_curves"] = p
        for p in plot_pr_curves(folds_data, out_dir):
            generated["pr_curves"] = p
        for p in plot_calibration(folds_data, out_dir):
            generated["calibration"] = p
        for p in plot_prob_distributions(folds_data, out_dir, thresholds):
            generated["prob_distributions"] = p
        for p in plot_lift_curve(folds_data, out_dir):
            generated["lift_curve"] = p
        for p in plot_confusion_matrix(folds_data, out_dir, thresholds):
            generated["confusion_matrices"] = p
        for p in plot_shap(folds_data, model_dir, out_dir):
            generated["shap_summary"] = p
        if perm_importance:
            for p in plot_permutation_importance(folds_data, model_dir, out_dir):
                generated["permutation_importance"] = p

    for p in plot_feature_importance(model_dir, out_dir):
        generated["feature_importance"] = p

    for p in plot_metrics_summary(metrics_path, out_dir):
        generated["metrics_per_fold"] = p

    log.info("ML plots generated: %d figures in %s", len(generated), out_dir)
    return generated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Generate v3 ML visualization plots")
    ap.add_argument("--manifest", default="data/training/v3/splits_manifest.json")
    ap.add_argument("--models",   default="models/v3")
    ap.add_argument("--metrics",  default="data/metrics/train_v3_report.json")
    ap.add_argument("--out",      default="data/metrics/v3/plots")
    ap.add_argument("--perm_importance", action="store_true",
                    help="Include permutation importance (slow)")
    args = ap.parse_args()

    generated = generate_all(
        manifest_path=Path(args.manifest),
        model_dir=Path(args.models),
        metrics_path=Path(args.metrics),
        out_dir=Path(args.out),
        perm_importance=args.perm_importance,
    )

    print(f"\n✅ {len(generated)} ML plots generated:")
    for name, p in sorted(generated.items()):
        print(f"   {name:<35} → {p}")


if __name__ == "__main__":
    main()
