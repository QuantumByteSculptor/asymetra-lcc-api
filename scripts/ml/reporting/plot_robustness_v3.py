"""
scripts/ml/reporting/plot_robustness_v3.py
==========================================
Scientific robustness upgrade for the v3 ML pipeline.

Objectives:
  1. Most-recent-fold performance table  → recent_fold_table.png
  2. Bootstrap CI on AUC + PR-AUC       → auc_bootstrap_hist.png
  3. Sharpe significance test (bootstrap → sharpe_bootstrap_hist.png
  4. Per-fold confusion metrics          → confusion_metrics_per_fold.png

All PNGs saved to data/metrics/v3/plots/  (same folder as plot_ml_v3.py).

Usage:
    python scripts/ml/reporting/plot_robustness_v3.py \\
        --metrics  data/metrics/train_v3_report.json \\
        --backtest data/metrics/backtest_v3.json \\
        --manifest data/training/v3/splits_manifest.json \\
        --models   models/v3 \\
        --out      data/metrics/v3/plots

Gracefully produces stub figures when model / val files are unavailable.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

log = logging.getLogger("plot_robustness_v3")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
DPI = 150

COLOR_DARK    = (30, 45, 70)
COLOR_ACCENT  = "#3498db"
COLOR_ORANGE  = "#e67e22"
COLOR_GREEN   = "#27ae60"
COLOR_RED     = "#c0392b"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int = DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", path)


def _no_data_stub(title: str, path: Path, msg: str = "Model/data not available") -> Path:
    """Generate a clearly-labelled stub figure when raw data is missing."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.4, f"[{msg}]", ha="center", va="center", fontsize=11,
            color="gray", style="italic", transform=ax.transAxes)
    ax.text(0.5, 0.25,
            "Run with full val.jsonl + trained model to obtain bootstrap estimates.",
            ha="center", va="center", fontsize=9, color="#999",
            transform=ax.transAxes)
    _savefig(fig, path)
    return path


# ── Bootstrap core ─────────────────────────────────────────────────────────────

def _bootstrap_metric(
    fn,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    min_pos: int = 5,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Bootstrap a scalar metric `fn(y_true, y_prob)` with replacement.

    Returns:
        (point_estimate, ci_lo, ci_hi, boot_samples)
        ci_lo / ci_hi: 2.5th / 97.5th percentiles (95% CI).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = fn(y_true, y_prob)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() < min_pos or (len(yt) - yt.sum()) < min_pos:
            samples.append(point)  # degenerate resample — reuse point estimate
            continue
        try:
            samples.append(fn(yt, yp))
        except Exception:
            samples.append(np.nan)
    samples_arr = np.array(samples)
    valid = samples_arr[np.isfinite(samples_arr)]
    if len(valid) == 0:
        return point, np.nan, np.nan, samples_arr
    ci_lo = float(np.percentile(valid, 2.5))
    ci_hi = float(np.percentile(valid, 97.5))
    return point, ci_lo, ci_hi, samples_arr


def _compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Weighted Expected Calibration Error.

    Bins are equal-width over [0,1].  Only non-empty bins contribute.
    Uses bin-size weighting to avoid small-sample bias.
    Never rounds to 0 because it operates on raw float arithmetic.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        ece += (n_bin / n) * abs(frac_pos - mean_prob)
    return float(ece)


def _roc_auc_fn(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def _pr_auc_fn(y, p):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p)


def _sharpe_fn(rets: np.ndarray, periods_year: float = 12.6) -> float:
    """Annualised Sharpe from array of period returns."""
    if len(rets) < 5:
        return np.nan
    std = rets.std(ddof=1)
    if std < 1e-12:
        return np.nan
    return (rets.mean() / std) * np.sqrt(periods_year)


# ── Obj 1 — Recent Fold Performance Table ─────────────────────────────────────

def plot_recent_fold_table(
    metrics_data: Dict,
    out_dir: Path,
    folds_data: Optional[Dict] = None,
) -> List[Path]:
    """
    Extract the most-recent fold metrics and render as a styled table PNG.

    Priority:
      1. folds_data (raw predictions from val.jsonl) — preferred.
         Most-recent fold = max(folds_data.keys()), e.g. fold 5 (2023-2025).
         ECE recomputed via _compute_ece (weighted, no empty-bin bias).
      2. metrics_data (train_v3_report.json) — fallback only.

    Columns: ROC-AUC, PR-AUC, Brier, ECE, Precision_non_ok, Recall_non_ok, FPR
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.metrics import confusion_matrix

    # ── Source 1: actual fold predictions (preferred) ──────────────────────────
    if folds_data:
        last_fk = max(folds_data.keys())
        y_true, y_prob, _ = folds_data[last_fk]
        fold_label = f"fold_{last_fk}"
        log.info("plot_recent_fold_table: using folds_data[%d] n=%d", last_fk, len(y_true))

        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float("nan")
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            brier = float("nan")

        ece = _compute_ece(y_true, y_prob)

        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)
        if y_true.sum() > 0 and (len(y_true) - y_true.sum()) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        recall_non_ok    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision_non_ok = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        fpr              = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        n_total = len(y_true)
        n_pos   = int(y_true.sum())
        n_neg   = n_total - n_pos

        last_fold_row = {
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "brier": brier,     "ece":    ece,
            "n": n_total, "n_pos": n_pos, "n_neg": n_neg,
        }
        calibrated = metrics_data.get("xgb", {}).get("final_calibrated", {})

    # ── Source 2: metrics JSON (fallback — may have ECE=0 artefact) ────────────
    else:
        fold_metrics = metrics_data.get("xgb", {}).get("fold_metrics", [])
        if not fold_metrics:
            log.warning("No fold data — skipping recent fold table")
            return []

        def _fold_sort_key(m):
            lbl = m.get("label", "")
            digits = "".join(filter(str.isdigit, lbl.split("fold")[-1]))
            return int(digits) if digits else 0

        last_fold_row = sorted(fold_metrics, key=_fold_sort_key)[-1]
        fold_label    = last_fold_row.get("label", "last fold")

        tp = last_fold_row.get("tp", 0)
        fp = last_fold_row.get("fp", 0)
        fn = last_fold_row.get("fn", 0)
        tn = last_fold_row.get("tn", 0)

        recall_non_ok    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision_non_ok = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        fpr              = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        calibrated = metrics_data.get("xgb", {}).get("final_calibrated", {})

    # Build display rows
    def _fmt(v, pct=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if pct:
            return f"{v:.1%}"
        return f"{v:.4f}"

    headers = ["Metric", f"Most-Recent Fold ({fold_label})", "XGB + Calibrated (final)"]
    col_widths_fig = [0.22, 0.39, 0.39]

    rows = [
        ["ROC-AUC",          _fmt(last_fold_row.get("roc_auc")),
                             _fmt(calibrated.get("roc_auc"))],
        ["PR-AUC",           _fmt(last_fold_row.get("pr_auc")),
                             _fmt(calibrated.get("pr_auc"))],
        ["Brier Score ↓",    _fmt(last_fold_row.get("brier")),
                             _fmt(calibrated.get("brier"))],
        ["ECE ↓ (weighted)", _fmt(last_fold_row.get("ece")),
                             _fmt(calibrated.get("ece"))],
        ["Precision non-OK", _fmt(precision_non_ok),          "—"],
        ["Recall non-OK",    _fmt(recall_non_ok),              "—"],
        ["FPR (at t=0.5)",   _fmt(fpr),                        _fmt(calibrated.get("fpr_at_tpr80"))],
        ["N (total)",        str(last_fold_row.get("n", "?")),     str(calibrated.get("n", "?"))],
        ["N pos / neg",
         f"{last_fold_row.get('n_pos','?')} / {last_fold_row.get('n_neg','?')}",
         f"{calibrated.get('n_pos','?')} / {calibrated.get('n_neg','?')}"],
    ]

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(rows) + 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    n_cols = len(headers)
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#1e2d46")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows
    highlight_rows = {4, 5, 6}  # Precision, Recall, FPR (new additions)
    for i in range(1, len(rows) + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            if i - 1 in highlight_rows:
                cell.set_facecolor("#fef9e7")  # soft yellow highlight for new metrics
            elif i % 2 == 0:
                cell.set_facecolor("#f5f8fc")
            else:
                cell.set_facecolor("white")

    # Set column widths proportionally
    for j, w in enumerate(col_widths_fig):
        for i in range(len(rows) + 1):
            table[(i, j)].set_width(w)

    ax.set_title(
        f"Most-Recent Regime Performance — {fold_label} (out-of-sample)",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    p = out_dir / "recent_fold_table.png"
    _savefig(fig, p)
    return [p]


# ── Obj 2 — Bootstrap CI: AUC + PR-AUC ────────────────────────────────────────

def plot_auc_bootstrap_hist(
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    n_boot: int = 1000,
) -> List[Path]:
    """
    Bootstrap ROC-AUC and PR-AUC on the most-recent fold (1000 resamples).
    Generates auc_bootstrap_hist.png with side-by-side histograms + 95% CI bands.
    """
    if not folds_data:
        return [_no_data_stub(
            "Bootstrap AUC / PR-AUC Confidence Intervals",
            out_dir / "auc_bootstrap_hist.png",
        )]

    last_fk = max(folds_data)
    y_true, y_prob, _ = folds_data[last_fk]
    fold_label = f"Fold {last_fk}"

    log.info("Bootstrapping AUC + PR-AUC (n_boot=%d, fold=%s)...", n_boot, fold_label)

    auc_pt, auc_lo, auc_hi, auc_samples = _bootstrap_metric(
        _roc_auc_fn, y_true, y_prob, n_boot=n_boot, seed=42,
    )
    ap_pt, ap_lo, ap_hi, ap_samples     = _bootstrap_metric(
        _pr_auc_fn, y_true, y_prob, n_boot=n_boot, seed=43,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def _hist_panel(ax, samples, point, ci_lo, ci_hi, title, color):
        valid = samples[np.isfinite(samples)]
        ax.hist(valid, bins=50, color=color, alpha=0.75, edgecolor="white", lw=0.3,
                density=True, label=f"Bootstrap distribution\n(n={n_boot:,})")
        ax.axvline(point, color="black", lw=2, ls="-",
                   label=f"Point estimate: {point:.4f}")
        ax.axvline(ci_lo, color="crimson", lw=1.5, ls="--",
                   label=f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        ax.axvline(ci_hi, color="crimson", lw=1.5, ls="--")
        ax.fill_betweenx(
            [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
            ci_lo, ci_hi, color="crimson", alpha=0.08,
        )
        ax.set(title=title, xlabel="Metric value", ylabel="Density")
        ax.legend(fontsize=8.5)

    _hist_panel(
        axes[0], auc_samples, auc_pt, auc_lo, auc_hi,
        f"ROC-AUC Bootstrap ({fold_label})", COLOR_ACCENT,
    )
    _hist_panel(
        axes[1], ap_samples, ap_pt, ap_lo, ap_hi,
        f"PR-AUC Bootstrap ({fold_label})", COLOR_ORANGE,
    )

    fig.suptitle(
        f"Bootstrap Confidence Intervals — {fold_label} (1000 resamples, 95% CI)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = out_dir / "auc_bootstrap_hist.png"
    _savefig(fig, p)

    # Log summary
    log.info("AUC: %.4f [%.4f, %.4f]", auc_pt, auc_lo, auc_hi)
    log.info("AP:  %.4f [%.4f, %.4f]", ap_pt,  ap_lo,  ap_hi)

    # Store results JSON alongside
    results = {
        "fold": int(last_fk),
        "n_boot": n_boot,
        "roc_auc": {"mean": float(auc_pt), "ci_lo_95": float(auc_lo), "ci_hi_95": float(auc_hi)},
        "pr_auc":  {"mean": float(ap_pt),  "ci_lo_95": float(ap_lo),  "ci_hi_95": float(ap_hi)},
    }
    (out_dir / "bootstrap_auc_ci.json").write_text(json.dumps(results, indent=2))

    return [p]


# ── Obj 3 — Sharpe Bootstrap + Significance Test ──────────────────────────────

def _build_signal_returns(
    df,
    t_lo: float,
    t_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From the financial signal DataFrame, compute 20d signal and baseline return arrays.
    Returns (signal_rets, baseline_rets).
    """
    df = df.copy()
    df["exposure"] = 1.0
    df.loc[df["proba_non_ok"] >= t_hi, "exposure"] = 0.0
    df.loc[(df["proba_non_ok"] >= t_lo) & (df["proba_non_ok"] < t_hi), "exposure"] = 0.5
    df["signal_ret"]   = df["forward_return_20d"] * df["exposure"]
    df["baseline_ret"] = df["forward_return_20d"]
    return (
        df["signal_ret"].dropna().values,
        df["baseline_ret"].dropna().values,
    )


def compute_sharpe_significance(
    signal_rets: np.ndarray,
    baseline_rets: np.ndarray,
    n_boot: int = 1000,
    seed: int = 44,
    periods_year: float = 12.6,
) -> Dict:
    """
    Bootstrap the difference in Sharpe: signal - baseline.

    Returns dict with:
        signal_sharpe, baseline_sharpe,
        mean_diff, ci_lo, ci_hi,
        p_value (fraction of bootstrap diffs <= 0)
    """
    rng = np.random.default_rng(seed)
    n_sig = len(signal_rets)
    n_bm  = len(baseline_rets)

    sig_sharpe = _sharpe_fn(signal_rets, periods_year)
    bm_sharpe  = _sharpe_fn(baseline_rets, periods_year)

    diffs = []
    for _ in range(n_boot):
        s_boot = signal_rets[rng.integers(0, n_sig, n_sig)]
        b_boot = baseline_rets[rng.integers(0, n_bm, n_bm)]
        sh_s = _sharpe_fn(s_boot, periods_year)
        sh_b = _sharpe_fn(b_boot, periods_year)
        if np.isfinite(sh_s) and np.isfinite(sh_b):
            diffs.append(sh_s - sh_b)

    diffs_arr = np.array(diffs)
    valid = diffs_arr[np.isfinite(diffs_arr)]

    if len(valid) == 0:
        return {
            "signal_sharpe": float(sig_sharpe), "baseline_sharpe": float(bm_sharpe),
            "mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "p_value": float("nan"), "n_boot": n_boot,
        }

    mean_diff = float(np.mean(valid))
    ci_lo     = float(np.percentile(valid, 2.5))
    ci_hi     = float(np.percentile(valid, 97.5))
    p_value   = float(np.mean(valid <= 0))   # fraction of bootstrap diffs <= 0

    return {
        "signal_sharpe":   float(sig_sharpe),
        "baseline_sharpe": float(bm_sharpe),
        "mean_diff":       mean_diff,
        "ci_lo":           ci_lo,
        "ci_hi":           ci_hi,
        "p_value":         p_value,
        "n_boot":          n_boot,
        "n_valid":         len(valid),
    }


def plot_sharpe_bootstrap_hist(
    df,           # Optional[pd.DataFrame] from _load_signal_series
    t_lo: float,
    t_hi: float,
    out_dir: Path,
    n_boot: int = 1000,
) -> List[Path]:
    """
    Bootstrap Sharpe of signal vs baseline and the difference distribution.
    Saves sharpe_bootstrap_hist.png and bootstrap_sharpe_significance.json.
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return [_no_data_stub(
            "Bootstrap Sharpe Ratio — Significance Test",
            out_dir / "sharpe_bootstrap_hist.png",
        )]

    signal_rets, baseline_rets = _build_signal_returns(df, t_lo, t_hi)

    if len(signal_rets) < 20:
        return [_no_data_stub(
            "Bootstrap Sharpe Ratio — Significance Test",
            out_dir / "sharpe_bootstrap_hist.png",
            msg=f"Insufficient data (n={len(signal_rets)})",
        )]

    log.info("Bootstrapping Sharpe difference (n_boot=%d, n=%d)...",
             n_boot, len(signal_rets))

    stats = compute_sharpe_significance(signal_rets, baseline_rets, n_boot=n_boot)

    # Re-run bootstrap for histogram
    rng = np.random.default_rng(44)
    n_sig, n_bm = len(signal_rets), len(baseline_rets)
    periods_year = 12.6

    sig_samples, bm_samples, diff_samples = [], [], []
    for _ in range(n_boot):
        s_b = signal_rets[rng.integers(0, n_sig, n_sig)]
        b_b = baseline_rets[rng.integers(0, n_bm, n_bm)]
        sh_s = _sharpe_fn(s_b, periods_year)
        sh_b = _sharpe_fn(b_b, periods_year)
        if np.isfinite(sh_s) and np.isfinite(sh_b):
            sig_samples.append(sh_s)
            bm_samples.append(sh_b)
            diff_samples.append(sh_s - sh_b)

    sig_arr  = np.array(sig_samples)
    bm_arr   = np.array(bm_samples)
    diff_arr = np.array(diff_samples)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _sharpe_hist(ax, samples, point, title, color):
        valid = samples[np.isfinite(samples)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            return
        ci_lo = np.percentile(valid, 2.5)
        ci_hi = np.percentile(valid, 97.5)
        ax.hist(valid, bins=50, color=color, alpha=0.75,
                edgecolor="white", lw=0.3, density=True)
        ax.axvline(point,  color="black", lw=2.0, ls="-",  label=f"Point: {point:.3f}")
        ax.axvline(ci_lo,  color="crimson", lw=1.5, ls="--")
        ax.axvline(ci_hi,  color="crimson", lw=1.5, ls="--",
                   label=f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.fill_betweenx([0, ax.get_ylim()[1] or 1], ci_lo, ci_hi,
                         color="crimson", alpha=0.08)
        ax.set(title=title, xlabel="Annualised Sharpe")
        ax.legend(fontsize=8.5)

    _sharpe_hist(axes[0], sig_arr,  stats["signal_sharpe"],
                 "Signal v3 Sharpe", COLOR_ACCENT)
    _sharpe_hist(axes[1], bm_arr,   stats["baseline_sharpe"],
                 "Always-OK Sharpe", COLOR_ORANGE)

    # Difference distribution
    ax_d = axes[2]
    valid_diff = diff_arr[np.isfinite(diff_arr)]
    if len(valid_diff) > 0:
        ax_d.hist(valid_diff, bins=50, color=COLOR_GREEN, alpha=0.75,
                  edgecolor="white", lw=0.3, density=True,
                  label=f"Diff distribution (n={n_boot:,})")
        ax_d.axvline(0, color="black", lw=2, ls="-", label="0 (no difference)")
        ax_d.axvline(stats["mean_diff"],  color="navy", lw=2, ls="-.",
                     label=f"Mean diff: {stats['mean_diff']:.3f}")
        ax_d.axvline(stats["ci_lo"], color="crimson", lw=1.5, ls="--")
        ax_d.axvline(stats["ci_hi"], color="crimson", lw=1.5, ls="--",
                     label=f"95% CI: [{stats['ci_lo']:.3f}, {stats['ci_hi']:.3f}]")
        # Shade negative region (null hypothesis)
        neg_x = np.linspace(valid_diff.min(), 0, 100)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(valid_diff)
            ax_d.fill_between(neg_x, kde(neg_x), alpha=0.2, color="crimson",
                               label=f"p-value area: {stats['p_value']:.3f}")
        except Exception:
            pass

        p_color = COLOR_GREEN if stats["p_value"] < 0.05 else COLOR_RED
        ax_d.set_title(
            f"Sharpe Difference (Signal - Baseline)\n"
            f"p-value = {stats['p_value']:.3f}  "
            f"({'significant' if stats['p_value'] < 0.05 else 'not significant'} at 5%)",
            fontsize=10, fontweight="bold",
            color=p_color,
        )
        ax_d.set(xlabel="Sharpe difference (ann.)")
        ax_d.legend(fontsize=8)

    fig.suptitle(
        f"Bootstrap Sharpe Analysis (n_boot={n_boot:,}, 95% CI) — Signal v3",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = out_dir / "sharpe_bootstrap_hist.png"
    _savefig(fig, p)

    # Save stats JSON
    stats_path = out_dir / "bootstrap_sharpe_significance.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Sharpe significance: diff=%.3f p=%.3f [%.3f, %.3f]",
             stats["mean_diff"], stats["p_value"], stats["ci_lo"], stats["ci_hi"])

    return [p]


# ── Obj 4 — Confusion Metrics per Fold ────────────────────────────────────────

def plot_confusion_metrics_per_fold(
    metrics_data: Dict,
    folds_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    threshold: float = 0.5,
) -> List[Path]:
    """
    Per-fold table + bar chart showing:
        Recall_non_ok, Precision_non_ok, FPR, F1_non_ok
    Sources:
        1. folds_data (raw predictions) — preferred, computed at given threshold
        2. fold_metrics from JSON (TP/FP/FN/TN already stored at t=0.5)
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    fold_metrics = metrics_data.get("xgb", {}).get("fold_metrics", [])
    if not fold_metrics and not folds_data:
        log.warning("No fold data — skipping confusion metrics per fold")
        return []

    # Build rows from the best available source
    rows = []   # {label, recall, precision, fpr, f1}

    if folds_data:
        # Compute from raw predictions at the given threshold
        for fk, (y, y_prob, _) in sorted(folds_data.items()):
            y_pred = (y_prob >= threshold).astype(int)
            if y.sum() == 0 or (len(y) - y.sum()) == 0:
                continue
            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            fpr_ = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else float("nan"))
            rows.append({
                "label": f"Fold {fk}",
                "recall":    rec,
                "precision": prec,
                "fpr":       fpr_,
                "f1":        f1,
            })
    else:
        # Fallback: derive from stored TP/FP/FN/TN in JSON
        def _sort_key(m):
            lbl = m.get("label", "")
            digits = "".join(filter(str.isdigit, lbl.split("fold")[-1]))
            return int(digits) if digits else 0

        for m in sorted(fold_metrics, key=_sort_key):
            tp = m.get("tp", 0)
            fp = m.get("fp", 0)
            fn = m.get("fn", 0)
            tn = m.get("tn", 0)
            rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            fpr_ = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            f1   = (2 * prec * rec / (prec + rec)
                    if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0)
                    else float("nan"))
            rows.append({
                "label": m.get("label", "?"),
                "recall":    rec,
                "precision": prec,
                "fpr":       fpr_,
                "f1":        f1,
            })

    if not rows:
        log.warning("No fold rows computed — skipping confusion_metrics_per_fold")
        return []

    labels     = [r["label"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    precisions = [r["precision"] for r in rows]
    fprs       = [r["fpr"]       for r in rows]
    f1s        = [r["f1"]        for r in rows]

    x = np.arange(len(labels))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: grouped bar chart ──
    ax = axes[0]
    bar_recall = ax.bar(x - 1.5 * width, recalls,    width=width,
                        label="Recall (non-OK)",    color="#4C72B0", alpha=0.88)
    bar_prec   = ax.bar(x - 0.5 * width, precisions, width=width,
                        label="Precision (non-OK)", color="#DD8452", alpha=0.88)
    bar_f1     = ax.bar(x + 0.5 * width, f1s,        width=width,
                        label="F1 (non-OK)",         color="#55A868", alpha=0.88)
    bar_fpr    = ax.bar(x + 1.5 * width, fprs,        width=width,
                        label="FPR (false alarm)",   color="#C44E52", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set(title=f"Confusion Metrics per Fold (threshold={threshold:.2f})",
           ylabel="Metric value", ylim=[0, 1.1])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.legend(fontsize=9, loc="upper right")

    for bars, vals in [(bar_recall, recalls), (bar_prec, precisions),
                       (bar_f1, f1s), (bar_fpr, fprs)]:
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        min(bar.get_height() + 0.015, 1.05),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    # ── Right panel: table ──
    ax_t = axes[1]
    ax_t.axis("off")

    def _pct(v):
        return f"{v:.1%}" if np.isfinite(v) else "N/A"

    table_headers = ["Fold", "Recall ↑", "Precision ↑", "FPR ↓", "F1 ↑"]
    table_rows = [
        [r["label"], _pct(r["recall"]), _pct(r["precision"]),
         _pct(r["fpr"]), _pct(r["f1"])]
        for r in rows
    ]
    # Summary row
    def _avg(vals):
        v = [x for x in vals if np.isfinite(x)]
        return np.mean(v) if v else float("nan")
    table_rows.append([
        "MEAN",
        _pct(_avg(recalls)), _pct(_avg(precisions)),
        _pct(_avg(fprs)), _pct(_avg(f1s)),
    ])

    t = ax_t.table(
        cellText=table_rows,
        colLabels=table_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)

    # Style
    for j in range(len(table_headers)):
        t[(0, j)].set_facecolor("#1e2d46")
        t[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_rows) + 1):
        for j in range(len(table_headers)):
            if i == len(table_rows):  # summary row
                t[(i, j)].set_facecolor("#d5e8d4")
                t[(i, j)].set_text_props(fontweight="bold")
            elif i % 2 == 0:
                t[(i, j)].set_facecolor("#f5f8fc")

    ax_t.set_title("Numeric Summary", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle("Per-Fold Confusion Metrics — v3 XGB (non-OK class)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "confusion_metrics_per_fold.png"
    _savefig(fig, p)
    return [p]


# ── Master entry ───────────────────────────────────────────────────────────────

def generate_all_robustness(
    folds_data: Dict,
    metrics_data: Dict,
    df_signal,          # Optional[pd.DataFrame]
    t_lo: float,
    t_hi: float,
    out_dir: Path,
    n_boot: int = 1000,
) -> Dict[str, Path]:
    """Run all robustness objectives and return {name: path} dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    log.info("Obj 1 — Recent fold performance table (fold_data priority)...")
    for p in plot_recent_fold_table(metrics_data, out_dir, folds_data=folds_data):
        generated["recent_fold_table"] = p

    log.info("Obj 2 — Bootstrap AUC CI (n_boot=%d)...", n_boot)
    for p in plot_auc_bootstrap_hist(folds_data, out_dir, n_boot=n_boot):
        generated["auc_bootstrap_hist"] = p

    log.info("Obj 3 — Bootstrap Sharpe significance...")
    for p in plot_sharpe_bootstrap_hist(df_signal, t_lo, t_hi, out_dir, n_boot=n_boot):
        generated["sharpe_bootstrap_hist"] = p

    log.info("Obj 4 — Confusion metrics per fold...")
    for p in plot_confusion_metrics_per_fold(metrics_data, folds_data, out_dir):
        generated["confusion_metrics_per_fold"] = p

    log.info("Robustness plots generated: %d figures", len(generated))
    return generated


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="v3 scientific robustness plots")
    ap.add_argument("--metrics",  default="data/metrics/train_v3_report.json")
    ap.add_argument("--backtest", default="data/metrics/backtest_v3.json")
    ap.add_argument("--manifest", default="data/training/v3/splits_manifest.json")
    ap.add_argument("--models",   default="models/v3")
    ap.add_argument("--out",      default="data/metrics/v3/plots")
    ap.add_argument("--n_boot",   type=int, default=1000)
    args = ap.parse_args()

    metrics_data  = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else {}
    backtest_data = json.loads(Path(args.backtest).read_text()) if Path(args.backtest).exists() else {}
    manifest_path = Path(args.manifest)
    model_dir     = Path(args.models)
    out_dir       = Path(args.out)

    # Load fold predictions (if model available)
    folds_data: Dict = {}
    try:
        from scripts.ml.reporting.plot_ml_v3 import load_fold_predictions
        if manifest_path.exists() and model_dir.exists():
            folds_data = load_fold_predictions(manifest_path, model_dir)
    except Exception as e:
        log.warning("Could not load fold predictions: %s", e)

    # Load signal DataFrame (if model + manifest available)
    df_signal = None
    try:
        from scripts.ml.reporting.plot_financial_v3 import _load_signal_series, _apply_signal
        if manifest_path.exists() and model_dir.exists():
            df_signal = _load_signal_series(manifest_path, model_dir)
    except Exception as e:
        log.warning("Could not load signal series: %s", e)

    # Thresholds
    t_lo, t_hi = 0.5, 0.65
    thr_path = model_dir / "v3_thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        t_lo = thr.get("t_lo", 0.5)
        t_hi = thr.get("t_hi", 0.65)

    if df_signal is not None:
        from scripts.ml.reporting.plot_financial_v3 import _apply_signal
        df_signal = _apply_signal(df_signal, t_lo, t_hi)

    generated = generate_all_robustness(
        folds_data=folds_data,
        metrics_data=metrics_data,
        df_signal=df_signal,
        t_lo=t_lo,
        t_hi=t_hi,
        out_dir=out_dir,
        n_boot=args.n_boot,
    )

    print(f"\n(OK) {len(generated)} robustness plots generated:")
    for name, p in sorted(generated.items()):
        print(f"   {name:<40} -> {p}")


if __name__ == "__main__":
    main()
