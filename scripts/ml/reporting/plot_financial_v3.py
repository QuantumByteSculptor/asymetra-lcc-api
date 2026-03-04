"""
scripts/ml/reporting/plot_financial_v3.py
==========================================
Phase 2 — Financial / quant visualizations for the v3 pipeline.

Generates:
  1. Cumulative log-return curves (signal_v3, always_ok, random)
  2. Drawdown comparison
  3. Return distribution by label class (ok / warn / block)
  4. Skip-rate over time (rolling 6-month)
  5. Rolling Sharpe 12 months
  6. Sharpe & CAGR by asset_type (barplot)

Outputs: data/metrics/v3/financial_plots/

Usage:
    python scripts/ml/reporting/plot_financial_v3.py \\
        --backtest data/metrics/backtest_v3.json \\
        --manifest data/training/v3/splits_manifest.json \\
        --models   models/v3 \\
        --out      data/metrics/v3/financial_plots

Gracefully falls back to JSON-only charts when JSONL + model are missing.
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
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

log = logging.getLogger("plot_financial_v3")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DPI = 150

_SIGNAL_COLOR  = "#1f77b4"
_ALWAYSOK_COLOR = "#ff7f0e"
_RANDOM_COLOR  = "#2ca02c"
_DRAWDOWN_COLORS = [_SIGNAL_COLOR, _ALWAYSOK_COLOR, _RANDOM_COLOR]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int = DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", path)


def _compute_dd_equity(simple_rets: np.ndarray) -> np.ndarray:
    """
    Drawdown path from simple-return array using the equity-curve method.
    Returns values in [-1, 0] (percentage units, e.g. -0.30 = -30%).
    """
    equity = np.cumprod(1.0 + np.clip(simple_rets, -0.999, 10.0))
    running_max = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(running_max, 1e-12)) - 1.0
    return dd


def _compute_series_metrics(
    sig_rets: np.ndarray,
    bm_rets: np.ndarray,
    periods_year: float = 252 / 20,
) -> dict:
    """
    Compute Sharpe, CAGR, MDD from the same return arrays used for the
    cumulative-return plot.  Used for consistency validation and metrics card.
    """
    def _sharpe(r):
        if len(r) < 5 or r.std(ddof=1) < 1e-12:
            return float("nan")
        return (r.mean() / r.std(ddof=1)) * np.sqrt(periods_year)

    def _cagr(r):
        if len(r) == 0:
            return float("nan")
        equity_end = float(np.prod(1.0 + np.clip(r, -0.999, 10.0)))
        n_years = len(r) / periods_year
        if n_years <= 0 or equity_end <= 0:
            return float("nan")
        return equity_end ** (1.0 / n_years) - 1.0

    def _mdd(r):
        dd = _compute_dd_equity(r)
        return float(np.min(dd)) if len(dd) > 0 else float("nan")

    return {
        "signal_sharpe":   _sharpe(sig_rets),
        "baseline_sharpe": _sharpe(bm_rets),
        "signal_cagr":     _cagr(sig_rets),
        "baseline_cagr":   _cagr(bm_rets),
        "signal_mdd":      _mdd(sig_rets),
        "baseline_mdd":    _mdd(bm_rets),
        "n_periods":       len(sig_rets),
    }


def _validate_finance_consistency(series_metrics: dict, backtest_json: dict) -> dict:
    """
    Cross-check time-series-derived metrics against the pre-computed JSON values.
    Returns a dict suitable for inclusion in sanity_report.json.
    """
    sig  = backtest_json.get("signal",    {})
    bm   = backtest_json.get("always_ok", {})

    sm = series_metrics
    checks = {}

    # MDD range
    checks["mdd_signal_valid_range"]   = (-1.0 <= sm.get("signal_mdd",   -999) <= 0.0)
    checks["mdd_baseline_valid_range"] = (-1.0 <= sm.get("baseline_mdd", -999) <= 0.0)
    checks["mdd_signal_json_valid"]    = (-1.0 <= sig.get("max_drawdown", -999) <= 0.0)

    # CAGR sign matches ending equity
    checks["signal_cagr_sign_ok"] = (
        (sm.get("signal_cagr", 0) > 0) == (sm.get("signal_mdd", 0) > -1.0)
        if sm.get("signal_cagr") is not None else "n/a"
    )

    # Sharpe comparison
    checks["signal_sharpe_series"]   = round(sm.get("signal_sharpe",   float("nan")), 4)
    checks["signal_sharpe_json"]     = round(float(sig.get("sharpe_ann", float("nan"))), 4)
    checks["baseline_sharpe_series"] = round(sm.get("baseline_sharpe", float("nan")), 4)
    checks["baseline_sharpe_json"]   = round(float(bm.get("sharpe_ann",  float("nan"))), 4)

    # MDD comparison (series-derived vs JSON)
    checks["signal_mdd_series"] = round(sm.get("signal_mdd",   float("nan")), 4)
    checks["signal_mdd_json"]   = round(float(sig.get("max_drawdown", float("nan"))), 4)

    checks["all_pass"] = (
        checks["mdd_signal_valid_range"]
        and checks["mdd_baseline_valid_range"]
    )
    return checks


def _rolling_sharpe(ret: pd.Series, window_periods: int, periods_year: float) -> pd.Series:
    mean  = ret.rolling(window_periods).mean()
    std   = ret.rolling(window_periods).std(ddof=1)
    return (mean / std.replace(0, np.nan)) * np.sqrt(periods_year)


# ── Load time-series signal from fold data ────────────────────────────────────

def _load_signal_series(
    manifest_path: Path,
    model_dir: Path,
    max_records: int = 60_000,
) -> Optional[pd.DataFrame]:
    """
    Load the last fold's val records, apply calibrated XGB model,
    reconstruct a time-indexed DataFrame:
      date | forward_return_20d | label | proba_non_ok | asset_type
    """
    try:
        import joblib
    except ImportError:
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    # Use last fold (largest, most recent)
    last_split = sorted(splits, key=lambda s: s["fold"])[-1]
    val_path = Path(last_split["val_jsonl"])
    if not val_path.exists():
        log.warning("Val file missing: %s", val_path)
        return None

    xgb_path  = model_dir / "v3_xgb_model.joblib"
    cal_path  = model_dir / "v3_calibrator.joblib"
    feat_path = model_dir / "v3_feature_names.joblib"

    if not xgb_path.exists():
        log.warning("XGB model missing — cannot reconstruct signal series")
        return None

    xgb_model  = joblib.load(xgb_path)
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    feat_cols  = joblib.load(feat_path) if feat_path.exists() else None
    if feat_cols is None:
        return None

    rows = []
    X_rows = []
    with val_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            fwd = rec.get("forward_return_20d")
            date = rec.get("window_end_date")
            label = rec.get("label", "ok")
            feats = rec.get("features", {})
            asset_type = feats.get("asset_type", "unknown") or "unknown"

            if date is None or fwd is None or not math.isfinite(fwd):
                continue

            feat_row = []
            for col in feat_cols:
                v = feats.get(col)
                if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                    feat_row.append(float(v))
                else:
                    feat_row.append(np.nan)

            rows.append({
                "date": pd.Timestamp(date),
                "forward_return_20d": fwd,
                "label": label,
                "asset_type": asset_type,
            })
            X_rows.append(feat_row)

    if not rows:
        return None

    X = np.array(X_rows, dtype=np.float32)
    xgb_raw = xgb_model.predict_proba(X)[:, 1]
    xgb_cal = calibrator.predict(xgb_raw) if calibrator is not None else xgb_raw

    df = pd.DataFrame(rows)
    df["proba_non_ok"] = xgb_cal
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _apply_signal(df: pd.DataFrame, t_lo: float, t_hi: float) -> pd.DataFrame:
    """Add exposure column based on thresholds."""
    def _exposure(p: float) -> float:
        if p >= t_hi:
            return 0.0   # block
        if p >= t_lo:
            return 0.5   # warn
        return 1.0       # ok

    df["exposure"] = df["proba_non_ok"].apply(_exposure)
    df["signal_ret"] = df["forward_return_20d"] * df["exposure"]
    return df


# ── 1. Cumulative Log-Return Curves ───────────────────────────────────────────

def plot_cumulative_returns(
    df: Optional[pd.DataFrame],
    backtest_json: Dict,
    out_dir: Path,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(12, 6))

    if df is not None and "signal_ret" in df.columns:
        # Use reconstructed time series
        df_grp = df.groupby("date").agg(
            signal_ret=("signal_ret", "mean"),
            always_ok_ret=("forward_return_20d", "mean"),
        ).reset_index().sort_values("date")

        sig_log = np.log1p(df_grp["signal_ret"].clip(-0.5, 2).values)
        ok_log  = np.log1p(df_grp["always_ok_ret"].clip(-0.5, 2).values)
        rng = np.random.default_rng(42)
        rand_log = np.log1p(rng.normal(0.0, df_grp["always_ok_ret"].std(), len(df_grp)).clip(-0.5, 2))

        dates = df_grp["date"].values

        ax.plot(dates, np.cumsum(sig_log),  color=_SIGNAL_COLOR,  lw=2.2,
                label=f"Signal v3 (CAGR≈{backtest_json['signal']['cagr']:.1%})")
        ax.plot(dates, np.cumsum(ok_log),   color=_ALWAYSOK_COLOR, lw=1.8,
                label=f"Always-OK (CAGR≈{backtest_json['always_ok']['cagr']:.1%})",
                alpha=0.8)
        ax.plot(dates, np.cumsum(rand_log), color=_RANDOM_COLOR,  lw=1.4,
                ls="--", alpha=0.6, label="Random baseline")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=30)
    else:
        # Aggregate bar chart from JSON
        sig = backtest_json.get("signal", {})
        bm  = backtest_json.get("always_ok", {})
        names = ["Signal v3", "Always-OK"]
        cagrs = [sig.get("cagr", 0), bm.get("cagr", 0)]
        ax.bar(names, cagrs, color=[_SIGNAL_COLOR, _ALWAYSOK_COLOR], alpha=0.85)
        ax.set_ylabel("CAGR")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax.set(title="Cumulative Log-Returns — Signal v3 vs Baselines",
           ylabel="Cumulative log-return")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    fig.tight_layout()
    p = out_dir / "cumulative_returns.png"
    _savefig(fig, p)
    return [p]


# ── 2. Drawdown Comparison ────────────────────────────────────────────────────

def plot_drawdown(
    df: Optional[pd.DataFrame],
    backtest_json: Dict,
    out_dir: Path,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(12, 5))

    if df is not None and "signal_ret" in df.columns:
        df_grp = df.groupby("date").agg(
            signal_ret=("signal_ret", "mean"),
            always_ok_ret=("forward_return_20d", "mean"),
        ).reset_index().sort_values("date")

        dates    = df_grp["date"].values
        sig_rets = df_grp["signal_ret"].values
        ok_rets  = df_grp["always_ok_ret"].values

        # Equity-curve drawdown in percentage units (correct method)
        sig_dd = _compute_dd_equity(sig_rets)
        ok_dd  = _compute_dd_equity(ok_rets)

        ax.fill_between(dates, sig_dd, 0,
                        color=_SIGNAL_COLOR, alpha=0.45, label="Signal v3")
        ax.fill_between(dates, ok_dd,  0,
                        color=_ALWAYSOK_COLOR, alpha=0.35, label="Always-OK")
        ax.plot(dates, sig_dd, color=_SIGNAL_COLOR, lw=1.5)
        ax.plot(dates, ok_dd,  color=_ALWAYSOK_COLOR, lw=1.2, alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        plt.xticks(rotation=30)

        sig_mdd = float(np.min(sig_dd))
        bm_mdd  = float(np.min(ok_dd))
    else:
        # Fallback bar chart — only shown when time-series data is unavailable
        sig = backtest_json.get("signal", {})
        bm  = backtest_json.get("always_ok", {})
        sig_mdd = sig.get("max_drawdown", 0)
        bm_mdd  = bm.get("max_drawdown", 0)
        ax.bar(["Signal v3", "Always-OK"], [sig_mdd, bm_mdd],
               color=[_SIGNAL_COLOR, _ALWAYSOK_COLOR], alpha=0.85)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax.set(title=f"Drawdown — Signal v3 (MDD={sig_mdd:.1%}) vs Always-OK (MDD={bm_mdd:.1%})",
           ylabel="Drawdown (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "drawdown.png"
    _savefig(fig, p)
    return [p]


# ── 3. Return Distribution by Label ──────────────────────────────────────────

def plot_return_distributions(
    df: Optional[pd.DataFrame],
    out_dir: Path,
) -> List[Path]:
    label_colors = {"ok": "#4C72B0", "warn": "#DD8452", "block": "#C44E52"}

    if df is None:
        log.warning("No time-series data — skipping return distributions")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: density by label
    ax = axes[0]
    for lbl, color in label_colors.items():
        subset = df.loc[df["label"] == lbl, "forward_return_20d"]
        if len(subset) == 0:
            continue
        # KDE via seaborn
        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.35,
                    label=f"{lbl} (n={len(subset):,}, μ={subset.mean():.3f})")
    ax.set(title="Return Distribution by Label (KDE)",
           xlabel="Forward return 20d", ylabel="Density")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)

    # Right: boxplot
    ax2 = axes[1]
    label_order = ["ok", "warn", "block"]
    data_for_box = [
        df.loc[df["label"] == lbl, "forward_return_20d"].clip(-0.5, 0.5).values
        for lbl in label_order
        if len(df.loc[df["label"] == lbl]) > 0
    ]
    lbl_present = [lbl for lbl in label_order
                   if len(df.loc[df["label"] == lbl]) > 0]
    bp = ax2.boxplot(data_for_box, patch_artist=True, notch=False,
                     medianprops=dict(color="black", lw=2))
    for patch, lbl in zip(bp["boxes"], lbl_present):
        patch.set_facecolor(label_colors[lbl])
        patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(lbl_present) + 1))
    ax2.set_xticklabels(lbl_present)
    ax2.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set(title="Return Box-Plot by Label (clipped ±50%)",
            xlabel="Label", ylabel="Forward return 20d")

    fig.suptitle("Return Distributions by Signal Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "return_distributions.png"
    _savefig(fig, p)
    return [p]


# ── 4. Skip-Rate Over Time (rolling 6 months) ────────────────────────────────

def plot_skip_rate(
    df: Optional[pd.DataFrame],
    backtest_json: Dict,
    t_lo: float,
    out_dir: Path,
    window_days: int = 126,  # ~6 months
) -> List[Path]:
    if df is None:
        return []

    df = df.set_index("date").sort_index()
    df["is_invested"] = (df["proba_non_ok"] < t_lo).astype(float)

    daily = df["is_invested"].resample("W").mean()
    rolling_skip = (1 - daily).rolling(window=window_days // 5, min_periods=4).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(rolling_skip.index, rolling_skip.values,
                    alpha=0.4, color="#9467bd")
    ax.plot(rolling_skip.index, rolling_skip.values, color="#9467bd", lw=1.8)
    ax.axhline(rolling_skip.mean(), color="gray", ls="--", lw=1.5,
               label=f"Mean skip = {rolling_skip.mean():.1%}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    ax.set(title=f"Rolling Skip-Rate ({window_days}d window, weekly resample)",
           ylabel="Fraction skipped (blocked+warned)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "skip_rate_rolling.png"
    _savefig(fig, p)
    return [p]


# ── 5. Rolling Sharpe (12 months) ────────────────────────────────────────────

def plot_rolling_sharpe(
    df: Optional[pd.DataFrame],
    backtest_json: Dict,
    out_dir: Path,
    window_days: int = 252,  # ~12 months
) -> List[Path]:
    if df is None:
        return []

    periods_year = 252 / 20  # 20d holding periods → ~12.6/year

    df_sorted = df.set_index("date").sort_index()
    daily_sig = df_sorted["signal_ret"].resample("W").mean()
    daily_bm  = df_sorted["forward_return_20d"].resample("W").mean()

    window = window_days // 5  # ~52 weeks
    roll_sig = _rolling_sharpe(daily_sig, window, periods_year)
    roll_bm  = _rolling_sharpe(daily_bm,  window, periods_year)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll_sig.index, roll_sig.values, color=_SIGNAL_COLOR, lw=2,
            label=f"Signal v3 (Sharpe ann.={backtest_json['signal']['sharpe_ann']:.2f})")
    ax.plot(roll_bm.index, roll_bm.values, color=_ALWAYSOK_COLOR, lw=1.6,
            alpha=0.8, label=f"Always-OK (Sharpe ann.={backtest_json['always_ok']['sharpe_ann']:.2f})")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.5, label="Sharpe=1 threshold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    ax.set(title=f"Rolling Sharpe ({window_days}d window, weekly series)",
           ylabel="Annualised Sharpe ratio")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "rolling_sharpe.png"
    _savefig(fig, p)
    return [p]


# ── 6. Performance by Asset Type ─────────────────────────────────────────────

def plot_performance_by_asset_type(
    backtest_json: Dict,
    out_dir: Path,
) -> List[Path]:
    by_type = backtest_json.get("by_asset_type", {})
    if not by_type:
        return []

    asset_types = list(by_type.keys())
    sharpes = [by_type[a].get("sharpe_ann", 0) for a in asset_types]
    cagrs   = [by_type[a].get("cagr", 0) for a in asset_types]
    mdds    = [by_type[a].get("max_drawdown", 0) for a in asset_types]
    n_recs  = [by_type[a].get("n_periods", 0) for a in asset_types]

    # Sort by Sharpe
    order = np.argsort(sharpes)[::-1]
    asset_types = [asset_types[i] for i in order]
    sharpes = [sharpes[i] for i in order]
    cagrs   = [cagrs[i] for i in order]
    mdds    = [mdds[i] for i in order]
    n_recs  = [n_recs[i] for i in order]

    palette = sns.color_palette("muted", len(asset_types))
    x = np.arange(len(asset_types))
    width = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sharpe
    ax = axes[0]
    bars = ax.bar(x, sharpes, width=0.6, color=palette, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set(title="Annualised Sharpe by Asset Type", ylabel="Sharpe", xticks=x)
    ax.set_xticklabels(asset_types, rotation=30, ha="right")
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if v >= 0 else -0.08),
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # CAGR
    ax2 = axes[1]
    bars2 = ax2.bar(x, [c * 100 for c in cagrs], width=0.6,
                    color=palette, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.set(title="CAGR by Asset Type", ylabel="CAGR (%)", xticks=x)
    ax2.set_xticklabels(asset_types, rotation=30, ha="right")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for bar, v in zip(bars2, cagrs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Max Drawdown
    ax3 = axes[2]
    bars3 = ax3.bar(x, [m * 100 for m in mdds], width=0.6,
                    color=palette, alpha=0.85, edgecolor="white")
    ax3.axhline(0, color="gray", ls="--", lw=1)
    ax3.set(title="Max Drawdown by Asset Type", ylabel="MDD (%)", xticks=x)
    ax3.set_xticklabels(asset_types, rotation=30, ha="right")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Sample sizes as subtitle
    sizes_str = "  ".join([f"{a}: n={n:,}" for a, n in zip(asset_types, n_recs)])
    fig.suptitle("Performance by Asset Type — Signal v3",
                 fontsize=13, fontweight="bold")
    plt.figtext(0.5, -0.01, sizes_str, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    p = out_dir / "performance_by_asset_type.png"
    _savefig(fig, p)
    return [p]


# ── Summary metrics card ──────────────────────────────────────────────────────

def plot_metrics_card(
    backtest_json: Dict,
    out_dir: Path,
    series_metrics: Optional[Dict] = None,
) -> List[Path]:
    """
    Single-page metrics summary — useful as PDF cover chart.

    When series_metrics is provided (from _compute_series_metrics), time-series-
    derived MDD and Sharpe values override the potentially incorrect JSON values.
    """
    sig = backtest_json.get("signal", {})
    bm  = backtest_json.get("always_ok", {})

    # Prefer time-series-derived MDD (guaranteed from equity curve, in [-1,0])
    # over the JSON value which can be corrupted by cross-sectional ordering
    if series_metrics is not None:
        sig_mdd = series_metrics.get("signal_mdd",   sig.get("max_drawdown", 0))
        bm_mdd  = series_metrics.get("baseline_mdd", bm.get("max_drawdown",  0))
        sig_sh  = series_metrics.get("signal_sharpe",   sig.get("sharpe_ann", 0))
        bm_sh   = series_metrics.get("baseline_sharpe", bm.get("sharpe_ann",  0))
        sig_cagr = series_metrics.get("signal_cagr",   sig.get("cagr", 0))
        bm_cagr  = series_metrics.get("baseline_cagr", bm.get("cagr",  0))
        mdd_note = "¹"
    else:
        sig_mdd = sig.get("max_drawdown", 0)
        bm_mdd  = bm.get("max_drawdown",  0)
        sig_sh  = sig.get("sharpe_ann", 0)
        bm_sh   = bm.get("sharpe_ann",  0)
        sig_cagr = sig.get("cagr", 0)
        bm_cagr  = bm.get("cagr",  0)
        mdd_note = ""

    # Clamp MDD to valid range for display
    sig_mdd = max(-1.0, min(0.0, float(sig_mdd))) if math.isfinite(float(sig_mdd)) else 0.0
    bm_mdd  = max(-1.0, min(0.0, float(bm_mdd)))  if math.isfinite(float(bm_mdd))  else 0.0

    calmar_sig = (sig_cagr / abs(sig_mdd)) if sig_mdd < 0 else float("nan")
    calmar_bm  = (bm_cagr  / abs(bm_mdd))  if bm_mdd  < 0 else float("nan")

    def _fmt_mdd(v):
        return f"{v:.1%}" if math.isfinite(v) else "N/A"

    def _fmt_calmar(v):
        return f"{v:.2f}" if math.isfinite(v) else "N/A"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    rows = [
        ["Metric", "Signal v3", "Always-OK", "Δ vs BM"],
        ["CAGR (fold time-series)",
         f"{sig_cagr:.1%}" if math.isfinite(float(sig_cagr)) else "N/A",
         f"{bm_cagr:.1%}"  if math.isfinite(float(bm_cagr))  else "N/A",
         f"{sig_cagr - bm_cagr:+.1%}" if math.isfinite(float(sig_cagr) + float(bm_cagr)) else "—"],
        [f"Sharpe (ann., fold){mdd_note}",
         f"{sig_sh:.2f}" if math.isfinite(float(sig_sh)) else "N/A",
         f"{bm_sh:.2f}"  if math.isfinite(float(bm_sh))  else "N/A",
         f"{sig_sh - bm_sh:+.2f}" if math.isfinite(float(sig_sh) + float(bm_sh)) else "—"],
        ["Sortino",
         f"{sig.get('sortino_ann', 0):.2f}", f"{bm.get('sortino_ann', 0):.2f}", "—"],
        [f"Max Drawdown (equity){mdd_note}",
         _fmt_mdd(sig_mdd), _fmt_mdd(bm_mdd),
         f"{sig_mdd - bm_mdd:+.1%}"],
        [f"Calmar (derived){mdd_note}",
         _fmt_calmar(calmar_sig), _fmt_calmar(calmar_bm), "—"],
        ["Hit Rate",
         f"{sig.get('hit_rate', 0):.1%}", f"{bm.get('hit_rate', 0):.1%}", "—"],
        ["Profit Factor",
         f"{sig.get('profit_factor', 0):.2f}", f"{bm.get('profit_factor', 0):.2f}", "—"],
        ["Avg Exposure",
         f"{sig.get('avg_exposure', 0):.1%}", "100%", "—"],
    ]
    if mdd_note:
        rows.append(["¹ MDD & Sharpe computed from fold 5 time-series (equity curve)", "", "", ""])

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color Δ column
    for i in range(1, len(rows)):
        delta_cell = table[(i, 3)]
        delta_txt = rows[i][3]
        if delta_txt.startswith("+"):
            delta_cell.set_facecolor("#d4edda")
        elif delta_txt.startswith("-"):
            delta_cell.set_facecolor("#f8d7da")

    subtitle = "¹ MDD/Sharpe from equity curve (fold 5 time-series)" if series_metrics else ""
    ax.set_title(
        f"Backtest Summary — Signal v3 vs Always-OK Baseline\n{subtitle}",
        fontsize=12, fontweight="bold", pad=15,
    )
    p = out_dir / "backtest_metrics_card.png"
    _savefig(fig, p)
    return [p]


# ── Main entry ─────────────────────────────────────────────────────────────────

def generate_all(
    backtest_path: Path,
    manifest_path: Path,
    model_dir: Path,
    out_dir: Path,
) -> Dict:
    """
    Generate all financial plots.

    Returns dict with keys:
      - plot name → Path  (generated plot files)
      - "consistency_checks" → dict  (validation results)
      - "series_metrics"     → dict  (time-series-derived metrics)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backtest JSON
    if not backtest_path.exists():
        log.error("Backtest JSON not found: %s", backtest_path)
        return {}

    backtest_json = json.loads(backtest_path.read_text(encoding="utf-8"))

    # Load thresholds
    t_lo, t_hi = 0.5, 0.65
    thr_path = model_dir / "v3_thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        t_lo = thr.get("t_lo", 0.5)
        t_hi = thr.get("t_hi", 0.65)

    # Load time-series signal
    df: Optional[pd.DataFrame] = None
    if manifest_path.exists():
        df = _load_signal_series(manifest_path, model_dir)
        if df is not None:
            df = _apply_signal(df, t_lo, t_hi)

    # Compute time-series-derived metrics for consistency validation
    series_metrics: Optional[Dict] = None
    consistency_checks: Dict = {}
    if df is not None and "signal_ret" in df.columns:
        df_grp = df.groupby("date").agg(
            signal_ret=("signal_ret", "mean"),
            always_ok_ret=("forward_return_20d", "mean"),
        ).reset_index().sort_values("date")
        series_metrics = _compute_series_metrics(
            df_grp["signal_ret"].values,
            df_grp["always_ok_ret"].values,
        )
        consistency_checks = _validate_finance_consistency(series_metrics, backtest_json)
        log.info("Finance consistency — MDD series: %.2f%% | JSON: %.2f%% | all_pass=%s",
                 series_metrics["signal_mdd"] * 100,
                 backtest_json.get("signal", {}).get("max_drawdown", float("nan")) * 100,
                 consistency_checks.get("all_pass"))

    generated: Dict = {}

    for p in plot_cumulative_returns(df, backtest_json, out_dir):
        generated["cumulative_returns"] = p

    for p in plot_drawdown(df, backtest_json, out_dir):
        generated["drawdown"] = p

    for p in plot_return_distributions(df, out_dir):
        generated["return_distributions"] = p

    for p in plot_skip_rate(df, backtest_json, t_lo, out_dir):
        generated["skip_rate_rolling"] = p

    for p in plot_rolling_sharpe(df, backtest_json, out_dir):
        generated["rolling_sharpe"] = p

    for p in plot_performance_by_asset_type(backtest_json, out_dir):
        generated["performance_by_asset_type"] = p

    for p in plot_metrics_card(backtest_json, out_dir, series_metrics=series_metrics):
        generated["backtest_metrics_card"] = p

    generated["consistency_checks"] = consistency_checks
    generated["series_metrics"]     = series_metrics or {}
    log.info("Financial plots generated: %d figures in %s", len(generated) - 2, out_dir)
    return generated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Generate v3 financial visualization plots")
    ap.add_argument("--backtest", default="data/metrics/backtest_v3.json")
    ap.add_argument("--manifest", default="data/training/v3/splits_manifest.json")
    ap.add_argument("--models",   default="models/v3")
    ap.add_argument("--out",      default="data/metrics/v3/financial_plots")
    args = ap.parse_args()

    generated = generate_all(
        backtest_path=Path(args.backtest),
        manifest_path=Path(args.manifest),
        model_dir=Path(args.models),
        out_dir=Path(args.out),
    )

    print(f"\n✅ {len(generated)} financial plots generated:")
    for name, p in sorted(generated.items()):
        print(f"   {name:<35} → {p}")


if __name__ == "__main__":
    main()
