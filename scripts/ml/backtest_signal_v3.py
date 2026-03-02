"""
scripts/ml/backtest_signal_v3.py
==================================
Phase 4 — Signal quality backtest for the v3 label system.

Evaluates the predictive quality of the OK/WARN/BLOCK signal by simulating
a simple risk-allocation policy over forward returns.

Policy tested:
  OK    → exposure = 1.0  (full position)
  WARN  → exposure = 0.5  (half position)
  BLOCK → exposure = 0.0  (no position)

Compared against baselines:
  always_ok    : exposure = 1.0 always (buy-and-hold)
  always_block : exposure = 0.0 always (cash — 0% return)

Horizons: 20d (primary) and 60d (secondary)

Metrics:
  - Mean return, volatility (period + annualised)
  - Sharpe / Sortino ratios
  - Max drawdown (on cumulative equity curve)
  - Hit rate, avg win / avg loss, profit factor
  - Calmar ratio

Outputs:
  data/metrics/backtest_v3.json  — machine-readable
  Console summary table

Usage:
    python scripts/ml/backtest_signal_v3.py \\
        --in  data/training/train_v3_all.jsonl \\
        --out data/metrics/backtest_v3.json \\
        --horizon 20

No API / prod impact.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("backtest_signal_v3")

# Annualisation factors (period returns)
_PERIODS_PER_YEAR_20D = 252 / 20   # ≈ 12.6
_PERIODS_PER_YEAR_60D = 252 / 60   # ≈ 4.2


# ---------------------------------------------------------------------------
# Loader (streaming)
# ---------------------------------------------------------------------------

def load_records(path: Path, horizon: int = 20) -> List[Dict[str, Any]]:
    """Load records that have both label and the requested forward return."""
    ret_key = f"forward_return_{horizon}d"
    records = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fwd = rec.get(ret_key)
                lbl = rec.get("label")
                if fwd is None or lbl is None:
                    bad += 1
                    continue
                if not isinstance(fwd, (int, float)) or not math.isfinite(float(fwd)):
                    bad += 1
                    continue
                records.append({
                    "label":      str(lbl),
                    "fwd_return": float(fwd),
                    "date_key":   (rec.get("label_start_date") or rec.get("window_end_date") or "")[:10],
                    "ticker":     rec.get("features", {}).get("ticker", ""),
                    "asset_type": rec.get("features", {}).get("asset_type", "unknown"),
                })
            except (json.JSONDecodeError, TypeError):
                bad += 1
    if bad:
        log.warning("Skipped %d records (missing label/return or bad JSON)", bad)
    log.info("Loaded %d usable records (horizon=%dd)", len(records), horizon)
    return records


def _sort_by_date(records: List[Dict]) -> List[Dict]:
    return sorted(records, key=lambda r: r["date_key"])


# ---------------------------------------------------------------------------
# Core metrics on a return series
# ---------------------------------------------------------------------------

def compute_metrics(returns: np.ndarray, label: str, periods_per_year: float) -> Dict[str, Any]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n == 0:
        return {"label": label, "n_periods": 0}

    n_years = n / periods_per_year

    # NOTE: This dataset is cross-sectional (many tickers, not a single portfolio path).
    # We treat records as sequential returns for signal quality purposes (factor backtest style).
    # Hard-clip returns to [-0.99, 2.0] to prevent cumulative overflow.
    r_clipped = np.clip(r, -0.99, 2.0)

    # CAGR proxy: geometric annualisation of mean per-period return
    # (log1p / n * periods_per_year, expm1)
    mean_log_r = float(np.mean(np.log1p(r_clipped)))
    cagr        = float(math.expm1(mean_log_r * periods_per_year))

    # Cumulative return on a SAMPLE window (last 500 periods or all if fewer)
    # to avoid overflow while preserving the shape of the equity curve
    sample = r_clipped[-min(len(r_clipped), 2000):]
    cum_return = float(np.prod(1.0 + sample) - 1.0) if len(sample) <= 2000 else float("nan")

    # Vol
    mean_r  = float(np.mean(r))
    vol_per = float(np.std(r, ddof=1)) if n > 1 else 0.0
    vol_ann = vol_per * math.sqrt(periods_per_year)

    # Sharpe (zero risk-free)
    sharpe = float(mean_r / (vol_per + 1e-12) * math.sqrt(periods_per_year)) if vol_per > 1e-12 else float("nan")

    # Sortino (downside vol only)
    neg_r = r[r < 0]
    down_dev = float(np.std(neg_r, ddof=1)) * math.sqrt(periods_per_year) if len(neg_r) > 1 else 1e-12
    sortino = float(mean_r * periods_per_year / (down_dev + 1e-12))

    # Max drawdown on clipped log-cumsum path (overflow-safe)
    log_path = np.cumsum(np.log1p(r_clipped))
    run_max  = np.maximum.accumulate(log_path)
    dd_log   = log_path - run_max
    max_dd   = float(math.expm1(float(dd_log.min())))  # back to return space

    # Calmar
    calmar = float(cagr / abs(max_dd)) if (math.isfinite(cagr) and abs(max_dd) > 1e-6) else float("nan")

    # Hit rate / win-loss
    wins   = r[r > 0]
    losses = r[r < 0]
    hit_rate      = float(len(wins) / n)
    avg_win        = float(wins.mean())  if len(wins)   else 0.0
    avg_loss       = float(losses.mean()) if len(losses) else 0.0
    sum_gains      = float(wins.sum())   if len(wins)   else 0.0
    sum_losses_abs = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor  = sum_gains / (sum_losses_abs + 1e-12) if sum_losses_abs > 1e-12 else (
        float("inf") if sum_gains > 0 else float("nan")
    )

    pct = np.percentile(r, [5, 25, 50, 75, 95]).tolist()

    return {
        "label":          label,
        "n_periods":      int(n),
        "n_years_approx": round(n_years, 2),
        "cagr":           round(cagr, 4)   if math.isfinite(cagr)    else None,
        "cum_return":     round(cum_return, 4),
        "mean_per_period":round(mean_r, 5),
        "vol_period":     round(vol_per, 5),
        "vol_ann":        round(vol_ann, 4),
        "sharpe_ann":     round(sharpe, 4) if math.isfinite(sharpe)  else None,
        "sortino_ann":    round(sortino, 4) if math.isfinite(sortino) else None,
        "max_drawdown":   round(max_dd, 4),
        "calmar":         round(calmar, 4) if math.isfinite(calmar)   else None,
        "hit_rate":       round(hit_rate, 4),
        "avg_win":        round(avg_win, 5),
        "avg_loss":       round(avg_loss, 5),
        "profit_factor":  round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "n_wins":         int(len(wins)),
        "n_losses":       int(len(losses)),
        "pct_p5":         round(pct[0], 5),
        "pct_p25":        round(pct[1], 5),
        "pct_p50":        round(pct[2], 5),
        "pct_p75":        round(pct[3], 5),
        "pct_p95":        round(pct[4], 5),
    }


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

EXPOSURE = {
    "ok":    1.0,
    "warn":  0.5,
    "block": 0.0,
    # baselines
    "always_ok":    1.0,
    "always_block": 0.0,
}


def simulate_strategy(
    records: List[Dict],
    mode: str,
    periods_per_year: float,
) -> Dict[str, Any]:
    """
    Simulate strategy returns.

    mode:
      "signal"       — use label-based exposure (OK=1, WARN=0.5, BLOCK=0)
      "always_ok"    — exposure = 1.0 always
      "always_block" — exposure = 0.0 always
    """
    sorted_recs = _sort_by_date(records)

    strat_returns: List[float] = []
    exposures: List[float] = []

    n_ok = n_warn = n_block = 0

    for rec in sorted_recs:
        fwd = rec["fwd_return"]
        lbl = rec["label"]

        if mode == "signal":
            exp = EXPOSURE.get(lbl, 0.5)
        elif mode == "always_ok":
            exp = 1.0
        elif mode == "always_block":
            exp = 0.0
        else:
            exp = 0.5

        strat_returns.append(fwd * exp)
        exposures.append(exp)

        if lbl == "ok":    n_ok    += 1
        elif lbl == "warn": n_warn  += 1
        elif lbl == "block": n_block += 1

    r_arr = np.array(strat_returns, dtype=float)
    e_arr = np.array(exposures, dtype=float)

    metrics = compute_metrics(r_arr, label=mode, periods_per_year=periods_per_year)
    metrics["avg_exposure"] = round(float(e_arr.mean()), 4)
    metrics["n_ok"]    = n_ok
    metrics["n_warn"]  = n_warn
    metrics["n_block"] = n_block

    return metrics


# ---------------------------------------------------------------------------
# Per-asset-type breakdown
# ---------------------------------------------------------------------------

def breakdown_by_asset_type(
    records: List[Dict], periods_per_year: float
) -> Dict[str, Any]:
    by_type: Dict[str, List[float]] = {}
    by_type_n: Dict[str, int] = {}
    by_type_blocked: Dict[str, int] = {}

    for rec in records:
        at = rec.get("asset_type", "unknown") or "unknown"
        exp = EXPOSURE.get(rec["label"], 0.5)
        by_type.setdefault(at, []).append(rec["fwd_return"] * exp)
        by_type_n[at] = by_type_n.get(at, 0) + 1
        if rec["label"] == "block":
            by_type_blocked[at] = by_type_blocked.get(at, 0) + 1

    results = {}
    for at, rets in by_type.items():
        r_arr = np.array(rets, dtype=float)
        m = compute_metrics(r_arr, label=at, periods_per_year=periods_per_year)
        m["n_total"]   = by_type_n[at]
        m["n_blocked"] = by_type_blocked.get(at, 0)
        m["block_rate"] = round(m["n_blocked"] / by_type_n[at], 4) if by_type_n[at] else 0.0
        results[at] = m

    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "  N/A"
    if pct:
        return f"{float(v)*100:+.2f}%"
    return f"{float(v):+.4f}"


def print_summary(result: Dict[str, Any], horizon: int) -> None:
    n = result["n_records"]
    print(f"\n{'='*70}")
    print(f"  BACKTEST SIGNAL V3  —  horizon={horizon}d  records={n:,}")
    print(f"{'='*70}")

    signal  = result["signal"]
    always  = result["always_ok"]
    cash    = result["always_block"]

    print(f"  {'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'HitRate':>8} {'PF':>8}")
    print(f"  {'─'*68}")

    for name, m in [("signal", signal), ("always_ok (BM)", always), ("always_block", cash)]:
        print(f"  {name:<22} "
              f"{_fmt(m.get('cagr'), pct=True):>8} "
              f"{_fmt(m.get('sharpe_ann')):>8} "
              f"{_fmt(m.get('sortino_ann')):>8} "
              f"{_fmt(m.get('max_drawdown'), pct=True):>8} "
              f"{_fmt(m.get('hit_rate')):>8} "
              f"{str(m.get('profit_factor', 'N/A')):>8}")

    # Delta vs benchmark
    print(f"\n  Alpha vs always_ok (signal − BM):")
    for k in ["cagr", "sharpe_ann", "max_drawdown", "hit_rate"]:
        sv = signal.get(k)
        bv = always.get(k)
        if sv is not None and bv is not None:
            delta = float(sv) - float(bv)
            sign = "✅" if (k != "max_drawdown" and delta > 0) or (k == "max_drawdown" and delta > 0) else "⚠️ "
            print(f"    {sign} {k:<20} {delta:+.4f}")

    print(f"\n  Exposure usage: ok={signal['n_ok']:,} warn={signal['n_warn']:,} "
          f"block={signal['n_block']:,}  avg_exp={signal['avg_exposure']:.2%}")

    # Verdict
    sharpe_ok = (signal.get("sharpe_ann") or 0) > (always.get("sharpe_ann") or 0)
    mdd_ok    = (signal.get("max_drawdown") or -1) > (always.get("max_drawdown") or -1)
    skip_rate = signal["n_block"] / n if n else 0
    skip_ok   = 0.10 <= skip_rate <= 0.60

    verdict = "✅ SIGNAL QUALITY CONFIRMED" if (sharpe_ok and mdd_ok and skip_ok) else "⚠️  MIXED SIGNAL"
    print(f"\n  {verdict}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(path: Path, horizon: int = 20) -> Dict[str, Any]:
    periods_per_year = _PERIODS_PER_YEAR_20D if horizon == 20 else _PERIODS_PER_YEAR_60D

    records = load_records(path, horizon=horizon)
    if not records:
        raise ValueError(f"No usable records for horizon={horizon}d in {path}")

    label_dist: Dict[str, int] = {}
    for rec in records:
        lbl = rec["label"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    signal     = simulate_strategy(records, "signal",       periods_per_year)
    always_ok  = simulate_strategy(records, "always_ok",    periods_per_year)
    always_blk = simulate_strategy(records, "always_block", periods_per_year)
    by_type    = breakdown_by_asset_type(records, periods_per_year)

    return {
        "generated_at": datetime.now().isoformat(),
        "source_file":  str(path),
        "horizon_days": horizon,
        "n_records":    len(records),
        "label_distribution": label_dist,
        "signal":       signal,
        "always_ok":    always_ok,
        "always_block": always_blk,
        "by_asset_type": by_type,
    }


def _sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Signal quality backtest for v3 labels")
    ap.add_argument("--in",      dest="input",  required=True,
                    help="Path to v3 JSONL dataset")
    ap.add_argument("--out",     dest="output", default="data/metrics/backtest_v3.json",
                    help="Output JSON path (default: data/metrics/backtest_v3.json)")
    ap.add_argument("--horizon", type=int, default=20, choices=[5, 10, 20, 60],
                    help="Forward return horizon in days (default: 20)")
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    result = run_backtest(input_path, horizon=args.horizon)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitise(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Backtest report written: %s", output_path)

    print_summary(result, horizon=args.horizon)


if __name__ == "__main__":
    main()
