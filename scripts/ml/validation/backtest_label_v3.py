"""
scripts/ml/validation/backtest_label_v3.py
==========================================
Phase 3 — Backtest of the v3 label quality.

Strategy:
  risk_on_if_ok : invest (use forward_return_20d) when target_non_ok == 0
                  skip (return = 0) when target_non_ok == 1

Compared to:
  always_ok     : always invest (buy-and-hold benchmark)

Metrics computed:
  - Sharpe ratio (annualised, assuming 252/20 ≈ 12.6 periods/year)
  - Max drawdown
  - Hit rate (% of periods with positive return)
  - Profit factor (sum gains / sum losses)
  - Avg win / avg loss
  - CAGR proxy
  - Distribution of gains

Chronologically ordered simulation (sorted by label_start_date).

Usage:
  python scripts/ml/validation/backtest_label_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --out_dir data/reports/

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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("backtest_label_v3")

# 20-day periods per year (approx)
_PERIODS_PER_YEAR = 252 / 20  # ≈ 12.6


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Only keep records with valid forward_return_20d and target_non_ok
                fwd = rec.get("forward_return_20d")
                tok = rec.get("target_non_ok")
                if fwd is None or tok is None:
                    continue
                if not isinstance(fwd, (int, float)) or not math.isfinite(float(fwd)):
                    continue
                rec["forward_return_20d"] = float(fwd)
                rec["target_non_ok"]      = int(tok)
                records.append(rec)
            except (json.JSONDecodeError, ValueError):
                continue

    log.info("Loaded %d usable records (with valid forward_return_20d + target_non_ok)", len(records))
    return records


def _sort_by_date(records: List[Dict]) -> List[Dict]:
    """Sort records chronologically by label_start_date, then window_end_date."""
    def _key(r):
        d = r.get("label_start_date") or r.get("window_end_date") or ""
        return str(d)[:10]
    return sorted(records, key=_key)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: np.ndarray, label: str = "") -> Dict[str, Any]:
    """
    Compute performance metrics for a return series.
    Returns are per 20-day period (not daily).
    """
    r = returns[np.isfinite(returns)]
    n = len(r)

    if n == 0:
        return {"label": label, "n_periods": 0, "error": "no data"}

    # CAGR proxy (compound annual growth rate assuming _PERIODS_PER_YEAR periods/year)
    cum_return = float(np.prod(1.0 + r) - 1.0)
    n_years    = n / _PERIODS_PER_YEAR
    cagr       = float((1.0 + cum_return) ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")

    # Volatility (annualised)
    vol_period = float(np.std(r, ddof=1)) if n > 1 else 0.0
    vol_ann    = vol_period * math.sqrt(_PERIODS_PER_YEAR)

    # Sharpe (annualised, zero risk-free rate)
    mean_period = float(np.mean(r))
    sharpe = (
        float(mean_period / (vol_period + 1e-12) * math.sqrt(_PERIODS_PER_YEAR))
        if vol_period > 1e-12 else float("nan")
    )

    # Max drawdown (from cumulative return path)
    cum_path = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cum_path)
    dd_series = cum_path / (running_max + 1e-12) - 1.0
    max_dd = float(dd_series.min())

    # Hit rate
    wins  = r[r > 0]
    losses = r[r < 0]
    hit_rate = float(np.sum(r > 0) / n) if n > 0 else float("nan")

    # Profit factor
    sum_gains  = float(np.sum(wins))  if len(wins)   > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        float(sum_gains / (sum_losses + 1e-12))
        if sum_losses > 1e-12 else float("inf") if sum_gains > 0 else float("nan")
    )

    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Return distribution percentiles
    pct = np.percentile(r, [5, 25, 50, 75, 95]).tolist()

    # Calmar ratio
    calmar = float(cagr / abs(max_dd)) if (
        np.isfinite(cagr) and abs(max_dd) > 1e-6
    ) else float("nan")

    return {
        "label":          label,
        "n_periods":      int(n),
        "n_years_approx": round(n_years, 2),
        "cagr":           round(cagr, 4)          if np.isfinite(cagr)    else None,
        "cum_return":     round(cum_return, 4),
        "vol_ann":        round(vol_ann, 4),
        "sharpe":         round(sharpe, 4)         if np.isfinite(sharpe)   else None,
        "max_drawdown":   round(max_dd, 4),
        "calmar":         round(calmar, 4)         if np.isfinite(calmar)   else None,
        "hit_rate":       round(hit_rate, 4),
        "profit_factor":  round(profit_factor, 4)  if np.isfinite(profit_factor) else None,
        "avg_win":        round(avg_win, 4),
        "avg_loss":       round(avg_loss, 4),
        "sum_wins":       round(sum_gains, 4),
        "sum_losses":     round(sum_losses, 4),
        "n_wins":         int(len(wins)),
        "n_losses":       int(len(losses)),
        "return_pct_p5":  round(pct[0], 4),
        "return_pct_p25": round(pct[1], 4),
        "return_pct_p50": round(pct[2], 4),
        "return_pct_p75": round(pct[3], 4),
        "return_pct_p95": round(pct[4], 4),
    }


# ---------------------------------------------------------------------------
# Backtest strategies
# ---------------------------------------------------------------------------

def run_strategy(
    records: List[Dict],
    strategy: str = "risk_on_if_ok",
) -> Dict[str, Any]:
    """
    Simulate a strategy over all records (chronologically sorted).

    Strategies:
      risk_on_if_ok : invest (forward_return_20d) if target_non_ok == 0, else 0
      always_ok     : always invest (benchmark)
      always_risk_off: always skip (returns 0 — sanity check)

    Returns dict with returns array + metrics.
    """
    sorted_recs = _sort_by_date(records)

    strategy_returns: List[float] = []
    benchmark_returns: List[float] = []
    skipped = 0
    invested = 0

    for rec in sorted_recs:
        fwd = rec["forward_return_20d"]
        tok = rec["target_non_ok"]
        benchmark_returns.append(fwd)

        if strategy == "risk_on_if_ok":
            if tok == 0:
                strategy_returns.append(fwd)
                invested += 1
            else:
                strategy_returns.append(0.0)
                skipped += 1
        elif strategy == "always_ok":
            strategy_returns.append(fwd)
            invested += 1
        elif strategy == "always_risk_off":
            strategy_returns.append(0.0)
            skipped += 1

    r_arr = np.array(strategy_returns, dtype=float)
    b_arr = np.array(benchmark_returns, dtype=float)

    metrics = compute_metrics(r_arr, label=strategy)
    metrics["n_invested"] = invested
    metrics["n_skipped"]  = skipped
    metrics["skip_rate"]  = round(skipped / len(sorted_recs), 4) if sorted_recs else 0.0

    return {
        "strategy":  strategy,
        "metrics":   metrics,
        "returns":   r_arr,
        "benchmark_returns": b_arr,
    }


# ---------------------------------------------------------------------------
# Per-asset-type breakdown
# ---------------------------------------------------------------------------

def breakdown_by_asset_type(records: List[Dict]) -> Dict[str, Any]:
    """Run risk_on_if_ok strategy per asset_type and report metrics."""
    by_type: Dict[str, List[Dict]] = {}
    for rec in records:
        at = rec.get("features", {}).get("asset_type", "unknown") or "unknown"
        by_type.setdefault(at, []).append(rec)

    results = {}
    for at, recs in by_type.items():
        r = np.array(
            [r["forward_return_20d"] if r["target_non_ok"] == 0 else 0.0
             for r in recs],
            dtype=float,
        )
        results[at] = compute_metrics(r, label=at)
        results[at]["n_total"] = len(recs)
        results[at]["n_blocked"] = sum(1 for r in recs if r["target_non_ok"] == 1)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pf_str(v) -> str:
    if v is None:
        return "N/A"
    if v == float("inf"):
        return "∞"
    return f"{v:.3f}"


def write_txt_report(result: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    a = lines.append

    a("=" * 70)
    a("DATASET V3 — LABEL BACKTEST REPORT")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("=" * 70)

    a(f"\nTotal usable records : {result['total_records']:,}")
    a(f"Date range           : {result['date_range']}")
    a(f"\nLabel distribution:")
    for k, v in result["label_dist"].items():
        a(f"  {k:<12} {v:,}  ({100*v/result['total_records']:.1f}%)")

    def _fmt_strategy(name: str, m: Dict) -> None:
        a(f"\n{'─'*70}")
        a(f"STRATEGY: {name.upper()}")
        a(f"{'─'*70}")
        a(f"  Periods invested   : {m.get('n_invested', m.get('n_periods', '?')):,}  "
          f"(skipped {m.get('n_skipped', 0):,}, skip_rate={m.get('skip_rate', 0):.1%})")
        a(f"  CAGR (proxy)       : {m.get('cagr', 'N/A')}")
        a(f"  Cumulative return  : {m.get('cum_return', 'N/A'):.2%}" if isinstance(m.get('cum_return'), float) else f"  Cumulative return  : N/A")
        a(f"  Sharpe (ann.)      : {m.get('sharpe', 'N/A')}")
        a(f"  Max drawdown       : {m.get('max_drawdown', 'N/A'):.2%}" if isinstance(m.get('max_drawdown'), float) else "  Max drawdown       : N/A")
        a(f"  Calmar ratio       : {_pf_str(m.get('calmar'))}")
        a(f"  Hit rate           : {m.get('hit_rate', 'N/A'):.1%}" if isinstance(m.get('hit_rate'), float) else "  Hit rate           : N/A")
        a(f"  Profit factor      : {_pf_str(m.get('profit_factor'))}")
        a(f"  Avg win            : {m.get('avg_win', 0):.4f}")
        a(f"  Avg loss           : {m.get('avg_loss', 0):.4f}")
        a(f"  Return pct [p5/p50/p95]: {m.get('return_pct_p5', 'N/A')} / "
          f"{m.get('return_pct_p50', 'N/A')} / {m.get('return_pct_p95', 'N/A')}")

    _fmt_strategy("risk_on_if_ok",  result["risk_on_if_ok"]["metrics"])
    _fmt_strategy("always_ok (BM)", result["always_ok"]["metrics"])

    # Delta
    a(f"\n{'─'*70}")
    a("ALPHA vs BENCHMARK (risk_on_if_ok − always_ok)")
    a(f"{'─'*70}")
    strat_m = result["risk_on_if_ok"]["metrics"]
    bench_m = result["always_ok"]["metrics"]

    def _delta(key: str) -> str:
        s = strat_m.get(key)
        b = bench_m.get(key)
        if s is None or b is None:
            return "N/A"
        return f"{s - b:+.4f}"

    a(f"  Sharpe delta       : {_delta('sharpe')}")
    a(f"  MaxDD delta        : {_delta('max_drawdown')}")
    a(f"  Hit rate delta     : {_delta('hit_rate')}")

    # Per-asset-type
    a(f"\n{'─'*70}")
    a("BREAKDOWN BY ASSET TYPE (risk_on_if_ok)")
    a(f"{'─'*70}")
    a(f"  {'asset_type':<14} {'n_total':>8} {'blocked':>8} {'sharpe':>8} "
      f"{'mdd':>8} {'hit_rt':>8} {'pf':>8}")
    for at, m in sorted(result["by_asset_type"].items()):
        a(
            f"  {at:<14} {m.get('n_total', 0):>8,} "
            f"{m.get('n_blocked', 0):>8,} "
            f"{str(m.get('sharpe', 'N/A')):>8} "
            f"{str(m.get('max_drawdown', 'N/A')):>8} "
            f"{str(m.get('hit_rate', 'N/A')):>8} "
            f"{_pf_str(m.get('profit_factor')):>8}"
        )

    # Verdict
    a(f"\n{'='*70}")
    a("VERDICT")
    a(f"{'─'*70}")
    sharpe_strat = strat_m.get("sharpe") or 0.0
    sharpe_bench = bench_m.get("sharpe") or 0.0
    mdd_strat    = strat_m.get("max_drawdown") or 0.0
    mdd_bench    = bench_m.get("max_drawdown") or 0.0

    signals = []
    if sharpe_strat > sharpe_bench:
        signals.append(f"✅ Sharpe improved ({sharpe_bench:.3f} → {sharpe_strat:.3f})")
    else:
        signals.append(f"⚠️  Sharpe NOT improved ({sharpe_bench:.3f} → {sharpe_strat:.3f})")

    if mdd_strat > mdd_bench:   # less negative = better
        signals.append(f"✅ MaxDD improved ({mdd_bench:.2%} → {mdd_strat:.2%})")
    else:
        signals.append(f"⚠️  MaxDD NOT improved ({mdd_bench:.2%} → {mdd_strat:.2%})")

    skip_rate = strat_m.get("skip_rate", 0.0)
    if 0.20 <= skip_rate <= 0.60:
        signals.append(f"✅ Skip rate in healthy range ({skip_rate:.1%})")
    elif skip_rate < 0.05:
        signals.append(f"⚠️  Skip rate very low ({skip_rate:.1%}) — label under-predicts risk")
    else:
        signals.append(f"⚠️  Skip rate high ({skip_rate:.1%}) — label over-predicts risk")

    for s in signals:
        a(f"  {s}")

    if all(s.startswith("✅") for s in signals):
        a("\n  ✅ Label shows genuine predictive signal — proceed to model training.")
    else:
        a("\n  ⚠️  Label has mixed signals — review thresholds or data coverage.")
    a("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("TXT report written: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_backtest(records: List[Dict]) -> Dict[str, Any]:
    sorted_recs = _sort_by_date(records)

    label_dist: Dict[str, int] = {}
    for rec in records:
        lbl = rec.get("label", "missing") or "missing"
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    dates = [
        (rec.get("label_start_date") or rec.get("window_end_date") or "")[:10]
        for rec in sorted_recs
        if (rec.get("label_start_date") or rec.get("window_end_date"))
    ]
    date_range = f"{min(dates)} → {max(dates)}" if dates else "unknown"

    risk_on   = run_strategy(records, "risk_on_if_ok")
    always_ok = run_strategy(records, "always_ok")
    by_at     = breakdown_by_asset_type(records)

    return {
        "total_records": len(records),
        "label_dist":    label_dist,
        "date_range":    date_range,
        "risk_on_if_ok": risk_on,
        "always_ok":     always_ok,
        "by_asset_type": by_at,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Backtest v3 dataset labels")
    ap.add_argument("--input",   required=True, help="Path to v3 JSONL file")
    ap.add_argument("--out_dir", default="data/reports",
                    help="Output directory for reports (default: data/reports)")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    if not records:
        log.error("No usable records in %s", args.input)
        sys.exit(1)

    log.info("Running backtest...")
    result = run_full_backtest(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialise (remove numpy arrays before JSON dump)
    def _serialisable(d):
        if isinstance(d, dict):
            return {k: _serialisable(v) for k, v in d.items()
                    if not isinstance(v, np.ndarray)}
        if isinstance(d, float) and not math.isfinite(d):
            return None
        if isinstance(d, np.integer):
            return int(d)
        if isinstance(d, np.floating):
            return float(d)
        return d

    json_path = out_dir / "backtest_report_v3.json"
    txt_path  = out_dir / "backtest_report_v3.txt"

    json_path.write_text(
        json.dumps(_serialisable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("JSON report written: %s", json_path)

    write_txt_report(result, txt_path)

    # Quick summary to stdout
    m_strat = result["risk_on_if_ok"]["metrics"]
    m_bench = result["always_ok"]["metrics"]
    print(f"\n{'='*50}")
    print(f"Total records : {result['total_records']:,}")
    print(f"Date range    : {result['date_range']}")
    print(f"\n{'Strategy':<22} {'Sharpe':>8} {'MaxDD':>8} {'HitRate':>8} {'PF':>8}")
    print(f"{'─'*50}")
    print(f"{'risk_on_if_ok':<22} "
          f"{str(m_strat.get('sharpe', 'N/A')):>8} "
          f"{str(m_strat.get('max_drawdown', 'N/A')):>8} "
          f"{str(m_strat.get('hit_rate', 'N/A')):>8} "
          f"{_pf_str(m_strat.get('profit_factor')):>8}")
    print(f"{'always_ok (BM)':<22} "
          f"{str(m_bench.get('sharpe', 'N/A')):>8} "
          f"{str(m_bench.get('max_drawdown', 'N/A')):>8} "
          f"{str(m_bench.get('hit_rate', 'N/A')):>8} "
          f"{_pf_str(m_bench.get('profit_factor')):>8}")
    print(f"{'='*50}")
    print(f"Reports → {args.out_dir}/backtest_report_v3.{{json,txt}}\n")


if __name__ == "__main__":
    main()
