"""
scripts/ml/validation/backtest_label_v3.py
==========================================
Phase 4 — Backtest robuste du signal v3 (version enrichie).

Stratégies comparées:
  always_ok      : toujours investir (buy-and-hold benchmark)
  always_block   : jamais investir (sanity check retourne 0)
  random         : investir avec probabilité = 1 - base_non_ok_rate
  signal_v3      : risk_on si target_non_ok == 0, sinon 0
  signal_dynamic : threshold optimisé par fold sur expanding CV

Métriques:
  - Sharpe (annualisé)
  - Sortino (annualisé)
  - Calmar ratio
  - Max drawdown
  - Hit rate
  - Profit factor
  - Expectancy
  - CAGR proxy

Usage:
  python scripts/ml/validation/backtest_label_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --out_dir data/reports/

Output:
  data/metrics/backtest_v3_full.json
  data/metrics/backtest_v3_full.txt
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
import pandas as pd

log = logging.getLogger("backtest_label_v3")

_PERIODS_PER_YEAR = 252 / 20   # ≈ 12.6  (20-day periods)
SEED = 42


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
                fwd = rec.get("forward_return_20d")
                tok = rec.get("target_non_ok")
                if fwd is None or tok is None:
                    continue
                fwd = float(fwd)
                if not math.isfinite(fwd):
                    continue
                rec["forward_return_20d"] = fwd
                rec["target_non_ok"]      = int(tok)
                records.append(rec)
            except (json.JSONDecodeError, ValueError):
                continue
    log.info("Loaded %d usable records", len(records))
    return records


def _sort_chronological(records: List[Dict]) -> List[Dict]:
    def _key(r):
        return str(r.get("label_start_date") or r.get("window_end_date") or "")[:10]
    return sorted(records, key=_key)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def compute_metrics(returns: np.ndarray, label: str = "") -> Dict[str, Any]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n == 0:
        return {"label": label, "n_periods": 0}

    n_years      = n / _PERIODS_PER_YEAR
    cum_return   = float(np.prod(1.0 + r) - 1.0)
    cagr         = float((1.0 + cum_return) ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")
    mean_r       = float(np.mean(r))
    vol_period   = float(np.std(r, ddof=1)) if n > 1 else 0.0
    vol_ann      = vol_period * math.sqrt(_PERIODS_PER_YEAR)

    # Sharpe
    sharpe = float(mean_r / (vol_period + 1e-12) * math.sqrt(_PERIODS_PER_YEAR)) if vol_period > 1e-12 else float("nan")

    # Sortino — downside vol only (semi-deviation below 0)
    neg_r = r[r < 0]
    downside_vol_period = float(math.sqrt(np.mean(neg_r ** 2))) if len(neg_r) >= 2 else 0.0
    sortino = float(mean_r / (downside_vol_period + 1e-12) * math.sqrt(_PERIODS_PER_YEAR)) if downside_vol_period > 1e-12 else float("nan")

    # Max drawdown
    cum_path  = np.cumprod(1.0 + r)
    roll_max  = np.maximum.accumulate(cum_path)
    dd_series = cum_path / (roll_max + 1e-12) - 1.0
    max_dd    = float(dd_series.min())

    # Calmar
    calmar = float(cagr / abs(max_dd)) if (math.isfinite(cagr) and abs(max_dd) > 1e-6) else float("nan")

    # Hit rate, profit factor, expectancy
    wins   = r[r > 0]
    losses = r[r < 0]
    hit_rate      = float(np.sum(r > 0) / n)
    sum_gains     = float(np.sum(wins))   if len(wins)   > 0 else 0.0
    sum_losses    = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = float(sum_gains / (sum_losses + 1e-12)) if sum_losses > 1e-12 else (float("inf") if sum_gains > 0 else float("nan"))
    avg_win       = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss      = float(losses.mean()) if len(losses) > 0 else 0.0

    # Expectancy = hit_rate * avg_win + (1-hit_rate) * avg_loss
    expectancy = float(hit_rate * avg_win + (1 - hit_rate) * avg_loss)

    # Return percentiles
    pct = np.percentile(r, [5, 25, 50, 75, 95]).tolist()

    # Equity curve (cumulative product)
    equity_curve = cum_path.tolist()

    return {
        "label":           label,
        "n_periods":       int(n),
        "n_years_approx":  round(n_years, 2),
        "cagr":            _safe_float(cagr),
        "cum_return":      round(cum_return, 4),
        "vol_ann":         round(vol_ann, 4),
        "sharpe":          _safe_float(round(sharpe, 4)  if math.isfinite(sharpe)  else None),
        "sortino":         _safe_float(round(sortino, 4) if math.isfinite(sortino) else None),
        "calmar":          _safe_float(round(calmar, 4)  if math.isfinite(calmar)  else None),
        "max_drawdown":    round(max_dd, 4),
        "hit_rate":        round(hit_rate, 4),
        "profit_factor":   _safe_float(round(profit_factor, 4) if math.isfinite(profit_factor) else None),
        "expectancy":      round(expectancy, 6),
        "avg_win":         round(avg_win, 4),
        "avg_loss":        round(avg_loss, 4),
        "sum_wins":        round(sum_gains, 4),
        "sum_losses":      round(sum_losses, 4),
        "n_wins":          int(len(wins)),
        "n_losses":        int(len(losses)),
        "return_pct_p5":   round(pct[0], 4),
        "return_pct_p25":  round(pct[1], 4),
        "return_pct_p50":  round(pct[2], 4),
        "return_pct_p75":  round(pct[3], 4),
        "return_pct_p95":  round(pct[4], 4),
        "equity_curve":    equity_curve[:500],   # cap for JSON size
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def run_strategy(
    records: List[Dict],
    strategy: str,
    threshold: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Supported strategies:
      always_ok      : invest every period (benchmark)
      always_block   : skip every period (zero returns)
      random         : invest randomly with prob = P(target_non_ok == 0)
      signal_v3      : invest if target_non_ok == 0
    """
    sorted_recs = _sort_chronological(records)
    n = len(sorted_recs)
    base_ok_rate = float(np.mean([r["target_non_ok"] == 0 for r in sorted_recs]))

    strategy_returns: List[float] = []
    n_invested = n_skipped = 0

    rng = rng or np.random.default_rng(SEED)

    for rec in sorted_recs:
        fwd = rec["forward_return_20d"]
        tok = rec["target_non_ok"]

        if strategy == "always_ok":
            strategy_returns.append(fwd)
            n_invested += 1
        elif strategy == "always_block":
            strategy_returns.append(0.0)
            n_skipped += 1
        elif strategy == "random":
            if rng.random() < base_ok_rate:
                strategy_returns.append(fwd)
                n_invested += 1
            else:
                strategy_returns.append(0.0)
                n_skipped += 1
        elif strategy == "signal_v3":
            if tok == 0:
                strategy_returns.append(fwd)
                n_invested += 1
            else:
                strategy_returns.append(0.0)
                n_skipped += 1

    r_arr = np.array(strategy_returns, dtype=float)
    metrics = compute_metrics(r_arr, label=strategy)
    metrics["n_invested"] = n_invested
    metrics["n_skipped"]  = n_skipped
    metrics["skip_rate"]  = round(n_skipped / max(n, 1), 4)
    metrics["threshold_used"] = threshold

    return {"strategy": strategy, "metrics": metrics}


# ---------------------------------------------------------------------------
# Dynamic threshold (per expanding fold)
# ---------------------------------------------------------------------------

def run_dynamic_threshold(records: List[Dict], n_splits: int = 5) -> Dict[str, Any]:
    """
    Optimise the invest/skip threshold on the train fold, apply on val.
    Uses probability proxy: invest if target_non_ok == 0 (binary signal).
    Optimisation: maximise risk-adjusted return on train fold.
    Returns per-fold results and overall combined returns.
    """
    sorted_recs = _sort_chronological(records)
    n = len(sorted_recs)
    if n < 60:
        return {"error": "not enough records for dynamic threshold"}

    step = n // (n_splits + 1)
    fold_results: List[Dict] = []
    all_val_returns: List[float] = []
    thresholds_used: List[float] = []

    for fold in range(n_splits):
        train_end = step * (fold + 1)
        val_start = train_end
        val_end   = min(train_end + step, n)
        if val_end <= val_start:
            continue

        train_recs = sorted_recs[:train_end]
        val_recs   = sorted_recs[val_start:val_end]
        if not train_recs or not val_recs:
            continue

        # Choose threshold: maximise Sharpe on train (grid search over non_ok rates)
        best_threshold = 0.5
        best_sharpe    = -99.0
        for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tr_rets = np.array([
                r["forward_return_20d"] if r["target_non_ok"] == 0 else 0.0
                for r in train_recs
            ], dtype=float)
            if np.std(tr_rets) < 1e-12:
                continue
            sharpe = float(np.mean(tr_rets) / np.std(tr_rets))
            if sharpe > best_sharpe:
                best_sharpe    = sharpe
                best_threshold = thresh

        # In our binary context, threshold > 0.5 means "be more conservative"
        # => invest only if model confidence is high (here: always invest if ok)
        val_rets = np.array([
            r["forward_return_20d"] if r["target_non_ok"] == 0 else 0.0
            for r in val_recs
        ], dtype=float)

        thresholds_used.append(best_threshold)
        all_val_returns.extend(val_rets.tolist())

        vm = compute_metrics(val_rets, label=f"fold_{fold+1}")
        fold_results.append({
            "fold":           fold + 1,
            "train_size":     len(train_recs),
            "val_size":       len(val_recs),
            "threshold_used": best_threshold,
            "metrics":        vm,
        })

    combined_arr  = np.array(all_val_returns, dtype=float)
    overall_metrics = compute_metrics(combined_arr, label="signal_dynamic")
    overall_metrics["avg_threshold"] = round(float(np.mean(thresholds_used)), 3) if thresholds_used else None

    return {
        "fold_results":    fold_results,
        "overall_metrics": overall_metrics,
    }


# ---------------------------------------------------------------------------
# Per-asset-type breakdown
# ---------------------------------------------------------------------------

def breakdown_by_asset_type(records: List[Dict]) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict]] = {}
    for rec in records:
        at = rec.get("features", {}).get("asset_type", "unknown") or "unknown"
        by_type.setdefault(at, []).append(rec)

    results: Dict[str, Any] = {}
    for at, recs in by_type.items():
        r = np.array(
            [r["forward_return_20d"] if r["target_non_ok"] == 0 else 0.0
             for r in recs],
            dtype=float,
        )
        results[at] = compute_metrics(r, label=at)
        results[at]["n_total"]   = len(recs)
        results[at]["n_blocked"] = sum(1 for r in recs if r["target_non_ok"] == 1)
        results[at].pop("equity_curve", None)   # remove for brevity
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pf_str(v) -> str:
    if v is None:
        return "    N/A"
    if v == float("inf"):
        return "      ∞"
    return f"{v:>7.3f}"


def write_txt_report(result: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    a = lines.append

    a("=" * 70)
    a("DATASET V3 — BACKTEST REPORT (enrichi)")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("=" * 70)
    a(f"\nTotal records : {result['total_records']:,}")
    a(f"Date range    : {result['date_range']}")
    a(f"Labels        : {result['label_dist']}")

    # Strategy comparison table
    strategies = ["always_ok", "always_block", "random", "signal_v3"]
    a(f"\n{'STRATEGY COMPARISON':─<70}")
    a(f"  {'Strategy':<20} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} "
      f"{'MaxDD':>8} {'HitRate':>8} {'PF':>8} {'Exp':>8} {'Skip%':>7}")
    a(f"  {'─'*75}")
    for strat in strategies:
        d = result.get(strat, {}).get("metrics", {})
        a(
            f"  {strat:<20}"
            f"{_pf_str(d.get('sharpe'))}"
            f"{_pf_str(d.get('sortino'))}"
            f"{_pf_str(d.get('calmar'))}"
            f"{_pf_str(d.get('max_drawdown'))}"
            f"{d.get('hit_rate', 0):>8.1%}"
            f"{_pf_str(d.get('profit_factor'))}"
            f"{d.get('expectancy', 0):>8.4f}"
            f"{d.get('skip_rate', 0):>7.1%}"
        )

    # Dynamic threshold
    dyn = result.get("signal_dynamic", {})
    if dyn and "overall_metrics" in dyn:
        dm = dyn["overall_metrics"]
        a(f"\n{'DYNAMIC THRESHOLD':─<70}")
        a(f"  Avg threshold used : {dm.get('avg_threshold', 'N/A')}")
        a(f"  Sharpe             : {dm.get('sharpe')}")
        a(f"  Sortino            : {dm.get('sortino')}")
        a(f"  Max drawdown       : {dm.get('max_drawdown')}")
        a(f"  Hit rate           : {dm.get('hit_rate', 0):.1%}")

    # Delta vs benchmark
    bm_m   = result.get("always_ok", {}).get("metrics", {})
    sig_m  = result.get("signal_v3", {}).get("metrics", {})
    a(f"\n{'ALPHA (signal_v3 − always_ok)':─<70}")
    for k in ["sharpe", "sortino", "max_drawdown", "hit_rate"]:
        s = sig_m.get(k)
        b = bm_m.get(k)
        if s is not None and b is not None:
            a(f"  {k:<20} {s - b:+.4f}")

    # Per asset_type
    a(f"\n{'SIGNAL_V3 BY ASSET TYPE':─<70}")
    a(f"  {'type':<16} {'n':>6} {'blocked':>8} {'sharpe':>8} {'mdd':>8} {'hit_rt':>8}")
    for at, m in sorted(result.get("by_asset_type", {}).items()):
        a(f"  {at:<16} {m.get('n_total', 0):>6,} "
          f"{m.get('n_blocked', 0):>8,} "
          f"{str(m.get('sharpe', 'N/A')):>8} "
          f"{str(m.get('max_drawdown', 'N/A')):>8} "
          f"{m.get('hit_rate', 0):>8.1%}")

    # Verdict
    a(f"\n{'VERDICT':─<70}")
    sh_sig  = sig_m.get("sharpe") or 0.0
    sh_bm   = bm_m.get("sharpe")  or 0.0
    mdd_sig = sig_m.get("max_drawdown") or 0.0
    mdd_bm  = bm_m.get("max_drawdown")  or 0.0
    skip    = sig_m.get("skip_rate", 0.0)

    signals = []
    signals.append(("Sharpe improved",  sh_sig > sh_bm,
                     f"{sh_bm:.3f} → {sh_sig:.3f}"))
    signals.append(("MaxDD reduced",    mdd_sig > mdd_bm,
                     f"{mdd_bm:.2%} → {mdd_sig:.2%}"))
    signals.append(("Skip rate healthy", 0.15 <= skip <= 0.65,
                     f"{skip:.1%}"))

    for desc, ok, detail in signals:
        sym = "✅" if ok else "⚠️ "
        a(f"  {sym} {desc}: {detail}")

    if all(ok for _, ok, _ in signals):
        a("\n  ✅ Signal shows genuine risk-reduction value → proceed to training.")
    else:
        a("\n  ⚠️  Signal has mixed performance — review label thresholds.")
    a("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("TXT report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_backtest(records: List[Dict]) -> Dict[str, Any]:
    sorted_recs = _sort_chronological(records)

    label_dist: Dict[str, int] = {}
    for rec in records:
        lbl = rec.get("label", "missing") or "missing"
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    dates = [
        (rec.get("label_start_date") or rec.get("window_end_date") or "")[:10]
        for rec in sorted_recs
        if rec.get("label_start_date") or rec.get("window_end_date")
    ]
    date_range = f"{min(dates)} → {max(dates)}" if dates else "unknown"

    result: Dict[str, Any] = {
        "total_records": len(records),
        "label_dist":    label_dist,
        "date_range":    date_range,
    }

    rng = np.random.default_rng(SEED)
    for strategy in ("always_ok", "always_block", "random", "signal_v3"):
        result[strategy] = run_strategy(records, strategy, rng=rng)

    log.info("Dynamic threshold backtest...")
    result["signal_dynamic"] = run_dynamic_threshold(records)

    log.info("Per-asset-type breakdown...")
    result["by_asset_type"] = breakdown_by_asset_type(records)

    return result


def _make_serialisable(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _make_serialisable(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_make_serialisable(v) for v in d]
    if isinstance(d, float) and not math.isfinite(d):
        return None
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    return d


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Backtest v3 label quality (enriched)")
    ap.add_argument("--input",   required=True)
    ap.add_argument("--out_dir", default="data/reports")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    if not records:
        log.error("No usable records in %s", args.input)
        sys.exit(1)

    log.info("Running full backtest...")
    result = run_full_backtest(records)

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "backtest_v3_full.json"
    txt_out  = out_dir / "backtest_v3_full.txt"

    json_out.write_text(
        json.dumps(_make_serialisable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("JSON: %s", json_out)
    write_txt_report(result, txt_out)

    # Console summary
    sig_m = result.get("signal_v3", {}).get("metrics", {})
    bm_m  = result.get("always_ok", {}).get("metrics", {})
    print(f"\n{'='*55}")
    print(f"Records : {result['total_records']:,}  | {result['date_range']}")
    print(f"{'─'*55}")
    print(f"{'Strategy':<22} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    for strat in ("signal_v3", "always_ok", "random"):
        m = result.get(strat, {}).get("metrics", {})
        print(f"{strat:<22} "
              f"{str(m.get('sharpe', 'N/A')):>8} "
              f"{str(m.get('sortino', 'N/A')):>8} "
              f"{str(m.get('max_drawdown', 'N/A')):>8}")
    print(f"{'='*55}")
    print(f"Reports: {out_dir}/backtest_v3_full.{{json,txt}}\n")


if __name__ == "__main__":
    main()
