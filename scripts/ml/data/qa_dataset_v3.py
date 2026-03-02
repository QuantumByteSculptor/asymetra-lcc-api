"""
scripts/ml/data/qa_dataset_v3.py
=================================
Phase 1 — QA + structural validation of a v3 JSONL dataset.

Runs all quality checks in a single pass and writes a compact JSON report:
  - n_rows, n_tickers, asset_type / market distribution
  - label / target_non_ok distribution
  - NaN rate per feature (top 30 most missing)
  - temporal coherence checks (window_start ≤ window_end < label_start ≤ label_end ≤ label_end_60d)
  - duplicate detection (ticker + window_end_date)
  - forward_return_* descriptive stats + outlier counts

Usage:
    python scripts/ml/data/qa_dataset_v3.py \\
        --in  data/training/train_v3_all.jsonl \\
        --out data/metrics/qa_v3_report.json

No API / prod impact. No heavy deps beyond stdlib + numpy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
log = logging.getLogger("qa_dataset_v3")

_HORIZON_FIELDS = [
    "forward_return_5d", "forward_return_10d",
    "forward_return_20d", "forward_return_60d",
]
_LABEL_FIELDS = ["label", "target_non_ok", "future_dd_20d", "future_vol_ratio"]

_META_KEYS = {"asset_type", "market", "ticker"}

# Macro / cross-asset / structurally-nullable features that are expected to have higher NaN rates.
# recovery_days / recovery_per_dd: null when asset hasn't recovered from drawdown (normal).
# abs_corr_mkt: derived from corr_spy; null when corr_spy unavailable.
_MACRO_FEATURES = {
    "vix_level", "vix_pct_60d", "rate_10y", "rate_2y", "term_spread",
    "credit_spread_hy", "credit_spread_ig", "vol_regime",
    "corr_spy", "corr_vix", "beta_market",
    # structurally nullable
    "abs_corr_mkt", "recovery_days", "recovery_per_dd",
}


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------

def stream_jsonl(path: Path):
    """Yield parsed records one-by-one (streaming — no full load into RAM)."""
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= 5:
                    log.warning("Line %d JSON error: %s", i + 1, e)
    if bad:
        log.warning("Total bad JSON lines: %d", bad)


def _parse_date(s: Optional[Any]) -> Optional[str]:
    """Return ISO date string YYYY-MM-DD or None."""
    if s is None:
        return None
    try:
        return str(s)[:10]
    except Exception:
        return None


def _date_le(a: Optional[str], b: Optional[str]) -> bool:
    """Return True if a <= b (both non-None ISO strings)."""
    if a is None or b is None:
        return False
    return a <= b


def _date_lt(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a < b


# ---------------------------------------------------------------------------
# Main QA pass (single-pass streaming)
# ---------------------------------------------------------------------------

def run_qa(path: Path) -> Dict[str, Any]:
    n_rows = 0
    n_bad_json = 0

    tickers: set = set()
    asset_type_counter: Counter = Counter()
    market_counter: Counter = Counter()
    label_counter: Counter = Counter()
    source_counter: Counter = Counter()
    non_ok_counter: Counter = Counter()

    # Temporal checks
    n_temporal_ok = 0
    temporal_violations: List[Dict] = []

    # Duplicates
    seen_keys: Dict[Tuple[str, str], int] = {}  # (ticker, window_end_date) -> first row idx
    duplicates: List[Dict] = []

    # Feature NaN tracking
    feat_nan: Dict[str, int] = defaultdict(int)
    feat_total: Dict[str, int] = defaultdict(int)

    # Label / horizon NaN tracking
    horizon_nan: Dict[str, int] = defaultdict(int)

    # forward_return stats accumulator
    ret_accum: Dict[str, List[float]] = {k: [] for k in _HORIZON_FIELDS}

    for rec in stream_jsonl(path):
        n_rows += 1
        feats = rec.get("features", {})
        ticker = feats.get("ticker", "")
        asset_type = feats.get("asset_type", "unknown")
        market = feats.get("market", "unknown")
        wed = _parse_date(rec.get("window_end_date"))
        wsd = _parse_date(rec.get("window_start_date"))
        lsd = _parse_date(rec.get("label_start_date"))
        led = _parse_date(rec.get("label_end_date"))
        l60 = _parse_date(rec.get("label_end_date_60d"))

        tickers.add(ticker)
        asset_type_counter[asset_type] += 1
        market_counter[market] += 1
        label_counter[rec.get("label", "missing")] += 1
        source_counter[rec.get("source", "unknown")] += 1
        non_ok_counter[str(rec.get("target_non_ok"))] += 1

        # --- temporal checks ---
        t_ok = True
        viols = []
        if wsd and wed and not _date_le(wsd, wed):
            viols.append(f"window_start({wsd}) > window_end({wed})")
            t_ok = False
        if wed and lsd and not _date_lt(wed, lsd):
            viols.append(f"window_end({wed}) >= label_start({lsd}) — feature/label overlap")
            t_ok = False
        if lsd and led and not _date_le(lsd, led):
            viols.append(f"label_start({lsd}) > label_end({led})")
            t_ok = False
        if led and l60 and not _date_lt(led, l60):
            viols.append(f"label_end({led}) >= label_end_60d({l60})")
            t_ok = False
        if not (wsd or wed or lsd or led):
            viols.append("all date fields missing")
            t_ok = False

        if t_ok:
            n_temporal_ok += 1
        elif len(temporal_violations) < 20:
            temporal_violations.append({
                "row": n_rows,
                "ticker": ticker,
                "window_start_date": wsd,
                "window_end_date": wed,
                "label_start_date": lsd,
                "label_end_date": led,
                "label_end_date_60d": l60,
                "violations": viols,
            })

        # --- duplicate check ---
        if ticker and wed:
            dup_key = (ticker, wed)
            if dup_key in seen_keys:
                if len(duplicates) < 50:
                    duplicates.append({
                        "ticker": ticker,
                        "window_end_date": wed,
                        "first_row": seen_keys[dup_key],
                        "dup_row": n_rows,
                    })
            else:
                seen_keys[dup_key] = n_rows

        # --- feature NaN tracking ---
        for k, v in feats.items():
            if k in _META_KEYS:
                continue
            feat_total[k] += 1
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                feat_nan[k] += 1

        # --- horizon NaN + accumulation ---
        for hk in _HORIZON_FIELDS:
            v = rec.get(hk)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                horizon_nan[hk] += 1
            elif isinstance(v, (int, float)):
                ret_accum[hk].append(float(v))

    # ----- post-processing -----

    # NaN rates per feature
    nan_rates = {}
    for k in feat_total:
        rate = 100.0 * feat_nan[k] / feat_total[k] if feat_total[k] else 0.0
        nan_rates[k] = round(rate, 2)
    # Sort by NaN rate descending, keep top 30
    top30_nan = dict(
        sorted(nan_rates.items(), key=lambda x: -x[1])[:30]
    )

    # forward_return stats
    ret_stats: Dict[str, Any] = {}
    for hk in _HORIZON_FIELDS:
        vals = ret_accum[hk]
        if not vals:
            ret_stats[hk] = {"n": 0}
            continue
        arr = np.array(vals, dtype=float)
        q = np.percentile(arr, [5, 25, 50, 75, 95]).tolist()
        outlier_lo = float(np.sum(arr < -0.5))
        outlier_hi = float(np.sum(arr > 1.0))
        ret_stats[hk] = {
            "n":          int(len(arr)),
            "nan_count":  int(horizon_nan[hk]),
            "mean":       round(float(arr.mean()), 5),
            "std":        round(float(arr.std(ddof=1)), 5) if len(arr) > 1 else 0.0,
            "min":        round(float(arr.min()), 5),
            "p5":         round(q[0], 5),
            "p25":        round(q[1], 5),
            "p50":        round(q[2], 5),
            "p75":        round(q[3], 5),
            "p95":        round(q[4], 5),
            "max":        round(float(arr.max()), 5),
            "outliers_below_minus50pct": int(outlier_lo),
            "outliers_above_100pct":     int(outlier_hi),
        }

    n = n_rows
    total_temporal_violations = (n - n_temporal_ok)
    n_duplicates = len(duplicates)

    # Verdict
    issues = []
    if total_temporal_violations > 0:
        issues.append(f"{total_temporal_violations} temporal violations")
    if n_duplicates > 0:
        issues.append(f"{n_duplicates} duplicates (ticker+window_end_date)")
    non_macro_high_nan = {k: v for k, v in nan_rates.items()
                          if v > 10.0 and k not in _MACRO_FEATURES}
    if non_macro_high_nan:
        issues.append(f"{len(non_macro_high_nan)} non-macro features with >10% NaN")

    report: Dict[str, Any] = {
        "generated_at":       datetime.now().isoformat(),
        "input_file":         str(path),
        "n_rows":             n,
        "n_tickers":          len(tickers),
        "by_asset_type":      dict(asset_type_counter.most_common()),
        "by_market":          dict(market_counter.most_common()),
        "label_distribution": dict(label_counter.most_common()),
        "label_pct": {k: round(100.0 * v / n, 2) for k, v in label_counter.items()} if n else {},
        "target_non_ok_distribution": dict(non_ok_counter),
        "by_source":          dict(source_counter.most_common()),
        "temporal_checks": {
            "n_ok":        n_temporal_ok,
            "n_violation": total_temporal_violations,
            "leakage_free": total_temporal_violations == 0,
            "violations_sample": temporal_violations,
        },
        "duplicates": {
            "n_duplicates": n_duplicates,
            "sample":       duplicates[:10],
        },
        "nan_top30_features":    top30_nan,
        "nan_macro_features":    {k: nan_rates.get(k, 0.0) for k in _MACRO_FEATURES},
        "non_macro_high_nan":    non_macro_high_nan,
        "forward_return_stats":  ret_stats,
        "verdict": {
            "ok":     len(issues) == 0,
            "issues": issues,
        },
    }

    return report


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(r: Dict[str, Any]) -> None:
    n = r["n_rows"]
    print(f"\n{'='*60}")
    print(f"  QA DATASET V3 — {r['input_file']}")
    print(f"{'='*60}")
    print(f"  Rows        : {n:,}")
    print(f"  Tickers     : {r['n_tickers']:,}")
    print(f"  Asset types : {r['by_asset_type']}")
    print(f"  Labels      : {r['label_distribution']}")
    print(f"  target_non_ok : {r['target_non_ok_distribution']}")

    tc = r["temporal_checks"]
    status = "CLEAN" if tc["leakage_free"] else f"VIOLATIONS ({tc['n_violation']})"
    print(f"  Temporal    : {status}")

    dup = r["duplicates"]
    print(f"  Duplicates  : {dup['n_duplicates']}")

    top_nan = list(r["nan_top30_features"].items())[:5]
    print(f"  Top-5 NaN   : {top_nan}")

    verdict = r["verdict"]
    if verdict["ok"]:
        print(f"\n  ✅ Dataset VALID — ready for split + training")
    else:
        print(f"\n  ⚠️  Issues: {'; '.join(verdict['issues'])}")

    print(f"{'='*60}\n")


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

    ap = argparse.ArgumentParser(
        description="QA + structural validation of a v3 JSONL dataset"
    )
    ap.add_argument("--in",  dest="input",  required=True,
                    help="Path to v3 JSONL dataset")
    ap.add_argument("--out", dest="output", default="data/metrics/qa_v3_report.json",
                    help="Output JSON report path (default: data/metrics/qa_v3_report.json)")
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Starting QA on %s ...", input_path)
    report = run_qa(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Report written: %s", output_path)

    print_summary(report)

    if not report["verdict"]["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
