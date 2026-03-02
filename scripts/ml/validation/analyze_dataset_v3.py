"""
scripts/ml/validation/analyze_dataset_v3.py
============================================
Phase 1 — Structural validation of a v3 JSONL dataset.

Checks:
  1. Distribution stats  (total, asset_type, year, label, target_non_ok)
  2. Temporal leakage    (window_end < label_start <= label_end < label_end_60d)
  3. NaN / inf analysis  (per feature, macro features, labels, zero-dominated)
  4. Multi-horizon correlation sanity

Outputs:
  dataset_report_v3.json
  dataset_report_v3.txt

Usage:
  python scripts/ml/validation/analyze_dataset_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --out_dir data/reports/

No heavy deps: only json, math, collections, pathlib, logging (stdlib) + numpy/pandas.
No API / prod impact.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
log = logging.getLogger("analyze_dataset_v3")

# Known macro / cross-asset feature names (expected higher NaN rates)
_MACRO_FEATURES = {
    "vix_level", "vix_pct_60d", "rate_10y", "rate_2y", "term_spread",
    "credit_spread_hy", "credit_spread_ig", "vol_regime",
    "corr_spy", "corr_vix", "beta_market",
}

# Features that are identity / not ML inputs
_META_FEATURES = {"asset_type", "market", "ticker", "tuw_pct"}

# Multi-horizon return fields at top level (not inside features)
_HORIZON_FIELDS = [
    "forward_return_5d", "forward_return_10d",
    "forward_return_20d", "forward_return_60d",
]
_LABEL_FIELDS = [
    "label", "target_non_ok", "future_dd_20d", "future_vol_ratio",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= 5:
                    log.warning("Line %d JSON error: %s", i + 1, e)
    if bad:
        log.warning("Total bad JSON lines: %d", bad)
    log.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Distribution stats
# ---------------------------------------------------------------------------

def compute_distribution_stats(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    asset_counter: Counter = Counter()
    year_counter: Counter = Counter()
    label_counter: Counter = Counter()
    source_counter: Counter = Counter()
    non_ok_counter: Counter = Counter()
    version_counter: Counter = Counter()

    for rec in records:
        feats = rec.get("features", {})
        asset_counter[feats.get("asset_type", "unknown")] += 1

        wed = rec.get("window_end_date")
        if wed:
            try:
                year_counter[str(wed)[:4]] += 1
            except Exception:
                pass

        label_counter[rec.get("label", "missing")] += 1
        source_counter[rec.get("source", "unknown")] += 1

        tok = rec.get("target_non_ok")
        non_ok_counter[str(tok)] += 1

        version_counter[rec.get("version", "unknown")] += 1

    return {
        "total_samples": n,
        "versions": dict(version_counter),
        "by_asset_type": dict(asset_counter.most_common()),
        "by_year": dict(sorted(year_counter.items())),
        "by_label": dict(label_counter.most_common()),
        "by_source": dict(source_counter.most_common()),
        "by_target_non_ok": dict(non_ok_counter),
        "label_balance": {
            k: round(100 * v / n, 2) if n else 0.0
            for k, v in label_counter.items()
        },
    }


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def _parse_date(s: Optional[str]) -> Optional[date]:
    if s is None:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


class TemporalLeakageError(ValueError):
    pass


def check_temporal_leakage(records: List[Dict]) -> Dict[str, Any]:
    """
    Verifies for every record:
      1. window_end_date  <  label_start_date     (features end before labels start)
      2. label_start_date <= label_end_date        (20d window valid)
      3. label_end_date   <  label_end_date_60d    (60d extends beyond 20d)

    Raises TemporalLeakageError if ANY violation is found.
    Returns a summary dict if all checks pass.
    """
    violations: List[Dict] = []
    n_missing_dates = 0
    n_ok = 0
    n_no_60d = 0  # records with null label_end_date_60d (not enough future data)

    for i, rec in enumerate(records):
        wed  = _parse_date(rec.get("window_end_date"))
        lsd  = _parse_date(rec.get("label_start_date"))
        led  = _parse_date(rec.get("label_end_date"))
        l60d = _parse_date(rec.get("label_end_date_60d"))
        ticker = rec.get("features", {}).get("ticker", "?")

        if wed is None or lsd is None or led is None:
            n_missing_dates += 1
            if n_missing_dates <= 5:
                log.warning(
                    "Record %d (%s): missing date fields "
                    "window_end=%s label_start=%s label_end=%s",
                    i, ticker, wed, lsd, led,
                )
            continue

        row_violations = []

        # Rule 1: features must end strictly before labels start
        if not (wed < lsd):
            row_violations.append(
                f"window_end({wed}) >= label_start({lsd}) — features overlap labels"
            )

        # Rule 2: 20d label window must be chronologically valid
        if not (lsd <= led):
            row_violations.append(
                f"label_start({lsd}) > label_end({led}) — inverted label window"
            )

        # Rule 3: 60d label end must extend beyond 20d (when present)
        if l60d is not None:
            if not (led < l60d):
                row_violations.append(
                    f"label_end({led}) >= label_end_60d({l60d}) — 60d horizon not wider than 20d"
                )
        else:
            n_no_60d += 1

        if row_violations:
            violations.append({
                "record_index": i,
                "ticker": ticker,
                "window_end_date": str(wed),
                "label_start_date": str(lsd),
                "label_end_date": str(led),
                "label_end_date_60d": str(l60d) if l60d else None,
                "violations": row_violations,
            })
        else:
            n_ok += 1

    result = {
        "total_checked": len(records),
        "records_ok": n_ok,
        "records_missing_dates": n_missing_dates,
        "records_without_60d": n_no_60d,
        "violations": violations[:20],   # cap for readability
        "n_violations": len(violations),
        "leakage_free": len(violations) == 0,
    }

    if violations:
        summary = "\n".join(
            f"  [{v['ticker']} @ {v['window_end_date']}] {v['violations']}"
            for v in violations[:5]
        )
        raise TemporalLeakageError(
            f"TEMPORAL LEAKAGE DETECTED — {len(violations)} violations:\n{summary}\n"
            "(showing first 5)"
        )

    log.info(
        "Leakage check: %d records OK, %d without 60d, %d missing dates, 0 violations",
        n_ok, n_no_60d, n_missing_dates,
    )
    return result


# ---------------------------------------------------------------------------
# NaN / inf / zero analysis
# ---------------------------------------------------------------------------

def compute_nan_stats(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    # Collect all feature keys
    all_keys: set = set()
    for rec in records:
        all_keys.update(rec.get("features", {}).keys())
    all_keys -= _META_FEATURES

    feat_nan: Dict[str, int] = defaultdict(int)
    feat_zero: Dict[str, int] = defaultdict(int)
    feat_total: Dict[str, int] = defaultdict(int)

    # Label-level NaN
    horizon_nan: Dict[str, int] = {k: 0 for k in _HORIZON_FIELDS + _LABEL_FIELDS}
    horizon_total: Dict[str, int] = {k: 0 for k in _HORIZON_FIELDS + _LABEL_FIELDS}

    for rec in records:
        feats = rec.get("features", {})
        for k in all_keys:
            v = feats.get(k)
            feat_total[k] += 1
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                feat_nan[k] += 1
            elif isinstance(v, (int, float)) and v == 0.0:
                feat_zero[k] += 1

        for k in _HORIZON_FIELDS + _LABEL_FIELDS:
            v = rec.get(k)
            horizon_total[k] += 1
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                horizon_nan[k] += 1

    # Per-feature NaN rate
    nan_rates = {
        k: round(100 * feat_nan[k] / feat_total[k], 2)
        for k in all_keys
        if feat_total[k] > 0
    }
    # Sort descending
    nan_rates = dict(sorted(nan_rates.items(), key=lambda x: -x[1]))

    # Zero-dominated features (>80%)
    zero_dominated = {
        k: round(100 * feat_zero[k] / feat_total[k], 2)
        for k in all_keys
        if feat_total[k] > 0 and feat_zero[k] / feat_total[k] > 0.80
    }

    # Macro features specifically
    macro_nan = {
        k: nan_rates.get(k, 0.0)
        for k in _MACRO_FEATURES
    }

    # Label NaN rates
    label_nan_rates = {
        k: round(100 * horizon_nan[k] / horizon_total[k], 2)
        for k in _HORIZON_FIELDS + _LABEL_FIELDS
        if horizon_total[k] > 0
    }

    # Features with high NaN (>10%)
    high_nan_features = {k: v for k, v in nan_rates.items() if v > 10.0}
    high_nan_non_macro = {k: v for k, v in high_nan_features.items() if k not in _MACRO_FEATURES}

    return {
        "n_features_tracked": len(all_keys),
        "nan_rate_by_feature": nan_rates,
        "nan_rate_macro_features": macro_nan,
        "nan_rate_labels_horizons": label_nan_rates,
        "high_nan_features": high_nan_features,
        "high_nan_non_macro_features": high_nan_non_macro,
        "zero_dominated_features": zero_dominated,
        "features_all_ok_non_macro": sum(
            1 for k, v in nan_rates.items()
            if v == 0.0 and k not in _MACRO_FEATURES
        ),
    }


# ---------------------------------------------------------------------------
# Multi-horizon correlation sanity
# ---------------------------------------------------------------------------

def compute_correlation_stats(records: List[Dict]) -> Dict[str, Any]:
    """
    Checks:
    - corr(fwd_5d, fwd_10d, fwd_20d, fwd_60d)  — should decrease monotonically with horizon distance
    - corr(target_non_ok, future_dd_20d)         — non_ok should correlate with bad drawdowns
    - corr(target_non_ok, future_vol_ratio)      — non_ok should correlate with vol spikes
    """
    horizon_keys = _HORIZON_FIELDS + ["target_non_ok", "future_dd_20d", "future_vol_ratio"]
    data: Dict[str, List[float]] = {k: [] for k in horizon_keys}

    for rec in records:
        for k in horizon_keys:
            v = rec.get(k)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                data[k].append(float(v))
            else:
                data[k].append(float("nan"))

    df = pd.DataFrame(data).dropna(subset=["forward_return_20d", "target_non_ok"])

    results: Dict[str, Any] = {}
    corr_matrix: Dict[str, Dict[str, float]] = {}

    # Horizon cross-correlations
    for h1 in _HORIZON_FIELDS:
        corr_matrix[h1] = {}
        for h2 in _HORIZON_FIELDS:
            sub = df[[h1, h2]].dropna()
            if len(sub) >= 30:
                c = float(sub[h1].corr(sub[h2]))
                corr_matrix[h1][h2] = round(c, 4)
            else:
                corr_matrix[h1][h2] = None

    results["horizon_correlation_matrix"] = corr_matrix

    # Diagnostic: adjacent horizon correlations (should be high and decrease with distance)
    pairs = [
        ("forward_return_5d",  "forward_return_10d"),
        ("forward_return_10d", "forward_return_20d"),
        ("forward_return_20d", "forward_return_60d"),
        ("forward_return_5d",  "forward_return_60d"),
    ]
    for h1, h2 in pairs:
        sub = df[[h1, h2]].dropna()
        if len(sub) >= 30:
            c = float(sub[h1].corr(sub[h2]))
            results[f"corr_{h1}_vs_{h2}"] = round(c, 4)

    # target_non_ok vs future drawdown/vol
    for predictor in ("future_dd_20d", "future_vol_ratio", "forward_return_20d"):
        sub = df[["target_non_ok", predictor]].dropna()
        if len(sub) >= 30:
            c = float(sub["target_non_ok"].corr(sub[predictor]))
            results[f"corr_target_non_ok_vs_{predictor}"] = round(c, 4)

    # Sanity checks
    checks: Dict[str, Any] = {}

    # future_dd_20d should be more negative for non_ok samples
    ok_dd    = df[df["target_non_ok"] == 0]["future_dd_20d"].dropna()
    non_ok_dd = df[df["target_non_ok"] == 1]["future_dd_20d"].dropna()
    if len(ok_dd) >= 10 and len(non_ok_dd) >= 10:
        checks["avg_dd_ok"]     = round(float(ok_dd.mean()), 4)
        checks["avg_dd_non_ok"] = round(float(non_ok_dd.mean()), 4)
        checks["dd_separation_ok"] = float(non_ok_dd.mean()) < float(ok_dd.mean())

    # forward_return_20d should be lower for non_ok
    ok_ret    = df[df["target_non_ok"] == 0]["forward_return_20d"].dropna()
    non_ok_ret = df[df["target_non_ok"] == 1]["forward_return_20d"].dropna()
    if len(ok_ret) >= 10 and len(non_ok_ret) >= 10:
        checks["avg_return_20d_ok"]     = round(float(ok_ret.mean()), 4)
        checks["avg_return_20d_non_ok"] = round(float(non_ok_ret.mean()), 4)
        checks["return_separation_ok"]  = float(non_ok_ret.mean()) < float(ok_ret.mean())

    results["label_sanity_checks"] = checks
    results["n_samples_used"] = int(len(df))

    return results


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt_bar(value: float, total: float, width: int = 30) -> str:
    if total <= 0:
        return ""
    filled = int(round(value / total * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:6.0f} ({100*value/total:.1f}%)"


def write_txt_report(stats: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    a = lines.append

    a("=" * 70)
    a("DATASET V3 — STRUCTURAL VALIDATION REPORT")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("=" * 70)

    # --- Distribution
    dist = stats["distribution"]
    n = dist["total_samples"]
    a(f"\n{'DISTRIBUTION':─<70}")
    a(f"Total samples     : {n:,}")
    a(f"Versions          : {dist['versions']}")
    a(f"\nBy asset_type:")
    for k, v in dist["by_asset_type"].items():
        a(f"  {k:<18} {_fmt_bar(v, n)}")
    a(f"\nBy year (window_end_date):")
    for k, v in dist["by_year"].items():
        a(f"  {k}  {_fmt_bar(v, n)}")
    a(f"\nBy label:")
    for k, v in dist["by_label"].items():
        a(f"  {k:<10} {_fmt_bar(v, n)}")
    a(f"\nBy source:")
    for k, v in dist["by_source"].items():
        a(f"  {k:<12} {_fmt_bar(v, n)}")

    # --- Leakage
    leakage = stats["leakage"]
    a(f"\n{'TEMPORAL LEAKAGE CHECK':─<70}")
    a(f"Status            : {'✅ CLEAN — no violations' if leakage['leakage_free'] else '❌ VIOLATIONS FOUND'}")
    a(f"Records checked   : {leakage['total_checked']:,}")
    a(f"Records OK        : {leakage['records_ok']:,}")
    a(f"Missing dates     : {leakage['records_missing_dates']}")
    a(f"No 60d label      : {leakage['records_without_60d']}")
    a(f"Violations        : {leakage['n_violations']}")
    if leakage.get("violations"):
        a("  (first violation):")
        v = leakage["violations"][0]
        a(f"  ticker={v['ticker']} window_end={v['window_end_date']} "
          f"label_start={v['label_start_date']} label_end={v['label_end_date']}")

    # --- NaN
    nan_stats = stats["nan_analysis"]
    a(f"\n{'NaN / INF ANALYSIS':─<70}")
    a(f"Features tracked  : {nan_stats['n_features_tracked']}")
    a(f"Non-macro OK(0%)  : {nan_stats['features_all_ok_non_macro']}")
    a(f"\nMacro features NaN rates (expected higher):")
    for k, v in nan_stats["nan_rate_macro_features"].items():
        a(f"  {k:<30} {v:.1f}%")
    a(f"\nLabel / horizon NaN rates:")
    for k, v in nan_stats["nan_rate_labels_horizons"].items():
        a(f"  {k:<30} {v:.1f}%")
    if nan_stats["high_nan_non_macro_features"]:
        a(f"\n⚠️  Non-macro features with >10% NaN (unexpected):")
        for k, v in sorted(nan_stats["high_nan_non_macro_features"].items(), key=lambda x: -x[1]):
            a(f"  {k:<30} {v:.1f}%")
    else:
        a(f"\n✅ No non-macro features with >10% NaN")
    if nan_stats["zero_dominated_features"]:
        a(f"\n⚠️  Zero-dominated features (>80% zero):")
        for k, v in nan_stats["zero_dominated_features"].items():
            a(f"  {k:<30} {v:.1f}%")
    else:
        a(f"\n✅ No zero-dominated features")

    # --- Correlations
    corr = stats["correlations"]
    a(f"\n{'MULTI-HORIZON CORRELATION SANITY':─<70}")
    a(f"Samples used      : {corr.get('n_samples_used', '?'):,}")
    for key in ["corr_forward_return_5d_vs_forward_return_10d",
                "corr_forward_return_10d_vs_forward_return_20d",
                "corr_forward_return_20d_vs_forward_return_60d",
                "corr_forward_return_5d_vs_forward_return_60d"]:
        v = corr.get(key)
        label = key.replace("corr_", "").replace("_vs_", " vs ")
        a(f"  {label:<45} {v}")
    a(f"\nLabel sanity (target_non_ok):")
    for key in ["corr_target_non_ok_vs_future_dd_20d",
                "corr_target_non_ok_vs_future_vol_ratio",
                "corr_target_non_ok_vs_forward_return_20d"]:
        v = corr.get(key)
        label = key.replace("corr_", "").replace("_vs_", " vs ")
        a(f"  {label:<45} {v}")
    sc = corr.get("label_sanity_checks", {})
    if sc:
        sep_ok = sc.get("dd_separation_ok")
        ret_ok = sc.get("return_separation_ok")
        a(f"\n  avg dd  ok={sc.get('avg_dd_ok')}   non_ok={sc.get('avg_dd_non_ok')}  "
          f"{'✅' if sep_ok else '⚠️ '}")
        a(f"  avg ret ok={sc.get('avg_return_20d_ok')}  non_ok={sc.get('avg_return_20d_non_ok')}  "
          f"{'✅' if ret_ok else '⚠️ '}")

    # --- Verdict
    leakage_ok = leakage["leakage_free"]
    critical_nan = bool(nan_stats.get("high_nan_non_macro_features"))
    a(f"\n{'VERDICT':─<70}")
    if leakage_ok and not critical_nan:
        a("✅ Dataset structurally VALID — ready for CV split and model training.")
    else:
        issues = []
        if not leakage_ok:
            issues.append(f"temporal leakage ({leakage['n_violations']} violations)")
        if critical_nan:
            issues.append(f"high NaN in non-macro features: {list(nan_stats['high_nan_non_macro_features'].keys())}")
        a(f"⚠️  Dataset has issues: {', '.join(issues)}")
    a("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("TXT report written: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(input_path: Path, out_dir: Path) -> Dict[str, Any]:
    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records loaded from {input_path}")

    log.info("Computing distribution stats...")
    dist = compute_distribution_stats(records)

    log.info("Checking temporal leakage...")
    try:
        leakage = check_temporal_leakage(records)
    except TemporalLeakageError as e:
        log.error("LEAKAGE: %s", e)
        leakage = {"leakage_free": False, "error": str(e),
                   "n_violations": -1, "total_checked": len(records),
                   "records_ok": 0, "records_missing_dates": 0,
                   "records_without_60d": 0, "violations": []}

    log.info("Analyzing NaN / inf...")
    nan_stats = compute_nan_stats(records)

    log.info("Computing multi-horizon correlations...")
    corr_stats = compute_correlation_stats(records)

    stats = {
        "input_file":   str(input_path),
        "distribution": dist,
        "leakage":      leakage,
        "nan_analysis": nan_stats,
        "correlations": corr_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_report_v3.json"
    txt_path  = out_dir / "dataset_report_v3.txt"

    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON report written: %s", json_path)

    write_txt_report(stats, txt_path)

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description="Validate a v3 dataset JSONL file")
    ap.add_argument("--input",   required=True, help="Path to v3 JSONL file")
    ap.add_argument("--out_dir", default="data/reports",
                    help="Output directory for reports (default: data/reports)")
    args = ap.parse_args()

    stats = analyze(Path(args.input), Path(args.out_dir))

    # Print brief summary to stdout
    n = stats["distribution"]["total_samples"]
    leakage_ok = stats["leakage"]["leakage_free"]
    critical_nan = bool(stats["nan_analysis"].get("high_nan_non_macro_features"))
    print(f"\n{'='*50}")
    print(f"Samples : {n:,}")
    print(f"Labels  : {stats['distribution']['by_label']}")
    print(f"Leakage : {'CLEAN ✅' if leakage_ok else 'VIOLATIONS ❌'}")
    print(f"NaN     : {'OK ✅' if not critical_nan else 'Issues ⚠️'}")
    print(f"Reports : {args.out_dir}/dataset_report_v3.{{json,txt}}")
    print("=" * 50)

    if not leakage_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
