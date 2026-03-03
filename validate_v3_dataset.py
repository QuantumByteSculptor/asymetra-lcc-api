# validate_v3_dataset.py
"""
Validation script for v3 JSONL datasets.

Usage:
  python validate_v3_dataset.py --input data/training/smoke_v3.jsonl \
      --out data/metrics/v3_smoke_report.json

Checks:
  - Sample counts (total, per asset_type, per label)
  - window_end_date < label_start_date (zero future leakage)
  - No empty timestamps
  - No NaN / inf in feature values
  - Null % per feature
  - Multi-horizon label coverage
  - Provider distribution
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────

def _is_bad(v: Any) -> bool:
    """Return True if value is NaN, Inf, or non-finite float."""
    if v is None:
        return False  # None → counted as null, not bad
    if isinstance(v, float):
        return not math.isfinite(v)
    return False


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {i} JSON decode error: {e}", file=sys.stderr)
    return records


def validate(records: List[Dict[str, Any]], run_start: float) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"error": "empty dataset", "valid": False}

    # ── Counts ────────────────────────────────────────────────────────────────
    by_asset: Counter = Counter()
    by_label: Counter = Counter()
    by_source: Counter = Counter()
    horizon_coverage: Dict[str, int] = {
        "forward_return_5d": 0,
        "forward_return_10d": 0,
        "forward_return_20d": 0,
        "forward_return_60d": 0,
    }

    # ── Temporal checks ───────────────────────────────────────────────────────
    ts_missing: int = 0         # any timestamp field None/empty
    leakage_violations: int = 0  # window_end_date >= label_start_date

    # ── Feature quality ───────────────────────────────────────────────────────
    null_counts: Dict[str, int] = defaultdict(int)
    inf_counts: Dict[str, int] = defaultdict(int)
    feature_names: set = set()

    for rec in records:
        feats = rec.get("features", {})
        asset_type = feats.get("asset_type", rec.get("asset_type", "unknown"))
        by_asset[asset_type] += 1
        by_label[rec.get("label", "?")] += 1
        by_source[rec.get("source", "unknown")] += 1

        # Horizon coverage
        for h in horizon_coverage:
            if rec.get(h) is not None:
                horizon_coverage[h] += 1

        # Timestamp checks
        ts_fields = {
            "window_start_date": rec.get("window_start_date"),
            "window_end_date":   rec.get("window_end_date"),
            "label_start_date":  rec.get("label_start_date"),
            "label_end_date":    rec.get("label_end_date"),
        }
        for fname, fval in ts_fields.items():
            if not fval:
                ts_missing += 1

        # Temporal ordering: window_end < label_start
        wed = rec.get("window_end_date")
        lsd = rec.get("label_start_date")
        if wed and lsd:
            try:
                if datetime.fromisoformat(wed) >= datetime.fromisoformat(lsd):
                    leakage_violations += 1
            except ValueError:
                ts_missing += 1

        # Feature null / inf
        for fname, fval in feats.items():
            if fname in ("asset_type", "market", "ticker"):
                continue
            feature_names.add(fname)
            if fval is None:
                null_counts[fname] += 1
            elif _is_bad(fval):
                inf_counts[fname] += 1

    # ── Null % per feature ────────────────────────────────────────────────────
    null_pct: Dict[str, float] = {
        k: round(100.0 * null_counts[k] / n, 2)
        for k in sorted(feature_names)
    }
    inf_pct: Dict[str, float] = {
        k: round(100.0 * inf_counts[k] / n, 2)
        for k in sorted(inf_counts)
        if inf_counts[k] > 0
    }

    # Features with high null rate (>50%)
    high_null = {k: v for k, v in null_pct.items() if v > 50.0}

    # ── Validity determination ────────────────────────────────────────────────
    issues: List[str] = []
    if ts_missing > 0:
        issues.append(f"{ts_missing} records have missing/empty timestamp fields")
    if leakage_violations > 0:
        issues.append(f"CRITICAL: {leakage_violations} temporal leakage violations "
                      f"(window_end >= label_start)")
    if inf_pct:
        issues.append(f"{len(inf_pct)} features have NaN/Inf values: {list(inf_pct.keys())[:10]}")
    if high_null:
        issues.append(f"{len(high_null)} features >50% null (expected for macro if --skip_macro): "
                      f"{list(high_null.keys())[:10]}")

    valid = leakage_violations == 0 and len(inf_pct) == 0

    # ── Suggestions ──────────────────────────────────────────────────────────
    suggestions: List[str] = []
    if high_null:
        macro_fields = {"vix_level","vix_pct_60d","rate_10y","rate_2y","term_spread",
                        "credit_spread_hy","vol_regime","corr_spy","corr_vix","beta_market"}
        macro_nulls = {k for k in high_null if k in macro_fields}
        if macro_nulls:
            suggestions.append(
                "Macro/cross-asset fields are 100% null — expected if --skip_macro was used. "
                "Run without --skip_macro for full run."
            )
        non_macro_nulls = {k: v for k, v in high_null.items() if k not in macro_fields}
        if non_macro_nulls:
            suggestions.append(
                f"Non-macro features with >50% null: {list(non_macro_nulls.keys())} — "
                "review window length or data quality."
            )
    if n < 50:
        suggestions.append(
            f"Only {n} samples — smoke test only. Run full pipeline for production dataset."
        )

    elapsed = round(time.time() - run_start, 2)

    return {
        "valid": valid,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": elapsed,
        "total_samples": n,
        "samples_per_asset_type": dict(by_asset),
        "samples_per_label": dict(by_label),
        "label_distribution_pct": {
            k: round(100.0 * v / n, 1) for k, v in by_label.items()
        },
        "horizon_coverage": {
            k: {"count": v, "pct": round(100.0 * v / n, 1)}
            for k, v in horizon_coverage.items()
        },
        "provider_distribution": dict(by_source),
        "temporal_checks": {
            "timestamp_fields_missing": ts_missing,
            "leakage_violations": leakage_violations,
            "leakage_free": leakage_violations == 0,
        },
        "feature_quality": {
            "total_features": len(feature_names),
            "features_with_any_inf_nan": len(inf_pct),
            "inf_nan_pct_per_feature": inf_pct,
            "null_pct_per_feature": null_pct,
            "features_high_null_gt50pct": list(high_null.keys()),
        },
        "issues": issues,
        "suggestions": suggestions,
    }


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Validate a v3 JSONL dataset")
    ap.add_argument("--input", required=True, help="Path to JSONL file")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {in_path} ...")
    records = load_records(in_path)
    print(f"  {len(records)} records loaded")

    print("Running validation checks ...")
    report = validate(records, run_start=t0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {out_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    status = "✅ VALID" if report["valid"] else "❌ INVALID"
    print(f"  {status}  — {report['total_samples']} samples")
    print(f"  Asset types: {report['samples_per_asset_type']}")
    print(f"  Labels: {report['label_distribution_pct']}")
    print(f"  Leakage violations: {report['temporal_checks']['leakage_violations']}")
    print(f"  Features with Inf/NaN: {report['feature_quality']['features_with_any_inf_nan']}")
    print(f"  Features >50% null: {len(report['feature_quality']['features_high_null_gt50pct'])}")
    if report["issues"]:
        print()
        print("  ISSUES:")
        for iss in report["issues"]:
            print(f"    ⚠️  {iss}")
    if report["suggestions"]:
        print()
        print("  SUGGESTIONS:")
        for sug in report["suggestions"]:
            print(f"    →  {sug}")
    print("=" * 60)

    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
