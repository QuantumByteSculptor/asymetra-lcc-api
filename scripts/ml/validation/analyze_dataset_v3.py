"""
scripts/ml/validation/analyze_dataset_v3.py
============================================
Phase 1 — Structural validation + QA avancée d'un dataset v3 JSONL.

Checks:
  1. Distribution (total, asset_type, year, label, target_non_ok, source)
  2. Leakage temporel  (window_end < label_start <= label_end < label_end_60d)
  3. NaN / inf / zéro par feature
  4. Corrélations multi-horizon + sanity labels
  5. [NEW] Distribution forward returns (quantiles 1/5/50/95/99)
  6. [NEW] Histogramme label par asset_type
  7. [NEW] Drift temporel intra-dataset (variance des features par année)
  8. [NEW] Top-20 features à variance quasi nulle (candidates à supprimer)

Outputs:
  data/reports/dataset_report_v3.json
  data/reports/dataset_report_v3.txt

Usage:
  python scripts/ml/validation/analyze_dataset_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --out_dir data/reports/
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

log = logging.getLogger("analyze_dataset_v3")

_MACRO_FEATURES = {
    "vix_level", "vix_pct_60d", "rate_10y", "rate_2y", "term_spread",
    "credit_spread_hy", "credit_spread_ig", "vol_regime",
    "corr_spy", "corr_vix", "beta_market",
}
_META_FEATURES = {"asset_type", "market", "ticker", "tuw_pct"}
_HORIZON_FIELDS = [
    "forward_return_5d", "forward_return_10d",
    "forward_return_20d", "forward_return_60d",
]
_LABEL_FIELDS = ["label", "target_non_ok", "future_dd_20d", "future_vol_ratio"]


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
                if bad <= 3:
                    log.warning("Line %d JSON error: %s", i + 1, e)
    if bad:
        log.warning("Total bad lines: %d", bad)
    log.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# 1. Distribution
# ---------------------------------------------------------------------------

def compute_distribution_stats(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    asset_c: Counter = Counter()
    year_c:  Counter = Counter()
    label_c: Counter = Counter()
    source_c: Counter = Counter()
    non_ok_c: Counter = Counter()
    version_c: Counter = Counter()

    for rec in records:
        feats = rec.get("features", {})
        asset_c[feats.get("asset_type", "unknown")] += 1
        wed = rec.get("window_end_date")
        if wed:
            year_c[str(wed)[:4]] += 1
        label_c[rec.get("label", "missing")] += 1
        source_c[rec.get("source", "unknown")] += 1
        non_ok_c[str(rec.get("target_non_ok"))] += 1
        version_c[rec.get("version", "unknown")] += 1

    return {
        "total_samples":   n,
        "versions":        dict(version_c),
        "by_asset_type":   dict(asset_c.most_common()),
        "by_year":         dict(sorted(year_c.items())),
        "by_label":        dict(label_c.most_common()),
        "by_source":       dict(source_c.most_common()),
        "by_target_non_ok": dict(non_ok_c),
        "label_balance":   {k: round(100 * v / n, 2) if n else 0.0
                            for k, v in label_c.items()},
    }


# ---------------------------------------------------------------------------
# 2. Leakage
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
    violations: List[Dict] = []
    n_missing = n_ok = n_no_60d = 0

    for i, rec in enumerate(records):
        wed  = _parse_date(rec.get("window_end_date"))
        lsd  = _parse_date(rec.get("label_start_date"))
        led  = _parse_date(rec.get("label_end_date"))
        l60d = _parse_date(rec.get("label_end_date_60d"))
        ticker = rec.get("features", {}).get("ticker", "?")

        if wed is None or lsd is None or led is None:
            n_missing += 1
            continue

        viol = []
        if not (wed < lsd):
            viol.append(f"window_end({wed}) >= label_start({lsd})")
        if not (lsd <= led):
            viol.append(f"label_start({lsd}) > label_end({led})")
        if l60d is not None and not (led < l60d):
            viol.append(f"label_end({led}) >= label_end_60d({l60d})")
        else:
            if l60d is None:
                n_no_60d += 1

        if viol:
            violations.append({"record_index": i, "ticker": ticker, "violations": viol})
        else:
            n_ok += 1

    result = {
        "total_checked": len(records),
        "records_ok": n_ok,
        "records_missing_dates": n_missing,
        "records_without_60d": n_no_60d,
        "n_violations": len(violations),
        "violations": violations[:20],
        "leakage_free": len(violations) == 0,
    }

    if violations:
        summary = "\n".join(
            f"  [{v['ticker']}] {v['violations']}" for v in violations[:3]
        )
        raise TemporalLeakageError(
            f"TEMPORAL LEAKAGE — {len(violations)} violations:\n{summary}"
        )

    log.info("Leakage: %d OK / %d missing dates / %d no-60d / 0 violations",
             n_ok, n_missing, n_no_60d)
    return result


# ---------------------------------------------------------------------------
# 3. NaN / inf / zero
# ---------------------------------------------------------------------------

def compute_nan_stats(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    all_keys: set = set()
    for rec in records:
        all_keys.update(rec.get("features", {}).keys())
    all_keys -= _META_FEATURES

    feat_nan   = defaultdict(int)
    feat_zero  = defaultdict(int)
    feat_total = defaultdict(int)

    horizon_nan   = {k: 0 for k in _HORIZON_FIELDS + _LABEL_FIELDS}
    horizon_total = {k: 0 for k in _HORIZON_FIELDS + _LABEL_FIELDS}

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

    nan_rates = dict(sorted(
        {k: round(100 * feat_nan[k] / max(feat_total[k], 1), 2) for k in all_keys}.items(),
        key=lambda x: -x[1],
    ))
    zero_dom = {
        k: round(100 * feat_zero[k] / max(feat_total[k], 1), 2)
        for k in all_keys
        if feat_total[k] > 0 and feat_zero[k] / max(feat_total[k], 1) > 0.80
    }
    macro_nan = {k: nan_rates.get(k, 0.0) for k in _MACRO_FEATURES}
    label_nan = {
        k: round(100 * horizon_nan[k] / max(horizon_total[k], 1), 2)
        for k in _HORIZON_FIELDS + _LABEL_FIELDS
    }
    high_nan = {k: v for k, v in nan_rates.items() if v > 10.0}
    high_nan_non_macro = {k: v for k, v in high_nan.items() if k not in _MACRO_FEATURES}

    return {
        "n_features_tracked": len(all_keys),
        "nan_rate_by_feature": nan_rates,
        "nan_rate_macro_features": macro_nan,
        "nan_rate_labels_horizons": label_nan,
        "high_nan_features": high_nan,
        "high_nan_non_macro_features": high_nan_non_macro,
        "zero_dominated_features": zero_dom,
        "features_all_ok_non_macro": sum(
            1 for k, v in nan_rates.items() if v == 0.0 and k not in _MACRO_FEATURES
        ),
    }


# ---------------------------------------------------------------------------
# 4. Multi-horizon correlations
# ---------------------------------------------------------------------------

def compute_correlation_stats(records: List[Dict]) -> Dict[str, Any]:
    horizon_keys = _HORIZON_FIELDS + ["target_non_ok", "future_dd_20d", "future_vol_ratio"]
    data: Dict[str, List[float]] = {k: [] for k in horizon_keys}

    for rec in records:
        for k in horizon_keys:
            v = rec.get(k)
            data[k].append(float(v) if (v is not None and math.isfinite(float(v) if isinstance(v, (int, float)) else float("nan"))) else float("nan"))

    df = pd.DataFrame(data).dropna(subset=["forward_return_20d", "target_non_ok"])
    results: Dict[str, Any] = {"n_samples_used": int(len(df))}

    # Horizon cross-correlations
    corr_matrix: Dict[str, Dict[str, float]] = {}
    for h1 in _HORIZON_FIELDS:
        corr_matrix[h1] = {}
        for h2 in _HORIZON_FIELDS:
            sub = df[[h1, h2]].dropna()
            if len(sub) >= 30:
                try:
                    a1 = sub[h1].to_numpy(dtype=float)
                    a2 = sub[h2].to_numpy(dtype=float)
                    c = float(np.corrcoef(a1, a2)[0, 1])
                    corr_matrix[h1][h2] = round(c, 4) if np.isfinite(c) else None
                except Exception:
                    corr_matrix[h1][h2] = None
            else:
                corr_matrix[h1][h2] = None

    results["horizon_correlation_matrix"] = corr_matrix

    for h1, h2 in [("forward_return_5d", "forward_return_10d"),
                   ("forward_return_10d", "forward_return_20d"),
                   ("forward_return_20d", "forward_return_60d"),
                   ("forward_return_5d", "forward_return_60d")]:
        sub = df[[h1, h2]].dropna()
        if len(sub) >= 30:
            try:
                c = float(np.corrcoef(sub[h1].to_numpy(float), sub[h2].to_numpy(float))[0, 1])
                results[f"corr_{h1}_vs_{h2}"] = round(c, 4) if np.isfinite(c) else None
            except Exception:
                pass

    for pred in ("future_dd_20d", "future_vol_ratio", "forward_return_20d"):
        sub = df[["target_non_ok", pred]].dropna()
        if len(sub) >= 30:
            try:
                c = float(np.corrcoef(sub["target_non_ok"].to_numpy(float),
                                      sub[pred].to_numpy(float))[0, 1])
                results[f"corr_target_non_ok_vs_{pred}"] = round(c, 4) if np.isfinite(c) else None
            except Exception:
                pass

    # Label separation
    sc: Dict[str, Any] = {}
    ok_dd    = df[df["target_non_ok"] == 0]["future_dd_20d"].dropna()
    nok_dd   = df[df["target_non_ok"] == 1]["future_dd_20d"].dropna()
    if len(ok_dd) >= 10 and len(nok_dd) >= 10:
        sc["avg_dd_ok"]     = round(float(ok_dd.mean()), 4)
        sc["avg_dd_non_ok"] = round(float(nok_dd.mean()), 4)
        sc["dd_separation_ok"] = float(nok_dd.mean()) < float(ok_dd.mean())
    ok_ret  = df[df["target_non_ok"] == 0]["forward_return_20d"].dropna()
    nok_ret = df[df["target_non_ok"] == 1]["forward_return_20d"].dropna()
    if len(ok_ret) >= 10 and len(nok_ret) >= 10:
        sc["avg_return_20d_ok"]     = round(float(ok_ret.mean()), 4)
        sc["avg_return_20d_non_ok"] = round(float(nok_ret.mean()), 4)
        sc["return_separation_ok"]  = float(nok_ret.mean()) < float(ok_ret.mean())
    results["label_sanity_checks"] = sc

    return results


# ---------------------------------------------------------------------------
# 5. [NEW] Distribution forward returns & future_dd_20d
# ---------------------------------------------------------------------------

def compute_return_distributions(records: List[Dict]) -> Dict[str, Any]:
    fields = {
        "forward_return_5d":  [],
        "forward_return_10d": [],
        "forward_return_20d": [],
        "forward_return_60d": [],
        "future_dd_20d":      [],
        "future_vol_ratio":   [],
    }
    for rec in records:
        for k in fields:
            v = rec.get(k)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                fields[k].append(float(v))

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    result: Dict[str, Any] = {}

    for field, vals in fields.items():
        if not vals:
            result[field] = {"n": 0}
            continue
        arr = np.array(vals, dtype=float)
        q   = np.quantile(arr, quantiles).tolist()
        result[field] = {
            "n":      len(vals),
            "mean":   round(float(arr.mean()), 6),
            "std":    round(float(arr.std(ddof=1)), 6),
            "min":    round(float(arr.min()), 6),
            "max":    round(float(arr.max()), 6),
            "q1pct":  round(q[0], 6),
            "q5pct":  round(q[1], 6),
            "q25pct": round(q[2], 6),
            "q50pct": round(q[3], 6),
            "q75pct": round(q[4], 6),
            "q95pct": round(q[5], 6),
            "q99pct": round(q[6], 6),
            "pct_positive": round(float(np.mean(arr > 0)), 4),
            "pct_negative": round(float(np.mean(arr < 0)), 4),
        }
    return result


# ---------------------------------------------------------------------------
# 6. [NEW] Label histogram by asset_type
# ---------------------------------------------------------------------------

def compute_label_by_asset_type(records: List[Dict]) -> Dict[str, Any]:
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        at  = rec.get("features", {}).get("asset_type", "unknown")
        lbl = rec.get("label", "missing")
        matrix[at][lbl] += 1

    result: Dict[str, Any] = {}
    for at, cntr in matrix.items():
        total = sum(cntr.values())
        result[at] = {
            "total": total,
            "by_label": dict(cntr),
            "pct": {k: round(100 * v / total, 1) for k, v in cntr.items()},
            "non_ok_rate": round(
                (cntr.get("warn", 0) + cntr.get("block", 0)) / max(total, 1), 4
            ),
        }
    return result


# ---------------------------------------------------------------------------
# 7. [NEW] Temporal drift detection (intra-dataset)
# ---------------------------------------------------------------------------

def compute_temporal_drift(records: List[Dict]) -> Dict[str, Any]:
    """
    Group records by year (window_end_date) and compute per-feature mean/std.
    Drift = std of yearly means relative to global std (coefficient of variation).
    High CV → feature mean shifts significantly over time.
    """
    year_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    numeric_feats: set = set()
    for rec in records[:1000]:   # sample to get feature list fast
        for k, v in rec.get("features", {}).items():
            if isinstance(v, (int, float)) and k not in _META_FEATURES:
                numeric_feats.add(k)

    for rec in records:
        wed = rec.get("window_end_date")
        if not wed:
            continue
        year = str(wed)[:4]
        feats = rec.get("features", {})
        for k in numeric_feats:
            v = feats.get(k)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
                year_data[year][k].append(float(v))

    if len(year_data) < 2:
        return {"n_years": len(year_data), "note": "not enough years for drift analysis"}

    years = sorted(year_data.keys())
    drift_scores: List[Dict] = []

    for feat in numeric_feats:
        yearly_means = [
            float(np.mean(year_data[y][feat]))
            for y in years
            if len(year_data[y][feat]) >= 10
        ]
        if len(yearly_means) < 2:
            continue
        arr = np.array(yearly_means)
        global_mean = float(arr.mean())
        if abs(global_mean) < 1e-12:
            cv = 0.0
        else:
            cv = float(arr.std(ddof=1) / abs(global_mean))

        drift_scores.append({
            "feature": feat,
            "cv_yearly_mean": round(cv, 4),
            "mean_range": round(float(arr.max() - arr.min()), 4),
            "yearly_means": {y: round(float(np.mean(year_data[y][feat])), 4)
                             for y in years if len(year_data[y][feat]) >= 10},
        })

    drift_scores.sort(key=lambda x: -x["cv_yearly_mean"])

    return {
        "n_years":           len(years),
        "years":             years,
        "top_drifting_features": drift_scores[:20],
        "high_drift_count":  sum(1 for d in drift_scores if d["cv_yearly_mean"] > 0.30),
    }


# ---------------------------------------------------------------------------
# 8. [NEW] Low-variance features
# ---------------------------------------------------------------------------

def compute_low_variance_features(records: List[Dict], threshold: float = 0.01) -> Dict[str, Any]:
    """
    Identify features with near-zero variance (useless for training).
    threshold: features with std < threshold × global_std_median are flagged.
    """
    feat_vals: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        for k, v in rec.get("features", {}).items():
            if (k not in _META_FEATURES and isinstance(v, (int, float))
                    and math.isfinite(float(v))):
                feat_vals[k].append(float(v))

    stds: Dict[str, float] = {}
    for k, vals in feat_vals.items():
        if len(vals) >= 30:
            stds[k] = float(np.std(vals, ddof=1))

    if not stds:
        return {}

    std_vals = list(stds.values())
    median_std = float(np.median([s for s in std_vals if s > 0])) if std_vals else 1.0
    abs_threshold = threshold * median_std

    low_var = {k: round(v, 8) for k, v in stds.items() if v < abs_threshold}
    low_var_sorted = dict(sorted(low_var.items(), key=lambda x: x[1]))

    all_sorted = dict(sorted(stds.items(), key=lambda x: x[1]))
    top20_lowest = dict(list(all_sorted.items())[:20])

    return {
        "median_feature_std":   round(median_std, 6),
        "threshold_applied":    round(abs_threshold, 8),
        "n_low_variance":       len(low_var),
        "low_variance_features": low_var_sorted,
        "top20_lowest_std":     {k: round(v, 6) for k, v in top20_lowest.items()},
        "recommendation":       (
            f"Consider removing {len(low_var)} near-constant features: "
            f"{list(low_var_sorted.keys())[:10]}"
        ) if low_var else "All features have sufficient variance.",
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt_bar(v: float, total: float, width: int = 28) -> str:
    if total <= 0:
        return ""
    filled = int(round(v / total * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {v:6.0f} ({100*v/total:.1f}%)"


def write_txt_report(stats: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    a = lines.append

    a("=" * 72)
    a("DATASET V3 — QA FULL REPORT")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("=" * 72)

    # Distribution
    dist = stats["distribution"]
    n = dist["total_samples"]
    a(f"\n{'DISTRIBUTION':─<72}")
    a(f"Total samples : {n:,}")
    a(f"Versions      : {dist['versions']}")
    a(f"\nBy asset_type:")
    for k, v in dist["by_asset_type"].items():
        a(f"  {k:<18} {_fmt_bar(v, n)}")
    a(f"\nBy year:")
    for k, v in dist["by_year"].items():
        a(f"  {k}  {_fmt_bar(v, n)}")
    a(f"\nBy label:")
    for k, v in dist["by_label"].items():
        a(f"  {k:<10} {_fmt_bar(v, n)}")
    a(f"\nBy source:")
    for k, v in dist["by_source"].items():
        a(f"  {k:<14} {_fmt_bar(v, n)}")

    # Leakage
    lk = stats["leakage"]
    a(f"\n{'TEMPORAL LEAKAGE':─<72}")
    a(f"  {'CLEAN ✅' if lk['leakage_free'] else 'VIOLATIONS ❌'}  "
      f"({lk['records_ok']:,} OK / {lk['n_violations']} violations / "
      f"{lk['records_missing_dates']} missing dates)")

    # NaN
    nn = stats["nan_analysis"]
    a(f"\n{'NaN / INF ANALYSIS':─<72}")
    a(f"  Features tracked  : {nn['n_features_tracked']}")
    a(f"  Non-macro OK(0%)  : {nn['features_all_ok_non_macro']}")
    a(f"\n  Macro NaN rates (expected higher in --skip_macro runs):")
    for k, v in nn["nan_rate_macro_features"].items():
        a(f"    {k:<30} {v:.1f}%")
    if nn["high_nan_non_macro_features"]:
        a(f"\n  ⚠️  Non-macro >10% NaN:")
        for k, v in sorted(nn["high_nan_non_macro_features"].items(), key=lambda x: -x[1]):
            a(f"    {k:<30} {v:.1f}%")
    if nn["zero_dominated_features"]:
        a(f"\n  ⚠️  Zero-dominated (>80%):")
        for k, v in nn["zero_dominated_features"].items():
            a(f"    {k:<30} {v:.1f}%")

    # Return distributions
    rd = stats.get("return_distributions", {})
    a(f"\n{'FORWARD RETURN DISTRIBUTIONS (quantiles)':─<72}")
    a(f"  {'Field':<28} {'q1%':>8} {'q5%':>8} {'q50%':>8} {'q95%':>8} {'q99%':>8} {'n':>7}")
    for field in _HORIZON_FIELDS + ["future_dd_20d"]:
        d = rd.get(field, {})
        if not d or d.get("n", 0) == 0:
            continue
        a(f"  {field:<28} "
          f"{d.get('q1pct', 'N/A'):>8.4f} "
          f"{d.get('q5pct', 'N/A'):>8.4f} "
          f"{d.get('q50pct', 'N/A'):>8.4f} "
          f"{d.get('q95pct', 'N/A'):>8.4f} "
          f"{d.get('q99pct', 'N/A'):>8.4f} "
          f"{d.get('n', 0):>7,}")

    # Label by asset_type
    lba = stats.get("label_by_asset_type", {})
    a(f"\n{'LABEL DISTRIBUTION BY ASSET TYPE':─<72}")
    a(f"  {'asset_type':<16} {'total':>7} {'ok%':>7} {'warn%':>7} {'block%':>7} {'non_ok%':>9}")
    for at, d in sorted(lba.items()):
        pct = d.get("pct", {})
        a(f"  {at:<16} {d['total']:>7,} "
          f"{pct.get('ok', 0):>7.1f} "
          f"{pct.get('warn', 0):>7.1f} "
          f"{pct.get('block', 0):>7.1f} "
          f"{d['non_ok_rate']*100:>9.1f}")

    # Drift
    dr = stats.get("temporal_drift", {})
    a(f"\n{'TEMPORAL DRIFT (intra-dataset, by year)':─<72}")
    a(f"  Years: {dr.get('years', [])}  |  Features with high drift: {dr.get('high_drift_count', '?')}")
    top_drift = dr.get("top_drifting_features", [])[:10]
    if top_drift:
        a(f"  {'Feature':<30} {'CV yearly mean':>15}")
        for d in top_drift:
            a(f"  {d['feature']:<30} {d['cv_yearly_mean']:>15.4f}")

    # Low variance
    lv = stats.get("low_variance", {})
    a(f"\n{'LOW-VARIANCE FEATURES (candidates to remove)':─<72}")
    a(f"  Median std        : {lv.get('median_feature_std', 'N/A')}")
    a(f"  Features flagged  : {lv.get('n_low_variance', 0)}")
    a(f"  Top-20 lowest std :")
    for k, v in list(lv.get("top20_lowest_std", {}).items())[:10]:
        a(f"    {k:<35} std={v}")

    # Correlations
    cr = stats.get("correlations", {})
    a(f"\n{'MULTI-HORIZON CORRELATION':─<72}")
    for key in ["corr_forward_return_5d_vs_forward_return_10d",
                "corr_forward_return_10d_vs_forward_return_20d",
                "corr_forward_return_20d_vs_forward_return_60d"]:
        lbl = key.replace("corr_", "").replace("_vs_", " vs ")
        a(f"  {lbl:<45} {cr.get(key, 'N/A')}")
    sc = cr.get("label_sanity_checks", {})
    if sc:
        a(f"\n  dd  ok={sc.get('avg_dd_ok')}  non_ok={sc.get('avg_dd_non_ok')}  "
          f"{'✅' if sc.get('dd_separation_ok') else '⚠️ '}")
        a(f"  ret ok={sc.get('avg_return_20d_ok')}  non_ok={sc.get('avg_return_20d_non_ok')}  "
          f"{'✅' if sc.get('return_separation_ok') else '⚠️ '}")

    # Verdict
    lk_ok = lk["leakage_free"]
    nan_ok = not bool(nn.get("high_nan_non_macro_features"))
    a(f"\n{'VERDICT':─<72}")
    if lk_ok and nan_ok:
        a("  ✅ Dataset structurally VALID — proceed to CV split and training.")
    else:
        issues = []
        if not lk_ok:
            issues.append(f"leakage ({lk['n_violations']} violations)")
        if not nan_ok:
            issues.append(f"high NaN in non-macro features: "
                          f"{list(nn['high_nan_non_macro_features'].keys())[:5]}")
        a(f"  ⚠️  Issues: {', '.join(issues)}")
    a("=" * 72)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("TXT report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(input_path: Path, out_dir: Path) -> Dict[str, Any]:
    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records in {input_path}")

    log.info("Distribution stats...")
    dist = compute_distribution_stats(records)

    log.info("Leakage check...")
    try:
        leakage = check_temporal_leakage(records)
    except TemporalLeakageError as e:
        log.error("LEAKAGE: %s", e)
        leakage = {"leakage_free": False, "error": str(e), "n_violations": -1,
                   "total_checked": len(records), "records_ok": 0,
                   "records_missing_dates": 0, "records_without_60d": 0, "violations": []}

    log.info("NaN analysis...")
    nan_stats = compute_nan_stats(records)

    log.info("Correlations...")
    corr = compute_correlation_stats(records)

    log.info("Return distributions...")
    ret_dist = compute_return_distributions(records)

    log.info("Label by asset_type...")
    lba = compute_label_by_asset_type(records)

    log.info("Temporal drift...")
    drift = compute_temporal_drift(records)

    log.info("Low-variance features...")
    low_var = compute_low_variance_features(records)

    stats = {
        "input_file":          str(input_path),
        "distribution":        dist,
        "leakage":             leakage,
        "nan_analysis":        nan_stats,
        "correlations":        corr,
        "return_distributions": ret_dist,
        "label_by_asset_type": lba,
        "temporal_drift":      drift,
        "low_variance":        low_var,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_report_v3.json"
    txt_path  = out_dir / "dataset_report_v3.txt"

    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON: %s", json_path)
    write_txt_report(stats, txt_path)

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description="Validate & QA a v3 dataset JSONL")
    ap.add_argument("--input",   required=True)
    ap.add_argument("--out_dir", default="data/reports")
    args = ap.parse_args()

    stats = analyze(Path(args.input), Path(args.out_dir))
    n = stats["distribution"]["total_samples"]
    lk_ok = stats["leakage"]["leakage_free"]
    nan_ok = not bool(stats["nan_analysis"].get("high_nan_non_macro_features"))

    print(f"\n{'='*55}")
    print(f"Samples : {n:,}")
    print(f"Labels  : {stats['distribution']['by_label']}")
    print(f"Leakage : {'CLEAN ✅' if lk_ok else 'VIOLATIONS ❌'}")
    print(f"NaN     : {'OK ✅' if nan_ok else 'Issues ⚠️'}")
    dr = stats.get("temporal_drift", {})
    print(f"Drift   : {dr.get('high_drift_count', '?')} high-drift features")
    lv = stats.get("low_variance", {})
    print(f"Low-var : {lv.get('n_low_variance', 0)} features flagged")
    print(f"Reports : {args.out_dir}/dataset_report_v3.{{json,txt}}")
    print("=" * 55)

    if not lk_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
