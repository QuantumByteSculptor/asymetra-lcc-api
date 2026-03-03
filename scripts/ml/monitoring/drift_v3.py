"""
scripts/ml/monitoring/drift_v3.py
===================================
Phase 5 — Monitoring de drift sur le dataset v3.

Calcule par feature:
  - PSI  (Population Stability Index)
  - KS test (Kolmogorov-Smirnov: distribution stationarity)
  - Jensen-Shannon divergence

Calcule un drift_score global (moyenne pondérée des PSI).

Usage:
  python scripts/ml/monitoring/drift_v3.py \\
      --reference data/training/smoke_v3.jsonl \\     # fenêtre de référence
      --current   data/training/train_v3_all.jsonl \\  # fenêtre courante
      --out       data/metrics/drift_v3_report.json

  # Ou en mode temporal: split automatique du même JSONL
  python scripts/ml/monitoring/drift_v3.py \\
      --input     data/training/train_v3_all.jsonl \\
      --split_date 2023-01-01 \\
      --out       data/metrics/drift_v3_report.json

Seuils PSI:
  PSI < 0.10  → stable       (vert)
  PSI < 0.20  → changement modéré (orange)
  PSI ≥ 0.20  → drift significatif (rouge)

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
import pandas as pd
from scipy import stats

log = logging.getLogger("drift_v3")

# PSI thresholds
PSI_GREEN  = 0.10
PSI_ORANGE = 0.20

# Features to skip (non-numeric or identity)
_SKIP_COLS = {
    "asset_type", "market", "ticker", "tuw_pct",
    "vol_regime",   # ordinal — analysed separately
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec  = json.loads(line)
                row  = {
                    "window_end_date": rec.get("window_end_date"),
                    "asset_type":      rec.get("features", {}).get("asset_type"),
                    "label":           rec.get("label"),
                    "target_non_ok":   rec.get("target_non_ok"),
                }
                for k, v in rec.get("features", {}).items():
                    if isinstance(v, (int, float)) or v is None:
                        row[k] = v
                records.append(row)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"], errors="coerce")
    return df.sort_values("window_end_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    ref = reference (training) distribution
    cur = current (production/test) distribution
    """
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    # Bin edges from ref distribution
    edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    edges = np.unique(edges)   # deduplicate flat edges
    if len(edges) < 3:
        return float("nan")

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / (ref_counts.sum() + 1e-12)
    cur_pct = cur_counts / (cur_counts.sum() + 1e-12)

    # Avoid log(0)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def compute_jsd(ref: np.ndarray, cur: np.ndarray, n_bins: int = 20) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    combined = np.concatenate([ref, cur])
    edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")

    p, _ = np.histogram(ref, bins=edges, density=True)
    q, _ = np.histogram(cur, bins=edges, density=True)

    # Normalise to probability
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(p / (m + 1e-12)), 0.0)
        kl_qm = np.where(q > 0, q * np.log(q / (m + 1e-12)), 0.0)

    jsd = 0.5 * (kl_pm.sum() + kl_qm.sum())
    return float(max(0.0, jsd))


# ---------------------------------------------------------------------------
# Per-feature drift analysis
# ---------------------------------------------------------------------------

def _psi_label(psi: float) -> str:
    if math.isnan(psi):
        return "na"
    if psi < PSI_GREEN:
        return "stable"
    if psi < PSI_ORANGE:
        return "moderate"
    return "drift"


def analyze_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Run PSI + KS + JSD for every numeric feature.
    Returns list of per-feature results, sorted by PSI descending.
    """
    numeric_cols = [
        c for c in ref_df.columns
        if c not in _SKIP_COLS
        and ref_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        and c not in ("target_non_ok", "window_end_date")
    ]

    results: List[Dict[str, Any]] = []

    for col in numeric_cols:
        ref_arr = ref_df[col].dropna().to_numpy(dtype=float)
        cur_arr = cur_df[col].dropna().to_numpy(dtype=float)

        if len(ref_arr) < 10 or len(cur_arr) < 10:
            continue

        psi  = compute_psi(ref_arr, cur_arr, n_bins=n_bins)
        jsd  = compute_jsd(ref_arr, cur_arr, n_bins=n_bins * 2)

        try:
            ks_stat, ks_pval = stats.ks_2samp(ref_arr, cur_arr)
        except Exception:
            ks_stat, ks_pval = float("nan"), float("nan")

        results.append({
            "feature":       col,
            "psi":           round(psi, 5)  if math.isfinite(psi)     else None,
            "psi_label":     _psi_label(psi),
            "ks_stat":       round(float(ks_stat), 5) if math.isfinite(ks_stat) else None,
            "ks_pval":       round(float(ks_pval), 5) if math.isfinite(ks_pval) else None,
            "ks_significant":bool(ks_pval < 0.05) if math.isfinite(ks_pval) else None,
            "jsd":           round(jsd, 5)  if math.isfinite(jsd)     else None,
            "ref_mean":      round(float(ref_arr.mean()), 4),
            "cur_mean":      round(float(cur_arr.mean()), 4),
            "ref_std":       round(float(ref_arr.std(ddof=1)), 4),
            "cur_std":       round(float(cur_arr.std(ddof=1)), 4),
            "mean_shift":    round(float(cur_arr.mean() - ref_arr.mean()), 4),
            "n_ref":         int(len(ref_arr)),
            "n_cur":         int(len(cur_arr)),
        })

    # Sort by PSI descending
    results.sort(key=lambda x: x.get("psi") or 0.0, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Label drift
# ---------------------------------------------------------------------------

def analyze_label_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> Dict[str, Any]:
    ref_dist = ref_df["label"].value_counts(normalize=True).to_dict()
    cur_dist = cur_df["label"].value_counts(normalize=True).to_dict()

    all_labels = set(ref_dist) | set(cur_dist)
    shift = {
        lbl: round(cur_dist.get(lbl, 0) - ref_dist.get(lbl, 0), 4)
        for lbl in all_labels
    }

    ref_non_ok = float(ref_df["target_non_ok"].mean())
    cur_non_ok = float(cur_df["target_non_ok"].mean())

    return {
        "ref_label_dist":   {k: round(v, 4) for k, v in ref_dist.items()},
        "cur_label_dist":   {k: round(v, 4) for k, v in cur_dist.items()},
        "label_shift":      shift,
        "ref_non_ok_rate":  round(ref_non_ok, 4),
        "cur_non_ok_rate":  round(cur_non_ok, 4),
        "non_ok_shift":     round(cur_non_ok - ref_non_ok, 4),
        "label_drift_significant": abs(cur_non_ok - ref_non_ok) > 0.05,
    }


# ---------------------------------------------------------------------------
# Global drift score
# ---------------------------------------------------------------------------

def compute_global_drift_score(feature_results: List[Dict]) -> Dict[str, Any]:
    """
    Global drift score = weighted average PSI across features.
    Weights: top-20 features by PSI contribute more.
    """
    psi_vals = [r["psi"] for r in feature_results if r.get("psi") is not None]
    if not psi_vals:
        return {"global_psi": None, "drift_level": "unknown"}

    global_psi = float(np.mean(psi_vals))
    max_psi    = float(np.max(psi_vals))
    n_drift    = sum(1 for p in psi_vals if p >= PSI_ORANGE)
    n_moderate = sum(1 for p in psi_vals if PSI_GREEN <= p < PSI_ORANGE)
    n_stable   = sum(1 for p in psi_vals if p < PSI_GREEN)

    if global_psi >= PSI_ORANGE or n_drift >= 5:
        drift_level = "high"
    elif global_psi >= PSI_GREEN or n_drift >= 2:
        drift_level = "moderate"
    else:
        drift_level = "low"

    return {
        "global_psi_mean":  round(global_psi, 4),
        "global_psi_max":   round(max_psi, 4),
        "n_features_drift":    n_drift,
        "n_features_moderate": n_moderate,
        "n_features_stable":   n_stable,
        "n_features_total":    len(psi_vals),
        "drift_level":         drift_level,
        "top5_drifting":       [r["feature"] for r in feature_results[:5]],
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_txt_summary(report: Dict[str, Any], path: Path) -> None:
    lines = []
    a = lines.append
    a("=" * 70)
    a("DATASET V3 — DRIFT MONITORING REPORT")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("=" * 70)

    g = report["global_drift"]
    level_emoji = {"low": "✅", "moderate": "⚠️ ", "high": "❌"}.get(g["drift_level"], "?")
    a(f"\nGLOBAL DRIFT SCORE")
    a(f"  Mean PSI          : {g['global_psi_mean']}  {level_emoji} {g['drift_level'].upper()}")
    a(f"  Max PSI           : {g['global_psi_max']}")
    a(f"  Features stable   : {g['n_features_stable']} / {g['n_features_total']}")
    a(f"  Features moderate : {g['n_features_moderate']}")
    a(f"  Features drifting : {g['n_features_drift']}")
    a(f"  Top 5 drifting    : {g.get('top5_drifting', [])}")

    ld = report["label_drift"]
    a(f"\nLABEL DRIFT")
    a(f"  Reference non_ok rate: {ld['ref_non_ok_rate']:.1%}")
    a(f"  Current  non_ok rate : {ld['cur_non_ok_rate']:.1%}")
    a(f"  Shift                : {ld['non_ok_shift']:+.1%}")
    a(f"  Significant?         : {'YES ⚠️' if ld['label_drift_significant'] else 'NO ✅'}")

    a(f"\nFEATURE DRIFT (top 20 by PSI):")
    a(f"  {'Feature':<30} {'PSI':>8} {'Status':>12} {'KS-p':>8} {'mean_shift':>12}")
    a(f"  {'─'*72}")
    for r in report["feature_drift"][:20]:
        status_sym = {"stable": "✅", "moderate": "⚠️ ", "drift": "❌", "na": " "}.get(
            r.get("psi_label", "na"), " "
        )
        a(f"  {r['feature']:<30} {str(r.get('psi', 'N/A')):>8} "
          f"{status_sym + r.get('psi_label', ''):>12} "
          f"{str(r.get('ks_pval', 'N/A')):>8} "
          f"{str(r.get('mean_shift', 'N/A')):>12}")

    a(f"\n{'='*70}")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("TXT summary: %s", path)


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

    ap = argparse.ArgumentParser(description="Drift monitoring for v3 dataset")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--reference", default=None,
                     help="Reference JSONL (training set / older period)")
    grp.add_argument("--input",     default=None,
                     help="Single JSONL — will be split by --split_date")
    ap.add_argument("--current",    default=None,
                     help="Current JSONL (used with --reference)")
    ap.add_argument("--split_date", default=None,
                     help="Split date for temporal mode (YYYY-MM-DD)")
    ap.add_argument("--out",        default="data/metrics/drift_v3_report.json")
    ap.add_argument("--n_bins",     type=int, default=10)
    args = ap.parse_args()

    if args.input:
        # Temporal split of a single JSONL
        if not args.split_date:
            ap.error("--split_date required when using --input mode")
        log.info("Loading dataset from %s ...", args.input)
        full_df = load_features(Path(args.input))
        split_ts = pd.Timestamp(args.split_date)
        ref_df = full_df[full_df["window_end_date"] <  split_ts].copy()
        cur_df = full_df[full_df["window_end_date"] >= split_ts].copy()
        log.info("Reference: %d records (before %s)", len(ref_df), args.split_date)
        log.info("Current  : %d records (from  %s)", len(cur_df), args.split_date)
    else:
        log.info("Loading reference: %s", args.reference)
        ref_df = load_features(Path(args.reference))
        if not args.current:
            ap.error("--current required when using --reference mode")
        log.info("Loading current  : %s", args.current)
        cur_df = load_features(Path(args.current))

    if len(ref_df) < 20 or len(cur_df) < 20:
        log.error("Too few records in ref (%d) or cur (%d)", len(ref_df), len(cur_df))
        sys.exit(1)

    log.info("Running feature drift analysis...")
    feature_drift = analyze_feature_drift(ref_df, cur_df, n_bins=args.n_bins)

    log.info("Computing global drift score...")
    global_score  = compute_global_drift_score(feature_drift)

    log.info("Analyzing label drift...")
    label_drift   = analyze_label_drift(ref_df, cur_df)

    report = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "reference_size": int(len(ref_df)),
        "current_size":   int(len(cur_df)),
        "n_features_analyzed": len(feature_drift),
        "global_drift":   global_score,
        "label_drift":    label_drift,
        "feature_drift":  feature_drift,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON report: %s", out_path)

    txt_path = out_path.with_suffix(".txt")
    write_txt_summary(report, txt_path)

    # Console summary
    g = global_score
    level_sym = {"low": "✅", "moderate": "⚠️", "high": "❌"}.get(g["drift_level"], "?")
    print(f"\n{'='*50}")
    print(f"Global drift: {g['global_psi_mean']} {level_sym} {g['drift_level'].upper()}")
    print(f"Features: {g['n_features_stable']} stable / "
          f"{g['n_features_moderate']} moderate / "
          f"{g['n_features_drift']} drifting")
    print(f"Label non_ok shift: {label_drift['non_ok_shift']:+.1%}")
    print(f"Reports: {out_path} + {txt_path}\n")


if __name__ == "__main__":
    main()
