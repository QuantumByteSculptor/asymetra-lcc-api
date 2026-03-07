"""
ml/credibility_v4_2/build_folds.py
────────────────────────────────────
Génère les splits expanding-window (5 folds temporels) avec purge + embargo.

Convention:
  - Chaque record dans dataset_raw.jsonl doit avoir features.window_end_date (YYYY-MM-DD).
  - Expanding: fold k s'entraîne sur [T0, T_k), valide sur [T_k + purge + embargo, T_{k+1}).
  - Purge  = 20 business days (labels à horizon 20j → fin du train peut regarder dans le val).
  - Embargo = 5 business days (buffer anti-leakage par corrélation sérielle).

Outputs dans out_dir/:
  - splits.json          (bornes + counts par fold)
  - fold_boundaries.csv  (tableau lisible)

Usage:
    python ml/credibility_v4_2/build_folds.py \
        --run_id v42_... \
        --out_dir artifacts/credibility_v4_2/v42_... \
        --purge_days 20 \
        --embargo_days 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Fold boundary definitions
# ═══════════════════════════════════════════════════════════════════════════════

# 5 fold boundaries: date at which each VALIDATION period begins (before purge/embargo).
# Training always starts from the global T0.
# These are calendar dates; actual record assignment uses window_end_date.
_DEFAULT_FOLD_BOUNDARIES = [
    # (fold_id, val_boundary_start, val_boundary_end)
    # val_boundary_start = start of validation zone (before purge/embargo offsets)
    (1, "2014-01-01", "2015-12-31"),
    (2, "2016-01-01", "2017-12-31"),
    (3, "2018-01-01", "2019-12-31"),
    (4, "2020-01-01", "2021-12-31"),
    (5, "2022-01-01", "2025-12-31"),
]
# T0: global dataset start (inclusive for training)
_T0 = "2010-01-01"


def add_business_days(d: date, n: int) -> date:
    """Add n business days to date d (pandas BDay)."""
    ts = pd.Timestamp(d) + pd.offsets.BDay(n)
    return ts.date()


def compute_fold_windows(
    boundaries: List[Tuple[int, str, str]],
    purge_days: int,
    embargo_days: int,
    t0: str,
) -> List[Dict[str, Any]]:
    """
    For each fold boundary (fold_id, raw_val_start, raw_val_end):

      train_start = T0
      train_end   = raw_val_start - 1 day        (last day of training)
      purge_start = raw_val_start                 (first day in purge zone)
      purge_end   = raw_val_start + purge_days bdays - 1
      embargo_start = purge_end + 1 bday
      embargo_end   = purge_end + embargo_days bdays
      val_start   = embargo_end + 1 bday          (first usable validation day)
      val_end     = raw_val_end

    Returns list of fold dicts (no counts yet).
    """
    folds = []
    for fold_id, raw_val_start_s, raw_val_end_s in boundaries:
        raw_val_start = pd.Timestamp(raw_val_start_s).date()
        raw_val_end = pd.Timestamp(raw_val_end_s).date()

        train_start = pd.Timestamp(t0).date()
        # Last training day = day before the fold boundary
        train_end = (pd.Timestamp(raw_val_start_s) - pd.offsets.BDay(1)).date()

        # Purge zone: last purge_days trading days of the training period
        # Records in this zone have labels that may overlap with validation window
        purge_start = (pd.Timestamp(train_end) - pd.offsets.BDay(purge_days - 1)).date()
        purge_end = train_end  # inclusive

        # Embargo zone: first embargo_days trading days after the boundary
        embargo_start = raw_val_start
        embargo_end = add_business_days(embargo_start, embargo_days - 1)

        # Actual validation start: day after embargo
        val_start = add_business_days(embargo_end, 1)
        val_end = raw_val_end

        folds.append({
            "fold_id": fold_id,
            "run_id": None,   # filled later
            "train_start": str(train_start),
            "train_end": str(train_end),
            "purge_start": str(purge_start),
            "purge_end": str(purge_end),
            "embargo_start": str(embargo_start),
            "embargo_end": str(embargo_end),
            "val_start": str(val_start),
            "val_end": str(val_end),
            "purge_days": purge_days,
            "embargo_days": embargo_days,
            # Counts filled after assignment
            "n_train": None,
            "n_val": None,
            "n_purge": None,
            "n_embargo": None,
            "label_dist_train": None,
            "label_dist_val": None,
            "asset_dist_val": None,
        })

    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# Record assignment
# ═══════════════════════════════════════════════════════════════════════════════

def assign_folds(
    records: List[Dict[str, Any]],
    folds: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assign each record to train or val for each fold.
    A record is assigned to fold k's TRAIN if:
        window_end_date in [train_start, train_end] AND
        window_end_date NOT in [purge_start, purge_end]   ← purged records excluded
    A record is assigned to fold k's VAL if:
        window_end_date in [val_start, val_end]

    Records in the purge or embargo zones are excluded from both.

    Returns: (folds_with_counts, records_with_fold_assignment)
    """
    # Build date lookup
    def to_date(s: str) -> date:
        return pd.Timestamp(s).date()

    fold_bounds = [
        {
            "fold_id": f["fold_id"],
            "train_start": to_date(f["train_start"]),
            "train_end": to_date(f["train_end"]),
            "purge_start": to_date(f["purge_start"]),
            "purge_end": to_date(f["purge_end"]),
            "embargo_start": to_date(f["embargo_start"]),
            "embargo_end": to_date(f["embargo_end"]),
            "val_start": to_date(f["val_start"]),
            "val_end": to_date(f["val_end"]),
        }
        for f in folds
    ]

    # Accumulators
    train_labels: List[List[str]] = [[] for _ in folds]
    val_labels: List[List[str]] = [[] for _ in folds]
    val_assets: List[List[str]] = [[] for _ in folds]
    purge_counts: List[int] = [0] * len(folds)
    embargo_counts: List[int] = [0] * len(folds)

    enriched: List[Dict[str, Any]] = []

    for rec in records:
        feats = rec.get("features", rec)
        wed_str = feats.get("window_end_date")
        if not wed_str:
            continue
        try:
            wed = pd.Timestamp(wed_str).date()
        except Exception:
            continue

        label = rec.get("label", feats.get("label_v2", "ok"))
        asset_type = str(feats.get("asset_type", "unknown"))

        rec_out = dict(rec)
        fold_assignments: Dict[str, str] = {}

        for k, fb in enumerate(fold_bounds):
            fid = fb["fold_id"]
            if fb["purge_start"] <= wed <= fb["purge_end"]:
                fold_assignments[f"fold{fid}"] = "purge"
                purge_counts[k] += 1
            elif fb["embargo_start"] <= wed <= fb["embargo_end"]:
                fold_assignments[f"fold{fid}"] = "embargo"
                embargo_counts[k] += 1
            elif fb["train_start"] <= wed <= fb["train_end"]:
                fold_assignments[f"fold{fid}"] = "train"
                train_labels[k].append(label)
            elif fb["val_start"] <= wed <= fb["val_end"]:
                fold_assignments[f"fold{fid}"] = "val"
                val_labels[k].append(label)
                val_assets[k].append(asset_type)
            else:
                fold_assignments[f"fold{fid}"] = "out"

        rec_out["fold_assignments"] = fold_assignments
        enriched.append(rec_out)

    # Fill counts into folds
    for k, f in enumerate(folds):
        f["n_train"] = len(train_labels[k])
        f["n_val"] = len(val_labels[k])
        f["n_purge"] = purge_counts[k]
        f["n_embargo"] = embargo_counts[k]
        f["label_dist_train"] = dict(Counter(train_labels[k]))
        f["label_dist_val"] = dict(Counter(val_labels[k]))
        f["asset_dist_val"] = dict(Counter(val_assets[k]))

    return folds, enriched


# ═══════════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════════

def write_splits_json(folds: List[Dict[str, Any]], path: Path, run_id: str) -> None:
    out = {
        "run_id": run_id,
        "schema_version": "v4.2",
        "purge_days": folds[0]["purge_days"] if folds else 20,
        "embargo_days": folds[0]["embargo_days"] if folds else 5,
        "n_folds": len(folds),
        "folds": folds,
    }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ {path}")


def write_fold_boundaries_csv(folds: List[Dict[str, Any]], path: Path) -> None:
    rows = []
    for f in folds:
        rows.append({
            "fold_id": f["fold_id"],
            "train_start": f["train_start"],
            "train_end": f["train_end"],
            "purge_start": f["purge_start"],
            "purge_end": f["purge_end"],
            "embargo_start": f["embargo_start"],
            "embargo_end": f["embargo_end"],
            "val_start": f["val_start"],
            "val_end": f["val_end"],
            "n_train": f["n_train"],
            "n_val": f["n_val"],
            "n_purge": f["n_purge"],
            "n_embargo": f["n_embargo"],
        })

    import csv
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Build expanding-window folds for Credibility v4.2")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--purge_days", type=int, default=20)
    ap.add_argument("--embargo_days", type=int, default=5)
    ap.add_argument("--t0", default=_T0, help="Global train start date")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dataset_path = out_dir / "dataset_raw.jsonl"
    splits_path = out_dir / "splits.json"
    csv_path = out_dir / "fold_boundaries.csv"

    if not dataset_path.exists():
        sys.exit(f"dataset_raw.jsonl not found: {dataset_path}")

    print(f"=== build_folds.py — run_id={args.run_id} ===")
    print(f"  purge={args.purge_days}bd  embargo={args.embargo_days}bd")

    # Load records
    records: List[Dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from dataset_raw.jsonl")

    if not records:
        sys.exit("ERROR: dataset_raw.jsonl is empty — run build_dataset.py first")

    # Check window_end_date presence
    missing_dates = sum(
        1 for r in records
        if not r.get("features", {}).get("window_end_date")
    )
    if missing_dates > 0:
        print(f"  ⚠  {missing_dates} records missing window_end_date — they will be skipped")

    # Build fold window definitions
    folds = compute_fold_windows(
        _DEFAULT_FOLD_BOUNDARIES,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        t0=args.t0,
    )
    for f in folds:
        f["run_id"] = args.run_id

    # Assign records
    folds, _enriched = assign_folds(records, folds)

    # Print summary
    print("\n  Fold summary:")
    print(f"  {'Fold':<6} {'TrainEnd':<12} {'ValStart':<12} {'ValEnd':<12} {'N_train':>8} {'N_val':>7}")
    for f in folds:
        print(
            f"  fold{f['fold_id']:<2} {f['train_end']:<12} {f['val_start']:<12} "
            f"{f['val_end']:<12} {f['n_train']:>8,} {f['n_val']:>7,}"
        )

    # Sanity: warn if any val fold has suspiciously few records
    for f in folds:
        if (f["n_val"] or 0) < 100:
            print(f"  ⚠  fold{f['fold_id']} val N={f['n_val']} is very low!")

    # Write outputs
    print()
    write_splits_json(folds, splits_path, args.run_id)
    write_fold_boundaries_csv(folds, csv_path)

    print("\n=== build_folds DONE ===")


if __name__ == "__main__":
    main()
