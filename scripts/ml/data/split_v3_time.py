"""
scripts/ml/data/split_v3_time.py
==================================
Phase 2 — Expanding-window temporal CV split for v3 datasets.

Key guarantees (finance-grade):
  - Split anchor = window_end_date
  - Expanding train: all records with window_end_date < (val_start - purge_days)
  - Purge gap: [val_start - purge_days, val_start) is excluded from train
    (removes samples whose features overlap the validation period)
  - Embargo gap (optional, applied after val_end): records within embargo_days
    of val_end are excluded from the NEXT fold's train start
  - No chronological overlap between train and val in any fold
  - Expanding property: each fold's train ⊇ previous fold's train

Outputs per fold:
  data/training/v3/fold_{k}/train.jsonl
  data/training/v3/fold_{k}/val.jsonl
Plus a manifest:
  data/training/v3/splits_manifest.json

Usage:
    python scripts/ml/data/split_v3_time.py \\
        --in  data/training/train_v3_all.jsonl \\
        --out_dir data/training/v3 \\
        --folds 5 \\
        --purge_days 20 \\
        --embargo_days 5

No API / prod impact. Stdlib + numpy + pandas only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("split_v3_time")


# ---------------------------------------------------------------------------
# Loader — streaming into a flat list (records are small, 54k × ~2 KB ≈ 100 MB)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a v3 JSONL file.
    Returns (records_list, raw_lines_list) for efficient JSONL rewrite.
    """
    records: List[Dict[str, Any]] = []
    raw_lines: List[str] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                rec = json.loads(stripped)
                wed = rec.get("window_end_date")
                if wed is None:
                    bad += 1
                    continue
                records.append(rec)
                raw_lines.append(stripped)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        log.warning("Skipped %d lines (bad JSON or missing window_end_date)", bad)
    log.info("Loaded %d records from %s", len(records), path)
    return records, raw_lines


def _parse_date(s: Any) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    try:
        return pd.Timestamp(str(s)[:10])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Expanding-window splits with purge + embargo
# ---------------------------------------------------------------------------

def generate_splits(
    records: List[Dict[str, Any]],
    n_folds: int = 5,
    purge_days: int = 20,
    embargo_days: int = 5,
    min_train_samples: int = 200,
) -> List[Dict[str, Any]]:
    """
    Generate expanding-window CV splits.

    Purge / embargo model:
      - purge: drop train records whose window_end_date is in
               [val_start - purge_days, val_start)
        (their feature windows might partially overlap the val period)
      - embargo: val records start after embargo_days past the last train cutoff
        (here implemented as a clean gap, not an additional hold-out from train)

    Returns list of fold dicts (without raw lines, indexed by record index).
    """
    # Build date index
    dates = []
    for i, rec in enumerate(records):
        d = _parse_date(rec.get("window_end_date"))
        dates.append((i, d))

    # Sort by date
    dates_sorted = sorted(
        [(i, d) for i, d in dates if d is not None],
        key=lambda x: x[1],
    )

    if not dates_sorted:
        raise ValueError("No records with valid window_end_date found")

    min_date = dates_sorted[0][1]
    max_date = dates_sorted[-1][1]
    total_days = (max_date - min_date).days

    if total_days < n_folds * 60:
        raise ValueError(
            f"Date range too narrow ({total_days} days) for {n_folds} folds. "
            f"Need at least {n_folds * 60} days."
        )

    # Divide timeline into (n_folds + 1) equal chunks
    # Each fold i: train on [min_date, cutoff_i - purge_days), val on [cutoff_i, cutoff_{i+1})
    step = total_days / (n_folds + 1)
    cutpoints = [
        min_date + timedelta(days=int(round(step * k)))
        for k in range(1, n_folds + 2)
    ]

    date_arr = np.array([d for _, d in dates_sorted], dtype="datetime64[ns]")
    idx_arr  = np.array([i for i, _ in dates_sorted], dtype=np.int64)

    splits: List[Dict[str, Any]] = []

    for fold_k in range(n_folds):
        val_start = cutpoints[fold_k]
        val_end   = cutpoints[fold_k + 1]

        # Train cutoff: records strictly before (val_start - purge_days)
        train_cutoff = val_start - timedelta(days=purge_days)

        train_mask = date_arr < np.datetime64(train_cutoff, "ns")
        val_mask   = (
            (date_arr >= np.datetime64(val_start, "ns")) &
            (date_arr <  np.datetime64(val_end, "ns"))
        )

        train_global_idx = idx_arr[train_mask].tolist()
        val_global_idx   = idx_arr[val_mask].tolist()

        if len(train_global_idx) < min_train_samples:
            log.warning(
                "Fold %d: only %d train samples (min=%d) — skipping",
                fold_k + 1, len(train_global_idx), min_train_samples,
            )
            continue

        if len(val_global_idx) == 0:
            log.warning("Fold %d: empty val set — skipping", fold_k + 1)
            continue

        # Label distribution helpers
        def _label_dist(indices: List[int]) -> Dict[str, int]:
            c: Counter = Counter()
            for idx in indices:
                c[records[idx].get("label", "missing")] += 1
            return dict(c.most_common())

        def _non_ok_rate(indices: List[int]) -> float:
            if not indices:
                return 0.0
            return sum(1 for idx in indices if records[idx].get("target_non_ok") == 1) / len(indices)

        train_dates = [records[i]["window_end_date"] for i in train_global_idx]
        val_dates   = [records[i]["window_end_date"] for i in val_global_idx]

        splits.append({
            "fold":             fold_k + 1,
            "train_start":      min(train_dates),
            "train_end":        max(train_dates),
            "purge_days":       purge_days,
            "embargo_days":     embargo_days,
            "train_cutoff":     str(train_cutoff.date()),
            "val_start":        str(val_start.date()),
            "val_end":          str(val_end.date()),
            "n_train":          len(train_global_idx),
            "n_val":            len(val_global_idx),
            "label_dist_train": _label_dist(train_global_idx),
            "label_dist_val":   _label_dist(val_global_idx),
            "non_ok_rate_train": round(_non_ok_rate(train_global_idx), 4),
            "non_ok_rate_val":   round(_non_ok_rate(val_global_idx), 4),
            "train_indices":    train_global_idx,  # removed before JSON manifest
            "val_indices":      val_global_idx,
        })

    return splits


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_splits(splits: List[Dict], records: List[Dict]) -> None:
    """
    Assert:
    1. No index overlap between train and val within any fold.
    2. All val dates > train_cutoff (purge enforced).
    3. Expanding-window property: each fold's train ⊇ previous.
    Raises ValueError on violation.
    """
    prev_train_set: set = set()

    for fold in splits:
        fi = fold["fold"]
        t_set = set(fold["train_indices"])
        v_set = set(fold["val_indices"])

        overlap = t_set & v_set
        if overlap:
            raise ValueError(f"Fold {fi}: {len(overlap)} indices in both train and val")

        # Purge check: no val record's date <= train_cutoff
        tc = pd.Timestamp(fold["train_cutoff"])
        for idx in fold["val_indices"]:
            d = _parse_date(records[idx].get("window_end_date"))
            if d is not None and d <= tc:
                raise ValueError(
                    f"Fold {fi}: val record idx={idx} date={d.date()} "
                    f"<= train_cutoff={tc.date()} — purge violated"
                )

        # Expanding property
        if prev_train_set and not prev_train_set.issubset(t_set):
            raise ValueError(
                f"Fold {fi}: train set is NOT a superset of fold {fi - 1} train "
                "— expanding-window property violated"
            )

        prev_train_set = t_set

    log.info("✅ Leakage / purge / expanding-window checks passed for all %d folds", len(splits))


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------

def write_fold_jsonl(
    splits: List[Dict],
    raw_lines: List[str],
    out_dir: Path,
) -> None:
    for fold in splits:
        fold_dir = out_dir / f"fold_{fold['fold']}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_path = fold_dir / "train.jsonl"
        val_path   = fold_dir / "val.jsonl"

        with train_path.open("w", encoding="utf-8") as f:
            for idx in fold["train_indices"]:
                f.write(raw_lines[idx] + "\n")

        with val_path.open("w", encoding="utf-8") as f:
            for idx in fold["val_indices"]:
                f.write(raw_lines[idx] + "\n")

        log.info(
            "Fold %d written: train=%d → %s | val=%d → %s",
            fold["fold"],
            fold["n_train"], train_path,
            fold["n_val"],   val_path,
        )


def write_manifest(
    splits: List[Dict],
    out_dir: Path,
    source_path: Path,
    n_folds: int,
    purge_days: int,
    embargo_days: int,
) -> Path:
    manifest_splits = []
    for fold in splits:
        s = {k: v for k, v in fold.items() if k not in ("train_indices", "val_indices")}
        s["train_jsonl"] = str(out_dir / f"fold_{fold['fold']}" / "train.jsonl")
        s["val_jsonl"]   = str(out_dir / f"fold_{fold['fold']}" / "val.jsonl")
        manifest_splits.append(s)

    manifest = {
        "generated_at":  datetime.now().isoformat(),
        "source_file":   str(source_path),
        "n_folds":       n_folds,
        "purge_days":    purge_days,
        "embargo_days":  embargo_days,
        "n_valid_folds": len(splits),
        "splits":        manifest_splits,
    }

    manifest_path = out_dir / "splits_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Manifest written: %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(splits: List[Dict]) -> None:
    print(f"\n{'='*76}")
    print(f"  EXPANDING-WINDOW CV — {len(splits)} folds")
    print(f"{'='*76}")
    print(f"  {'Fold':<5} {'Train start':<12} {'Train end':<12} {'Cutoff':<12} "
          f"{'Val start':<12} {'Val end':<12} {'N train':>8} {'N val':>7}")
    print(f"  {'─'*75}")
    for s in splits:
        print(
            f"  {s['fold']:<5} {s['train_start']:<12} {s['train_end']:<12} "
            f"{s['train_cutoff']:<12} {s['val_start']:<12} {s['val_end']:<12} "
            f"{s['n_train']:>8,} {s['n_val']:>7,}"
        )
    print(f"\n  Label distribution per fold (val):")
    for s in splits:
        d = s["label_dist_val"]
        nv = s["n_val"]
        d_str = "  ".join(f"{k}:{v}({100*v/nv:.0f}%)" for k, v in sorted(d.items()))
        print(f"    Fold {s['fold']} → {d_str}  non_ok_rate={s['non_ok_rate_val']:.1%}")
    print(f"{'='*76}\n")


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
        description="Generate expanding-window temporal CV splits for v3 dataset"
    )
    ap.add_argument("--in",           dest="input",    required=True,
                    help="Path to v3 JSONL dataset")
    ap.add_argument("--out_dir",      dest="out_dir",  default="data/training/v3",
                    help="Output directory (default: data/training/v3)")
    ap.add_argument("--folds",        type=int, default=5,
                    help="Number of CV folds (default: 5)")
    ap.add_argument("--purge_days",   type=int, default=20,
                    help="Days excluded from train before val_start (default: 20)")
    ap.add_argument("--embargo_days", type=int, default=5,
                    help="Post-val embargo stored in manifest (default: 5)")
    ap.add_argument("--min_train",    type=int, default=200,
                    help="Minimum train samples per fold (default: 200)")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    records, raw_lines = load_jsonl(input_path)

    log.info("Generating %d splits (purge=%dd, embargo=%dd)...",
             args.folds, args.purge_days, args.embargo_days)
    splits = generate_splits(
        records,
        n_folds=args.folds,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_train_samples=args.min_train,
    )

    if not splits:
        log.error("No valid folds generated — check date range and parameters")
        sys.exit(1)

    log.info("Verifying split integrity...")
    verify_splits(splits, records)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_fold_jsonl(splits, raw_lines, out_dir)
    manifest_path = write_manifest(
        splits, out_dir, input_path,
        args.folds, args.purge_days, args.embargo_days,
    )

    print_summary(splits)
    print(f"✅ {len(splits)} folds written to {out_dir}")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
