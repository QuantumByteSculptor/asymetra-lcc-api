"""
scripts/ml/validation/time_series_split_v3.py
===============================================
Phase 2 — Expanding-window temporal CV split for v3 datasets.

Core function:
    generate_expanding_splits(df, n_splits=5, embargo_days=20)

Rules:
  - Split anchor = window_end_date (feature window close date)
  - Expanding train set: all records with window_end_date < fold_val_start
  - Embargo gap: records within embargo_days of val_start are dropped from train
    (prevents leakage through autocorrelated features at fold boundaries)
  - Validation set: records with val_start <= window_end_date < val_end
  - Verified: no chronological overlap between train and val

Usage (standalone):
  python scripts/ml/validation/time_series_split_v3.py \\
      --input data/training/train_v3_all.jsonl \\
      --n_splits 5 \\
      --embargo_days 20

Usage (as module):
  from scripts.ml.validation.time_series_split_v3 import generate_expanding_splits
  splits = generate_expanding_splits(df, n_splits=5, embargo_days=20)

No API / prod impact.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("time_series_split_v3")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_jsonl_as_df(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                flat = {
                    "window_end_date":  rec.get("window_end_date"),
                    "label_start_date": rec.get("label_start_date"),
                    "label_end_date":   rec.get("label_end_date"),
                    "label":            rec.get("label"),
                    "target_non_ok":    rec.get("target_non_ok"),
                    "asset_type":       rec.get("features", {}).get("asset_type"),
                    "ticker":           rec.get("features", {}).get("ticker"),
                    "forward_return_20d": rec.get("forward_return_20d"),
                    "source":           rec.get("source"),
                }
                records.append(flat)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)

    # Parse dates
    for col in ("window_end_date", "label_start_date", "label_end_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["window_end_date"]).copy()
    df = df.sort_values("window_end_date").reset_index(drop=True)

    log.info("Loaded %d records, date range %s → %s",
             len(df),
             df["window_end_date"].min().date(),
             df["window_end_date"].max().date())
    return df


# ---------------------------------------------------------------------------
# Expanding-window CV
# ---------------------------------------------------------------------------

FoldType = Tuple[np.ndarray, np.ndarray]  # (train_indices, val_indices)


def generate_expanding_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 20,
    min_train_samples: int = 100,
) -> List[Dict]:
    """
    Generate expanding-window temporal CV splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column `window_end_date` (datetime).
        Assumed to be sorted by window_end_date.
    n_splits : int
        Number of folds.
    embargo_days : int
        Gap between the last training sample's window_end_date and the
        first validation sample's window_end_date. Prevents feature
        autocorrelation leakage at fold boundaries.
    min_train_samples : int
        Minimum number of train samples required for a fold to be valid.

    Returns
    -------
    List of dicts, one per fold:
        {
          "fold": int,
          "train_start": date,
          "train_end": date,              # last train window_end_date
          "embargo_end": date,            # train_end + embargo_days
          "val_start": date,
          "val_end": date,
          "n_train": int,
          "n_val": int,
          "train_indices": np.ndarray,
          "val_indices": np.ndarray,
          "label_dist_train": dict,
          "label_dist_val": dict,
        }
    """
    dates = df["window_end_date"].sort_values().dropna()
    if len(dates) == 0:
        raise ValueError("DataFrame has no valid window_end_date values")

    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days

    if total_days < n_splits * 30:
        raise ValueError(
            f"Date range too narrow ({total_days} days) for {n_splits} splits. "
            f"Need at least {n_splits * 30} days."
        )

    # Divide the timeline into n_splits+1 equal parts
    # Each fold i: train on [min_date, cutoff_i), val on [cutoff_i, cutoff_{i+1})
    # With n_splits=5 we get 5 cutpoints → 5 folds
    step = total_days / (n_splits + 1)
    cutpoints = [
        min_date + timedelta(days=int(round(step * k)))
        for k in range(1, n_splits + 2)
    ]

    splits: List[Dict] = []

    for fold_idx in range(n_splits):
        val_start = cutpoints[fold_idx]
        val_end   = cutpoints[fold_idx + 1]

        embargo_cutoff = val_start - timedelta(days=embargo_days)

        # Train: all records with window_end_date < embargo_cutoff (expanding)
        train_mask = df["window_end_date"] < embargo_cutoff
        train_idx  = np.where(train_mask.to_numpy())[0]

        # Val: records in [val_start, val_end)
        val_mask = (
            (df["window_end_date"] >= val_start) &
            (df["window_end_date"] <  val_end)
        )
        val_idx = np.where(val_mask.to_numpy())[0]

        if len(train_idx) < min_train_samples:
            log.warning(
                "Fold %d: only %d train samples (min=%d) — skipping",
                fold_idx + 1, len(train_idx), min_train_samples,
            )
            continue

        if len(val_idx) == 0:
            log.warning("Fold %d: empty validation set — skipping", fold_idx + 1)
            continue

        # Label distributions
        def _dist(idx: np.ndarray) -> Dict[str, int]:
            if "label" not in df.columns:
                return {}
            counts = df.iloc[idx]["label"].value_counts().to_dict()
            return {str(k): int(v) for k, v in counts.items()}

        train_dates = df.iloc[train_idx]["window_end_date"]
        val_dates   = df.iloc[val_idx]["window_end_date"]

        splits.append({
            "fold":           fold_idx + 1,
            "train_start":    str(train_dates.min().date()),
            "train_end":      str(train_dates.max().date()),
            "embargo_end":    str(embargo_cutoff.date()),
            "val_start":      str(val_dates.min().date()),
            "val_end":        str(val_dates.max().date()),
            "n_train":        int(len(train_idx)),
            "n_val":          int(len(val_idx)),
            "train_indices":  train_idx,
            "val_indices":    val_idx,
            "label_dist_train": _dist(train_idx),
            "label_dist_val":   _dist(val_idx),
        })

    return splits


# ---------------------------------------------------------------------------
# Overlap verification
# ---------------------------------------------------------------------------

def verify_no_overlap(splits: List[Dict], df: pd.DataFrame) -> bool:
    """
    Verify that:
    1. No sample appears in both train and val within any fold.
    2. No val sample has window_end_date <= embargo_end.
    3. Train sets are truly expanding (each fold's train ⊇ previous).

    Returns True if all checks pass; raises ValueError otherwise.
    """
    ok = True

    for fold in splits:
        fi = fold["fold"]
        train_idx = set(fold["train_indices"].tolist())
        val_idx   = set(fold["val_indices"].tolist())

        # Check 1: no index overlap
        overlap = train_idx & val_idx
        if overlap:
            ok = False
            log.error("Fold %d: %d samples in both train and val", fi, len(overlap))

        # Check 2: val samples chronologically after embargo
        embargo_end = pd.Timestamp(fold["embargo_end"])
        val_dates   = df.iloc[sorted(fold["val_indices"])]["window_end_date"]
        violations  = val_dates[val_dates <= embargo_end]
        if not violations.empty:
            ok = False
            log.error(
                "Fold %d: %d val samples inside embargo window (before %s)",
                fi, len(violations), embargo_end.date(),
            )

    # Check 3: expanding property (each fold train ⊇ previous)
    for i in range(1, len(splits)):
        prev_train = set(splits[i - 1]["train_indices"].tolist())
        curr_train = set(splits[i]["train_indices"].tolist())
        if not prev_train.issubset(curr_train):
            ok = False
            log.error(
                "Fold %d train is NOT a superset of fold %d train — "
                "expanding-window property violated",
                splits[i]["fold"], splits[i - 1]["fold"],
            )

    if ok:
        log.info("✅ All overlap / embargo / expanding-window checks passed")
    else:
        raise ValueError("CV split integrity checks FAILED — see log above")

    return ok


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(n: int, total: int, width: int = 20) -> str:
    filled = int(round(n / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_fold_summary(splits: List[Dict], df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\n{'='*72}")
    print(f"EXPANDING-WINDOW CV — {len(splits)} folds")
    print(f"{'='*72}")
    print(f"{'Fold':<5} {'Train start':<12} {'Train end':<12} {'Embargo end':<12} "
          f"{'Val start':<12} {'Val end':<12} {'N train':>8} {'N val':>7}")
    print(f"{'─'*72}")

    for s in splits:
        print(
            f"{s['fold']:<5} {s['train_start']:<12} {s['train_end']:<12} "
            f"{s['embargo_end']:<12} {s['val_start']:<12} {s['val_end']:<12} "
            f"{s['n_train']:>8,} {s['n_val']:>7,}"
        )

    print(f"\n{'Label distribution per fold':}")
    print(f"{'─'*72}")
    for s in splits:
        d_train = s["label_dist_train"]
        d_val   = s["label_dist_val"]
        n_train = s["n_train"]
        n_val   = s["n_val"]
        t_str = "  ".join(
            f"{k}:{v}({100*v/n_train:.0f}%)" if n_train else f"{k}:{v}"
            for k, v in sorted(d_train.items())
        )
        v_str = "  ".join(
            f"{k}:{v}({100*v/n_val:.0f}%)" if n_val else f"{k}:{v}"
            for k, v in sorted(d_val.items())
        )
        print(f"  Fold {s['fold']}  TRAIN [{t_str}]")
        print(f"         VAL   [{v_str}]")

    print(f"\n{'='*72}\n")

    # Asset-type coverage per fold
    if "asset_type" in df.columns:
        print("Asset-type coverage (val fold):")
        print(f"{'─'*50}")
        for s in splits:
            val_df = df.iloc[s["val_indices"]]
            at_counts = val_df["asset_type"].value_counts().to_dict()
            print(f"  Fold {s['fold']}: {at_counts}")
        print()


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

    ap = argparse.ArgumentParser(description="Generate expanding-window CV splits for v3 dataset")
    ap.add_argument("--input",         required=True, help="Path to v3 JSONL file")
    ap.add_argument("--n_splits",      type=int, default=5)
    ap.add_argument("--embargo_days",  type=int, default=20,
                    help="Days gap between train end and val start (default 20)")
    ap.add_argument("--min_train",     type=int, default=100,
                    help="Minimum train samples per fold (default 100)")
    ap.add_argument("--out",           default=None,
                    help="Optional output JSON file for split metadata")
    args = ap.parse_args()

    df = load_jsonl_as_df(Path(args.input))

    splits = generate_expanding_splits(
        df,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
        min_train_samples=args.min_train,
    )

    print_fold_summary(splits, df)

    # Integrity verification
    verify_no_overlap(splits, df)

    # Optional JSON output (without numpy arrays)
    if args.out:
        serializable = [
            {k: v for k, v in s.items() if k not in ("train_indices", "val_indices")}
            for s in splits
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Split metadata written: %s", args.out)

    print(f"\n✅ {len(splits)} valid folds generated. Embargo={args.embargo_days}d. No overlaps.")


if __name__ == "__main__":
    main()
