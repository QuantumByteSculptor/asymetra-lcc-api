"""
scripts/build_v2_split.py
─────────────────────────
Produit un split train/holdout propre à partir de train_v2_all.jsonl.

Split stratifié par (asset_type, label) — 80% train, 20% holdout.
Aucune fuite temporelle : les fenêtres d'un même ticker ne se retrouvent
pas de chaque côté du split (group-aware).

Usage :
    python scripts/build_v2_split.py \
        --input  data/training/train_v2_all.jsonl \
        --train  data/training/train_v2_split.jsonl \
        --holdout data/training/holdout_v2.jsonl \
        --holdout_ratio 0.20 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def make_split(
    records: List[dict],
    holdout_ratio: float,
    seed: int,
) -> tuple[List[dict], List[dict]]:
    """
    Group-aware stratified split.

    Strategy:
    1. Group windows by ticker (prevents leakage between train/holdout).
    2. Within each (asset_type, label) stratum, sample ~holdout_ratio of
       tickers (not windows) into holdout.
    3. All windows of a ticker stay together (no temporal leakage).
    """
    rng = random.Random(seed)

    # Group records by ticker
    ticker_records: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        feats = rec.get("features", rec)
        ticker = str(feats.get("ticker") or feats.get("asset_type", "unknown"))
        ticker_records[ticker].append(rec)

    # For each ticker, pick its dominant label (majority vote)
    def dominant_label(recs: List[dict]) -> str:
        from collections import Counter
        labels = [r.get("label") or r.get("features", {}).get("label_v2", "ok") for r in recs]
        return Counter(labels).most_common(1)[0][0]

    def asset_type(recs: List[dict]) -> str:
        return str(recs[0].get("features", {}).get("asset_type", "global")).strip().lower()

    # Stratify tickers by (asset_type, dominant_label)
    strata: Dict[tuple, List[str]] = defaultdict(list)
    for ticker, recs in ticker_records.items():
        key = (asset_type(recs), dominant_label(recs))
        strata[key].append(ticker)

    holdout_tickers: set = set()
    for key, tickers in strata.items():
        rng.shuffle(tickers)
        n_hold = max(1, round(len(tickers) * holdout_ratio))
        holdout_tickers.update(tickers[:n_hold])

    train_recs, holdout_recs = [], []
    for ticker, recs in ticker_records.items():
        if ticker in holdout_tickers:
            holdout_recs.extend(recs)
        else:
            train_recs.extend(recs)

    # Shuffle both sets
    rng.shuffle(train_recs)
    rng.shuffle(holdout_recs)
    return train_recs, holdout_recs


def write_jsonl(records: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="data/training/train_v2_all.jsonl")
    ap.add_argument("--train",   default="data/training/train_v2_split.jsonl")
    ap.add_argument("--holdout", default="data/training/holdout_v2.jsonl")
    ap.add_argument("--holdout_ratio", type=float, default=0.20)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    from collections import Counter

    print(f"Loading {args.input} …")
    records = load_jsonl(args.input)
    print(f"  Total records: {len(records)}")

    label_dist = Counter(r.get("label", "?") for r in records)
    asset_dist = Counter(r.get("features", {}).get("asset_type", "?") for r in records)
    print(f"  Labels: {dict(label_dist)}")
    print(f"  Assets: {dict(asset_dist)}")

    train_recs, holdout_recs = make_split(records, args.holdout_ratio, args.seed)

    # Report split quality
    t_lab = Counter(r.get("label", "?") for r in train_recs)
    h_lab = Counter(r.get("label", "?") for r in holdout_recs)
    t_at  = Counter(r.get("features", {}).get("asset_type", "?") for r in train_recs)
    h_at  = Counter(r.get("features", {}).get("asset_type", "?") for r in holdout_recs)

    print(f"\nTrain  : {len(train_recs):6d} records  labels={dict(t_lab)}  assets={dict(t_at)}")
    print(f"Holdout: {len(holdout_recs):6d} records  labels={dict(h_lab)}  assets={dict(h_at)}")

    actual_ratio = len(holdout_recs) / len(records)
    print(f"\nActual holdout ratio: {actual_ratio:.3f}  (target={args.holdout_ratio})")

    # Check no ticker overlap
    def tickers_in(recs):
        return {r.get("features", {}).get("ticker", "") for r in recs}

    overlap = tickers_in(train_recs) & tickers_in(holdout_recs)
    if overlap:
        print(f"⚠️  {len(overlap)} tickers in BOTH splits (ticker reuse) — check grouping")
    else:
        print("✅ Zero ticker overlap between train and holdout")

    write_jsonl(train_recs, args.train)
    write_jsonl(holdout_recs, args.holdout)
    print(f"\n✅ Saved train  → {args.train}")
    print(f"✅ Saved holdout → {args.holdout}")


if __name__ == "__main__":
    main()
