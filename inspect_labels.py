#!/usr/bin/env python3
# inspect_labels.py
import sys, json
from collections import Counter, defaultdict
from pathlib import Path

def inspect(path, n_sample=5):
    p = Path(path)
    if not p.exists():
        print("File missing:", path); return

    total = 0
    key_counts = Counter()
    numeric_counts = Counter()
    sample_vals = defaultdict(list)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print("JSON error line", total, e); continue
            feats = obj.get("features", obj)
            for k in ("z_if","z_lof","z_gap_if_lof","max_dd","max_drawdown"):
                if k in feats:
                    key_counts[k] += 1
                    v = feats.get(k)
                    try:
                        # treat pandas Series etc. as non-numeric
                        isnum = isinstance(v, (int,float)) and (not (v != v))  # not NaN
                    except Exception:
                        isnum = False
                    if isnum:
                        numeric_counts[k] += 1
                        if len(sample_vals[k]) < n_sample:
                            sample_vals[k].append(v)
                else:
                    # not present
                    pass

    print(f"File: {path}")
    print("Total records:", total)
    print("\nPresence counts (any value present):")
    for k in ("z_if","z_lof","z_gap_if_lof","max_dd","max_drawdown"):
        print(f"  {k:14s} : {key_counts.get(k,0)}")

    print("\nNumeric (finite) counts:")
    for k in ("z_if","z_lof","z_gap_if_lof","max_dd","max_drawdown"):
        print(f"  {k:14s} : {numeric_counts.get(k,0)}")

    print("\nSamples (first finite values found):")
    for k,vals in sample_vals.items():
        print(f"  {k:14s} -> {vals}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_labels.py <path.jsonl>")
        sys.exit(1)
    inspect(sys.argv[1])

