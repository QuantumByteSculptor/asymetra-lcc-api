import json
from collections import Counter, defaultdict
from pathlib import Path

LOG_PATH = Path("logs/lcc_shadow_log.jsonl")

def bucket(pb: float) -> str:
    # buckets: 0.00-0.05, 0.05-0.10, ... 0.95-1.00
    b = int(pb * 20) / 20
    return f"{b:.2f}+"

def main():
    if not LOG_PATH.exists():
        raise SystemExit(f"Missing log file: {LOG_PATH}")

    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    print("rows:", len(rows))

    final_status = Counter()
    by_asset = Counter()
    by_market = Counter()

    disagree = 0
    disagree_hi = 0
    pb_bins = Counter()
    soft_warn = 0

    # focus: unsup OK but xgb predicts BLOCK
    for r in rows:
        fs = r.get("final_status")
        final_status[fs] += 1

        at = r.get("asset_type") or "unknown"
        mk = r.get("market") or "unknown"
        by_asset[at] += 1
        by_market[mk] += 1

        unsup = r.get("unsup") or {}
        u = unsup.get("status")
        xgb = r.get("xgb_shadow") or {}
        xpred = xgb.get("pred")
        probs = xgb.get("probs") or {}
        pb = probs.get("BLOCK")

        reasons = r.get("reasons") or []
        if "XGB_SOFT_WARN" in reasons:
            soft_warn += 1

        if u == "OK" and xpred == "BLOCK":
            disagree += 1
            if isinstance(pb, (int, float)):
                pb_bins[bucket(pb)] += 1
                if pb >= 0.90:
                    disagree_hi += 1

    print("final_status:", dict(final_status))
    print("soft_warn count:", soft_warn)
    print("disagree (unsup OK / xgb BLOCK):", disagree)
    print("...with P(block)>=0.90:", disagree_hi)
    print("P(block) buckets:", dict(pb_bins))

    print("\nTop asset_type:")
    for k, v in by_asset.most_common(10):
        print(f"  - {k}: {v}")

    print("\nTop market:")
    for k, v in by_market.most_common(10):
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()

