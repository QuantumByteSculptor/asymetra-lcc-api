import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def safe_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/lcc_shadow_log.jsonl")
    ap.add_argument("--out", default="exports/disagree.csv")
    ap.add_argument("--pb_min", type=float, default=0.80)  # seuil “disagree” (p(block) >= pb_min)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    print("rows:", len(rows))

    final_status = Counter()
    soft_warn_count = 0

    disagree = 0
    disagree_hi = 0
    pb_bins = defaultdict(int)

    # CSV export rows
    export_rows = []

    for r in rows:
        fs = r.get("final_status")
        final_status[fs] += 1

        reasons = r.get("reasons") or []
        if "XGB_SOFT_WARN" in reasons:
            soft_warn_count += 1

        u_status = safe_get(r, ["unsup", "status"])
        x_pred = safe_get(r, ["xgb_shadow", "pred"])
        pb = safe_get(r, ["xgb_shadow", "probs", "BLOCK"])

        if u_status == "OK" and x_pred == "BLOCK":
            # option: ne garder que si p(block) >= pb_min
            if isinstance(pb, (int, float)) and pb >= args.pb_min:
                disagree += 1
                b = int(pb * 20) / 20  # buckets 0.80, 0.85, ...
                pb_bins[f"{b:.2f}+"] += 1
                if pb >= 0.90:
                    disagree_hi += 1

                # champs “lisibles”
                payload_asset = r.get("asset_type")
                payload_mkt = r.get("market")

                export_rows.append({
                    "ts": r.get("ts"),
                    "final_status": r.get("final_status"),
                    "asset_type": payload_asset,
                    "market": payload_mkt,

                    "unsup_status": u_status,
                    "unsup_score": safe_get(r, ["unsup", "ensemble"]),
                    "unsup_z_if": safe_get(r, ["unsup", "z_if"]),
                    "unsup_z_lof": safe_get(r, ["unsup", "z_lof"]),

                    "xgb_pred": x_pred,
                    "xgb_p_ok": safe_get(r, ["xgb_shadow", "probs", "OK"]),
                    "xgb_p_warn": safe_get(r, ["xgb_shadow", "probs", "WARN"]),
                    "xgb_p_block": pb,

                    "reasons": "|".join(reasons),
                })

    print("final_status:", dict(final_status))
    print("soft_warn count:", soft_warn_count)
    print("disagree (unsup OK / xgb BLOCK) with P(block)>=pb_min:", disagree)
    print("...with P(block)>=0.90:", disagree_hi)
    print("P(block) buckets:", dict(pb_bins))

    # write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if export_rows:
        fieldnames = list(export_rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(export_rows)
        print(f"✅ wrote CSV: {out_path} (rows={len(export_rows)})")
    else:
        print("ℹ️ no rows to export (maybe pb_min too high?)")


if __name__ == "__main__":
    main()



