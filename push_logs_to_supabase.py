#!/usr/bin/env python3
import os
import json
import requests
from pathlib import Path

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LOG_PATH = os.getenv("SHADOW_LOG_PATH", "logs/lcc_shadow_log.jsonl")
TABLE = os.getenv("SUPABASE_TABLE", "lcc_shadow_logs")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Please set SUPABASE_URL and SUPABASE_KEY environment variables.")

def read_jsonl(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)

def prepare_row(j):
    # Map fields to columns in the SQL schema above
    row = {
        "ts": j.get("ts"),
        "asset_type": j.get("asset_type"),
        "market": j.get("market"),
        "final_status": j.get("final_status") or (j.get("unsup") or {}).get("status"),
        "deterministic_reasons": json.dumps(j.get("deterministic_reasons") or j.get("deterministic_reasons") or []),
        "unsup_raw_if": (j.get("unsup") or {}).get("raw_if"),
        "unsup_raw_lof": (j.get("unsup") or {}).get("raw_lof"),
        "unsup_z_if": (j.get("unsup") or {}).get("z_if"),
        "unsup_z_lof": (j.get("unsup") or {}).get("z_lof"),
        "unsup_ensemble": (j.get("unsup") or {}).get("ensemble"),
        "unsup_status": (j.get("unsup") or {}).get("status"),
        "unsup_thresholds": json.dumps((j.get("unsup") or {}).get("thresholds") or {}),
        "asset_thresholds_used": (j.get("unsup") or {}).get("asset_thresholds_used", False),
        "xgb_pred": (j.get("xgb_shadow") or {}).get("pred") if isinstance(j.get("xgb_shadow"), dict) else None,
        "xgb_probs": json.dumps((j.get("xgb_shadow") or {}).get("probs") or None),
        "payload_raw": json.dumps(j),
        "reasons": json.dumps(j.get("reasons") or []),
    }
    # remove keys with None values -> keeps payload lean
    return {k: v for k, v in row.items() if v is not None}

def push_batch(rows):
    endpoint = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"  # speed: don't return inserted rows
    }
    r = requests.post(endpoint, headers=headers, json=rows)
    if not r.ok:
        print("ERROR pushing batch:", r.status_code, r.text)
    return r

def main():
    path = Path(LOG_PATH)
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")
    rows = []
    for j in read_jsonl(LOG_PATH):
        rows.append(prepare_row(j))
        if len(rows) >= 50:
            push_batch(rows)
            print("pushed 50 rows")
            rows = []
    if rows:
        push_batch(rows)
        print("pushed last batch")

if __name__ == "__main__":
    main()