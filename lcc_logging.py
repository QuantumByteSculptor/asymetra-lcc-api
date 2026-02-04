# lcc_logging.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


SHADOW_LOG_PATH = os.getenv("SHADOW_LOG_PATH", "logs/lcc_shadow_log.jsonl")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_LCC_TABLE", "lcc_shadow_logs")


# -----------------------
# Stable minimal reasons
# -----------------------
def stable_reasons(
    payload: Dict[str, Any],
    deterministic_reasons: List[str],
    unsup_score: Optional[float],
    unsup_status: Optional[str],
    xgb_probs: Optional[Dict[str, float]],
) -> List[str]:
    """
    Goal: stable, minimal, same output ordering always.
    - Deterministic reasons always first (already stable)
    - Then 0-2 ML hints max (never spam)
    """
    out: List[str] = []

    # 1) deterministic reasons (stable order)
    for r in deterministic_reasons:
        if r and r not in out:
            out.append(r)

    # 2) unsup status hint (only if WARN/BLOCK)
    if unsup_status in ("WARN", "BLOCK"):
        out.append(f"UNSUP_{unsup_status}")

    # 3) xgb hint (only if it strongly disagrees / or strongly suggests block)
    # expecting keys ok/warn/block
    if xgb_probs:
        pb = float(xgb_probs.get("block", 0.0))
        pw = float(xgb_probs.get("warn", 0.0))
        # minimal, very conservative
        if pb >= 0.90:
            out.append("XGB_HIGH_BLOCK_PROB")
        elif pw >= 0.80:
            out.append("XGB_HIGH_WARN_PROB")

    # Keep it short (max 4)
    return out[:4]


# -----------------------
# JSONL backup logging
# -----------------------
def log_shadow_jsonl(event: Dict[str, Any]) -> None:
    try:
        p = Path(SHADOW_LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -----------------------
# Supabase insert logging
# -----------------------
def log_shadow_supabase(event: Dict[str, Any]) -> None:
    """
    Safe insert into Supabase. Never throws.
    Uses Service Role Key (server-side only).
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return

    try:
        import requests

        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        # Map to your DB columns (match the ALTER TABLE I gave)
        payload = {
            "ts": event.get("ts"),
            "asset_type": event.get("asset_type"),
            "market": event.get("market"),
            "final_status": event.get("final_status"),
            "deterministic_reasons": event.get("deterministic_reasons"),
            "reasons": event.get("reasons"),
            "unsup_score": (event.get("unsup") or {}).get("ensemble"),
            "unsup_status": (event.get("unsup") or {}).get("status"),
            "unsup_thresholds": (event.get("unsup") or {}).get("thresholds"),
            "xgb_pred": ((event.get("xgb_shadow") or {}) or {}).get("pred"),
            "xgb_probs": ((event.get("xgb_shadow") or {}) or {}).get("probs"),
            "raw_payload": event.get("raw_payload"),
        }

        # Remove None to avoid mismatches
        payload = {k: v for k, v in payload.items() if v is not None}

        requests.post(url, headers=headers, data=json.dumps(payload), timeout=2.0)

    except Exception:
        pass


def log_shadow(event: Dict[str, Any]) -> None:
    # Always keep a local trail
    log_shadow_jsonl(event)
    # Best effort supabase
    log_shadow_supabase(event)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
