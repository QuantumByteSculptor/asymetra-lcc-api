from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from features import features_to_vector  # ton existant

API_KEY = os.getenv("API_KEY")

UNSUP_PATH = os.getenv("MODEL_PATH", "models/unsup_bundle.joblib")
SUP_PATH = os.getenv("SUP_MODEL_PATH", "models/sup_bundle.joblib")
SHADOW_LOG_PATH = os.getenv("SHADOW_LOG_PATH", "logs/lcc_shadow_log.jsonl")
ENABLE_XGB_SHADOW = os.getenv("ENABLE_XGB_SHADOW", "1") == "1"

app = FastAPI(title="Asymetra LCC API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://asymetra.lovable.app", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------
# Startup load
# -----------------------------
unsup_bundle: dict[str, Any] | None = None
sup_bundle: dict[str, Any] | None = None


@app.on_event("startup")
def load_models():
    global unsup_bundle, sup_bundle

    unsup_bundle = joblib.load(UNSUP_PATH)

    sup_bundle = None
    try:
        if Path(SUP_PATH).exists():
            sup_bundle = joblib.load(SUP_PATH)
    except Exception:
        sup_bundle = None


# -----------------------------
# Helpers
# -----------------------------
def check_api_key(req: Request):
    if API_KEY:
        key = req.headers.get("x-api-key")
        if key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")


def now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_shadow(event: dict) -> None:
    try:
        p = Path(SHADOW_LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # never break scoring because of logging
        pass


# -----------------------------
# Deterministic rules
# -----------------------------
def deterministic_lcc(features: dict) -> list[str]:
    errors: list[str] = []

    # --- Robust VaR/ES ordering (supports negative-loss or positive-magnitude conventions) ---
    v95 = features.get("var95", None)
    v99 = features.get("var99", None)
    e95 = features.get("es95", None)
    e99 = features.get("es99", None)

    def _order_ok(a, b) -> bool:
        # Returns True if "b is more extreme than a" according to sign convention
        # negative losses: b <= a
        # positive magnitudes: b >= a
        if a is None or b is None:
            return True
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return True
        if a == 0 or b == 0:
            return True
        if (a < 0 and b < 0):
            return b <= a
        if (a > 0 and b > 0):
            return b >= a
        return False  # mixed sign

    if not _order_ok(v95, v99):
        errors.append("VAR_ORDER_VIOLATION")
    if not _order_ok(e95, e99):
        errors.append("ES_ORDER_VIOLATION")

    if (isinstance(v95, (int, float)) and isinstance(v99, (int, float)) and ((v95 < 0) != (v99 < 0))):
        errors.append("VAR_MIXED_SIGN")
    if (isinstance(e95, (int, float)) and isinstance(e99, (int, float)) and ((e95 < 0) != (e99 < 0))):
        errors.append("ES_MIXED_SIGN")

    # Other sanity checks (exemples)
    if features.get("tuw_pct", 0) < 0 or features.get("tuw_pct", 0) > 100:
        errors.append("TUW_OUT_OF_RANGE")

    if (features.get("asset_type") or "").strip().lower() == "equity":
        vol_ann = features.get("vol_ann", 0)
        if isinstance(vol_ann, (int, float)):
            if vol_ann < 0.03:
                errors.append("VOL_TOO_LOW_FOR_EQUITY")
            if vol_ann > 1.5:
                errors.append("VOL_TOO_HIGH_FOR_EQUITY")

    return errors


# -----------------------------
# Unsupervised scoring
# -----------------------------
def _z(score: float, mu: float, sigma: float) -> float:
    return (score - mu) / (sigma + 1e-9)


def unsup_score_payload(payload: dict) -> dict:
    """
    Returns:
      raw_if, raw_lof, z_if, z_lof, ensemble, status, thresholds_used
    """
    assert unsup_bundle is not None, "unsup bundle not loaded"

    cfg = unsup_bundle["config"]
    X = features_to_vector(payload, cfg).reshape(1, -1)

    if_model = unsup_bundle["models"]["iforest"]
    lof_model = unsup_bundle["models"]["lof"]

    imp = if_model.named_steps["imputer"]
    sca = if_model.named_steps["scaler"]

    Xi = sca.transform(imp.transform(X))

    raw_if = float(-if_model.named_steps["iforest"].score_samples(Xi)[0])
    raw_lof = float(-lof_model.named_steps["lof"].score_samples(Xi)[0])

    score_norm = unsup_bundle.get("score_norm") or {}
    mu_if = score_norm.get("if", {}).get("mu", 0.0)
    sd_if = score_norm.get("if", {}).get("sigma", 1.0)
    mu_lof = score_norm.get("lof", {}).get("mu", 0.0)
    sd_lof = score_norm.get("lof", {}).get("sigma", 1.0)

    z_if = float(_z(raw_if, mu_if, sd_if))
    z_lof = float(_z(raw_lof, mu_lof, sd_lof))

    w = unsup_bundle.get("ensemble_weights", {"if": 0.6, "lof": 0.4})
    ens = float(w.get("if", 0.6) * z_if + w.get("lof", 0.4) * z_lof)

    thr = unsup_bundle["thresholds_global"]
    asset_type = (payload.get("asset_type") or "").strip().lower()
    asset_thr = (unsup_bundle.get("thresholds_per_asset_type") or {}).get(asset_type)

    warn = float(asset_thr["warn"]) if asset_thr else float(thr["warn"])
    block = float(asset_thr["block"]) if asset_thr else float(thr["block"])

    status = "OK"
    if ens >= warn:
        status = "WARN"
    if ens >= block:
        status = "BLOCK"

    return {
        "raw_if": raw_if,
        "raw_lof": raw_lof,
        "z_if": z_if,
        "z_lof": z_lof,
        "ensemble": ens,
        "status": status,
        "thresholds": {"warn": warn, "block": block},
        "asset_thresholds_used": bool(asset_thr),
    }


# -----------------------------
# Stable reasons
# -----------------------------
def stable_reasons(
    payload: dict,
    deterministic_reasons: list[str] | None = None,
    unsup_status: str | None = None,
    unsup_score: float | None = None,
    xgb_probs: dict | None = None,
) -> list[str]:
    deterministic_reasons = deterministic_reasons or []
    reasons: list[str] = []
    reasons.extend(deterministic_reasons)

    if unsup_status in ("WARN", "BLOCK") and unsup_score is not None:
        reasons.append(f"UNSUP_{unsup_status}")

    if xgb_probs and isinstance(xgb_probs, dict):
        pb = xgb_probs.get("BLOCK")
        if isinstance(pb, (int, float)) and pb >= 0.90:
            reasons.append("XGB_VERY_HIGH_BLOCK_PROB")

    # example of a “disagree high” tag (no bump here, juste log)
    if unsup_status == "OK" and xgb_probs and isinstance(xgb_probs, dict):
        pb = xgb_probs.get("BLOCK")
        if isinstance(pb, (int, float)) and pb >= 0.80:
            reasons.append("XGB_DISAGREE_HIGH")

    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for r in reasons:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


# -----------------------------
# XGB shadow inference (bundle v2: models.xgb + prep.*)
# -----------------------------
def xgb_shadow(payload: dict) -> dict | None:
    if not ENABLE_XGB_SHADOW or sup_bundle is None:
        return None

    try:
        model = (sup_bundle.get("models") or {}).get("xgb")
        prep = sup_bundle.get("prep") or {}
        numeric_cols = prep.get("numeric_cols") or []
        medians = prep.get("medians") or {}
        feat_cols = prep.get("feature_columns") or []

        labels = (sup_bundle.get("labels") or {}).get("inv") or {0: "OK", 1: "WARN", 2: "BLOCK"}

        if model is None or not feat_cols:
            return None

        # Build numeric vector (same base as unsup config)
        assert unsup_bundle is not None, "unsup bundle not loaded"
        cfg = sup_bundle.get("config") or unsup_bundle.get("config")
        x_num = features_to_vector(payload, cfg).reshape(1, -1)

        # Convert numeric vector into dict by numeric_cols order.
        # Assumption: numeric_cols correspond to vector_columns(cfg) used in train.
        # If lengths mismatch, we fill what we can.
        row = {c: 0.0 for c in numeric_cols}
        for i, c in enumerate(numeric_cols):
            if i < x_num.shape[1]:
                val = float(x_num[0, i])
                if np.isnan(val):
                    val = float(medians.get(c, 0.0))
                row[c] = val
            else:
                row[c] = float(medians.get(c, 0.0))

        # Apply medians for any missing numeric col
        for c in numeric_cols:
            if row.get(c) is None or (isinstance(row.get(c), float) and np.isnan(row[c])):
                row[c] = float(medians.get(c, 0.0))

        # One-hot asset/market like in train script (prefix asset_ / mkt_)
        at = (payload.get("asset_type") or "").strip().lower()
        mk = (payload.get("market") or "").strip().upper()

        # Build full feature vector with final one-hot columns
        full = {c: 0.0 for c in feat_cols}

        # Fill numeric into full where names match
        for c in numeric_cols:
            if c in full:
                full[c] = float(row[c])

        # Set one-hot if columns exist
        # train: pd.get_dummies columns=["_asset_type","_market"], prefix=["asset","mkt"]
        at_col = f"asset_{at}"
        mk_col = f"mkt_{mk}"
        if at_col in full:
            full[at_col] = 1.0
        if mk_col in full:
            full[mk_col] = 1.0

        X_final = np.array([[full[c] for c in feat_cols]], dtype=float)

        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_final)[0].tolist()

        pred = None
        if hasattr(model, "predict"):
            pred = int(model.predict(X_final)[0])

        # Map probs
        prob_map = None
        if probs is not None and len(probs) == len(labels):
            prob_map = {labels[i]: float(probs[i]) for i in range(len(probs))}

        pred_label = labels.get(pred, str(pred)) if pred is not None else None

        return {"pred": pred_label, "probs": prob_map}

    except Exception:
        return None


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "service": "Asymetra LCC API",
        "status": "running",
        "endpoints": ["/health", "/score", "/shadow_logs"],
        "models": {
            "unsupervised": UNSUP_PATH,
            "supervised": SUP_PATH if Path(SUP_PATH).exists() else None,
            "xgb_shadow_enabled": ENABLE_XGB_SHADOW,
        },
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "unsup_loaded": unsup_bundle is not None,
        "sup_loaded": sup_bundle is not None,
    }


@app.get("/shadow_logs")
def shadow_logs(req: Request, limit: int = 50):
    check_api_key(req)
    limit = max(1, min(int(limit), 500))

    p = Path(SHADOW_LOG_PATH)
    if not p.exists():
        return {"rows": [], "count": 0}

    # Read last N lines (simple + ok for small logs)
    lines = p.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:]

    rows = []
    for ln in tail:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass

    return {"rows": rows, "count": len(rows)}


@app.post("/score")
async def score(req: Request):
    check_api_key(req)
    payload = await req.json()

    asset_type = (payload.get("asset_type") or "").strip().lower()
    market = (payload.get("market") or "").strip().upper()

    # 1) Deterministic block
    det_errors = deterministic_lcc(payload)
    if det_errors:
        xgb = xgb_shadow(payload)  # log even if deterministic blocks

        reasons = stable_reasons(
            payload=payload,
            deterministic_reasons=det_errors,
            unsup_status=None,
            unsup_score=None,
            xgb_probs=(xgb or {}).get("probs") if xgb else None,
        )

        evt = {
            "ts": now_iso_z(),
            "asset_type": asset_type,
            "market": market,
            "final_status": "BLOCK_DETERMINISTIC",
            "deterministic_reasons": det_errors,
            "unsup": None,
            "xgb_shadow": xgb,
            "reasons": reasons,
        }
        log_shadow(evt)

        return {"status": "BLOCK_DETERMINISTIC", "reasons": det_errors, "ml": None}

    # 2) Unsupervised (authoritative)
    u = unsup_score_payload(payload)
    final_status = u["status"]

    # 3) XGB shadow (non-authoritative)
    xgb = xgb_shadow(payload)

    # 4) Stable reasons
    reasons = stable_reasons(
        payload=payload,
        deterministic_reasons=None,
        unsup_status=u["status"],
        unsup_score=float(u["ensemble"]),
        xgb_probs=(xgb or {}).get("probs") if xgb else None,
    )

    # 5) ✅ Soft “bump” (conservateur) — PROMOTE seulement vers WARN
    if final_status == "OK" and xgb and isinstance(xgb.get("probs"), dict):
        pb = xgb["probs"].get("BLOCK")
        if isinstance(pb, (int, float)):
            near_warn = float(u["ensemble"]) >= float(u["thresholds"]["warn"]) - 0.30
            if pb >= 0.90 and near_warn:
                final_status = "WARN"
                if "XGB_SOFT_WARN" not in reasons:
                    reasons.append("XGB_SOFT_WARN")

    # 6) Log shadow event
    evt = {
        "ts": now_iso_z(),
        "asset_type": asset_type,
        "market": market,
        "unsup": {
            "raw_if": u["raw_if"],
            "raw_lof": u["raw_lof"],
            "z_if": u["z_if"],
            "z_lof": u["z_lof"],
            "ensemble": u["ensemble"],
            "status": u["status"],
            "thresholds": u["thresholds"],
            "asset_thresholds_used": u["asset_thresholds_used"],
        },
        "xgb_shadow": xgb,
        "final_status": final_status,
        "reasons": reasons,
    }
    log_shadow(evt)

    # 7) Return
    return {
        "status": final_status,
        "anomaly_score": float(u["ensemble"]),
        "thresholds": u["thresholds"],
        "reasons": reasons,
        "shadow": {"xgb": xgb} if xgb is not None else None,
    }








