from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import numpy as np

from features import features_to_vector

API_KEY = os.getenv("API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "models/unsup_bundle.joblib")

app = FastAPI(title="Asymetra LCC API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://asymetra.lovable.app"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def load_model():
    global bundle
    bundle = joblib.load(MODEL_PATH)

def check_api_key(req: Request):
    if API_KEY:
        key = req.headers.get("x-api-key")
        if key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

def deterministic_lcc(features: dict):
    errors = []

    if features.get("var99", 0) < features.get("var95", 0):
        errors.append("VAR_ORDER_VIOLATION")
    if features.get("es99", 0) < features.get("es95", 0):
        errors.append("ES_ORDER_VIOLATION")
    if features.get("tuw_pct", 0) < 0 or features.get("tuw_pct", 0) > 100:
        errors.append("TUW_OUT_OF_RANGE")
    if features.get("asset_type") == "equity":
        if features.get("vol_ann", 0) < 0.03:
            errors.append("VOL_TOO_LOW_FOR_EQUITY")
        if features.get("vol_ann", 0) > 1.5:
            errors.append("VOL_TOO_HIGH_FOR_EQUITY")

    return errors

@app.post("/score")
async def score(req: Request):
    check_api_key(req)
    payload = await req.json()

    # 1) Deterministic LCC
    errors = deterministic_lcc(payload)
    if errors:
        return {
            "status": "BLOCK_DETERMINISTIC",
            "reasons": errors,
            "ml": None,
        }

    # 2) ML scoring
    cfg = bundle["config"]
    cols = bundle["columns"]

    X = features_to_vector(payload, cfg).reshape(1, -1)

    if_model = bundle["models"]["iforest"]
    lof_model = bundle["models"]["lof"]

    imp = if_model.named_steps["imputer"]
    sca = if_model.named_steps["scaler"]

    Xi = sca.transform(imp.transform(X))
    s_if = -if_model.named_steps["iforest"].score_samples(Xi)
    s_lof = -lof_model.named_steps["lof"].score_samples(Xi)

    score = float(0.6 * s_if[0] + 0.4 * s_lof[0])

    thr = bundle["thresholds_global"]
    asset_thr = bundle.get("thresholds_per_asset_type", {}).get(payload.get("asset_type"))

    warn = asset_thr["warn"] if asset_thr else thr["warn"]
    block = asset_thr["block"] if asset_thr else thr["block"]

    status = "OK"
    if score >= warn:
        status = "WARN"
    if score >= block:
        status = "BLOCK"

    return {
        "status": status,
        "anomaly_score": score,
        "thresholds": {"warn": warn, "block": block},
        "reasons": [],
    }
