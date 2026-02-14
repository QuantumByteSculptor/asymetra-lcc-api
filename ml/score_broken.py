# score_broken.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from features import DEFAULT_CONFIG, features_to_row, vector_columns


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        out.append({"raw": obj, "features": feats})
    return out


# -----------------------------
# Build matrix aligned to bundle schema
# -----------------------------
def _bundle_columns(bundle: Dict[str, Any], fallback_cfg: Dict[str, Any]) -> List[str]:
    """
    Source of truth for feature order:
    - prefer bundle["feature_columns"] (new)
    - else bundle["columns"] (old)
    - else fallback to current code's vector_columns(DEFAULT_CONFIG)
    """
    cols = bundle.get("feature_columns") or bundle.get("columns")
    if cols and isinstance(cols, list) and all(isinstance(c, str) for c in cols):
        return cols
    return vector_columns(fallback_cfg)


def build_frame(items: List[Dict[str, Any]], cols: List[str]) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for it in items:
        feats = it["features"]
        rec = features_to_row(feats, cfg=DEFAULT_CONFIG)

        # meta for per-asset thresholds
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        rec["_market"] = (feats.get("market") or "").strip().upper()

        recs.append(rec)

    df = pd.DataFrame.from_records(recs)

    # ensure all expected columns exist, in the bundle's exact order
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[cols + ["_asset_type", "_market"]].copy()


# -----------------------------
# Scoring utilities
# -----------------------------
def _z_with_params(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (x - float(mu)) / (float(sigma) + 1e-12)


def _score_anomaly(model, Xp: np.ndarray) -> np.ndarray:
    """
    Return "bigger = more anomalous" scores for both IF and LOF.
    - IF: decision_function higher = less anomalous => negate
    - LOF: score_samples higher = less anomalous => negate
    """
    if hasattr(model, "decision_function"):
        return (-model.decision_function(Xp)).astype(float)
    if hasattr(model, "score_samples"):
        return (-model.score_samples(Xp)).astype(float)
    raise TypeError(f"Unsupported model type: {type(model)}")


def anomaly_scores(bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    # pull components
    models = bundle["models"]
    if_model = models["iforest"]
    lof_model = models["lof"]

    w_if = float(bundle["ensemble_weights"]["if"])
    w_lof = float(bundle["ensemble_weights"]["lof"])

    # imputer (serialized as bundle["imputer"]["object"])
    imp = bundle.get("imputer", {})
    imputer = imp.get("object", None)
    if imputer is None:
        raise KeyError("bundle['imputer']['object'] missing (retrain unsup bundle with patched trainer)")

    # IMPORTANT: align X to what the model expects (feature count)
    # If you dropped all-NaN columns during training, bundle columns is smaller.
    # Our build_frame uses bundle columns already, so X should match.
    X_imp = imputer.transform(X)

    # compute raw "anomaly" (higher=more anomalous)
    s_if_raw = _score_anomaly(if_model, X_imp)
    s_lof_raw = _score_anomaly(lof_model, X_imp)

    # normalize with training params
    norm = bundle.get("score_norm", {})
    n_if = norm.get("if", {})
    n_lof = norm.get("lof", {})

    z_if = _z_with_params(s_if_raw, n_if.get("mu", 0.0), n_if.get("sigma", 1.0))
    z_lof = _z_with_params(s_lof_raw, n_lof.get("mu", 0.0), n_lof.get("sigma", 1.0))

    ens = (w_if * z_if) + (w_lof * z_lof)
    return ens.astype(float)


def status_from_score(score: float, warn: float, block: float) -> str:
    if score >= block:
        return "BLOCK"
    if score >= warn:
        return "WARN"
    return "OK"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/unsup_bundle.joblib", help="Path to unsup bundle")
    ap.add_argument("--tests", default="lcc_broken_tests.jsonl", help="Broken tests jsonl")
    ap.add_argument("--use_per_asset_thresholds", action="store_true", help="Use per-asset thresholds if available")
    ap.add_argument("--top", type=int, default=10, help="Show top-N anomalies")
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    cols = _bundle_columns(bundle, DEFAULT_CONFIG)

    items = load_jsonl(args.tests)
    df = build_frame(items, cols)

    X = df[cols].to_numpy(dtype=float)
    scores = anomaly_scores(bundle, X)

    global_warn = float(bundle["thresholds_global"]["warn"])
    global_block = float(bundle["thresholds_global"]["block"])
    per_asset = bundle.get("thresholds_per_asset_type", {}) or {}

    statuses: List[str] = []
    warn_used: List[float] = []
    block_used: List[float] = []

    for i in range(len(df)):
        at = str(df.iloc[i]["_asset_type"] or "")
        warn = global_warn
        block = global_block
        if args.use_per_asset_thresholds and at in per_asset:
            warn = float(per_asset[at]["warn"])
            block = float(per_asset[at]["block"])
        statuses.append(status_from_score(float(scores[i]), warn, block))
        warn_used.append(warn)
        block_used.append(block)

    out = df[["_asset_type", "_market"]].copy()
    out["score"] = scores
    out["status"] = statuses
    out["warn_thr"] = warn_used
    out["block_thr"] = block_used

    # Summary report
    counts = out["status"].value_counts().to_dict()
    total = int(len(out))
    ok = int(counts.get("OK", 0))
    warn = int(counts.get("WARN", 0))
    block = int(counts.get("BLOCK", 0))

    print(f"Bundle: {args.bundle}")
    print(f"Using feature columns: {len(cols)}")
    print(f"Global thresholds: WARN>={global_warn:.4f}, BLOCK>={global_block:.4f}")
    print("Per-asset thresholds: " + ("ENABLED" if args.use_per_asset_thresholds else "DISABLED") + "\n")

    for i, row in out.iterrows():
        print(
            f"[{i+1:02d}] status={row['status']:5s} score={row['score']:.4f}  "
            f"asset_type={row['_asset_type']} market={row['_market']}  "
            f"thr(W,B)=({row['warn_thr']:.3f},{row['block_thr']:.3f})"
        )

    print("\nSummary:")
    print(f"OK={ok} WARN={warn} BLOCK={block} total={total}")
    if total > 0:
        print(f"Rates: OK={ok/total:.1%} WARN={warn/total:.1%} BLOCK={block/total:.1%}")

    # Top anomalies
    top = out.sort_values("score", ascending=False).head(int(args.top))
    print(f"\nTop {min(int(args.top), total)} anomalies:")
    for i, row in top.iterrows():
        print(
            f"  - idx={i+1:02d} score={row['score']:.4f} status={row['status']} "
            f"asset_type={row['_asset_type']} market={row['_market']}"
        )

    # Simple pass/fail criterion for broken tests
    if ok > 0:
        print("\n⚠️ FAIL: Some broken cases are still OK.")
        print("Next actions:")
        print("  1) Add deterministic hard rules for unit/invariant violations.")
        print("  2) Keep ML as soft detector; improve features ratios (already included).")
        print("  3) Consider per-asset thresholds + separate crypto model.")
    else:
        print("\n✅ PASS: No broken case returned OK.")


if __name__ == "__main__":
    main()

