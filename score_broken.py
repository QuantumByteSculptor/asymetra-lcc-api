# score_broken.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from features import DEFAULT_CONFIG, features_to_row, vector_columns


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        out.append({"raw": obj, "features": feats})
    return out


def build_frame(items: List[Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for it in items:
        feats = it["features"]
        rec = features_to_row(feats, cfg=DEFAULT_CONFIG)
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        rec["_market"] = (feats.get("market") or "").strip().upper()
        recs.append(rec)
    df = pd.DataFrame.from_records(recs)
    cols = vector_columns(DEFAULT_CONFIG)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols + ["_asset_type", "_market"]]


def z(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def anomaly_scores(bundle, X: np.ndarray) -> np.ndarray:
    if_model = bundle["models"]["iforest"]
    lof_model = bundle["models"]["lof"]
    w_if = bundle["ensemble_weights"]["if"]
    w_lof = bundle["ensemble_weights"]["lof"]

    # Use pipeline internals
    def score_if(pipe):
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        m = pipe.named_steps["iforest"]
        X2 = scaler.transform(imputer.transform(X))
        return -m.score_samples(X2)

    def score_lof(pipe):
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        m = pipe.named_steps["lof"]
        X2 = scaler.transform(imputer.transform(X))
        return -m.score_samples(X2)

    s_if = score_if(if_model)
    s_lof = score_lof(lof_model)
    ens = w_if * z(s_if) + w_lof * z(s_lof)
    return ens


def status_from_score(score: float, warn: float, block: float) -> str:
    if score >= block:
        return "BLOCK"
    if score >= warn:
        return "WARN"
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/unsup_bundle.joblib", help="Path to unsup bundle")
    ap.add_argument("--tests", default="lcc_broken_tests.jsonl", help="Broken tests jsonl")
    ap.add_argument("--use_per_asset_thresholds", action="store_true", help="Use per-asset thresholds if available")
    ap.add_argument("--top", type=int, default=10, help="Show top-N anomalies")
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    items = load_jsonl(args.tests)
    df = build_frame(items)

    cols = vector_columns(DEFAULT_CONFIG)
    X = df[cols].to_numpy(dtype=float)

    scores = anomaly_scores(bundle, X)

    global_warn = bundle["thresholds_global"]["warn"]
    global_block = bundle["thresholds_global"]["block"]
    per_asset = bundle.get("thresholds_per_asset_type", {})

    statuses = []
    warn_used = []
    block_used = []
    for i in range(len(df)):
        at = df.iloc[i]["_asset_type"]
        warn = global_warn
        block = global_block
        if args.use_per_asset_thresholds and at in per_asset:
            warn = per_asset[at]["warn"]
            block = per_asset[at]["block"]
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
    total = len(out)
    ok = counts.get("OK", 0)
    warn = counts.get("WARN", 0)
    block = counts.get("BLOCK", 0)

    print(f"Bundle: {args.bundle}")
    print(f"Global thresholds: WARN>={global_warn:.4f}, BLOCK>={global_block:.4f}")
    if args.use_per_asset_thresholds:
        print("Per-asset thresholds: ENABLED (if available)\n")
    else:
        print("Per-asset thresholds: DISABLED\n")

    for i, row in out.iterrows():
        print(f"[{i+1:02d}] status={row['status']:5s} score={row['score']:.4f}  asset_type={row['_asset_type']} market={row['_market']}  thr(W,B)=({row['warn_thr']:.3f},{row['block_thr']:.3f})")

    print("\nSummary:")
    print(f"OK={ok} WARN={warn} BLOCK={block} total={total}")
    print(f"Rates: OK={ok/total:.1%} WARN={warn/total:.1%} BLOCK={block/total:.1%}")

    # Top anomalies
    top = out.sort_values("score", ascending=False).head(args.top)
    print(f"\nTop {min(args.top, len(out))} anomalies:")
    for i, row in top.iterrows():
        print(f"  - idx={i+1:02d} score={row['score']:.4f} status={row['status']} asset_type={row['_asset_type']} market={row['_market']}")

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