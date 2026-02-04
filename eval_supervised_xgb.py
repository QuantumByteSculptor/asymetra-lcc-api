# eval_supervised_xgb.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from features import features_to_row, vector_columns, DEFAULT_CONFIG


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        items.append({"raw": obj, "features": feats})
    return items


def build_frame(items: List[Dict[str, Any]], cfg: dict) -> pd.DataFrame:
    recs = []
    for it in items:
        feats = it["features"]
        rec = features_to_row(feats, cfg=cfg)
        # keep asset_type for debugging / optional slicing
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        recs.append(rec)

    df = pd.DataFrame.from_records(recs)

    # Ensure stable numeric columns exist (from cfg)
    num_cols = vector_columns(cfg)
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Put numeric first, then meta
    df = df[num_cols + ["_asset_type"]]
    return df


# -----------------------------
# Preprocess aligned with training bundle
# -----------------------------
def apply_training_prep(
    df: pd.DataFrame,
    numeric_cols: List[str],
    feature_columns: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """
    Rebuild the exact feature matrix expected by the trained XGB:
    - take numeric cols
    - median-impute using training medians
    - one-hot encode _asset_type
    - align columns to feature_columns (add missing, drop extra, order)
    """
    X_num = df[numeric_cols].copy()

    # median-impute with training medians (fallback: current median, then 0)
    for c in numeric_cols:
        if c in medians and medians[c] is not None:
            X_num[c] = X_num[c].fillna(float(medians[c]))
        else:
            m = X_num[c].median()
            if pd.isna(m):
                m = 0.0
            X_num[c] = X_num[c].fillna(float(m))

    # one-hot asset type
    # produce columns like asset_equity, asset_fx, ...
    at = df["_asset_type"].fillna("").astype(str)
    X_cat = pd.get_dummies(at, prefix="asset")
    # Merge
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    # Align to training feature columns
    for c in feature_columns:
        if c not in X.columns:
            X[c] = 0.0

    X = X[feature_columns].copy()

    # Ensure numeric dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X


# -----------------------------
# Eval
# -----------------------------
def pretty_counts(pred: np.ndarray) -> str:
    # expected classes: 0=OK,1=WARN,2=BLOCK (common convention)
    vals, cnts = np.unique(pred, return_counts=True)
    mapping = {0: "OK", 1: "WARN", 2: "BLOCK"}
    parts = []
    for v, c in zip(vals, cnts):
        parts.append(f"{mapping.get(int(v), str(int(v)))}={int(c)}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, help="Path to holdout jsonl (e.g. lcc_clean_holdout.jsonl)")
    ap.add_argument("--sup", required=True, help="Path to sup bundle joblib (e.g. models/sup_bundle.joblib)")
    ap.add_argument("--show_top", type=int, default=15, help="Show top N most suspicious (highest prob of BLOCK)")
    args = ap.parse_args()

    sup = joblib.load(args.sup)

    # --- Solution A: read keys from your real bundle structure ---
    cfg = sup.get("config") or DEFAULT_CONFIG

    models = sup.get("models", {})
    model = models.get("xgb")

    prep = sup.get("prep", {})
    feature_columns = prep.get("feature_columns")
    numeric_cols = prep.get("numeric_cols")
    medians = prep.get("medians", {})

    if model is None or not feature_columns or not numeric_cols or cfg is None:
        raise SystemExit(
            "sup bundle missing required keys: "
            "models['xgb'], prep['feature_columns'], prep['numeric_cols'], config"
        )

    # Load & featurize holdout
    items = load_jsonl(args.holdout)
    df = build_frame(items, cfg=cfg)

    # Build X aligned
    X = apply_training_prep(
        df=df,
        numeric_cols=list(numeric_cols),
        feature_columns=list(feature_columns),
        medians=dict(medians),
    )

    # Predict
    # XGBoost sklearn API: predict_proba gives probabilities per class
    proba = model.predict_proba(X.values)
    pred = np.argmax(proba, axis=1)

    print("✅ Loaded:", args.sup)
    print("Holdout:", args.holdout)
    print("Rows:", len(df))
    print("Pred counts:", pretty_counts(pred))

    # Show top anomalies: highest P(BLOCK)
    # assume class index 2 is BLOCK, otherwise fallback to max prob
    block_idx = 2 if proba.shape[1] >= 3 else int(np.argmax(np.mean(proba, axis=0)))
    p_block = proba[:, block_idx]

    top_n = min(args.show_top, len(df))
    top_idx = np.argsort(-p_block)[:top_n]

    print(f"\nTop {top_n} most suspicious by P(BLOCK):")
    for rank, i in enumerate(top_idx, start=1):
        at = df.loc[i, "_asset_type"]
        pb = float(p_block[i])
        pw = float(proba[i, 1]) if proba.shape[1] > 1 else 0.0
        po = float(proba[i, 0]) if proba.shape[1] > 0 else 0.0
        label = {0: "OK", 1: "WARN", 2: "BLOCK"}.get(int(pred[i]), str(int(pred[i])))
        print(f"  [{rank:02d}] pred={label:5s}  P(block)={pb:.3f}  P(warn)={pw:.3f}  P(ok)={po:.3f}  asset_type={at}")

    # Optional: quick sanity on distribution per asset_type
    print("\nBy asset_type:")
    tmp = pd.DataFrame({"asset_type": df["_asset_type"], "pred": pred})
    for atype, g in tmp.groupby("asset_type"):
        vals, cnts = np.unique(g["pred"].values, return_counts=True)
        mapping = {0: "OK", 1: "WARN", 2: "BLOCK"}
        line = ", ".join([f"{mapping.get(int(v), v)}={int(c)}" for v, c in zip(vals, cnts)])
        print(f"  - {atype or '(empty)'}: {line}")


if __name__ == "__main__":
    main()



