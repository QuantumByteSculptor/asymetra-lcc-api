# train_supervised_xgb.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from features import DEFAULT_CONFIG, features_to_row, vector_columns


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


# -----------------------------
# Feature building
# -----------------------------
def build_frame(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []

    for it in items:
        feats = it.get("features", it)

        # base engineered vector (legacy pipeline)
        rec = features_to_row(feats, cfg=cfg)

        # -----------------------------
        # 🔧 PATCH: robust aliasing + keep unsup z-features
        # -----------------------------
        # 1) max_dd alias (dataset uses max_drawdown)
        if rec.get("max_dd") is None or (isinstance(rec.get("max_dd"), float) and np.isnan(rec.get("max_dd"))):
            md = feats.get("max_dd")
            if md is None:
                md = feats.get("max_drawdown")
            if isinstance(md, (int, float)) and np.isfinite(md):
                rec["max_dd"] = float(md)

        # 2) Bring z_if / z_lof / z_gap_if_lof from dataset (built in build_dataset_daily --unsup_bundle)
        z_if = feats.get("z_if")
        z_lof = feats.get("z_lof")
        z_gap = feats.get("z_gap_if_lof")

        if isinstance(z_if, (int, float)) and np.isfinite(z_if):
            rec["z_if"] = float(z_if)
        if isinstance(z_lof, (int, float)) and np.isfinite(z_lof):
            rec["z_lof"] = float(z_lof)

        if isinstance(z_gap, (int, float)) and np.isfinite(z_gap):
            rec["z_gap_if_lof"] = float(z_gap)
        else:
            # compute if possible
            if isinstance(z_if, (int, float)) and np.isfinite(z_if) and isinstance(z_lof, (int, float)) and np.isfinite(z_lof):
                rec["z_gap_if_lof"] = float(abs(float(z_if) - float(z_lof)))

        # categorical fields for one-hot
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        rec["_market"] = (feats.get("market") or "").strip().upper()

        recs.append(rec)

    df = pd.DataFrame.from_records(recs)

    # Ensure stable numeric columns baseline
    cols = vector_columns(cfg)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Also ensure our patched columns exist (even if not in vector_columns(cfg))
    for c in ["max_dd", "z_if", "z_lof", "z_gap_if_lof"]:
        if c not in df.columns:
            df[c] = np.nan

    # Keep order: base vector + our extra + categoricals
    keep_cols = cols + [c for c in ["max_dd", "z_if", "z_lof", "z_gap_if_lof"] if c not in cols] + ["_asset_type", "_market"]
    df = df[keep_cols]

    return df


def median_impute_inplace(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Median impute numeric cols; drop cols that are 100% NaN to avoid warnings & shape drift."""
    df = df.copy()

    all_nan = [c for c in numeric_cols if c in df.columns and df[c].isna().all()]
    if all_nan:
        print(f"⚠️ Dropping fully-NaN numeric columns: {all_nan}")
        df = df.drop(columns=all_nan)
        numeric_cols = [c for c in numeric_cols if c not in all_nan]

    medians: Dict[str, float] = {}
    for c in numeric_cols:
        arr = df[c].to_numpy(dtype=float)
        if np.isfinite(arr).sum() == 0:
            m = 0.0
        else:
            m = float(np.nanmedian(arr))
            if not np.isfinite(m):
                m = 0.0
        medians[c] = m
        df[c] = df[c].astype(float).fillna(m)

    return df, medians


def make_design_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    assert "_asset_type" in df.columns, "Missing _asset_type — build_frame() must add it"
    assert "_market" in df.columns, "Missing _market — build_frame() must add it"

    base = df[numeric_cols + ["_asset_type", "_market"]].copy()
    X = pd.get_dummies(
        base,
        columns=["_asset_type", "_market"],
        prefix=["asset", "mkt"],
        dummy_na=False,
    )
    feat_names = X.columns.tolist()
    return X, feat_names


# -----------------------------
# Main
# -----------------------------
LABELS = {"ok": 0, "warn": 1, "block": 2}
INV_LABELS = {v: k for k, v in LABELS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok", required=True, help="JSONL path for OK samples")
    ap.add_argument("--warn", required=True, help="JSONL path for WARN samples")
    ap.add_argument("--block", required=True, help="JSONL path for BLOCK samples")
    ap.add_argument("--unsup_bundle", required=True, help="Path to unsupervised bundle (joblib)")
    ap.add_argument("--out", default="models/sup_bundle.joblib", help="Output bundle path")

    # model knobs
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=350)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--min_child_weight", type=float, default=2.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)

    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load unsup bundle (config parity)
    unsup = joblib.load(args.unsup_bundle)
    cfg = unsup.get("config", DEFAULT_CONFIG)

    # Load datasets
    ok_items = load_jsonl(args.ok)
    warn_items = load_jsonl(args.warn)
    block_items = load_jsonl(args.block)

    ok_df = build_frame(ok_items, cfg=cfg)
    warn_df = build_frame(warn_items, cfg=cfg)
    block_df = build_frame(block_items, cfg=cfg)

    X_all = pd.concat([ok_df, warn_df, block_df], ignore_index=True)
    y = np.array(
        [LABELS["ok"]] * len(ok_df) + [LABELS["warn"]] * len(warn_df) + [LABELS["block"]] * len(block_df),
        dtype=int,
    )

    # Numeric columns = base vector columns that exist + our extra columns if present
    base_numeric_cols = [c for c in vector_columns(cfg) if c in X_all.columns and not c.startswith("_")]
    extra = [c for c in ["max_dd", "z_if", "z_lof", "z_gap_if_lof"] if c in X_all.columns]
    numeric_cols = list(dict.fromkeys(base_numeric_cols + extra))  # stable unique order

    # Impute (and drop fully-NaN)
    X_all, medians = median_impute_inplace(X_all, numeric_cols)
    numeric_cols = [c for c in numeric_cols if c in X_all.columns]  # after drops

    # warn if z_gap missing
    if "z_gap_if_lof" not in numeric_cols:
        print("⚠️ z_gap_if_lof not included (dropped as fully-NaN or never present)")
    else:
        print("✅ z_gap_if_lof INCLUDED")

    # One-hot encode
    X_feat, feat_names = make_design_matrix(X_all, numeric_cols)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    model.fit(X_feat.to_numpy(dtype=float), y)

    proba = model.predict_proba(X_feat.to_numpy(dtype=float))
    pred = proba.argmax(axis=1)
    acc = float((pred == y).mean())

    bundle = {
        "config": cfg,
        "models": {"xgb": model},
        "prep": {
            "numeric_cols": numeric_cols,
            "medians": medians,
            "feature_columns": feat_names,
        },
        "labels": {"map": LABELS, "inv": INV_LABELS},
        "meta": {
            "seed": args.seed,
            "version": "sup_xgb_v3_patch_alias_zgap",
            "train_counts": {"ok": len(ok_df), "warn": len(warn_df), "block": len(block_df)},
            "train_accuracy": acc,
        },
    }

    joblib.dump(bundle, args.out)
    print(f"✅ Saved supervised bundle to {args.out}")
    print(f"Train counts: OK={len(ok_df)} WARN={len(warn_df)} BLOCK={len(block_df)}")
    print(f"Train accuracy (sanity): {acc:.3f}")
    print(f"Final feature columns: {len(feat_names)} (numeric={len(numeric_cols)} + one-hot)")


if __name__ == "__main__":
    main()





