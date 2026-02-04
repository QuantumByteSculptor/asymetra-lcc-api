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
# Helpers (unsup normalization)
# -----------------------------
def _z(x: float, mu: float, sigma: float) -> float:
    return (x - mu) / (sigma + 1e-9)


def compute_unsup_z_if_lof(
    feats: Dict[str, Any],
    unsup_bundle: Dict[str, Any],
) -> Tuple[float | None, float | None]:
    """
    Compute z_if and z_lof for a payload using unsup_bundle normalization stats.
    Requires running iforest/lof score_samples through the unsup pipeline.
    Returns (z_if, z_lof) or (None, None) if anything missing.
    """
    try:
        cfg = unsup_bundle.get("config", DEFAULT_CONFIG)

        # Build the same numeric vector as unsup expects
        # Here we rely on features_to_row + vector_columns(cfg) for stable numeric set.
        # IMPORTANT: Your unsup training used features_to_vector; but we don't import it here.
        # We'll reconstruct numeric vector from features_to_row + vector_columns.
        rec = features_to_row(feats, cfg=cfg)
        cols = vector_columns(cfg)
        x = np.array([[float(rec.get(c, np.nan)) for c in cols]], dtype=float)

        if_model = unsup_bundle["models"]["iforest"]
        lof_model = unsup_bundle["models"]["lof"]

        imp = if_model.named_steps["imputer"]
        sca = if_model.named_steps["scaler"]

        Xi = sca.transform(imp.transform(x))

        raw_if = float(-if_model.named_steps["iforest"].score_samples(Xi)[0])
        raw_lof = float(-lof_model.named_steps["lof"].score_samples(Xi)[0])

        score_norm = unsup_bundle.get("score_norm") or {}
        mu_if = float(score_norm.get("if", {}).get("mu", 0.0))
        sd_if = float(score_norm.get("if", {}).get("sigma", 1.0))
        mu_lof = float(score_norm.get("lof", {}).get("mu", 0.0))
        sd_lof = float(score_norm.get("lof", {}).get("sigma", 1.0))

        z_if = float(_z(raw_if, mu_if, sd_if))
        z_lof = float(_z(raw_lof, mu_lof, sd_lof))
        return z_if, z_lof
    except Exception:
        return None, None


# -----------------------------
# Feature building (same spirit as unsupervised)
# -----------------------------
def build_frame(items: List[Dict[str, Any]], unsup_bundle: Dict[str, Any]) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for it in items:
        feats = it.get("features", it)

        rec = features_to_row(feats, cfg=DEFAULT_CONFIG)

        # categorical for one-hot
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        rec["_market"] = (feats.get("market") or "").strip().upper()

        # ✅ add unsup-derived features (z_if, z_lof, z_gap_if_lof)
        z_if, z_lof = compute_unsup_z_if_lof(feats, unsup_bundle)
        rec["z_if"] = z_if
        rec["z_lof"] = z_lof
        rec["z_gap_if_lof"] = (z_if - z_lof) if (isinstance(z_if, (int, float)) and isinstance(z_lof, (int, float))) else np.nan

        recs.append(rec)

    df = pd.DataFrame.from_records(recs)

    # Ensure stable base numeric columns exist
    cols = vector_columns(DEFAULT_CONFIG)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure our added columns exist
    for c in ["z_if", "z_lof", "z_gap_if_lof"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols + ["z_if", "z_lof", "z_gap_if_lof", "_asset_type", "_market"]]
    return df


# -----------------------------
# Preprocess
# -----------------------------
def median_impute_inplace(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Median impute numeric cols.
    ✅ Drops columns that are 100% NaN BEFORE attempting imputation (no warnings, stable shape).
    Returns (df, medians, kept_numeric_cols)
    """
    df = df.copy()

    # Drop missing or fully-NaN columns BEFORE imputation
    kept = []
    for c in numeric_cols:
        if c not in df.columns:
            continue
        if df[c].isna().all():
            continue
        kept.append(c)

    dropped = [c for c in numeric_cols if c not in kept]
    if dropped:
        df = df.drop(columns=dropped, errors="ignore")

    medians: Dict[str, float] = {}
    for c in kept:
        arr = df[c].to_numpy(dtype=float)
        m = float(np.nanmedian(arr)) if not np.isnan(arr).all() else 0.0
        if np.isnan(m):
            m = 0.0
        medians[c] = m
        df[c] = df[c].astype(float).fillna(m)

    return df, medians, kept


def make_design_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode _asset_type and _market, keep numeric cols.
    Return (X_df, feature_names)
    """
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
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--min_child_weight", type=float, default=2.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)

    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load unsup bundle (for z_if/z_lof + config parity)
    unsup = joblib.load(args.unsup_bundle)
    cfg = unsup.get("config", DEFAULT_CONFIG)

    # Load datasets
    ok_items = load_jsonl(args.ok)
    warn_items = load_jsonl(args.warn)
    block_items = load_jsonl(args.block)

    ok_df = build_frame(ok_items, unsup)
    warn_df = build_frame(warn_items, unsup)
    block_df = build_frame(block_items, unsup)

    # Stack
    X_all = pd.concat([ok_df, warn_df, block_df], ignore_index=True)
    y = np.array(
        [LABELS["ok"]] * len(ok_df) + [LABELS["warn"]] * len(warn_df) + [LABELS["block"]] * len(block_df),
        dtype=int,
    )

    # Numeric columns = engineered vector cols + our added unsup z-features
    base_numeric_cols = [c for c in vector_columns(cfg) if c in X_all.columns and not c.startswith("_")]
    base_numeric_cols += ["z_if", "z_lof", "z_gap_if_lof"]

    # Impute (drops fully-NaN numeric cols first => no warnings)
    X_all, medians, numeric_cols = median_impute_inplace(X_all, base_numeric_cols)

    # One-hot encode
    X_feat, feat_names = make_design_matrix(X_all, numeric_cols)

    # XGBoost classifier (3 classes)
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

    # quick training sanity
    proba = model.predict_proba(X_feat.to_numpy(dtype=float))
    pred = proba.argmax(axis=1)
    acc = float((pred == y).mean())

    # sanity: confirm z_gap_if_lof is actually present after prep
    included_gap = "z_gap_if_lof" in numeric_cols or any(c.startswith("z_gap_if_lof") for c in feat_names)

    bundle = {
        "config": cfg,
        "models": {
            "xgb": model,
        },
        "prep": {
            "numeric_cols": numeric_cols,
            "medians": medians,             # runtime impute
            "feature_columns": feat_names,  # final columns after get_dummies
        },
        "labels": {
            "map": LABELS,
            "inv": INV_LABELS,
            "class_names": ["OK", "WARN", "BLOCK"],
        },
        "meta": {
            "seed": args.seed,
            "version": "sup_xgb_v2_zgap",
            "train_counts": {"ok": len(ok_df), "warn": len(warn_df), "block": len(block_df)},
            "train_accuracy": acc,
            "included_z_gap_if_lof": bool(included_gap),
        },
    }

    joblib.dump(bundle, args.out)
    print(f"✅ Saved supervised bundle to {args.out}")
    print(f"Train counts: OK={len(ok_df)} WARN={len(warn_df)} BLOCK={len(block_df)}")
    print(f"Train accuracy (sanity): {acc:.3f}")
    print(f"Final feature columns: {len(feat_names)} (numeric={len(numeric_cols)} + one-hot)")
    if included_gap:
        print("✅ z_gap_if_lof INCLUDED")
    else:
        print("⚠️ z_gap_if_lof NOT included (likely dropped because it's 100% NaN) — check unsup_bundle score_norm/models")


if __name__ == "__main__":
    main()



