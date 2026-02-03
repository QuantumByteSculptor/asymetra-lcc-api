# train_unsupervised.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from features import DEFAULT_CONFIG, features_to_row, vector_columns


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        rows.append({"raw": obj, "features": feats})
    return rows


def build_frame(items: List[Dict[str, Any]]) -> pd.DataFrame:
    # Expand to dataframe of engineered features
    recs = []
    for it in items:
        feats = it["features"]
        rec = features_to_row(feats, cfg=DEFAULT_CONFIG)
        # Keep for per-class calibration
        rec["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
        rec["_market"] = (feats.get("market") or "").strip().upper()
        recs.append(rec)
    df = pd.DataFrame.from_records(recs)
    # ensure stable column order
    cols = vector_columns(DEFAULT_CONFIG)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols + ["_asset_type", "_market"]]
    return df


def fit_models(X: np.ndarray, contamination: float, seed: int):
    if_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("iforest", IsolationForest(
                n_estimators=400,
                contamination=contamination,
                random_state=seed,
                n_jobs=-1
            )),
        ]
    )
    if_model.fit(X)

    # LOF: cannot "predict" on new data unless novelty=True (scikit-learn supports novelty)
    lof_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lof", LocalOutlierFactor(
                n_neighbors=25,
                novelty=True,
                contamination=contamination,
                n_jobs=-1
            )),
        ]
    )
    lof_model.fit(X)

    return if_model, lof_model


def score_anomaly(model: Pipeline, X: np.ndarray, kind: str) -> np.ndarray:
    """
    Convert model score -> anomaly score where higher = more anomalous.
    IsolationForest.score_samples: higher is more normal -> negate
    LOF.score_samples: higher is more normal -> negate
    """
    if kind == "if":
        s = model.named_steps["iforest"].score_samples(model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(X)))
        return -s
    if kind == "lof":
        s = model.named_steps["lof"].score_samples(model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(X)))
        return -s
    raise ValueError("unknown kind")


def calibrate_thresholds(scores: np.ndarray, warn_q: float, block_q: float) -> Tuple[float, float]:
    warn = float(np.quantile(scores, warn_q))
    block = float(np.quantile(scores, block_q))
    return warn, block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to lcc_runs.jsonl")
    ap.add_argument("--out", default="models/unsup_bundle.joblib", help="Output bundle path")
    ap.add_argument("--contamination", type=float, default=0.03, help="Expected anomaly rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warn_q", type=float, default=0.97, help="Warn quantile on training scores")
    ap.add_argument("--block_q", type=float, default=0.995, help="Block quantile on training scores")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(args.input)
    df = build_frame(items)

    cols = vector_columns(DEFAULT_CONFIG)
    X = df[cols].to_numpy(dtype=float)

    if_model, lof_model = fit_models(X, contamination=args.contamination, seed=args.seed)

    # Ensemble score = average of normalized scores (simple)
    if_scores = score_anomaly(if_model, X, "if")
    lof_scores = score_anomaly(lof_model, X, "lof")

    # normalize to z-scores for combining
    def z(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)

    ens = 0.6 * z(if_scores) + 0.4 * z(lof_scores)

    warn, block = calibrate_thresholds(ens, args.warn_q, args.block_q)

    # Per-asset-type threshold calibration (more stable in practice)
    per_asset = {}
    for atype in sorted(set(df["_asset_type"].tolist())):
        if not atype:
            continue
        mask = df["_asset_type"] == atype
        if mask.sum() < 50:
            continue
        warn_a, block_a = calibrate_thresholds(ens[mask.to_numpy()], args.warn_q, args.block_q)
        per_asset[atype] = {"warn": warn_a, "block": block_a, "n": int(mask.sum())}

    bundle = {
        "config": DEFAULT_CONFIG,
        "columns": cols,
        "models": {
            "iforest": if_model,
            "lof": lof_model,
        },
        "ensemble_weights": {"if": 0.6, "lof": 0.4},
        "thresholds_global": {"warn": warn, "block": block, "warn_q": args.warn_q, "block_q": args.block_q},
        "thresholds_per_asset_type": per_asset,
        "meta": {
            "contamination": args.contamination,
            "seed": args.seed,
            "version": "unsup_v1",
        }
    }

    joblib.dump(bundle, args.out)
    print(f"Saved bundle to {args.out}")
    print(f"Global thresholds: WARN>={warn:.4f}, BLOCK>={block:.4f}")
    if per_asset:
        print("Per-asset thresholds:")
        for k, v in per_asset.items():
            print(f"  - {k}: WARN>={v['warn']:.4f}, BLOCK>={v['block']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()

