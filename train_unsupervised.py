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

    df = df[cols + ["_asset_type", "_market"]]
    return df


def fit_models(X: np.ndarray, contamination: float, seed: int) -> Tuple[Pipeline, Pipeline]:
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


def _transform_X(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Apply same preprocessing as training pipeline (imputer + scaler)."""
    imp = pipe.named_steps["imputer"]
    sca = pipe.named_steps["scaler"]
    return sca.transform(imp.transform(X))


def anomaly_scores_from_models(if_model: Pipeline, lof_model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return anomaly scores where higher = more anomalous.
    Both IsolationForest.score_samples and LOF.score_samples: higher => more normal, so we negate.
    """
    Xi_if = _transform_X(if_model, X)
    s_if = -if_model.named_steps["iforest"].score_samples(Xi_if)

    Xi_lof = _transform_X(lof_model, X)
    s_lof = -lof_model.named_steps["lof"].score_samples(Xi_lof)

    return s_if, s_lof


def calibrate_thresholds(scores: np.ndarray, warn_q: float, block_q: float) -> Tuple[float, float]:
    warn = float(np.quantile(scores, warn_q))
    block = float(np.quantile(scores, block_q))
    return warn, block


def safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    if not np.isfinite(sigma) or sigma < 1e-9:
        sigma = 1e-9
    return mu, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to training JSONL (e.g. lcc_train.jsonl)")
    ap.add_argument("--out", default="models/unsup_bundle.joblib", help="Output bundle path")
    ap.add_argument("--contamination", type=float, default=0.03, help="Expected anomaly rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warn_q", type=float, default=0.97, help="Warn quantile on training ensemble scores")
    ap.add_argument("--block_q", type=float, default=0.995, help="Block quantile on training ensemble scores")
    ap.add_argument("--w_if", type=float, default=0.6, help="Weight for IsolationForest in ensemble")
    ap.add_argument("--w_lof", type=float, default=0.4, help="Weight for LOF in ensemble")
    args = ap.parse_args()

    if abs((args.w_if + args.w_lof) - 1.0) > 1e-6:
        raise ValueError("w_if + w_lof must equal 1.0")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(args.input)
    df = build_frame(items)

    cols = vector_columns(DEFAULT_CONFIG)
    X = df[cols].to_numpy(dtype=float)

    if_model, lof_model = fit_models(X, contamination=args.contamination, seed=args.seed)

    # Raw anomaly scores (higher = more anomalous)
    if_scores, lof_scores = anomaly_scores_from_models(if_model, lof_model, X)

    # Freeze normalization params on TRAIN only (critical!)
    if_mu, if_sigma = safe_mean_std(if_scores)
    lof_mu, lof_sigma = safe_mean_std(lof_scores)

    # Ensemble in normalized space (this is what thresholds will be calibrated on)
    ens = (
        args.w_if * ((if_scores - if_mu) / if_sigma) +
        args.w_lof * ((lof_scores - lof_mu) / lof_sigma)
    )

    warn, block = calibrate_thresholds(ens, args.warn_q, args.block_q)

    # Per-asset thresholds
    per_asset: Dict[str, Dict[str, Any]] = {}
    for atype in sorted(set(df["_asset_type"].tolist())):
        if not atype:
            continue
        mask = (df["_asset_type"] == atype)
        n = int(mask.sum())
        if n < 80:
            continue
        warn_a, block_a = calibrate_thresholds(ens[mask.to_numpy()], args.warn_q, args.block_q)
        per_asset[atype] = {"warn": float(warn_a), "block": float(block_a), "n": n}

    bundle = {
        "config": DEFAULT_CONFIG,
        "columns": cols,
        "models": {
            "iforest": if_model,
            "lof": lof_model,
        },
        "ensemble_weights": {"if": float(args.w_if), "lof": float(args.w_lof)},
        "score_norm": {
            "if": {"mu": if_mu, "sigma": if_sigma},
            "lof": {"mu": lof_mu, "sigma": lof_sigma},
        },
        "thresholds_global": {
            "warn": float(warn),
            "block": float(block),
            "warn_q": float(args.warn_q),
            "block_q": float(args.block_q),
        },
        "thresholds_per_asset_type": per_asset,
        "meta": {
            "contamination": float(args.contamination),
            "seed": int(args.seed),
            "version": "unsup_v2_normed",
        }
    }

    joblib.dump(bundle, args.out)

    print(f"Saved bundle to {args.out}")
    print(f"Global thresholds: WARN>={warn:.4f}, BLOCK>={block:.4f}")
    print(f"Score norm: IF(mu={if_mu:.6f}, sigma={if_sigma:.6f}) LOF(mu={lof_mu:.6f}, sigma={lof_sigma:.6f})")

    if per_asset:
        print("Per-asset thresholds:")
        for k, v in per_asset.items():
            print(f"  - {k}: WARN>={v['warn']:.4f}, BLOCK>={v['block']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()


