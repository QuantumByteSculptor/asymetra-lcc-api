# train_unsupervised.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

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
        items.append(feats)
    return items


# -----------------------------
# Build matrix
# -----------------------------
def build_df(feats_list: List[Dict[str, Any]], cfg: Dict[str, Any]) -> pd.DataFrame:
    cols = vector_columns(cfg)
    recs: List[Dict[str, Any]] = []
    for f in feats_list:
        row = features_to_row(f, cfg=cfg)
        recs.append(row)
    df = pd.DataFrame.from_records(recs)

    # assure toutes les colonnes (ordre stable)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].copy()
    return df


def drop_all_nan_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    all_nan = [c for c in cols if df[c].isna().all()]
    if all_nan:
        print(f"⚠️ Dropping fully-NaN columns before impute: {all_nan}")
        df = df.drop(columns=all_nan)
        cols = [c for c in cols if c not in all_nan]
    return df, cols


def compute_z_norm_params(x: np.ndarray) -> Dict[str, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-12)
    return {"mu": mu, "sigma": sigma}


def choose_thresholds(scores: np.ndarray, warn_q: float, block_q: float) -> Dict[str, float]:
    # scores = ensemble (plus grand = plus anormal chez toi)
    warn = float(np.quantile(scores, warn_q))
    block = float(np.quantile(scores, block_q))
    return {"warn": warn, "block": block}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL path for OK samples (unsup training)")
    ap.add_argument("--out", default="models/unsup_bundle.joblib", help="Output bundle path")

    # models
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--if_estimators", type=int, default=300)
    ap.add_argument("--if_contamination", type=float, default=0.01)

    ap.add_argument("--lof_neighbors", type=int, default=35)

    # thresholds (quantiles sur OK)
    ap.add_argument("--warn_q", type=float, default=0.95)
    ap.add_argument("--block_q", type=float, default=0.99)

    # ensemble
    ap.add_argument("--w_if", type=float, default=0.5)
    ap.add_argument("--w_lof", type=float, default=0.5)

    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG

    feats_list = load_jsonl(args.input)
    if not feats_list:
        raise SystemExit("No rows loaded. Check your --input JSONL.")

    # build numeric matrix
    cols = vector_columns(cfg)
    df = build_df(feats_list, cfg=cfg)

    # drop fully-NaN columns (=> supprime tes warnings)
    df, cols = drop_all_nan_columns(df, cols)

    X = df.to_numpy(dtype=float)

    # impute median
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # models
    iforest = IsolationForest(
        n_estimators=args.if_estimators,
        contamination=args.if_contamination,
        random_state=args.seed,
        n_jobs=-1,
    )

    # IMPORTANT: novelty=True pour scorer sur de nouvelles lignes en production
    lof = LocalOutlierFactor(
        n_neighbors=args.lof_neighbors,
        novelty=True,
        metric="minkowski",
    )

    iforest.fit(X_imp)
    lof.fit(X_imp)

    # raw scores
    # IF: score_samples (higher = less anomalous chez sklearn), mais toi tu z-normalises ensuite
    raw_if = iforest.score_samples(X_imp).astype(float)

    # LOF: score_samples existe avec novelty=True (higher = less anomalous)
    raw_lof = lof.score_samples(X_imp).astype(float)

    # z-normalisation
    norm_if = compute_z_norm_params(raw_if)
    norm_lof = compute_z_norm_params(raw_lof)

    z_if = (raw_if - norm_if["mu"]) / (norm_if["sigma"] + 1e-12)
    z_lof = (raw_lof - norm_lof["mu"]) / (norm_lof["sigma"] + 1e-12)

    # ensemble (ton ADN: mélange IF + LOF)
    w_if = float(args.w_if)
    w_lof = float(args.w_lof)
    s_ens = (w_if * z_if) + (w_lof * z_lof)

    # thresholds global (sur OK)
    thr_global = choose_thresholds(s_ens, args.warn_q, args.block_q)

    # thresholds per asset_type si présent dans la source (sinon global)
    per_asset: Dict[str, Dict[str, float]] = {}
    # on essaie de récupérer asset_type depuis feats_list
    by_at: Dict[str, List[float]] = {}
    for f, s in zip(feats_list, s_ens):
        at = (f.get("asset_type") or "").strip().lower()
        if not at:
            continue
        by_at.setdefault(at, []).append(float(s))

    for at, arr in by_at.items():
        a = np.array(arr, dtype=float)
        if len(a) < 50:
            continue
        per_asset[at] = choose_thresholds(a, args.warn_q, args.block_q)

    bundle = {
        "config": cfg,
        # ✅ colonnes ACTIVES (après drop des fully-NaN)
        "columns": cols,
        "models": {
            "iforest": iforest,
            "lof": lof,
        },
        "imputer": {
            # SimpleImputer ne se sérialise pas toujours bien selon versions, on stocke l’objet + stats
            "object": imputer,
            "statistics": getattr(imputer, "statistics_", None),
        },
        # ✅ keys en minuscules (évite ton bug 'IF')
        "score_norm": {
            "if": norm_if,
            "lof": norm_lof,
        },
        "ensemble_weights": {
            "if": w_if,
            "lof": w_lof,
        },
        "thresholds_global": thr_global,
        "thresholds_per_asset_type": per_asset,
        "meta": {
            "version": "unsup_v2_drop_allnan_cols",
            "n_rows": int(X_imp.shape[0]),
            "n_features": int(X_imp.shape[1]),
            "warn_q": float(args.warn_q),
            "block_q": float(args.block_q),
            "seed": int(args.seed),
        },
    }

    joblib.dump(bundle, args.out)

    print(f"Saved bundle to {args.out}")
    print(f"Global thresholds: WARN>={thr_global['warn']:.4f}, BLOCK>={thr_global['block']:.4f}")
    print(f"Score norm: IF(mu={norm_if['mu']:.6f}, sigma={norm_if['sigma']:.6f}) LOF(mu={norm_lof['mu']:.6f}, sigma={norm_lof['sigma']:.6f})")
    if per_asset:
        print("Per-asset thresholds:")
        for k in sorted(per_asset.keys()):
            t = per_asset[k]
            print(f"  - {k}: WARN>={t['warn']:.4f}, BLOCK>={t['block']:.4f} (n={len(by_at.get(k, []))})")
    else:
        print("Per-asset thresholds: none (not enough data per asset_type)")


if __name__ == "__main__":
    main()
