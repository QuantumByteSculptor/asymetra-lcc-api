# train_lcc.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

from lcc_model import features_to_vector


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_matrix(data: List[Dict[str, Any]], only_status: List[str] | None = None) -> np.ndarray:
    X = []
    for r in data:
        if only_status and r.get("status_rules") not in only_status:
            continue
        feats = r.get("features", {})
        X.append(features_to_vector(feats))
    if not X:
        raise ValueError("No samples found after filtering.")
    return np.vstack(X)


def calibrate_thresholds(scores: np.ndarray) -> tuple[float, float]:
    """
    Définir des seuils pragmatiques:
    - WARN = 95e percentile des scores anormaux
    - BLOCK = 99e percentile
    Ajuste ensuite selon ton taux de faux positifs.
    """
    warn = float(np.quantile(scores, 0.95))
    block = float(np.quantile(scores, 0.99))
    return warn, block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to lcc_runs.jsonl")
    ap.add_argument("--out", default="unsup_bundle.joblib", help="Output model path")
    ap.add_argument("--contamination", type=float, default=0.03, help="IsolationForest contamination")
    args = ap.parse_args()

    data = load_jsonl(Path(args.input))

    # ✅ On entraîne sur les runs "propres" (OK et éventuellement WARN)
    X_train = build_matrix(data, only_status=["OK"])

    # Pipeline: imputation + scaling + IsolationForest
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=300,
            contamination=args.contamination,
            random_state=42,
        )),
    ])

    model.fit(X_train)

    # IsolationForest: score_samples -> plus grand = plus normal.
    # On veut un "anomaly_score" où plus grand = plus anormal.
    normal_scores = model.named_steps["iforest"].score_samples(
        model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(X_train)
        )
    )
    anomaly_scores = -normal_scores  # ✅ inverse

    warn, block = calibrate_thresholds(anomaly_scores)

    payload = {
        "model": model,
        "threshold_warn": warn,
        "threshold_block": block,
        "meta": {
            "trained_on": len(X_train),
            "contamination": args.contamination,
        }
    }

    joblib.dump(payload, args.out)
    print(f"Saved model to {args.out}")
    print(f"Thresholds: WARN>={warn:.4f}, BLOCK>={block:.4f}")


if __name__ == "__main__":
    main()