#!/usr/bin/env python3
# ml/threshold_sweep_bin.py

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def _force_local_features_module() -> None:
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    local_features_path = os.path.join(project_root, "features.py")

    if not os.path.exists(local_features_path):
        raise FileNotFoundError(f"Local features.py not found at: {local_features_path}")

    spec = importlib.util.spec_from_file_location("features", local_features_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for local features.py")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["features"] = mod
    spec.loader.exec_module(mod)  # type: ignore

    print(f"[threshold_sweep] using local features.py")


def _load_features_api():
    from features import DEFAULT_CONFIG, features_to_row  # type: ignore
    return DEFAULT_CONFIG, features_to_row


def load_dataset(path: str, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    DEFAULT_CONFIG, features_to_row = _load_features_api()

    X: List[List[float]] = []
    y: List[int] = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            feats = r.get("features", {}) or {}
            row = features_to_row(feats, cfg=DEFAULT_CONFIG)
            X.append([float(row.get(c, 0.0) or 0.0) for c in cols])
            y.append(0 if r.get("label") == "ok" else 1)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--t_lo", type=float, default=0.4)
    parser.add_argument("--t_hi", type=float, default=0.8)
    args = parser.parse_args()

    _force_local_features_module()

    bundle = joblib.load(args.model)
    cols: List[str] = bundle["cols"]

    print(f"[threshold_sweep] columns={len(cols)}")

    X, y = load_dataset(args.input, cols)
    unique, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[threshold_sweep] samples={len(y)}  class_dist={dist}")

    # Use calibrated proba if available
    calibrator = bundle.get("calibrator", None)
    if calibrator is not None:
        p = calibrator.predict_proba(X)[:, 1]
    else:
        model = bundle["model"]
        p = model.predict_proba(X)[:, 1]

    print("\n--- THRESHOLD SWEEP (predict non_ok if p>=t) ---")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pred = (p >= t).astype(int)
        cm = confusion_matrix(y, pred)
        tn, fp, fn, tp = cm.ravel()
        rec1 = tp / (tp + fn + 1e-12)
        prec1 = tp / (tp + fp + 1e-12)
        print(f"t={t:.2f}  FP={fp:4d}  FN={fn:4d}  prec1={prec1:.3f}  rec1={rec1:.3f}")

    # zones
    t_lo = args.t_lo
    t_hi = args.t_hi
    zone_ok = int(np.sum(p < t_lo))
    zone_warn = int(np.sum((p >= t_lo) & (p < t_hi)))
    zone_block = int(np.sum(p >= t_hi))
    print("\n--- ZONES ---")
    print(f"t_lo={t_lo:.2f} t_hi={t_hi:.2f}  => ok={zone_ok} warn={zone_warn} block={zone_block}")

    # report at t_lo (binary: ok vs non_ok)
    pred_lo = (p >= t_lo).astype(int)
    print("\n--- REPORT @ t_lo (treat warn+block as non_ok) ---")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, pred_lo))
    print("\nClassification report:")
    print(classification_report(y, pred_lo, digits=4))


if __name__ == "__main__":
    main()



