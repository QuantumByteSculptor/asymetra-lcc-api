# ml/compare_calibration.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, log_loss, brier_score_loss


def _force_local_features_module() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    if not os.path.exists(os.path.join(root, "features.py")):
        raise FileNotFoundError("features.py not found at repo root")
    print("[compare_calibration] using local features.py")


_force_local_features_module()
from features import DEFAULT_CONFIG, features_to_row  # type: ignore


def load_xy(path: str, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            feats = r.get("features", {}) or {}
            row = features_to_row(feats, cfg=DEFAULT_CONFIG)
            X.append([float(row.get(c, 0.0) or 0.0) for c in cols])
            y.append(0 if r.get("label") == "ok" else 1)
    return np.asarray(X, float), np.asarray(y, int)


def eval_bundle(bundle_path: str, holdout_path: str, t: float) -> None:
    b: Dict[str, Any] = joblib.load(bundle_path)
    cols = b["cols"]
    model = b["model"]
    calibrated = b.get("calibrated", False)
    calib_method = b.get("calib_method", None)

    X, y = load_xy(holdout_path, cols)

    p = model.predict_proba(X)[:, 1]
    pred = (p >= t).astype(int)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    ok = int((y == 0).sum())
    non_ok = int((y == 1).sum())

    print(f"\n=== {os.path.basename(bundle_path)} ===")
    print(f"calibrated={calibrated} method={calib_method}")
    print(f"samples={len(y)} ok={ok} non_ok={non_ok}")
    print(f"log_loss={log_loss(y, p):.4f}  brier={brier_score_loss(y, p):.4f}")
    print(f"@t={t:.2f}  FP={fp} ({fp/ok:.1%} of OK)  FN={fn}  TP={tp}  TN={tn}")
    print("confusion matrix:\n", cm)
    print("\nreport:\n", classification_report(y, pred, digits=4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmoid", required=True)
    ap.add_argument("--isotonic", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--t", type=float, default=0.50)
    args = ap.parse_args()

    eval_bundle(args.sigmoid, args.holdout, args.t)
    eval_bundle(args.isotonic, args.holdout, args.t)


if __name__ == "__main__":
    main()




