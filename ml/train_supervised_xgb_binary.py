# ml/train_supervised_xgb_binary.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError("xgboost is required. pip install xgboost") from e


def _force_local_features_module() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    local_features_path = os.path.join(root, "features.py")
    if not os.path.exists(local_features_path):
        raise FileNotFoundError(f"Local features.py not found at: {local_features_path}")

    print("[train_binary] using local features.py")


_force_local_features_module()
from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore


def load_dataset(path: str, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            feats = r.get("features", {}) or {}
            row = features_to_row(feats, cfg=DEFAULT_CONFIG)
            X.append([float(row.get(c, 0.0) or 0.0) for c in cols])
            y.append(0 if r.get("label") == "ok" else 1)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def build_base_model(scale_pos_weight: float) -> Any:
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_method", default="none", choices=["none", "sigmoid", "isotonic"])
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--calib_cv", type=int, default=3)
    ap.add_argument("--t_eval", type=float, default=0.50)
    args = ap.parse_args()

    cols = vector_columns(DEFAULT_CONFIG)
    print(f"[train_binary] columns={len(cols)}")

    X, y = load_dataset(args.input, cols)
    print(f"[train_binary] samples={len(y)} X={X.shape}")

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"[train_binary] class dist (0=ok,1=non_ok): {{0:{n0}, 1:{n1}}}")
    spw = (n0 / max(n1, 1))
    print(f"[train_binary] scale_pos_weight={spw:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"[train_binary] split sizes: train_full={len(y_train)} test={len(y_test)}")

    base = build_base_model(scale_pos_weight=spw)

    if args.calib_method == "none":
        print("[train_binary] fitting base model (no calibration)...")
        base.fit(X_train, y_train)
        model = base
        calibrated = False
        calib_method = None
    else:
        print(f"[train_binary] calibrating probabilities: method={args.calib_method} cv={args.calib_cv} ...")
        model = CalibratedClassifierCV(base, method=args.calib_method, cv=args.calib_cv)
        model.fit(X_train, y_train)
        calibrated = True
        calib_method = args.calib_method

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= args.t_eval).astype(int)

    print(f"\n[train_binary] HOLDOUT ({'calibrated' if calibrated else 'base'}) @ t={args.t_eval:.2f}")
    cm = confusion_matrix(y_test, pred)
    print("[train_binary] confusion matrix:")
    print(cm)
    print("\n[train_binary] classification report:")
    print(classification_report(y_test, pred, digits=4))

    bundle: Dict[str, Any] = {
        "cols": cols,
        "cfg": DEFAULT_CONFIG,
        "model": model,
        "calibrated": calibrated,
        "calib_method": calib_method,
    }
    joblib.dump(bundle, args.out)
    print(f"\n[train_binary] saved: {args.out}")


if __name__ == "__main__":
    main()










