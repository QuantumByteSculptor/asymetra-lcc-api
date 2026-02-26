#!/usr/bin/env python3
# ml/optimize_threshold.py

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

# --- local features.py loader (same pattern as your other scripts) ---
def _force_local_features_module() -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    local_features_path = root / "features.py"
    if not local_features_path.exists():
        raise FileNotFoundError(f"Local features.py not found at: {local_features_path}")

    spec = importlib.util.spec_from_file_location("features", str(local_features_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["features"] = mod  # critical for pickle + dataclasses
    spec.loader.exec_module(mod)  # type: ignore

    print(f"[optimize_threshold] using local features.py: {local_features_path}")


_force_local_features_module()
from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore  # noqa: E402


# -----------------------------
# Bundle interface (robust)
# -----------------------------
def _get_model_and_cols(bundle: Any) -> Tuple[Any, List[str]]:
    """
    Supports multiple bundle dict/dataclass shapes:
      - {"model": clf, "cols": [...]}
      - {"clf": clf, "cols": [...]}
      - dataclass with .model/.clf and .cols
    """
    cols = None
    model = None

    if isinstance(bundle, dict):
        cols = bundle.get("cols") or bundle.get("columns")
        model = bundle.get("model") or bundle.get("clf") or bundle.get("estimator")
    else:
        cols = getattr(bundle, "cols", None) or getattr(bundle, "columns", None)
        model = getattr(bundle, "model", None) or getattr(bundle, "clf", None) or getattr(bundle, "estimator", None)

    if cols is None:
        cols = vector_columns(DEFAULT_CONFIG)

    if model is None:
        # Sometimes the bundle itself *is* the estimator
        model = bundle

    return model, list(cols)


# -----------------------------
# Data loading
# -----------------------------
def load_dataset_jsonl(path: str, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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


# -----------------------------
# Threshold search
# -----------------------------
@dataclass
class SweepBest:
    t: float
    fp: int
    fn: int
    tp: int
    tn: int
    prec1: float
    recall1: float
    fpr_ok: float  # FP / OK_count


def optimize_threshold_under_fp(
    p1: np.ndarray,
    y: np.ndarray,
    alpha: float,
    t_min: float = 0.01,
    t_max: float = 0.99,
    step: float = 0.001,
) -> SweepBest:
    """
    Choose threshold t that maximizes recall(class=1) under constraint FP/OK <= alpha.
    Break ties by higher precision(class=1), then lower FP.
    """
    ok_mask = (y == 0)
    ok_count = int(ok_mask.sum())
    if ok_count <= 0:
        raise ValueError("No OK samples in input; cannot constrain FP/OK.")

    best: SweepBest | None = None

    # scan thresholds
    t = t_min
    while t <= t_max + 1e-12:
        pred = (p1 >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr_ok = fp / (ok_count + 1e-12)
        if fpr_ok <= alpha + 1e-12:
            prec1 = tp / (tp + fp + 1e-12)
            recall1 = tp / (tp + fn + 1e-12)

            cand = SweepBest(
                t=float(t),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                tn=int(tn),
                prec1=float(prec1),
                recall1=float(recall1),
                fpr_ok=float(fpr_ok),
            )

            if best is None:
                best = cand
            else:
                # primary: higher recall1
                if cand.recall1 > best.recall1 + 1e-12:
                    best = cand
                # tie: higher precision
                elif abs(cand.recall1 - best.recall1) <= 1e-12 and cand.prec1 > best.prec1 + 1e-12:
                    best = cand
                # tie: lower FP
                elif (
                    abs(cand.recall1 - best.recall1) <= 1e-12
                    and abs(cand.prec1 - best.prec1) <= 1e-12
                    and cand.fp < best.fp
                ):
                    best = cand

        t += step

    if best is None:
        # No threshold satisfies constraint -> pick the one with minimum FP rate
        # (pragmatic fallback)
        best_t = 0.99
        pred = (p1 >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        ok_count = int((y == 0).sum())
        prec1 = tp / (tp + fp + 1e-12)
        recall1 = tp / (tp + fn + 1e-12)
        best = SweepBest(
            t=float(best_t),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            tn=int(tn),
            prec1=float(prec1),
            recall1=float(recall1),
            fpr_ok=float(fp / (ok_count + 1e-12)),
        )

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to joblib bundle (or estimator).")
    ap.add_argument("--input", required=True, help="JSONL with {'label', 'features'}")
    ap.add_argument("--alpha", type=float, default=0.25, help="Constraint: FP/OK <= alpha")
    ap.add_argument("--t_min", type=float, default=0.01)
    ap.add_argument("--t_max", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.001)
    ap.add_argument("--t_hi", type=float, default=0.85, help="Suggested high threshold for block zone.")
    ap.add_argument("--out_json", default=None, help="Write chosen thresholds + metrics to this JSON file.")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, cols = _get_model_and_cols(bundle)

    X, y = load_dataset_jsonl(args.input, cols)

    # probabilities
    if hasattr(model, "predict_proba"):
        p1 = model.predict_proba(X)[:, 1]
    else:
        # fallback: decision function -> sigmoid
        if not hasattr(model, "decision_function"):
            raise ValueError("Model has neither predict_proba nor decision_function.")
        z = model.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-z))

    best = optimize_threshold_under_fp(
        p1=p1,
        y=y,
        alpha=args.alpha,
        t_min=args.t_min,
        t_max=args.t_max,
        step=args.step,
    )

    ok_count = int((y == 0).sum())

    print("\n--- OPTIMAL THRESHOLD UNDER CONSTRAINT ---")
    print(f"alpha(FP/OK) <= {args.alpha}")
    print(
        f"t*={best.t:.3f}  FP={best.fp} ({best.fpr_ok*100:.1f}% of OK={ok_count})  "
        f"FN={best.fn}  prec1={best.prec1:.3f}  rec1={best.recall1:.3f}"
    )

    if args.out_json:
        out = {
            "t_lo": float(best.t),
            "t_hi": float(args.t_hi),
            "alpha": float(args.alpha),
            "metrics": {
                "fp": int(best.fp),
                "fn": int(best.fn),
                "tp": int(best.tp),
                "tn": int(best.tn),
                "precision_non_ok": float(best.prec1),
                "recall_non_ok": float(best.recall1),
                "fp_rate_ok": float(best.fpr_ok),
                "ok_count": int(ok_count),
                "n_samples": int(len(y)),
            },
            "model_path": str(args.model),
            "input_path": str(args.input),
            "columns": cols,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()








