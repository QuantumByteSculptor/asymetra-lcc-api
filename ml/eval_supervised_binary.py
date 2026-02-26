# ml/eval_supervised_binary.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _bin_label(label: str) -> int:
    return 0 if label == "ok" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    bundle: Dict[str, Any] = joblib.load(args.model)
    clf = bundle["model"]
    cols: List[str] = bundle.get("columns") or vector_columns(DEFAULT_CONFIG)
    thr = float(args.threshold if args.threshold is not None else bundle.get("threshold", 0.5))

    X_rows: List[List[float]] = []
    y: List[int] = []

    for rec in _iter_jsonl(args.input):
        label = rec.get("label")
        feats = rec.get("features") or {}
        if label not in ("ok", "warn", "block"):
            continue
        row = features_to_row(feats, cols, DEFAULT_CONFIG)
        if row is None:
            continue
        X_rows.append(row)
        y.append(_bin_label(label))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y, dtype=int)

    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    acc = float((pred == y).mean())

    # confusion matrix rows=true cols=pred
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    # metrics
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]

    prec = float(tp / max(1, (tp + fp)))
    rec = float(tp / max(1, (tp + fn)))
    f1 = float((2 * prec * rec) / max(1e-12, (prec + rec)))

    print(f"Samples: {len(y)}  threshold={thr:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision(non_ok): {prec:.4f}")
    print(f"Recall(non_ok): {rec:.4f}")
    print(f"F1(non_ok): {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()

