# eval_supervised.py

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def _find_featureconfig_file(project_root: Path) -> Path:
    """
    Cherche dans le projet un fichier .py qui contient 'class FeatureConfig'.
    On ignore .venv, site-packages, etc.
    """
    ignore_parts = {
        ".venv",
        "venv",
        "__pycache__",
        "site-packages",
        "dist-packages",
        ".git",
        ".idea",
        "node_modules",
        "build",
        "dist",
    }

    candidates: List[Path] = []
    for p in project_root.rglob("*.py"):
        parts = set(p.parts)
        if parts & ignore_parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "class FeatureConfig" in txt:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Impossible de trouver un fichier qui définit 'class FeatureConfig'. "
            "Cherché dans le projet (hors .venv/site-packages)."
        )

    # Heuristique: on préfère un fichier qui s'appelle features.py ou contient 'features' dans le nom
    candidates.sort(key=lambda x: (0 if x.name == "features.py" else 1, 0 if "feature" in x.name.lower() else 1, len(str(x))))
    return candidates[0]


def _force_local_features_module() -> None:
    """
    Force sys.modules['features'] à pointer vers ton module local (celui qui définit FeatureConfig),
    pour que joblib/pickle retrouve features.FeatureConfig au load.
    """
    project_root = Path(__file__).resolve().parents[1]
    local_path = _find_featureconfig_file(project_root)

    spec = importlib.util.spec_from_file_location("features", str(local_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import local features module from: {local_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["features"] = module
    spec.loader.exec_module(module)

    # mini sanity check
    if not hasattr(module, "FeatureConfig"):
        raise RuntimeError(f"Loaded {local_path} but FeatureConfig is missing (unexpected).")

    print(f"[eval_supervised] using local features module: {local_path}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        feats["label"] = obj.get("label")
        out.append(feats)
    return out


def _safe_float(v: Any, default: float) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def build_matrix(rows: List[Dict[str, Any]], bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    prep = bundle["prep"]
    numeric_cols: List[str] = prep["numeric_cols"]
    feature_cols: List[str] = prep["feature_columns"]
    medians: Dict[str, float] = prep.get("medians", {})

    label_map: Dict[str, int] = bundle["labels"]["map"]

    X_rows: List[np.ndarray] = []
    y: List[int] = []

    for r in rows:
        base: Dict[str, Any] = {}

        for c in numeric_cols:
            base[c] = _safe_float(r.get(c), medians.get(c, 0.0))

        base["_asset_type"] = (r.get("asset_type") or "").strip().lower()
        base["_market"] = (r.get("market") or "").strip().upper()

        df = pd.DataFrame([base])
        df = pd.get_dummies(
            df,
            columns=["_asset_type", "_market"],
            prefix=["asset", "mkt"],
            dummy_na=False,
        )

        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_cols]

        X_rows.append(df.iloc[0].to_numpy(dtype=float))
        y.append(label_map.get(r.get("label", "ok"), 0))

    return np.vstack(X_rows), np.array(y, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    _force_local_features_module()   # <= AVANT joblib.load()

    bundle = joblib.load(args.model)
    rows = load_jsonl(args.input)

    X, y_true = build_matrix(rows, bundle)

    model = bundle["models"]["xgb"]
    proba = model.predict_proba(X)
    y_pred = proba.argmax(axis=1)

    acc = float((y_pred == y_true).mean())

    print("Samples:", len(y_true))
    print("Accuracy:", round(acc, 4))

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()










