# ml/feature_importance_bin.py
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib


def _force_local_features_module() -> None:
    """
    Ensure the project root (../) is on sys.path and that the local features.py
    is importable as module name 'features' (needed for joblib/pickle).
    """
    # project root = parent of this file's directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    local_features_path = os.path.join(root_dir, "features.py")
    if not os.path.exists(local_features_path):
        raise FileNotFoundError(f"Local features.py not found at: {local_features_path}")

    # Load it explicitly as module name "features"
    spec = importlib.util.spec_from_file_location("features", local_features_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for local features.py")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["features"] = mod  # crucial for pickle resolution
    spec.loader.exec_module(mod)   # type: ignore

    print(f"[feature_importance_bin] using local features.py: {local_features_path}")


def _get_model_and_cols(bundle: Any) -> Tuple[Any, List[str]]:
    """
    Be tolerant to bundle structure variations.
    Expected keys: model, cols/columns.
    """
    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("clf") or bundle.get("estimator")
        cols = bundle.get("cols") or bundle.get("columns") or bundle.get("feature_names")
        if model is None:
            raise KeyError("Bundle dict missing 'model' (or clf/estimator).")
        if cols is None:
            raise KeyError("Bundle dict missing 'cols' (or columns/feature_names).")
        return model, list(cols)

    # fallback: attribute-based
    model = getattr(bundle, "model", None) or getattr(bundle, "clf", None)
    cols = getattr(bundle, "cols", None) or getattr(bundle, "columns", None)
    if model is None or cols is None:
        raise TypeError("Unknown bundle format: cannot find model/cols.")
    return model, list(cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to supervised binary bundle joblib")
    parser.add_argument("--top", type=int, default=30, help="Top N features to display")
    args = parser.parse_args()

    _force_local_features_module()

    bundle = joblib.load(args.model)
    model, cols = _get_model_and_cols(bundle)

    # XGBoost sklearn wrapper: feature_importances_ exists
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError("Model has no feature_importances_. Are you sure it's an XGBClassifier?")

    pairs = list(zip(cols, importances))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)

    topn = max(1, args.top)
    print(f"[feature_importance_bin] top {topn} features:")
    for name, score in pairs[:topn]:
        print(f"{name:30s}  {float(score):.6f}")


if __name__ == "__main__":
    main()



