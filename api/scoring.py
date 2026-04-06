# api/scoring.py
"""
Expert model loading and scoring layer.

Provides lazy-loaded per-asset-type expert bundles (IF+LOF unsup + XGB sup_bin).
Falls back to global_bundle.joblib if no expert exists for the given asset_type.

Feature flag: EXPERTS_ENABLED env var (default "0" = OFF).
When disabled, score_expert() returns None and no bundles are loaded.

Public API:
    is_experts_enabled()            -> bool
    score_expert(feats, asset_type) -> Optional[Dict]
    list_loaded_experts()           -> List[str]
    preload_experts()               -> None  (optional warm-up)
    experts_health()                -> Dict   (structured status for /health)
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Ensure repo root on sys.path so we can import features.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import DEFAULT_CONFIG, features_to_row  # type: ignore

# ---------------------------------------------------------------------------
# Feature flag: EXPERTS_ENABLED (default OFF)
# ---------------------------------------------------------------------------
def is_experts_enabled() -> bool:
    """Return True unless EXPERTS_ENABLED env var is explicitly set to '0'."""
    return os.getenv("EXPERTS_ENABLED", "1").strip() not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Module-level expert cache (lazy, populated on first use)
# ---------------------------------------------------------------------------
_EXPERT_CACHE: Dict[str, Any] = {}   # asset_type → bundle or None
_EXPERTS_DIR: str = os.getenv("EXPERTS_DIR", "models/experts")


def _expert_path(asset_type: str) -> Path:
    return Path(_EXPERTS_DIR) / f"{asset_type}_bundle.joblib"


def _load_bundle(path: Path) -> Optional[Dict[str, Any]]:
    try:
        b = joblib.load(path)
        if not isinstance(b, dict) or "cols" not in b:
            logger.warning("scoring: invalid bundle at %s (missing 'cols')", path)
            return None
        return b
    except Exception as exc:
        logger.warning("scoring: failed to load %s — %s", path, exc)
        return None


def load_expert_bundle(asset_type: str) -> Optional[Dict[str, Any]]:
    """
    Return the expert bundle for asset_type (with cache).
    Falls back to global_bundle.  Returns None if nothing is available.
    Returns None immediately if EXPERTS_ENABLED != 1.
    """
    if not is_experts_enabled():
        return None

    at = (asset_type or "").strip().lower()

    if at in _EXPERT_CACHE:
        return _EXPERT_CACHE[at]

    # Try specific expert first
    specific = _expert_path(at)
    if specific.exists():
        b = _load_bundle(specific)
        if b is not None:
            logger.info("scoring: loaded expert bundle for '%s' from %s", at, specific)
            _EXPERT_CACHE[at] = b
            return b

    # Fallback: global bundle
    global_path = _expert_path("global")
    if "global" not in _EXPERT_CACHE:
        gb = _load_bundle(global_path) if global_path.exists() else None
        _EXPERT_CACHE["global"] = gb
        if gb is not None:
            logger.info("scoring: loaded global fallback bundle from %s", global_path)

    _EXPERT_CACHE[at] = _EXPERT_CACHE.get("global")  # may be None
    return _EXPERT_CACHE[at]


def list_loaded_experts() -> List[str]:
    """Return list of expert asset_types currently cached (excludes None entries)."""
    return [k for k, v in _EXPERT_CACHE.items() if v is not None]


def preload_experts() -> None:
    """Eagerly load all bundles found in EXPERTS_DIR. No-op if disabled."""
    if not is_experts_enabled():
        logger.info("scoring: EXPERTS_ENABLED=0, skipping preload")
        return
    experts_dir = Path(_EXPERTS_DIR)
    if not experts_dir.is_dir():
        logger.info("scoring: experts dir not found (%s), skipping preload", experts_dir)
        return
    for p in sorted(experts_dir.glob("*_bundle.joblib")):
        at = p.stem.replace("_bundle", "")
        if at not in _EXPERT_CACHE:
            b = _load_bundle(p)
            _EXPERT_CACHE[at] = b


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _build_feature_vector(feats: Dict[str, Any], cols: List[str]) -> np.ndarray:
    """Build a 1×N feature matrix from a features dict, handling NaN → 0."""
    row = features_to_row(feats, cfg=DEFAULT_CONFIG)
    vec = np.array([float(row.get(c, 0.0) or 0.0) for c in cols], dtype=float)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)


def _unsup_score(bundle: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    """Compute IF+LOF ensemble score and status from unsup sub-bundle."""
    unsup = bundle.get("unsup", {})
    iforest = unsup.get("iforest")
    lof = unsup.get("lof")
    score_norm = unsup.get("score_norm", {})
    weights = unsup.get("weights", {"if": 0.5, "lof": 0.5})
    thresholds = unsup.get("thresholds", {})

    if iforest is None or lof is None:
        return {}

    imputer = unsup.get("imputer")
    X_imp = imputer.transform(X) if imputer is not None else X

    raw_if = float(iforest.score_samples(X_imp)[0])
    raw_lof = float(lof.score_samples(X_imp)[0])

    mu_if = score_norm.get("if", {}).get("mu", 0.0)
    sg_if = score_norm.get("if", {}).get("sigma", 1.0) or 1.0
    mu_lof = score_norm.get("lof", {}).get("mu", 0.0)
    sg_lof = score_norm.get("lof", {}).get("sigma", 1.0) or 1.0

    z_if = (raw_if - mu_if) / sg_if
    z_lof = (raw_lof - mu_lof) / sg_lof

    w_if = float(weights.get("if", 0.5))
    w_lof = float(weights.get("lof", 0.5))
    score = w_if * z_if + w_lof * z_lof

    thr_warn = thresholds.get("warn", float("inf"))
    thr_block = thresholds.get("block", float("inf"))

    if score >= thr_block:
        status = "block"
    elif score >= thr_warn:
        status = "warn"
    else:
        status = "ok"

    return {
        "unsup_score": round(score, 6),
        "unsup_status": status,
        "raw_if": round(raw_if, 6),
        "raw_lof": round(raw_lof, 6),
        "z_if": round(z_if, 6),
        "z_lof": round(z_lof, 6),
    }


def _sup_bin_score(bundle: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    """Compute XGB binary probability and status from sup_bin sub-bundle."""
    sup = bundle.get("sup_bin", {})
    model = sup.get("model")
    thresholds = sup.get("thresholds", {})

    if model is None:
        return {}

    try:
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.warning("scoring: sup_bin predict_proba failed — %s", exc)
        return {}

    t_lo = thresholds.get("t_lo", 0.5)
    t_hi = thresholds.get("t_hi", 0.8)

    if prob >= t_hi:
        status = "block"
    elif prob >= t_lo:
        status = "warn"
    else:
        status = "ok"

    return {
        "expert_prob_non_ok": round(prob, 6),
        "expert_status": status,
        "expert_t_lo": round(t_lo, 4),
        "expert_t_hi": round(t_hi, 4),
    }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def score_expert(
    feats: Dict[str, Any],
    asset_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Score a features dict using the appropriate expert bundle.

    Returns a dict with keys:
        expert_loaded       : bool
        expert_asset_type   : str  (which expert was used, may be 'global')
        expert_status       : 'ok'|'warn'|'block'
        expert_prob_non_ok  : float
        unsup_score, unsup_status, raw_if, raw_lof, z_if, z_lof

    Returns None only if no bundle at all is available.
    """
    bundle = load_expert_bundle(asset_type)
    if bundle is None:
        return None

    cols = bundle.get("cols", [])
    if not cols:
        return None

    X = _build_feature_vector(feats, cols)

    result: Dict[str, Any] = {
        "expert_loaded": True,
        "expert_asset_type": bundle.get("asset_type", "global"),
        "feature_version": bundle.get("feature_version", "v1"),
    }

    try:
        result.update(_unsup_score(bundle, X))
    except Exception as exc:
        logger.warning("scoring: unsup scoring failed — %s", exc)

    try:
        result.update(_sup_bin_score(bundle, X))
    except Exception as exc:
        logger.warning("scoring: sup_bin scoring failed — %s", exc)

    # Ensure expert_status is always present
    if "expert_status" not in result:
        result["expert_status"] = result.get("unsup_status", "ok")

    return result


# ---------------------------------------------------------------------------
# Health / observability
# ---------------------------------------------------------------------------

def experts_health() -> Dict[str, Any]:
    """
    Structured status dict for /health endpoint.

    Always returns a stable shape regardless of enabled/disabled state.
    """
    enabled = is_experts_enabled()
    experts_dir = Path(_EXPERTS_DIR)
    bundle_paths: Dict[str, str] = {}
    bundle_exists: Dict[str, bool] = {}

    # Scan for bundle files on disk (even if disabled — useful for ops)
    if experts_dir.is_dir():
        for p in sorted(experts_dir.glob("*_bundle.joblib")):
            at = p.stem.replace("_bundle", "")
            bundle_paths[at] = str(p)
            bundle_exists[at] = True

    loaded = list_loaded_experts() if enabled else []

    # Per-bundle detail
    bundles_detail: Dict[str, Dict[str, Any]] = {}
    for at, path_str in bundle_paths.items():
        detail: Dict[str, Any] = {"path": path_str, "exists": True, "loaded": at in loaded}
        if at in _EXPERT_CACHE and _EXPERT_CACHE[at] is not None:
            b = _EXPERT_CACHE[at]
            detail["n_cols"] = len(b.get("cols", []))
            detail["feature_version"] = b.get("feature_version", "unknown")
            detail["has_unsup"] = "unsup" in b and b["unsup"] is not None
            detail["has_sup_bin"] = "sup_bin" in b and b["sup_bin"] is not None
            sup = b.get("sup_bin", {})
            detail["thresholds_loaded"] = bool(sup.get("thresholds"))
            detail["n_train"] = b.get("meta", {}).get("n_train", None)
        bundles_detail[at] = detail

    return {
        "enabled": enabled,
        "experts_dir": str(experts_dir),
        "bundles_on_disk": list(bundle_paths.keys()),
        "bundles_loaded": loaded,
        "n_bundles_on_disk": len(bundle_paths),
        "n_bundles_loaded": len(loaded),
        "bundles": bundles_detail,
    }
