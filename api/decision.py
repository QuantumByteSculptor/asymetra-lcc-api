# api/decision.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

# ------------------------------------------------------------------
# Ensure repo root is on sys.path so we import local features.py
# (avoid collision with pip package named "features")
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse your feature builder (same as api.main)
from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore

# -----------------------------
# Config
# -----------------------------
BIN_ENABLED = os.getenv("BIN_ENABLED", "1").strip() not in ("0", "false", "False")
BIN_BUNDLE_PATH = os.getenv("BIN_BUNDLE_PATH", "models/bin_sigmoid.joblib")
BIN_THRESHOLDS_PATH = os.getenv("BIN_THRESHOLDS_PATH", "models/threshold_sigmoid.json")
BIN_T_HI_DEFAULT = float(os.getenv("BIN_T_HI_DEFAULT", "0.85"))

DEBUG_RESPONSE = os.getenv("DEBUG_RESPONSE", "0").strip() in ("1", "true", "True")

# -----------------------------
# Caches
# -----------------------------
_BIN_BUNDLE: Optional[Dict[str, Any]] = None
_THRESHOLDS: Optional[Dict[str, Any]] = None

# -----------------------------
# Small utils
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _compact_thresholds_source(src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enlève les gros champs (ex: raw.columns) pour éviter des réponses JSON énormes.
    """
    if not isinstance(src, dict):
        return {"path": None, "fallback_used": False}

    out: Dict[str, Any] = {
        "path": src.get("path"),
        "fallback_used": bool(src.get("fallback_used", False)),
    }

    raw = src.get("raw")
    if isinstance(raw, dict):
        raw2 = dict(raw)
        # kill huge payloads
        raw2.pop("columns", None)
        out["raw"] = raw2

    return out


def _compact_decision(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return d

    out: Dict[str, Any] = {
        "p_non_ok": d.get("p_non_ok"),
        "decision": d.get("decision"),
        "thresholds": d.get("thresholds"),
        "thresholds_source": _compact_thresholds_source(d.get("thresholds_source", {})),
    }
    if DEBUG_RESPONSE:
        out["debug"] = d.get("debug")
    return out


# -----------------------------
# Loaders (used by /health too)
# -----------------------------
def _load_bin_bundle() -> Dict[str, Any]:
    """
    Loads models/bin_sigmoid.joblib (cached).
    Expected shape (flexible):
      - may include: model, cols (or columns), calibrated, calib_method, alpha, config
    NEVER crashes: if missing/corrupt, returns {}.
    """
    global _BIN_BUNDLE
    if _BIN_BUNDLE is not None:
        return _BIN_BUNDLE

    p = Path(BIN_BUNDLE_PATH)
    if not p.exists():
        _BIN_BUNDLE = {}
        return _BIN_BUNDLE

    try:
        _BIN_BUNDLE = joblib.load(p)
    except Exception:
        _BIN_BUNDLE = {}
        return _BIN_BUNDLE

    if not isinstance(_BIN_BUNDLE, dict):
        # normalize
        _BIN_BUNDLE = {"model": _BIN_BUNDLE}

    return _BIN_BUNDLE


def _load_thresholds() -> Dict[str, Any]:
    """
    Loads thresholds json (cached). Returns a normalized dict:
      {
        "t_lo": float,
        "t_hi": float,
        "raw": dict|None,
        "path": str,
        "fallback_used": bool
      }
    """
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS

    p = Path(BIN_THRESHOLDS_PATH)
    if not p.exists():
        _THRESHOLDS = {
            "t_lo": 0.5,
            "t_hi": BIN_T_HI_DEFAULT,
            "raw": None,
            "path": str(p),
            "fallback_used": True,
        }
        return _THRESHOLDS

    raw = json.loads(p.read_text(encoding="utf-8"))
    t_lo = float(raw.get("t_lo", 0.5))
    t_hi = float(raw.get("t_hi", BIN_T_HI_DEFAULT))

    _THRESHOLDS = {
        "t_lo": t_lo,
        "t_hi": t_hi,
        "raw": raw,
        "path": str(p),
        "fallback_used": False,
    }
    return _THRESHOLDS


# -----------------------------
# Vectorization (robust)
# -----------------------------
def _vector_for_cols(feats: Dict[str, Any], cols: list[str], cfg: Dict[str, Any]) -> np.ndarray:
    """
    Builds a 1xN numeric vector for the bin model.
    Uses features_to_row output; missing/non-numeric -> 0.0.
    """
    row_dict = features_to_row(feats, cfg=cfg)

    row: list[float] = []
    for c in cols:
        v = row_dict.get(c, None)

        # allow aliasing max_dd / max_drawdown in case columns use either
        if v is None and c == "max_dd":
            v = row_dict.get("max_drawdown", None)
        if v is None and c == "max_drawdown":
            v = row_dict.get("max_dd", None)

        fv = _safe_float(v)
        row.append(float(fv) if fv is not None else 0.0)

    X = np.asarray([row], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# -----------------------------
# Public API
# -----------------------------
def decide(feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "p_non_ok": float,
        "decision": "OK"|"WARN"|"BLOCK",
        "thresholds": {"t_lo":..., "t_hi":...},
        "thresholds_source": {...},
        "debug": {...}   # only if DEBUG_RESPONSE=1
      }
    """
    if not BIN_ENABLED:
        return _compact_decision(
            {
                "p_non_ok": None,
                "decision": "OK",
                "thresholds": {"t_lo": 0.5, "t_hi": BIN_T_HI_DEFAULT},
                "thresholds_source": {"path": BIN_THRESHOLDS_PATH, "fallback_used": True, "raw": None},
                "debug": {"disabled": True},
            }
        )

    b = _load_bin_bundle()
    thr = _load_thresholds()

    # Resolve columns
    cols = b.get("cols") or b.get("columns")
    if not cols:
        # fallback to feature builder columns (should be rare)
        cfg = b.get("config", DEFAULT_CONFIG)
        cols = vector_columns(cfg)
    cols = list(cols)

    # Resolve model
    # (bundle may store under "model" OR under "models":{"bin":...})
    model = b.get("model")
    if model is None:
        model = (b.get("models") or {}).get("bin") or (b.get("models") or {}).get("model")
    if model is None:
        return _compact_decision(
            {
                "p_non_ok": None,
                "decision": "OK",
                "thresholds": {"t_lo": float(thr.get("t_lo", 0.5)), "t_hi": float(thr.get("t_hi", BIN_T_HI_DEFAULT))},
                "thresholds_source": {"path": thr.get("path"), "fallback_used": True, "raw": thr.get("raw")},
                "debug": {"error": "missing model in bin bundle"},
            }
        )

    cfg = b.get("config", DEFAULT_CONFIG)
    X = _vector_for_cols(feats, cols, cfg=cfg)

    # Predict probability of NON-OK robustly
    p_non_ok: Optional[float] = None
    try:
        proba = model.predict_proba(X)[0]
        proba = np.asarray(proba, dtype=float)

        # Preferred: explicit index from bundle
        non_ok_index = b.get("non_ok_index")
        if isinstance(non_ok_index, int) and 0 <= non_ok_index < len(proba):
            p_non_ok = float(proba[non_ok_index])
        else:
            # Next best: use model.classes_ if present
            if hasattr(model, "classes_"):
                classes = list(getattr(model, "classes_"))
                # Try common label conventions first
                if "NON_OK" in classes:
                    p_non_ok = float(proba[classes.index("NON_OK")])
                elif "non_ok" in classes:
                    p_non_ok = float(proba[classes.index("non_ok")])
                elif 1 in classes:
                    p_non_ok = float(proba[classes.index(1)])
                else:
                    # fallback heuristic
                    if len(proba) == 2:
                        p_non_ok = float(proba[1])
                    else:
                        ok_index = b.get("ok_index")
                        if isinstance(ok_index, int) and 0 <= ok_index < len(proba):
                            p_non_ok = float(1.0 - proba[ok_index])
                        else:
                            p_non_ok = float(np.max(proba[1:])) if len(proba) > 1 else float(proba[0])
            else:
                # Final fallback heuristic
                if len(proba) == 2:
                    p_non_ok = float(proba[1])
                else:
                    ok_index = b.get("ok_index")
                    if isinstance(ok_index, int) and 0 <= ok_index < len(proba):
                        p_non_ok = float(1.0 - proba[ok_index])
                    else:
                        p_non_ok = float(np.max(proba[1:])) if len(proba) > 1 else float(proba[0])

    except Exception as e:
        return _compact_decision(
            {
                "p_non_ok": None,
                "decision": "OK",
                "thresholds": {"t_lo": float(thr.get("t_lo", 0.5)), "t_hi": float(thr.get("t_hi", BIN_T_HI_DEFAULT))},
                "thresholds_source": {"path": thr.get("path"), "fallback_used": True, "raw": thr.get("raw")},
                "debug": {"predict_error": f"{type(e).__name__}: {e}"},
            }
        )

    t_lo = float(thr.get("t_lo", 0.5))
    t_hi = float(thr.get("t_hi", BIN_T_HI_DEFAULT))

    decision_str = "OK"
    if p_non_ok is not None:
        if p_non_ok >= t_hi:
            decision_str = "BLOCK"
        elif p_non_ok >= t_lo:
            decision_str = "WARN"

    thresholds_source = {
        "path": thr.get("path", BIN_THRESHOLDS_PATH),
        "fallback_used": bool(thr.get("fallback_used", False)),
        "raw": thr.get("raw"),
    }

    debug_block = {
        "bundle_path": BIN_BUNDLE_PATH,
        "calibrated": bool(b.get("calibrated", False)),
        "calib_method": b.get("calib_method"),
        "n_cols": len(cols),
    }

    return _compact_decision(
        {
            "p_non_ok": float(p_non_ok) if p_non_ok is not None else None,
            "decision": decision_str,
            "thresholds": {"t_lo": t_lo, "t_hi": t_hi},
            "thresholds_source": thresholds_source,
            "debug": debug_block,
        }
    )




