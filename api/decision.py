# api/decision.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

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

def _load_bin_bundle_safe(bundle_path: str) -> Dict[str, Any]:
    """Charge un bundle joblib et normalise les clés possibles."""
    info: Dict[str, Any] = {
        "bundle_loaded": False,
        "bundle_calibrated": False,
        "calib_method": None,
        "n_cols": 0,
        "cols_preview": [],
        "bundle_meta": {},
    }
    try:
        if not bundle_path or not os.path.exists(bundle_path):
            return info

        obj = joblib.load(bundle_path)
        info["bundle_loaded"] = True

        if isinstance(obj, dict):
            cols = obj.get("cols") or obj.get("columns") or obj.get("columns_") or obj.get("columns_list")
            if not cols and isinstance(obj.get("cfg"), dict):
                cols = obj["cfg"].get("cols") or obj["cfg"].get("columns")

            if isinstance(cols, (list, tuple)):
                info["n_cols"] = len(cols)
                info["cols_preview"] = list(cols)[:20]
            else:
                try:
                    info["n_cols"] = int(obj.get("n_cols") or obj.get("ncols") or 0)
                except Exception:
                    info["n_cols"] = 0

            info["bundle_calibrated"] = bool(obj.get("calibrated") or obj.get("bundle_calibrated") or False)
            info["calib_method"] = obj.get("calib_method") or None

            # meta safe
            info["bundle_meta"] = {k: obj.get(k) for k in ("cfg", "meta", "version") if k in obj}

        return info

    except Exception as e:
        info["bundle_loaded"] = False
        info["bundle_meta"] = {"load_error": str(e)}
        return info


def _load_thresholds_safe(thresholds_path: str) -> Dict[str, Any]:
    """Charge thresholds json et normalise t_lo/t_hi/alpha."""
    out: Dict[str, Any] = {
        "path": thresholds_path,
        "loaded": False,
        "t_lo": None,
        "t_hi": None,
        "alpha": None,
        "fallback_used": True,
    }
    try:
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path, "r", encoding="utf-8") as f:
                j = json.load(f)

            if isinstance(j, dict):
                t_lo = j.get("t_lo")
                t_hi = j.get("t_hi")
                alpha = j.get("alpha")

                if t_lo is None and isinstance(j.get("thresholds"), dict):
                    t_lo = j["thresholds"].get("t_lo")
                    t_hi = j["thresholds"].get("t_hi")
                    alpha = j["thresholds"].get("alpha", alpha)

                out.update({"loaded": True, "t_lo": t_lo, "t_hi": t_hi, "alpha": alpha, "fallback_used": False})
        return out

    except Exception as e:
        out["loaded"] = False
        out["error"] = str(e)
        return out

# -----------------------------
def _load_bin_bundle() -> Dict[str, Any]:
    """
    Loads models/bin_sigmoid.joblib (cached).
    Expected shape (flexible):
      - may include: model, cols (or columns), calibrated, calib_method, alpha, config
    """
    global _BIN_BUNDLE
    if _BIN_BUNDLE is not None:
        return _BIN_BUNDLE

    p = Path(BIN_BUNDLE_PATH)
    if not p.exists():
        _BIN_BUNDLE = {}
        return _BIN_BUNDLE

    _BIN_BUNDLE = joblib.load(p)
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

    # Predict probability of NON-OK (assume binary proba [OK, NONOK] or [NONOK, OK] unknown)
    p_non_ok: Optional[float] = None
    try:
        proba = model.predict_proba(X)[0]
        proba = np.asarray(proba, dtype=float)

        # Heuristic:
        # - If 2 classes: take max of last column as non-ok if bundle says so, else assume class1 is non-ok.
        # If bundle includes "non_ok_index", use it.
        non_ok_index = b.get("non_ok_index")
        if isinstance(non_ok_index, int) and 0 <= non_ok_index < len(proba):
            p_non_ok = float(proba[non_ok_index])
        else:
            if len(proba) == 2:
                p_non_ok = float(proba[1])
            else:
                # if multi-class, treat "non-ok" as 1 - P(OK) if "ok_index" exists, else max non-0
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



