# features.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# -----------------------------
# Feature spec (stable ordering)
# -----------------------------

NUMERIC_BASE: List[str] = [
    "vol_ann",
    "var95",
    "var99",
    "es95",
    "es99",
    "max_dd",
    "tuw_pct",
    "n_used",
    "missing_pct",
    "tail_obs_99",
]

# Derived features (ratios/consistency/quality)
DERIVED: List[str] = [
    "var99_div_var95",
    "es99_div_es95",
    "es95_div_var95",
    "es99_div_var99",
    "vol_to_var95_scaled",     # vol_ann / (var95*sqrt(252))
    "var95_vs_vol_daily",      # var95 / (vol_ann/sqrt(252))
    "dd_to_var99",
    "tuw_per_dd",
    "tail_obs_99_ratio",
    "missing_pct_clamped",
]

# Categorical one-hot (keep small; expand later)
ASSET_TYPES: List[str] = ["equity", "etf", "bond", "fx", "commodity", "crypto"]
MARKETS: List[str] = [
    "US", "EU", "UK", "JP", "HK", "CN", "IN", "AU", "CA", "BR", "MX", "ZA",
    "G10", "EM",
    # commodity/cross markets that sometimes appear in your synthetic data
    "XAU", "WTI", "BRENT", "NG", "HG", "SILVER", "WHEAT", "CORN", "COFFEE",
    "SOYBEAN", "SUGAR", "COTTON", "ALUMINUM", "ALT", "BTC", "ETH",
]

EPS = 1e-12


@dataclass(frozen=True)
class FeatureConfig:
    numeric_base: Tuple[str, ...] = tuple(NUMERIC_BASE)
    derived: Tuple[str, ...] = tuple(DERIVED)
    asset_types: Tuple[str, ...] = tuple(ASSET_TYPES)
    markets: Tuple[str, ...] = tuple(MARKETS)

    @property
    def numeric_columns(self) -> List[str]:
        return list(self.numeric_base) + list(self.derived)

    @property
    def onehot_columns(self) -> List[str]:
        return [f"asset_type__{a}" for a in self.asset_types] + [f"market__{m}" for m in self.markets]

    @property
    def all_columns(self) -> List[str]:
        return self.numeric_columns + self.onehot_columns


DEFAULT_CONFIG = FeatureConfig()


def _safe_div(a: float, b: float) -> float:
    if b is None or abs(b) < EPS:
        return float("nan")
    return a / b


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_derived(feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived features from raw base metrics.
    All returns are floats; may contain NaN if insufficient data.
    """
    vol_ann = _to_float(feats.get("vol_ann"))
    var95 = _to_float(feats.get("var95"))
    var99 = _to_float(feats.get("var99"))
    es95 = _to_float(feats.get("es95"))
    es99 = _to_float(feats.get("es99"))
    max_dd = _to_float(feats.get("max_dd"))
    tuw_pct = _to_float(feats.get("tuw_pct"))
    n_used = _to_float(feats.get("n_used"))
    missing_pct = _to_float(feats.get("missing_pct"))
    tail_obs_99 = _to_float(feats.get("tail_obs_99"))

    sqrt252 = math.sqrt(252.0)

    vol_daily = _safe_div(vol_ann, sqrt252)  # approx
    vol_to_var95_scaled = _safe_div(vol_ann, var95 * sqrt252)  # should be O(1-10)
    var95_vs_vol_daily = _safe_div(var95, vol_daily)           # should be O(0.5-3) depending on tails

    out = {
        "var99_div_var95": _safe_div(var99, var95),
        "es99_div_es95": _safe_div(es99, es95),
        "es95_div_var95": _safe_div(es95, var95),
        "es99_div_var99": _safe_div(es99, var99),
        "vol_to_var95_scaled": vol_to_var95_scaled,
        "var95_vs_vol_daily": var95_vs_vol_daily,
        "dd_to_var99": _safe_div(max_dd, var99),
        "tuw_per_dd": _safe_div(tuw_pct, max_dd),
        "tail_obs_99_ratio": _safe_div(tail_obs_99, n_used),
        "missing_pct_clamped": float("nan") if math.isnan(missing_pct) else max(0.0, min(100.0, missing_pct)),
    }
    return out


def onehot(asset_type: str | None, market: str | None, cfg: FeatureConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    d: Dict[str, float] = {}
    at = (asset_type or "").strip().lower()
    mk = (market or "").strip().upper()

    for a in cfg.asset_types:
        d[f"asset_type__{a}"] = 1.0 if at == a else 0.0
    for m in cfg.markets:
        d[f"market__{m}"] = 1.0 if mk == m else 0.0
    return d


def features_to_row(feats: Dict[str, Any], cfg: FeatureConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Turn raw feature dict -> full row dict (numeric + derived + onehot).
    """
    row: Dict[str, float] = {}
    # base
    for k in cfg.numeric_base:
        row[k] = _to_float(feats.get(k))
    # derived
    row.update(compute_derived(feats))
    # onehot
    row.update(onehot(feats.get("asset_type"), feats.get("market"), cfg=cfg))
    return row


def row_to_vector(row: Dict[str, float], cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    Dict row -> stable ordered vector.
    """
    cols = cfg.all_columns
    return np.array([row.get(c, float("nan")) for c in cols], dtype=float)


def features_to_vector(feats: Dict[str, Any], cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    return row_to_vector(features_to_row(feats, cfg=cfg), cfg=cfg)


def vector_columns(cfg: FeatureConfig = DEFAULT_CONFIG) -> List[str]:
    return cfg.all_columns


def rule_flags(feats: Dict[str, Any]) -> Dict[str, int]:
    """
    Optional: deterministic rule flags as features if you want.
    Not used by default in unsupervised; useful for supervised.
    """
    v = {k: _to_float(feats.get(k)) for k in NUMERIC_BASE}
    flags = {}

    # Invariants
    flags["inv_var99_ge_var95"] = int(not math.isnan(v["var99"]) and not math.isnan(v["var95"]) and v["var99"] >= v["var95"])
    flags["inv_es99_ge_es95"] = int(not math.isnan(v["es99"]) and not math.isnan(v["es95"]) and v["es99"] >= v["es95"])
    flags["inv_es95_ge_var95"] = int(not math.isnan(v["es95"]) and not math.isnan(v["var95"]) and v["es95"] >= v["var95"])
    flags["inv_es99_ge_var99"] = int(not math.isnan(v["es99"]) and not math.isnan(v["var99"]) and v["es99"] >= v["var99"])
    flags["tuw_in_0_100"] = int(not math.isnan(v["tuw_pct"]) and 0.0 <= v["tuw_pct"] <= 100.0)
    flags["missing_pct_reasonable"] = int(math.isnan(v["missing_pct"]) or (0.0 <= v["missing_pct"] <= 10.0))
    flags["tail_obs_99_ok"] = int(math.isnan(v["tail_obs_99"]) or v["tail_obs_99"] >= 5)

    return flags
