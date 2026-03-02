# features.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-12


# -----------------------------
# Feature spec (stable ordering)
# -----------------------------
# IMPORTANT:
# - Keep this list stable once you deploy, or version your bundles.
# - We include all keys you currently generate (seen in your JSONL)
#   + extra “distribution/dynamics” keys you’re about to add in build_dataset_daily.py.

NUMERIC_BASE: List[str] = [
    # core risk metrics
    "vol_ann",
    "vol_20d",
    "vol_60d",
    "vol_120d",
    "var95",
    "var99",
    "es95",
    "es99",
    "max_dd",          # prefer this canonical name
    "tuw_pct",
    "n_used",
    "missing_pct",
    "tail_obs_99",

    # distribution shape (to be added by you)
    "skew",
    "kurtosis_excess",

    # drawdown dynamics (to be added by you)
    "dd_duration",
    "recovery_days",

    # extra signals already present in your dataset (sample keys)
    "corr_mkt",
    "rsi",
    "raw_if",
    "raw_lof",
    "z_if",
    "z_lof",
    "z_gap_if_lof",

    # v2: downside risk & volatility dynamics
    "downside_dev",
    "semivariance",
    "vol_of_vol",
    "worst_5d_ret",
    "worst_20d_ret",
    "autocorr_1",
    "vol_ewma_ann",
    "stress_var99",
    "stress_multiplier",
]


# Derived features (ratios/consistency/quality)
DERIVED: List[str] = [
    # tail structure ratios
    "var99_div_var95",
    "es99_div_es95",
    "es95_div_var95",
    "es99_div_var99",

    # volatility consistency
    "vol_to_var95_scaled",     # vol_ann / (var95*sqrt(252))
    "var95_vs_vol_daily",      # var95 / (vol_ann/sqrt(252))
    "vol20_to_volann",         # vol_20d / vol_ann (stability)
    "vol60_to_volann",         # vol_60d / vol_ann
    "vol120_to_volann",        # vol_120d / vol_ann
    "vol20_to_vol60",          # short vs mid
    "vol60_to_vol120",         # mid vs long

    # drawdown vs tail risk
    "dd_to_var99",
    "tuw_per_dd",
    "tail_obs_99_ratio",
    "missing_pct_clamped",

    # quality/consistency helpers
    "rsi_centered",            # rsi-50
    "abs_corr_mkt",            # |corr_mkt|
    "log_n_used",              # log(1+n_used)
    "dd_duration_per_n",        # dd_duration / n_used
    "recovery_per_dd_dur",      # recovery_days / dd_duration

    # v2 derived
    "downside_div_vol",         # downside_dev / vol_ann (tail asymmetry index)
    "worst_5d_vs_var99",        # abs(worst_5d_ret) / var99 (stress vs normal tail risk)
]


# Categorical one-hot
ASSET_TYPES: List[str] = ["equity", "etf", "bond", "fx", "commodity", "crypto"]
MARKETS: List[str] = [
    "US", "EU", "UK", "JP", "HK", "CN", "IN", "AU", "CA", "BR", "MX", "ZA",
    "G10", "EM",
    # commodity/cross markets that sometimes appear in your synthetic data
    "XAU", "WTI", "BRENT", "NG", "HG", "SILVER", "WHEAT", "CORN", "COFFEE",
    "SOYBEAN", "SUGAR", "COTTON", "ALUMINUM", "ALT", "BTC", "ETH",
]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class FeatureConfig:
    numeric_base: Tuple[str, ...] = tuple(NUMERIC_BASE)
    derived: Tuple[str, ...] = tuple(DERIVED)
    asset_types: Tuple[str, ...] = tuple(ASSET_TYPES)
    markets: Tuple[str, ...] = tuple(MARKETS)
    version: str = "v2"

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


# -----------------------------
# utils
# -----------------------------
def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _safe_div(a: float, b: float) -> float:
    if b is None:
        return float("nan")
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < EPS:
        return float("nan")
    return a / b


def _clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return max(lo, min(hi, x))


def _nan_to(x: float, default: float = 0.0) -> float:
    return default if (x is None or not math.isfinite(x)) else float(x)


# -----------------------------
# derived computation
# -----------------------------
def compute_derived(feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived features from raw base metrics.
    All returns are floats; may contain NaN if insufficient data.
    """
    # pull base
    vol_ann = _to_float(feats.get("vol_ann"))
    vol_20d = _to_float(feats.get("vol_20d"))
    vol_60d = _to_float(feats.get("vol_60d"))
    vol_120d = _to_float(feats.get("vol_120d"))

    var95 = _to_float(feats.get("var95"))
    var99 = _to_float(feats.get("var99"))
    es95 = _to_float(feats.get("es95"))
    es99 = _to_float(feats.get("es99"))

    max_dd = _to_float(feats.get("max_dd"))
    tuw_pct = _to_float(feats.get("tuw_pct"))
    n_used = _to_float(feats.get("n_used"))
    missing_pct = _to_float(feats.get("missing_pct"))
    tail_obs_99 = _to_float(feats.get("tail_obs_99"))

    corr_mkt = _to_float(feats.get("corr_mkt"))
    rsi = _to_float(feats.get("rsi"))

    dd_duration = _to_float(feats.get("dd_duration"))
    recovery_days = _to_float(feats.get("recovery_days"))

    downside_dev = _to_float(feats.get("downside_dev"))
    worst_5d_ret = _to_float(feats.get("worst_5d_ret"))

    sqrt252 = math.sqrt(252.0)

    # volatility normalization helpers
    vol_daily = _safe_div(vol_ann, sqrt252)  # approx
    vol_to_var95_scaled = _safe_div(vol_ann, var95 * sqrt252)  # should be O(1-10)
    var95_vs_vol_daily = _safe_div(var95, vol_daily)           # should be O(0.5-3) depending on tails

    # ratios for multi-horizon vol
    vol20_to_volann = _safe_div(vol_20d, vol_ann)
    vol60_to_volann = _safe_div(vol_60d, vol_ann)
    vol120_to_volann = _safe_div(vol_120d, vol_ann)
    vol20_to_vol60 = _safe_div(vol_20d, vol_60d)
    vol60_to_vol120 = _safe_div(vol_60d, vol_120d)

    out: Dict[str, float] = {
        # tail structure
        "var99_div_var95": _safe_div(var99, var95),
        "es99_div_es95": _safe_div(es99, es95),
        "es95_div_var95": _safe_div(es95, var95),
        "es99_div_var99": _safe_div(es99, var99),

        # consistency
        "vol_to_var95_scaled": vol_to_var95_scaled,
        "var95_vs_vol_daily": var95_vs_vol_daily,
        "vol20_to_volann": vol20_to_volann,
        "vol60_to_volann": vol60_to_volann,
        "vol120_to_volann": vol120_to_volann,
        "vol20_to_vol60": vol20_to_vol60,
        "vol60_to_vol120": vol60_to_vol120,

        # drawdown vs tails
        "dd_to_var99": _safe_div(max_dd, var99),
        "tuw_per_dd": _safe_div(tuw_pct, max_dd),
        "tail_obs_99_ratio": _safe_div(tail_obs_99, n_used),

        # cleanliness
        "missing_pct_clamped": float("nan") if not math.isfinite(missing_pct) else _clamp(missing_pct, 0.0, 100.0),

        # misc helpers
        "rsi_centered": float("nan") if not math.isfinite(rsi) else (rsi - 50.0),
        "abs_corr_mkt": float("nan") if not math.isfinite(corr_mkt) else abs(corr_mkt),
        "log_n_used": float("nan") if not math.isfinite(n_used) else math.log1p(max(0.0, n_used)),

        # dd dynamics normalization
        "dd_duration_per_n": _safe_div(dd_duration, n_used),
        "recovery_per_dd_dur": _safe_div(recovery_days, dd_duration),

        # v2 derived
        "downside_div_vol": _safe_div(downside_dev, vol_ann),
        "worst_5d_vs_var99": _safe_div(abs(worst_5d_ret) if math.isfinite(worst_5d_ret) else float("nan"), var99),
    }

    return out


# -----------------------------
# one-hot
# -----------------------------
def onehot(asset_type: str | None, market: str | None, cfg: FeatureConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    d: Dict[str, float] = {}
    at = (asset_type or "").strip().lower()
    mk = (market or "").strip().upper()

    for a in cfg.asset_types:
        d[f"asset_type__{a}"] = 1.0 if at == a else 0.0
    for m in cfg.markets:
        d[f"market__{m}"] = 1.0 if mk == m else 0.0
    return d


# -----------------------------
# main adapters
# -----------------------------
def features_to_row(feats: Dict[str, Any], cfg: FeatureConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Turn raw feature dict -> full row dict (numeric + derived + onehot).
    Handles backward-compat aliases (e.g., max_drawdown -> max_dd).
    """
    # alias normalization (keep it minimal + safe)
    if "max_dd" not in feats and "max_drawdown" in feats:
        feats = dict(feats)  # don't mutate caller
        feats["max_dd"] = feats.get("max_drawdown")

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


# -----------------------------
# optional deterministic flags
# -----------------------------
def rule_flags(feats: Dict[str, Any]) -> Dict[str, int]:
    """
    Optional: deterministic rule flags as features if you want.
    Not used by default here.
    """
    v = {k: _to_float(feats.get(k)) for k in [
        "vol_ann", "var95", "var99", "es95", "es99", "max_dd",
        "tuw_pct", "missing_pct", "tail_obs_99", "n_used",
    ]}
    flags: Dict[str, int] = {}

    flags["inv_var99_ge_var95"] = int(math.isfinite(v["var99"]) and math.isfinite(v["var95"]) and v["var99"] >= v["var95"])
    flags["inv_es99_ge_es95"] = int(math.isfinite(v["es99"]) and math.isfinite(v["es95"]) and v["es99"] >= v["es95"])
    flags["inv_es95_ge_var95"] = int(math.isfinite(v["es95"]) and math.isfinite(v["var95"]) and v["es95"] >= v["var95"])
    flags["inv_es99_ge_var99"] = int(math.isfinite(v["es99"]) and math.isfinite(v["var99"]) and v["es99"] >= v["var99"])
    flags["tuw_in_0_100"] = int(math.isfinite(v["tuw_pct"]) and 0.0 <= v["tuw_pct"] <= 100.0)
    flags["missing_pct_reasonable"] = int((not math.isfinite(v["missing_pct"])) or (0.0 <= v["missing_pct"] <= 10.0))
    flags["tail_obs_99_ok"] = int((not math.isfinite(v["tail_obs_99"])) or (v["tail_obs_99"] >= 5))

    return flags







