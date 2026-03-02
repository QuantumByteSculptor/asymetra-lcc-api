# feature_utils.py  (root level)
"""
Shared risk feature computation functions.
Used by:
  - api/main.py          (Oracle computation)
  - build_dataset_daily.py  (training data pipeline)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Downside risk
# ---------------------------------------------------------------------------

def compute_downside_dev(returns: np.ndarray, ann: int = 252) -> float:
    """Annualized downside deviation (Sortino denominator)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    neg = r[r < 0]
    if len(neg) < 5:
        return float("nan")
    return float(np.sqrt(np.mean(neg ** 2) * ann))


def compute_semivariance(returns: np.ndarray, ann: int = 252) -> float:
    """Annualized semivariance (downside variance, without sqrt)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    neg = r[r < 0]
    if len(neg) < 5:
        return float("nan")
    return float(np.mean(neg ** 2) * ann)


# ---------------------------------------------------------------------------
# Volatility dynamics
# ---------------------------------------------------------------------------

def compute_vol_of_vol(returns: np.ndarray, window: int = 20) -> float:
    """
    Coefficient of variation of rolling-window volatility.
    Measures how unstable the volatility regime is.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < window * 2:
        return float("nan")

    vols = np.array(
        [np.std(r[i - window : i], ddof=1) for i in range(window, n + 1)],
        dtype=float,
    )
    vols = vols[np.isfinite(vols) & (vols > 0)]
    if len(vols) < 5:
        return float("nan")

    mu = float(np.mean(vols))
    if mu < 1e-12:
        return float("nan")
    return float(np.std(vols, ddof=1) / mu)


def compute_ewma_vol_ann(returns: np.ndarray, lam: float = 0.94, ann: int = 252) -> float:
    """EWMA (RiskMetrics) volatility, annualized."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return float("nan")
    var = float(np.var(r, ddof=1))
    for x in r:
        var = lam * var + (1.0 - lam) * (x * x)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(ann))


# ---------------------------------------------------------------------------
# Worst rolling returns
# ---------------------------------------------------------------------------

def compute_worst_rolling_return(returns: np.ndarray, n: int) -> float:
    """
    Worst n-day rolling cumulative return.
    Returns a negative float (e.g., -0.08 for an 8% drawdown in n days).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < n:
        return float("nan")

    worst = 0.0
    for i in range(len(r) - n + 1):
        w = np.clip(r[i : i + n], -0.9, 10.0)
        cum = float(np.prod(1.0 + w) - 1.0)
        if cum < worst:
            worst = cum
    return float(worst)


# ---------------------------------------------------------------------------
# Serial correlation
# ---------------------------------------------------------------------------

def compute_autocorr(returns: np.ndarray, lag: int = 1) -> float:
    """Lag-n Pearson autocorrelation of returns (momentum / mean-reversion signal)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < lag + 10:
        return float("nan")

    r1, r2 = r[:-lag], r[lag:]
    mu1, mu2 = r1.mean(), r2.mean()
    c1, c2 = r1 - mu1, r2 - mu2
    num = float(np.mean(c1 * c2))
    den = float(np.std(r1, ddof=1) * np.std(r2, ddof=1))
    if den < 1e-12:
        return float("nan")
    return float(np.clip(num / den, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Drawdown dynamics
# ---------------------------------------------------------------------------

def compute_dd_duration_recovery(prices: np.ndarray) -> Tuple[int, int]:
    """
    Returns (dd_duration, recovery_days) for the maximum drawdown period.

    - dd_duration  : consecutive trading days from peak to trough
    - recovery_days: days from trough until price first exceeds the peak again
                     (0 if no recovery occurred within the sample)
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    n = len(p)
    if n < 10:
        return (0, 0)

    peak_val = p[0]
    peak_idx = 0
    max_dd = 0.0
    best_peak_idx = 0
    best_trough_idx = 0

    for i in range(1, n):
        if p[i] > peak_val:
            peak_val = p[i]
            peak_idx = i
        dd = (p[i] - peak_val) / (peak_val + 1e-12)
        if dd < max_dd:
            max_dd = dd
            best_peak_idx = peak_idx
            best_trough_idx = i

    if best_trough_idx == 0 or max_dd >= -1e-6:
        return (0, 0)

    dd_duration = int(best_trough_idx - best_peak_idx)
    peak_val_at_best = float(p[best_peak_idx])

    recovery_days = 0
    for i in range(best_trough_idx + 1, n):
        if p[i] >= peak_val_at_best:
            recovery_days = i - best_trough_idx
            break

    return (dd_duration, recovery_days)


# ---------------------------------------------------------------------------
# Stress features
# ---------------------------------------------------------------------------

def compute_stress_features(
    returns: np.ndarray,
    base_var99: Optional[float],
    window: int = 20,
    q: float = 0.99,
) -> Dict[str, Any]:
    """
    Finds the worst rolling `window`-day period and computes:
      - stress_var99     : 99th-pct daily loss VaR in that window
      - stress_multiplier: stress_var99 / base_var99
      - stress_cumret    : cumulative return of the worst window
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < max(60, window + 10):
        return {"stress_var99": None, "stress_multiplier": None, "stress_cumret": None}

    worst_i: Optional[int] = None
    worst_cum = 1e9
    for i in range(0, n - window + 1):
        w = np.clip(r[i : i + window], -0.9, 10.0)
        cum = float(np.prod(1.0 + w) - 1.0)
        if cum < worst_cum:
            worst_cum = cum
            worst_i = i

    if worst_i is None:
        return {"stress_var99": None, "stress_multiplier": None, "stress_cumret": float(worst_cum)}

    losses = -r[worst_i : worst_i + window]
    v = float(np.quantile(losses, q))
    stress_var = v if np.isfinite(v) else float("nan")

    mult: Optional[float] = None
    if (
        base_var99 is not None
        and np.isfinite(base_var99)
        and base_var99 > 1e-12
        and np.isfinite(stress_var)
    ):
        mult = float(stress_var / base_var99)

    return {
        "stress_var99": float(stress_var) if np.isfinite(stress_var) else None,
        "stress_multiplier": float(mult) if mult is not None and np.isfinite(mult) else None,
        "stress_cumret": float(worst_cum),
        "stress_window_days": int(window),
    }


# ---------------------------------------------------------------------------
# Convenience: compute all v2 features from a returns array
# ---------------------------------------------------------------------------

def compute_all_v2_features(
    returns: np.ndarray,
    prices: np.ndarray,
    base_var99: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One-shot computation of all v2 features.
    `returns` and `prices` must be finite-cleaned before passing.
    """
    stress = compute_stress_features(returns, base_var99=base_var99)
    dd_dur, rec = compute_dd_duration_recovery(prices)
    return {
        "downside_dev": compute_downside_dev(returns),
        "semivariance": compute_semivariance(returns),
        "vol_of_vol": compute_vol_of_vol(returns),
        "worst_5d_ret": compute_worst_rolling_return(returns, 5),
        "worst_20d_ret": compute_worst_rolling_return(returns, 20),
        "autocorr_1": compute_autocorr(returns),
        "vol_ewma_ann": compute_ewma_vol_ann(returns),
        "stress_var99": stress.get("stress_var99"),
        "stress_multiplier": stress.get("stress_multiplier"),
        "stress_cumret": stress.get("stress_cumret"),
        "dd_duration": dd_dur if dd_dur > 0 else None,
        "recovery_days": rec if rec > 0 else None,
    }
