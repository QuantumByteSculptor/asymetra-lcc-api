# build_dataset_v3.py
"""
Dataset v3 pipeline — temporally rigorous, multi-asset, macro-enriched.

Key improvements over v2:
  - Timestamps on every window (window_start, window_end, label_start, label_end)
  - Multi-horizon labels (5d, 10d, 20d, 60d forward returns)
  - Macro/regime features (VIX, US 10Y, credit spread)
  - Handles FX, commodities, crypto, indices (not just equity/etf)
  - Stationarised features (log-returns, z-scores, winsorisation)
  - Cross-asset features (corr_vs_spy, corr_vs_vix, beta_market)

Usage:
  python build_dataset_v3.py --universe data/universe.json --out data/training/train_v3_all.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from feature_utils import (
    compute_downside_dev, compute_semivariance, compute_vol_of_vol,
    compute_worst_rolling_return, compute_autocorr, compute_ewma_vol_ann,
    compute_dd_duration_recovery, compute_stress_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _skew_kurtosis_np(x: np.ndarray):
    """Pure-numpy skew and excess kurtosis (no scipy dependency)."""
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan"), float("nan")
    m = x.mean()
    c = x - m
    s2 = float(np.mean(c * c))
    if s2 <= 1e-18:
        return 0.0, -3.0
    s = np.sqrt(s2)
    skew = float(np.mean(c ** 3) / (s ** 3 + 1e-12))
    kurt_excess = float(np.mean(c ** 4) / (s2 ** 2 + 1e-12) - 3.0)
    return skew, kurt_excess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOOKBACK = 252        # trading days for feature window
_HORIZON_PRIMARY = 20  # primary label horizon
_STEP_DAYS = 10        # step between windows
_MAX_PER_TICKER = 80   # max windows per ticker
_MIN_HISTORY = 400     # min days needed (lookback + horizon + margin)


# ===================================================================
# MACRO DATA — downloaded once, shared across all tickers
# ===================================================================

def _fred_series(series_id: str, start: str = "2017-01-01") -> pd.Series:
    """Download a FRED time series (public CSV endpoint, no API key needed)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # FRED returns two columns: DATE and the series id
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col]).set_index(date_col).sort_index()
        return df[val_col]
    except Exception as e:
        log.warning("FRED %s download failed: %s", series_id, e)
        return pd.Series(dtype=float)


def download_macro_data(start: str = "2017-01-01") -> Dict[str, pd.Series]:
    """
    Download key macro indicators from FRED.
    Returns dict of {name: pd.Series(date-indexed, daily-ffilled)}.
    """
    macro = {}
    series_map = {
        "vix": "VIXCLS",           # CBOE VIX
        "rate_10y": "DGS10",       # US 10Y Treasury
        "rate_2y": "DGS2",         # US 2Y Treasury
        "fed_funds": "DFF",        # Fed Funds effective rate
        "credit_spread_hy": "BAMLH0A0HYM2",  # ICE BofA HY OAS
        "credit_spread_ig": "BAMLC0A0CM",     # ICE BofA IG OAS
    }

    for name, fred_id in series_map.items():
        log.info("Downloading FRED %s (%s)...", name, fred_id)
        s = _fred_series(fred_id, start=start)
        time.sleep(0.5)  # gentle rate limit between FRED requests
        if not s.empty:
            # Forward-fill to daily (FRED has gaps on weekends/holidays)
            idx = pd.date_range(s.index.min(), s.index.max(), freq="B")
            s = s.reindex(idx).ffill().bfill()
            macro[name] = s
            log.info("  %s: %d points [%s → %s]", name, len(s),
                     s.index.min().date(), s.index.max().date())
        else:
            log.warning("  %s: EMPTY — will use NaN fallback", name)

    return macro


def download_spy_returns(start: str = "2017-01-01") -> pd.Series:
    """Download SPY daily returns for cross-asset features."""
    try:
        # Use direct Yahoo API (yfinance 0.2.40 is broken for most tickers)
        s = _yahoo_direct_download("SPY", start)
        ret = s.pct_change().dropna()
        return ret
    except Exception as e:
        log.warning("SPY download failed: %s", e)
    return pd.Series(dtype=float)


# ===================================================================
# PRICE DOWNLOAD — handles all asset types
# ===================================================================

def _stooq_download(ticker: str, market: str) -> pd.Series:
    """Download close prices from Stooq."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    candidates = []
    t = ticker.strip()
    m = (market or "").strip().upper()

    # Build symbol candidates based on ticker format
    if "." not in t and "=" not in t and not t.startswith("^"):
        if m == "US":
            candidates.append(f"{t}.US")
        candidates.append(t)
        if m != "US":
            candidates.append(f"{t}.US")
    else:
        candidates.append(t)

    # De-dup
    seen = set()
    cands = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            cands.append(c)

    last_err = None
    for sym in cands:
        try:
            url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt or txt.lower().startswith("no data"):
                continue

            first_line = txt.splitlines()[0].strip()
            if not first_line.lower().startswith("date,open,high,low,close"):
                continue

            df = pd.read_csv(io.StringIO(txt))
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
            idx = pd.DatetimeIndex(df["Date"].to_numpy())
            close = pd.Series(df["Close"].to_numpy(dtype=float), index=idx).dropna()
            if not close.empty:
                return close
        except Exception as e:
            last_err = e

    raise RuntimeError(f"stooq failed for {ticker}: {last_err}")


def _yahoo_direct_download(ticker: str, start: str) -> pd.Series:
    """
    Download close prices via Yahoo Finance chart API v8 (no yfinance dependency).
    More reliable for FX (=X), futures (=F), crypto (-USD), indices (^).
    """
    from datetime import timezone as _tz
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=_tz.utc).timestamp())
    end_ts = int(datetime.now(tz=_tz.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()
    result = (j.get("chart") or {}).get("result") or []
    if not result:
        err = (j.get("chart") or {}).get("error")
        raise RuntimeError(f"Yahoo API no result for {ticker}: {err}")
    timestamps = result[0].get("timestamp") or []
    quotes = result[0].get("indicators", {}).get("quote", [{}])
    closes = (quotes[0] if quotes else {}).get("close") or []
    if not timestamps or not closes:
        raise RuntimeError(f"Yahoo API empty data for {ticker}")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    s = pd.Series(
        [float(c) if c is not None else float("nan") for c in closes],
        index=dates,
    ).dropna()
    if s.empty:
        raise RuntimeError(f"Yahoo API all-NaN closes for {ticker}")
    return s


def _yf_download(ticker: str, start: str, max_tries: int = 3) -> pd.Series:
    """
    Download close prices from Yahoo Finance.
    Tries direct chart API first (reliable for =X/=F/^/-USD tickers),
    falls back to yfinance library.
    """
    # Direct API — works for all ticker types including FX, futures, crypto, indices
    try:
        return _yahoo_direct_download(ticker, start)
    except Exception as e_direct:
        pass

    # Fallback: yfinance library (works for plain equity/etf tickers)
    import yfinance as yf
    last_err = None
    for attempt in range(max_tries):
        try:
            df = yf.download(ticker, start=start, interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0] if ticker not in close.columns else close[ticker]
                s = pd.Series(close).dropna()
                if not s.empty:
                    return s
            last_err = RuntimeError("empty close")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (1.4 ** attempt))

    raise RuntimeError(f"yfinance failed for {ticker}: direct={e_direct}; lib={last_err}")


def download_close(
    ticker: str, market: str, start: str,
    asset_type: str = "",
) -> Tuple[pd.Series, str]:
    """
    Download close prices with provider fallback.
    Returns (close_series, source_name).
    """
    t = ticker.strip()

    # For FX/futures/indices, use yfinance directly (stooq doesn't handle these well)
    yf_first = (
        t.endswith("=X") or t.endswith("=F") or t.startswith("^")
        or t.endswith("-USD")  # crypto
        or asset_type in ("fx", "crypto", "rate")
    )

    errs = []
    if not yf_first:
        try:
            s = _stooq_download(ticker, market)
            if s is not None and not s.empty:
                return s, "stooq"
        except Exception as e:
            errs.append(f"stooq: {e}")

    try:
        s = _yf_download(ticker, start)
        if s is not None and not s.empty:
            return s, "yfinance"
    except Exception as e:
        errs.append(f"yf: {e}")

    if yf_first:
        try:
            s = _stooq_download(ticker, market)
            if s is not None and not s.empty:
                return s, "stooq"
        except Exception as e:
            errs.append(f"stooq: {e}")

    raise RuntimeError(f"all providers failed for {ticker}: {'; '.join(errs)}")


# ===================================================================
# FEATURE ENGINEERING V3
# ===================================================================

def _safe(x) -> Optional[float]:
    """Convert to float, return None if NaN/inf."""
    if x is None:
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _winsorise(x: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """IQR-based winsorisation."""
    finite = x[np.isfinite(x)]
    if len(finite) < 10:
        return x
    lo, hi = np.quantile(finite, [lower_q, upper_q])
    return np.clip(x, lo, hi)


def rsi(prices: pd.Series, period: int = 14) -> float:
    x = prices.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    dd = prices / (peak + 1e-12) - 1.0
    return float(dd.min())


def _bollinger_distance(prices: pd.Series, window: int = 20) -> float:
    """Distance from current price to Bollinger mid band, normalised by band width."""
    if len(prices) < window + 5:
        return float("nan")
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std(ddof=1)
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = upper.iloc[-1] - lower.iloc[-1]
    if width < 1e-12:
        return 0.0
    return float((prices.iloc[-1] - mid.iloc[-1]) / (width + 1e-12))


def _macd_histogram(prices: pd.Series) -> float:
    """MACD histogram (12/26/9)."""
    if len(prices) < 35:
        return float("nan")
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    # Normalise by price level
    px = float(prices.iloc[-1])
    return float(hist.iloc[-1] / (px + 1e-12))


def _sma_slope(prices: pd.Series, window: int) -> float:
    """Normalised SMA slope (annualised rate of change)."""
    if len(prices) < window + 5:
        return float("nan")
    sma = prices.rolling(window).mean()
    if sma.iloc[-1] < 1e-12:
        return 0.0
    slope = (sma.iloc[-1] - sma.iloc[-window]) / (sma.iloc[-1] + 1e-12)
    return float(slope * 252 / window)  # annualised


def _jump_indicator(returns: np.ndarray, threshold_sigma: float = 3.0) -> float:
    """Fraction of returns exceeding threshold_sigma standard deviations."""
    r = returns[np.isfinite(returns)]
    if len(r) < 20:
        return float("nan")
    sigma = np.std(r, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(np.abs(r) > threshold_sigma * sigma))


def _hill_estimator(returns: np.ndarray, k: int = 20) -> float:
    """Hill tail index estimator on losses."""
    losses = -returns[np.isfinite(returns)]
    losses = losses[losses > 0]
    if len(losses) < k + 5:
        return float("nan")
    sorted_losses = np.sort(losses)[::-1][:k]
    if sorted_losses[-1] < 1e-12:
        return float("nan")
    log_ratios = np.log(sorted_losses / sorted_losses[-1])
    return float(1.0 / (np.mean(log_ratios) + 1e-12))


def build_features_v3(
    ticker: str,
    asset_type: str,
    market: str,
    closes: pd.Series,
    returns: pd.Series,
    macro: Dict[str, pd.Series],
    spy_returns: pd.Series,
    window_end_date: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Build v3 feature vector for a single window.
    Returns empty dict if insufficient data.
    """
    ret = returns.to_numpy(dtype=float)
    ret = ret[np.isfinite(ret)]
    px = closes.to_numpy(dtype=float)

    if len(ret) < 60 or len(px) < 60:
        return {}

    # Winsorise returns for robust stats
    ret_w = _winsorise(ret)

    # --- BASE RISK METRICS ---
    vol_ann = float(np.std(ret_w, ddof=1) * np.sqrt(252)) if len(ret_w) > 10 else None

    ret20 = ret_w[-20:] if len(ret_w) >= 20 else ret_w
    ret60 = ret_w[-60:] if len(ret_w) >= 60 else ret_w
    ret120 = ret_w[-120:] if len(ret_w) >= 120 else ret_w

    vol_20d = float(np.std(ret20, ddof=1)) if len(ret20) >= 10 else None
    vol_60d = float(np.std(ret60, ddof=1) * np.sqrt(252)) if len(ret60) >= 20 else None
    vol_120d = float(np.std(ret120, ddof=1) * np.sqrt(252)) if len(ret120) >= 40 else None

    # VaR/ES (positive loss convention)
    losses = -ret_w
    var95 = float(np.quantile(losses, 0.95)) if len(losses) >= 30 else None
    var99 = float(np.quantile(losses, 0.99)) if len(losses) >= 30 else None
    tail95 = losses[losses >= (var95 or 1e9)] if var95 is not None else np.array([])
    tail99 = losses[losses >= (var99 or 1e9)] if var99 is not None else np.array([])
    es95 = float(tail95.mean()) if len(tail95) > 0 else var95
    es99 = float(tail99.mean()) if len(tail99) > 0 else var99

    mdd = max_drawdown(closes)
    n_used = len(ret)
    missing_pct = max(0.0, 1.0 - n_used / 252.0)
    tail_obs_99 = int(np.sum(losses >= (var99 or 1e9))) if var99 is not None else 0

    # --- DISTRIBUTION SHAPE ---
    skew, kurt_excess = _skew_kurtosis_np(ret)

    # --- MOMENTUM & TREND ---
    rsi_val = rsi(closes) if len(closes) >= 20 else None
    bb_dist = _bollinger_distance(closes)
    macd_hist = _macd_histogram(closes)
    sma_slope_20 = _sma_slope(closes, 20)
    sma_slope_60 = _sma_slope(closes, 60)

    # --- V2 FEATURES ---
    downside_dev = compute_downside_dev(ret)
    semivariance = compute_semivariance(ret)
    vol_of_vol = compute_vol_of_vol(ret)
    worst_5d = compute_worst_rolling_return(ret, 5)
    worst_10d = compute_worst_rolling_return(ret, 10)
    worst_20d = compute_worst_rolling_return(ret, 20)
    autocorr_1 = compute_autocorr(ret, lag=1)
    autocorr_5 = compute_autocorr(ret, lag=5)
    vol_ewma = compute_ewma_vol_ann(ret)
    dd_dur, recovery = compute_dd_duration_recovery(px)
    stress = compute_stress_features(ret, base_var99=var99)

    # --- TAIL METRICS ---
    jump_ind = _jump_indicator(ret, threshold_sigma=3.0)
    hill_est = _hill_estimator(ret, k=min(20, max(5, len(ret) // 20)))

    # --- CROSS-ASSET FEATURES ---
    corr_spy = None
    beta_mkt = None
    if not spy_returns.empty:
        # Align dates
        common_idx = closes.index.intersection(spy_returns.index)
        if len(common_idx) >= 60:
            r_asset = returns.reindex(common_idx).dropna()
            r_spy = spy_returns.reindex(common_idx).dropna()
            common = r_asset.index.intersection(r_spy.index)
            if len(common) >= 60:
                a = r_asset.reindex(common).to_numpy(dtype=float)[-252:]
                s = r_spy.reindex(common).to_numpy(dtype=float)[-252:]
                mask = np.isfinite(a) & np.isfinite(s)
                a, s = a[mask], s[mask]
                if len(a) >= 30:
                    corr_spy = float(np.corrcoef(a, s)[0, 1])
                    var_s = float(np.var(s, ddof=1))
                    if var_s > 1e-12:
                        beta_mkt = float(np.cov(a, s, ddof=1)[0, 1] / var_s)

    # --- MACRO FEATURES (as of window_end_date) ---
    def _macro_val(name: str) -> Optional[float]:
        s = macro.get(name)
        if s is None or s.empty:
            return None
        # Get most recent value on or before window_end_date
        valid = s[s.index <= window_end_date]
        if valid.empty:
            return None
        return float(valid.iloc[-1])

    def _macro_percentile(name: str, lookback: int = 60) -> Optional[float]:
        s = macro.get(name)
        if s is None or s.empty:
            return None
        valid = s[s.index <= window_end_date]
        if len(valid) < lookback:
            return None
        window_data = valid.iloc[-lookback:]
        current = float(valid.iloc[-1])
        return float((window_data <= current).mean())

    vix_level = _macro_val("vix")
    vix_pct_60d = _macro_percentile("vix", 60)
    rate_10y = _macro_val("rate_10y")
    rate_2y = _macro_val("rate_2y")
    term_spread = None
    if rate_10y is not None and rate_2y is not None:
        term_spread = rate_10y - rate_2y
    credit_spread_hy = _macro_val("credit_spread_hy")

    # Volatility regime (based on VIX percentile)
    vol_regime = None
    if vix_pct_60d is not None:
        if vix_pct_60d < 0.33:
            vol_regime = 0  # low
        elif vix_pct_60d < 0.67:
            vol_regime = 1  # medium
        else:
            vol_regime = 2  # high

    # Correlation vs VIX
    corr_vix = None
    vix_series = macro.get("vix")
    if vix_series is not None and not vix_series.empty:
        common_idx = closes.index.intersection(vix_series.index)
        if len(common_idx) >= 60:
            r_asset = returns.reindex(common_idx).dropna()
            vix_ret = vix_series.reindex(common_idx).pct_change().dropna()
            common = r_asset.index.intersection(vix_ret.index)
            if len(common) >= 60:
                a = r_asset.reindex(common).to_numpy(dtype=float)[-120:]
                v = vix_ret.reindex(common).to_numpy(dtype=float)[-120:]
                mask = np.isfinite(a) & np.isfinite(v)
                a, v = a[mask], v[mask]
                if len(a) >= 30:
                    corr_vix = float(np.corrcoef(a, v)[0, 1])

    # --- DERIVED RATIOS ---
    def _ratio(a, b):
        if a is None or b is None:
            return None
        try:
            fa, fb = float(a), float(b)
            if abs(fb) < 1e-12 or not np.isfinite(fa) or not np.isfinite(fb):
                return None
            return float(fa / fb)
        except:
            return None

    # Build feature dict
    feats: Dict[str, Any] = {
        # Metadata (not features, but needed for splits)
        "asset_type": (asset_type or "").strip().lower(),
        "market": (market or "").strip().upper(),
        "ticker": (ticker or "").strip(),

        # Base risk
        "vol_ann": _safe(vol_ann),
        "vol_20d": _safe(vol_20d),
        "vol_60d": _safe(vol_60d),
        "vol_120d": _safe(vol_120d),
        "var95": _safe(var95),
        "var99": _safe(var99),
        "es95": _safe(es95),
        "es99": _safe(es99),
        "max_dd": _safe(mdd),
        "max_drawdown": _safe(mdd),
        "n_used": int(n_used),
        "missing_pct": float(missing_pct),
        "tuw_pct": 95.0,
        "tail_obs_99": int(tail_obs_99),

        # Distribution shape
        "skew": _safe(skew),
        "kurtosis_excess": _safe(kurt_excess),

        # Momentum & trend
        "rsi": _safe(rsi_val),
        "bb_distance": _safe(bb_dist),
        "macd_hist": _safe(macd_hist),
        "sma_slope_20": _safe(sma_slope_20),
        "sma_slope_60": _safe(sma_slope_60),

        # Downside risk
        "downside_dev": _safe(downside_dev),
        "semivariance": _safe(semivariance),

        # Volatility dynamics
        "vol_of_vol": _safe(vol_of_vol),
        "vol_ewma_ann": _safe(vol_ewma),

        # Worst returns
        "worst_5d_ret": _safe(worst_5d),
        "worst_10d_ret": _safe(worst_10d),
        "worst_20d_ret": _safe(worst_20d),

        # Serial correlation
        "autocorr_1": _safe(autocorr_1),
        "autocorr_5": _safe(autocorr_5),

        # Drawdown dynamics
        "dd_duration": dd_dur if dd_dur > 0 else None,
        "recovery_days": recovery if recovery > 0 else None,

        # Stress
        "stress_var99": _safe(stress.get("stress_var99")),
        "stress_multiplier": _safe(stress.get("stress_multiplier")),

        # Tail metrics (NEW v3)
        "jump_indicator": _safe(jump_ind),
        "hill_tail_index": _safe(hill_est),

        # Cross-asset (NEW v3)
        "corr_spy": _safe(corr_spy),
        "corr_vix": _safe(corr_vix),
        "beta_market": _safe(beta_mkt),

        # Macro (NEW v3)
        "vix_level": _safe(vix_level),
        "vix_pct_60d": _safe(vix_pct_60d),
        "rate_10y": _safe(rate_10y),
        "rate_2y": _safe(rate_2y),
        "term_spread": _safe(term_spread),
        "credit_spread_hy": _safe(credit_spread_hy),
        "vol_regime": vol_regime,

        # Derived ratios
        "var99_var95": _safe(_ratio(var99, var95)),
        "es99_es95": _safe(_ratio(es99, es95)),
        "es95_var95": _safe(_ratio(es95, var95)),
        "es99_var99": _safe(_ratio(es99, var99)),
        "vol_to_var95": _safe(_ratio(vol_ann, var95 * np.sqrt(252) if var95 else None)),
        "vol20_vol_ann": _safe(_ratio(vol_20d, vol_ann / np.sqrt(252) if vol_ann else None)),
        "vol60_vol_ann": _safe(_ratio(vol_60d, vol_ann)),
        "vol120_vol_ann": _safe(_ratio(vol_120d, vol_ann)),
        "vol20_vol60": _safe(_ratio(
            vol_20d, vol_60d / np.sqrt(252) if vol_60d else None
        )),
        "dd_to_var99": _safe(_ratio(abs(mdd) if mdd else None, var99)),
        "rsi_centered": _safe((rsi_val - 50.0) / 50.0 if rsi_val is not None else None),
        "abs_corr_mkt": _safe(abs(corr_spy) if corr_spy is not None else None),
        "log_n_used": _safe(np.log1p(n_used)),
        "downside_div_vol": _safe(_ratio(downside_dev, vol_ann)),
        "worst_5d_vs_var99": _safe(_ratio(
            abs(worst_5d) if worst_5d else None, var99
        )),
        "dd_duration_per_n": _safe(_ratio(dd_dur, n_used) if dd_dur else None),
        "recovery_per_dd": _safe(_ratio(recovery, dd_dur) if recovery and dd_dur else None),
    }

    return feats


# ===================================================================
# LABELS — multi-horizon, temporally clean
# ===================================================================

def compute_labels(
    ret_past: pd.Series,
    px_future: pd.Series,
    ret_future: pd.Series,
) -> Dict[str, Any]:
    """
    Compute multi-horizon labels.
    Returns dict with forward returns and classification target.
    """
    result: Dict[str, Any] = {}

    # Multi-horizon forward returns
    for horizon in [5, 10, 20, 60]:
        if len(ret_future) >= horizon:
            fwd_ret = float(np.prod(1.0 + ret_future.iloc[:horizon].to_numpy(dtype=float)) - 1.0)
            result[f"forward_return_{horizon}d"] = round(fwd_ret, 6)
        else:
            result[f"forward_return_{horizon}d"] = None

    # Primary classification target: is next 20d "non-ok"?
    # Using same rules as v2 for backward compatibility
    if len(ret_future) >= 20 and len(px_future) >= 20:
        # Future drawdown
        fut_dd = max_drawdown(px_future.iloc[:20])
        # Volatility ratio
        v_past = float(np.std(ret_past.to_numpy(dtype=float), ddof=1) * np.sqrt(252))
        v_fut = float(np.std(ret_future.iloc[:20].to_numpy(dtype=float), ddof=1) * np.sqrt(252))
        vol_ratio = v_fut / (v_past + 1e-12) if np.isfinite(v_past) and v_past > 1e-12 else float("inf")

        if fut_dd <= -0.12 or vol_ratio >= 2.5:
            label = "block"
        elif fut_dd <= -0.07 or vol_ratio >= 1.8:
            label = "warn"
        else:
            label = "ok"

        result["label"] = label
        result["target_non_ok"] = 0 if label == "ok" else 1
        result["future_dd_20d"] = round(float(fut_dd), 6)
        result["future_vol_ratio"] = round(float(vol_ratio), 4) if np.isfinite(vol_ratio) else None
    else:
        result["label"] = None
        result["target_non_ok"] = None
        result["future_dd_20d"] = None
        result["future_vol_ratio"] = None

    return result


# ===================================================================
# WORKER — processes one ticker
# ===================================================================

# Module-level shared data (set before Pool starts)
_SHARED_MACRO: Dict[str, pd.Series] = {}
_SHARED_SPY: pd.Series = pd.Series(dtype=float)


def _init_worker(macro_dict: Dict[str, pd.Series], spy_ret: pd.Series):
    """Initializer for pool workers — sets shared macro data."""
    global _SHARED_MACRO, _SHARED_SPY
    _SHARED_MACRO = macro_dict
    _SHARED_SPY = spy_ret


def process_ticker(task: Dict[str, Any]) -> Tuple[List[str], str, Optional[str]]:
    """
    Process one ticker into v3 JSONL records.
    Returns (json_lines, ticker, error_msg | None).
    """
    ticker = task["ticker"]
    asset_type = task["asset_type"]
    market = task["market"]
    start = task["start"]
    lookback = task.get("lookback_days", _LOOKBACK)
    horizon = task.get("horizon_days", _HORIZON_PRIMARY)
    step = task.get("step_days", _STEP_DAYS)
    max_windows = task.get("max_per_ticker", _MAX_PER_TICKER)

    try:
        close, source = download_close(ticker, market, start, asset_type)
        close = pd.Series(close).dropna()
        if close.empty:
            return [], ticker, "empty close series"

        ret = close.pct_change()

        min_len = lookback + horizon + 60 + 30  # extra margin for 60d forward + warmup
        if len(close) < min_len:
            return [], ticker, f"too short ({len(close)}<{min_len}) [{source}]"

        # Generate window end indices
        min_end = lookback
        max_end = len(close) - horizon - 60 - 1  # ensure room for 60d forward
        if max_end <= min_end + 10:
            return [], ticker, f"not enough window room [{source}]"

        end_ixs = list(range(min_end, max_end + 1, step))
        if len(end_ixs) > max_windows:
            idx = np.linspace(0, len(end_ixs) - 1, max_windows, dtype=int)
            end_ixs = [end_ixs[i] for i in idx]

        lines: List[str] = []
        for end_ix in end_ixs:
            # Window slices
            past_slice = slice(max(0, end_ix - lookback), end_ix)
            # For labels: use up to 60d forward (max horizon)
            fut_slice = slice(end_ix, min(end_ix + 60 + 5, len(close)))

            px_past = close.iloc[past_slice]
            ret_past = ret.iloc[past_slice].dropna()
            px_fut = close.iloc[fut_slice]
            ret_fut = ret.iloc[fut_slice].dropna()

            if len(ret_past) < 60 or len(ret_fut) < 20:
                continue

            # Dates
            window_start = px_past.index[0]
            window_end = px_past.index[-1]
            label_start = px_fut.index[0] if len(px_fut) > 0 else None
            label_end = px_fut.index[min(19, len(px_fut) - 1)] if len(px_fut) > 0 else None

            # Build features
            feats = build_features_v3(
                ticker=ticker,
                asset_type=asset_type,
                market=market,
                closes=px_past,
                returns=ret_past,
                macro=_SHARED_MACRO,
                spy_returns=_SHARED_SPY,
                window_end_date=window_end,
            )
            if not feats:
                continue

            # Labels
            labels = compute_labels(ret_past, px_fut, ret_fut)

            # Skip if no primary label
            if labels.get("label") is None:
                continue

            # Build record
            rec = {
                "version": "v3",
                "label": labels["label"],
                "target_non_ok": labels["target_non_ok"],
                "window_start_date": str(window_start.date()),
                "window_end_date": str(window_end.date()),
                "label_start_date": str(label_start.date()) if label_start is not None else None,
                "label_end_date": str(label_end.date()) if label_end is not None else None,
                "forward_return_5d": labels.get("forward_return_5d"),
                "forward_return_10d": labels.get("forward_return_10d"),
                "forward_return_20d": labels.get("forward_return_20d"),
                "forward_return_60d": labels.get("forward_return_60d"),
                "future_dd_20d": labels.get("future_dd_20d"),
                "future_vol_ratio": labels.get("future_vol_ratio"),
                "source": source,
                "features": feats,
            }

            lines.append(json.dumps(rec, ensure_ascii=False))

        return lines, ticker, None

    except Exception as e:
        return [], ticker, str(e)


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset v3 — temporally rigorous, multi-asset, macro-enriched")
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--out", default="data/training/train_v3_all.jsonl")
    ap.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    ap.add_argument("--lookback_years", type=int, default=7, help="Years of history (default 7)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=10)
    ap.add_argument("--max_per_ticker", type=int, default=80)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep_ticker", type=float, default=0.0)
    ap.add_argument("--skip_macro", action="store_true", help="Skip FRED macro download (for testing)")
    args = ap.parse_args()

    # Start date
    if args.start:
        start_date = args.start
    else:
        cutoff = datetime.today() - timedelta(days=int(args.lookback_years) * 365 + 300)
        start_date = cutoff.strftime("%Y-%m-%d")

    # Load universe
    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe not found: {uni_path}")
    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    log.info("Universe: %d tickers", len(uni))

    # Download macro data
    if args.skip_macro:
        log.info("Skipping macro data (--skip_macro)")
        macro = {}
        spy_ret = pd.Series(dtype=float)
    else:
        log.info("Downloading macro data from FRED...")
        macro = download_macro_data(start=start_date)
        log.info("Downloading SPY returns...")
        spy_ret = download_spy_returns(start=start_date)
        log.info("SPY returns: %d points", len(spy_ret))

    # Build task list — v3 does NOT skip FX/futures/indices
    tasks: List[Dict[str, Any]] = []
    for item in uni:
        ticker = str(item.get("ticker", "")).strip()
        if not ticker:
            continue
        tasks.append({
            "ticker": ticker,
            "asset_type": str(item.get("asset_type", "")).strip(),
            "market": str(item.get("market", "")).strip(),
            "start": start_date,
            "lookback_days": args.lookback_days,
            "horizon_days": args.horizon_days,
            "step_days": args.step_days,
            "max_per_ticker": args.max_per_ticker,
        })

    log.info("Tasks: %d tickers to process", len(tasks))
    log.info("History: start=%s, end=%s", start_date, args.end or "today")
    log.info("Windows: lookback=%dd, horizon=%dd, step=%dd, max=%d/ticker",
             args.lookback_days, args.horizon_days, args.step_days, args.max_per_ticker)

    # Output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"ok": 0, "warn": 0, "block": 0}
    n_tickers_ok = 0
    n_tickers_fail = 0
    total_windows = 0

    workers = max(1, args.workers)

    with out_path.open("w", encoding="utf-8") as out_file:
        if workers > 1:
            # Multi-process: use module-level _init_worker (picklable on macOS spawn)
            # pd.Series with DatetimeIndex is directly picklable — no need to serialise
            with multiprocessing.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(macro, spy_ret),
            ) as pool:
                for lines, ticker, err in pool.imap_unordered(process_ticker, tasks, chunksize=1):
                    if err:
                        n_tickers_fail += 1
                        log.warning("FAIL %s: %s", ticker, err)
                    else:
                        n_tickers_ok += 1
                        for ln in lines:
                            out_file.write(ln + "\n")
                            rec = json.loads(ln)
                            lab = rec.get("label", "ok")
                            counts[lab] = counts.get(lab, 0) + 1
                            total_windows += 1
                        if lines:
                            log.info("OK %s: %d windows", ticker, len(lines))
        else:
            # Sequential — set global macro
            global _SHARED_MACRO, _SHARED_SPY
            _SHARED_MACRO = macro
            _SHARED_SPY = spy_ret

            for task in tasks:
                lines, ticker, err = process_ticker(task)
                if err:
                    n_tickers_fail += 1
                    log.warning("FAIL %s: %s", ticker, err)
                else:
                    n_tickers_ok += 1
                    for ln in lines:
                        out_file.write(ln + "\n")
                        rec = json.loads(ln)
                        lab = rec.get("label", "ok")
                        counts[lab] = counts.get(lab, 0) + 1
                        total_windows += 1
                    if lines:
                        log.info("OK %s: %d windows", ticker, len(lines))

                if args.sleep_ticker > 0:
                    time.sleep(args.sleep_ticker)

    # Summary
    log.info("=" * 60)
    log.info("DATASET V3 COMPLETE")
    log.info("Output: %s", out_path)
    log.info("Tickers: %d ok, %d failed, %d total", n_tickers_ok, n_tickers_fail, len(tasks))
    log.info("Windows: %d total", total_windows)
    log.info("Labels: %s", counts)
    if total_windows > 0:
        pct = {k: f"{100*v/total_windows:.1f}%" for k, v in counts.items()}
        log.info("Distribution: %s", pct)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
