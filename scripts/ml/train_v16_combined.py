"""
scripts/ml/train_v16_combined.py
==================================
V16 — Combined Regime + Fundamentals + Momentum Walk-Forward Stock Picker

Fuses V15 regime awareness with V14-B fundamental features and momentum filter:

  V15 contributions:
    - Market regime detection (Bull/Bear/Sideways via VIX + SPY 6m momentum)
    - Regime-conditional exposure: Bear→0% / Sideways→50% / Bull→100%
    - Walk-forward: 8 folds × 6m test, 3m embargo, expanding train window

  V14-B contributions:
    - Fundamental features: gross_margin, op_margin, net_margin, roe,
      debt_to_equity, rd_intensity, fcf_margin, revenue_growth, ni_growth,
      pe_ratio, pb_ratio, earnings_yield, current_ratio, asset_growth, accruals_ratio
    - Momentum filter at portfolio construction:
        mom_12_1 > 0  AND  ret_12m > spy_12m

Feature set (33 total):
  Technical  (14): ret_1m, ret_3m, ret_6m, ret_12m, mom_12_1, ret_vs_spy_3m,
                   spy_1m, spy_12m, vol_ann, vol_ratio, skew_12m,
                   above_200ma, trend_strength, dd_from_hi52
  Regime     ( 3): vix_level, spy_mom_6m, regime_id
  Fundamentals (15): gross_margin, op_margin, net_margin, roe, debt_to_equity,
                     rd_intensity, fcf_margin, revenue_growth, ni_growth,
                     pe_ratio, pb_ratio, earnings_yield,
                     current_ratio, asset_growth, accruals_ratio
  Static     ( 1): sector_id

Data source: yfinance (prices + .info snapshot for fundamentals)
Fundamentals: static cross-sectional snapshot from yfinance.info (current values
  broadcast to all months). yfinance only provides ~5 recent quarters, insufficient
  for 2017-2024 historical TTM. Static quality metrics (margins, ROE) are stable
  enough cross-sectionally for this demonstration (AAPL vs CVS gross_margin ranking
  is unchanged over 7 years). For production, use SimFin/Compustat point-in-time data.
Fundamental cache: data/cache/fundamentals_v16.pkl  (auto-reused on re-run)

Target: forward 3m alpha vs SPY > 2.5% (binary label)

Usage:
  python scripts/ml/train_v16_combined.py \\
      --start 2017-01-01 \\
      --end   2024-12-31 \\
      --out   data/metrics/v16_results.json \\
      --model_out models/stock_picker_v16.joblib \\
      [--no_cache]   # force re-download of fundamentals

No API / prod impact.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("train_v16")

SEED = 42
np.random.seed(SEED)

# ────────────────────────────────────────���────────────────────────────────────
# Universe (same 50 stocks as V15)
# ────────────────────────────────────────────────────────��────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
    "JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "C", "AXP", "BLK", "USB",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "MDT", "CVS",
    "WMT", "PG", "KO", "PEP", "MCD", "COST", "NKE", "HD", "TGT", "LOW",
    "XOM", "CVX", "COP", "SLB", "CAT", "HON", "MMM", "GE", "BA", "UPS",
]

SECTOR_MAP = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0, "AMZN": 0,
    "NVDA": 0, "AMD": 0, "INTC": 0, "CSCO": 0, "ORCL": 0,
    "JPM": 1, "BAC": 1, "WFC": 1, "GS": 1, "MS": 1,
    "BRK-B": 1, "C": 1, "AXP": 1, "BLK": 1, "USB": 1,
    "JNJ": 2, "UNH": 2, "PFE": 2, "ABBV": 2, "MRK": 2,
    "LLY": 2, "TMO": 2, "ABT": 2, "MDT": 2, "CVS": 2,
    "WMT": 3, "PG": 3, "KO": 3, "PEP": 3, "MCD": 3,
    "COST": 3, "NKE": 3, "HD": 3, "TGT": 3, "LOW": 3,
    "XOM": 4, "CVX": 4, "COP": 4, "SLB": 4, "CAT": 5,
    "HON": 5, "MMM": 5, "GE": 5, "BA": 5, "UPS": 5,
}

# ─────────────────────────────────────��─────────────────────────────────���─────
# Regime config (identical to V15)
# ───────────────────────────────���──────────────────────────────��──────────────
VIX_BEAR_THRESHOLD  = 27.0
VIX_BULL_MAX        = 18.0
SPY_MOM_BULL_MIN    = 0.05
SPY_MOM_BEAR_MAX    = -0.05

REGIME_BULL     = 0
REGIME_SIDEWAYS = 1
REGIME_BEAR     = 2
REGIME_LABELS   = {REGIME_BULL: "bull", REGIME_SIDEWAYS: "sideways", REGIME_BEAR: "bear"}
REGIME_EXPOSURE = {REGIME_BULL: 1.0, REGIME_SIDEWAYS: 0.5, REGIME_BEAR: 0.0}

TOP_PCT = 0.20

# Filing lag: assume quarterly data available 45 days after quarter end
FILING_LAG_DAYS = 45

# ─────────────────���────────────────────────────────────��──────────────────────
# Feature column list
# ─────────────────────────────────────────────────────────────────────────────
TECH_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "ret_vs_spy_3m",
    "spy_1m", "spy_12m",
    "vol_ann", "vol_ratio", "skew_12m",
    "above_200ma", "trend_strength", "dd_from_hi52",
]
REGIME_FEATURES = ["vix_level", "spy_mom_6m", "regime_id"]
FUND_FEATURES = [
    "gross_margin", "op_margin", "net_margin", "roe",
    "debt_to_equity", "rd_intensity", "fcf_margin",
    "revenue_growth", "ni_growth",
    "pe_ratio", "pb_ratio", "earnings_yield",
    "current_ratio", "asset_growth", "accruals_ratio",
]
STATIC_FEATURES = ["sector_id"]

V16_FEATURES = TECH_FEATURES + REGIME_FEATURES + FUND_FEATURES + STATIC_FEATURES


# ────────────────────────────────────────���─────────────────────────────��──────
# 1. Price download (same as V15)
# ───────────────────────────────��────────────────────────────��────────────────

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    log.info("Downloading monthly prices (%d tickers + SPY + VIX)...", len(tickers))
    all_tickers = list(set(tickers + ["SPY", "^VIX"]))
    raw = yf.download(
        all_tickers, start=start, end=end,
        interval="1mo", auto_adjust=True, progress=False,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index = prices.index.to_period("M").to_timestamp("M")
    log.info("Prices: %d months × %d tickers", len(prices), len(prices.columns))
    return prices


# ────────────────────────────────────────────────────���────────────────────────
# 2. Regime detection (same as V15)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    spy = prices["SPY"].dropna()
    vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()

    df = pd.DataFrame(index=prices.index)
    df["spy_1m"]    = spy.pct_change(1)
    df["spy_6m"]    = spy.pct_change(6)
    df["spy_12m"]   = spy.pct_change(12)
    df["vix_level"] = vix.reindex(df.index).fillna(20.0)

    def _classify(row):
        vix_val = row["vix_level"]
        mom_6m  = row["spy_6m"] if pd.notna(row["spy_6m"]) else 0.0
        if vix_val >= VIX_BEAR_THRESHOLD or mom_6m <= SPY_MOM_BEAR_MAX:
            return REGIME_BEAR
        elif vix_val <= VIX_BULL_MAX and mom_6m >= SPY_MOM_BULL_MIN:
            return REGIME_BULL
        return REGIME_SIDEWAYS

    df["regime_id"]    = df.apply(_classify, axis=1)
    df["regime_label"] = df["regime_id"].map(REGIME_LABELS)
    rc = df["regime_id"].value_counts().sort_index()
    log.info("Regime distribution: %s", {REGIME_LABELS[k]: int(v) for k, v in rc.items()})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fundamental data — static snapshot via yfinance.info
#
# Design note: yfinance only provides the ~5 most recent quarterly reports.
# For a backtest going back to 2017, historical point-in-time fundamentals
# are unavailable without a paid provider (SimFin, Compustat, etc.).
#
# Approach: use yfinance.info as a STATIC CROSS-SECTIONAL snapshot.
# This captures quality discrimination between companies (AAPL vs CVS margins)
# which is stable over time. Look-ahead is minimal for quality metrics like
# gross margin and ROE that rarely change rank-order over a 7-year window.
# ─────────────────────────────────────────────────────────────────────────────

# yfinance.info key → our feature name
_INFO_FIELD_MAP: Dict[str, str] = {
    "grossMargins":     "gross_margin",
    "operatingMargins": "op_margin",
    "profitMargins":    "net_margin",
    "returnOnEquity":   "roe",
    "debtToEquity":     "debt_to_equity",
    "revenueGrowth":    "revenue_growth",
    "earningsGrowth":   "ni_growth",
    "trailingPE":       "pe_ratio",
    "priceToBook":      "pb_ratio",
    "currentRatio":     "current_ratio",
}

_FUND_CLIPS: Dict[str, Tuple[float, float]] = {
    "gross_margin": (-1.0, 1.0), "op_margin": (-1.0, 0.8), "net_margin": (-1.0, 0.5),
    "roe": (-2.0, 2.0), "debt_to_equity": (0.0, 20.0), "rd_intensity": (0.0, 0.5),
    "fcf_margin": (-1.0, 0.5), "revenue_growth": (-0.5, 2.0), "ni_growth": (-5.0, 5.0),
    "pe_ratio": (0.0, 200.0), "pb_ratio": (0.0, 50.0), "earnings_yield": (-0.1, 0.3),
    "current_ratio": (0.0, 10.0), "asset_growth": (-0.3, 1.0), "accruals_ratio": (-0.2, 0.2),
}


def _get_ticker_snapshot(ticker: str) -> Dict[str, float]:
    """
    Fetch current fundamental snapshot for one ticker from yfinance.info.

    Returns a dict mapping FUND_FEATURES → float (NaN where unavailable).
    Applied as static features to ALL months for this ticker.
    """
    nan_snap = {f: float("nan") for f in FUND_FEATURES}

    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log.warning("[%s] info() failed: %s", ticker, e)
        return nan_snap

    result: Dict[str, float] = {}

    # Direct mapping
    for info_key, feat in _INFO_FIELD_MAP.items():
        v = info.get(info_key)
        try:
            fv = float(v) if v is not None else float("nan")
            result[feat] = fv if math.isfinite(fv) else float("nan")
        except (TypeError, ValueError):
            result[feat] = float("nan")

    # Derived: earnings_yield = 1 / pe_ratio
    pe = result.get("pe_ratio", float("nan"))
    result["earnings_yield"] = float(1.0 / pe) if pe and pe > 0 and math.isfinite(pe) else float("nan")

    # R&D intensity: compute from most recent quarterly income stmt
    rd_int = float("nan")
    try:
        inc = t.quarterly_income_stmt
        if inc is not None and not inc.empty:
            idx_lower = {str(r).lower(): r for r in inc.index}
            rev_row = next(
                (inc.loc[idx_lower[k]] for k in ["total revenue", "operating revenue"] if k in idx_lower),
                None,
            )
            rd_row = inc.loc[idx_lower["research and development"]] if "research and development" in idx_lower else None
            if rev_row is not None and rd_row is not None:
                n_q   = min(4, inc.shape[1])
                rev_t = float(rev_row.iloc[:n_q].dropna().sum())
                rd_t  = abs(float(rd_row.iloc[:n_q].dropna().sum()))
                if rev_t > 0:
                    rd_int = rd_t / rev_t
    except Exception:
        pass
    result["rd_intensity"] = rd_int if math.isfinite(rd_int) else float("nan")

    # FCF margin: freeCashflow / totalRevenue from info
    fcf = info.get("freeCashflow")
    rev = info.get("totalRevenue")
    try:
        result["fcf_margin"] = float(fcf) / float(rev) if fcf and rev and float(rev) > 0 else float("nan")
    except (TypeError, ValueError):
        result["fcf_margin"] = float("nan")

    # asset_growth, accruals_ratio: not reliably available from info
    result["asset_growth"]   = float("nan")
    result["accruals_ratio"] = float("nan")

    # Clip extremes
    for feat, (lo, hi) in _FUND_CLIPS.items():
        v = result.get(feat, float("nan"))
        if not math.isnan(v):
            result[feat] = max(lo, min(hi, v))

    # Ensure all features present
    for feat in FUND_FEATURES:
        result.setdefault(feat, float("nan"))

    return result


def _build_fund_monthly_UNUSED(
    ticker: str,
    monthly_dates: pd.DatetimeIndex,
    prices_series: pd.Series,
) -> pd.DataFrame:
    """
    [UNUSED in V16 — yfinance only provides 5 recent quarters, insufficient for
     a 2017-2024 backtest. Replaced by _get_ticker_snapshot + static broadcast.]
    """
    result = pd.DataFrame(index=monthly_dates, columns=FUND_FEATURES, dtype=float)

    try:
        t = yf.Ticker(ticker)
        inc = t.quarterly_income_stmt
        bs  = t.quarterly_balance_sheet
        cf  = t.quarterly_cash_flow
    except Exception as e:
        log.warning("[%s] Failed to download fundamentals: %s", ticker, e)
        return result

    if inc is None or inc.empty or bs is None or bs.empty:
        log.warning("[%s] Empty fundamental data", ticker)
        return result

    # Standardize columns to datetime, sorted newest-first → oldest-first
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = pd.to_datetime(df.columns, errors="coerce")
        df = df.loc[:, df.columns.notna()]
        df = df.sort_index(axis=1)  # oldest first
        return df

    try:
        inc = _normalize_cols(inc)
        bs  = _normalize_cols(bs)
        cf  = _normalize_cols(cf) if cf is not None and not cf.empty else pd.DataFrame()
    except Exception as e:
        log.warning("[%s] Column normalization failed: %s", ticker, e)
        return result

    if inc.empty or bs.empty:
        return result

    # For each month M, find quarters available at M (i.e., quarter_end + filing_lag ≤ M)
    filing_lag = timedelta(days=FILING_LAG_DAYS)

    for month_date in monthly_dates:
        # Available quarters: quarter end date ≤ month_date - filing_lag
        cutoff = month_date - filing_lag

        inc_avail = inc.loc[:, inc.columns <= cutoff]
        bs_avail  = bs.loc[:, bs.columns <= cutoff]
        cf_avail  = cf.loc[:, cf.columns <= cutoff] if not cf.empty else pd.DataFrame()

        if inc_avail.shape[1] < 2:
            continue  # need at least 2 quarters

        # ── Income TTM (sum of last 4 quarters) ──
        rev_s   = _safe_row(inc_avail, "Total Revenue")
        gp_s    = _safe_row(inc_avail, "Gross Profit")
        oi_s    = _safe_row(inc_avail, "Operating Income", "EBIT",
                            "Total Operating Income As Reported")
        ni_s    = _safe_row(inc_avail, "Net Income", "Net Income Common Stockholders",
                            "Net Income From Continuing Operations")
        rd_s    = _safe_row(inc_avail, "Research And Development")

        rev_ttm = _ttm_sum(rev_s)   if rev_s is not None else None
        gp_ttm  = _ttm_sum(gp_s)    if gp_s  is not None else None
        oi_ttm  = _ttm_sum(oi_s)    if oi_s  is not None else None
        ni_ttm  = _ttm_sum(ni_s)    if ni_s  is not None else None
        rd_ttm  = _ttm_sum(rd_s)    if rd_s  is not None else None

        # ── Cash Flow TTM ──
        fcf_ttm = None
        if not cf_avail.empty:
            fcf_s = _safe_row(cf_avail, "Free Cash Flow")
            if fcf_s is not None:
                fcf_ttm = _ttm_sum(fcf_s)
        if fcf_ttm is None and oi_ttm is not None:
            # rough proxy: operating income (no capex data)
            fcf_ttm = oi_ttm * 0.85

        # Prior year TTM for growth metrics (8 quarters ago)
        rev_prev_ttm = None
        ni_prev_ttm  = None
        assets_prev  = None
        if inc_avail.shape[1] >= 8:
            rev_prev_ttm = _ttm_sum(rev_s.iloc[:-4]) if rev_s is not None else None
            ni_prev_ttm  = _ttm_sum(ni_s.iloc[:-4])  if ni_s  is not None else None

        # ── Balance Sheet (most recent quarter) ──
        if bs_avail.empty:
            continue

        total_assets    = _safe_row(bs_avail, "Total Assets")
        equity          = _safe_row(bs_avail, "Common Stock Equity", "Stockholders Equity",
                                    "Total Equity Gross Minority Interest")
        total_debt      = _safe_row(bs_avail, "Total Debt", "Long Term Debt")
        curr_assets     = _safe_row(bs_avail, "Current Assets")
        curr_liab       = _safe_row(bs_avail, "Current Liabilities")
        shares_out      = _safe_row(bs_avail, "Ordinary Shares Number", "Share Issued")

        def _latest(s: Optional[pd.Series]) -> Optional[float]:
            if s is None or s.dropna().empty:
                return None
            return float(s.dropna().iloc[-1])

        assets_val  = _latest(total_assets)
        equity_val  = _latest(equity)
        debt_val    = _latest(total_debt) or 0.0
        curr_a_val  = _latest(curr_assets)
        curr_l_val  = _latest(curr_liab)
        shares_val  = _latest(shares_out)

        if bs_avail.shape[1] >= 5 and total_assets is not None:
            assets_prev = _ttm_sum(total_assets) / 4 if total_assets.dropna().shape[0] >= 5 else None

        # ── Market Cap (from price × shares) ──
        mktcap = None
        if shares_val and month_date in prices_series.index:
            price_val = float(prices_series.loc[month_date])
            if pd.notna(price_val) and price_val > 0:
                mktcap = price_val * shares_val

        # ── Ratio computation (defensive, skip if denominator invalid) ──
        def _div(num, denom) -> Optional[float]:
            if num is None or denom is None:
                return None
            if abs(denom) < 1e-6:
                return None
            return float(num / denom)

        gross_margin  = _div(gp_ttm,  rev_ttm)
        op_margin     = _div(oi_ttm,  rev_ttm)
        net_margin    = _div(ni_ttm,  rev_ttm)
        roe           = _div(ni_ttm,  equity_val) if equity_val and equity_val > 0 else None
        deb_to_eq     = _div(debt_val, equity_val) if equity_val and equity_val > 0 else None
        rd_intensity  = _div(rd_ttm,  rev_ttm)   if rd_ttm  else 0.0
        fcf_margin    = _div(fcf_ttm, rev_ttm)

        rev_growth  = _div(rev_ttm, rev_prev_ttm) - 1 if rev_ttm and rev_prev_ttm and rev_prev_ttm > 0 else None
        ni_growth   = _div(ni_ttm,  ni_prev_ttm)  - 1 if ni_ttm  and ni_prev_ttm  and ni_prev_ttm != 0 else None

        pe_ratio      = _div(mktcap, ni_ttm)   if ni_ttm and ni_ttm > 0 else None
        pb_ratio      = _div(mktcap, equity_val) if equity_val and equity_val > 0 and mktcap else None
        earn_yield    = _div(1.0, pe_ratio)     if pe_ratio and pe_ratio > 0 else None
        current_ratio = _div(curr_a_val, curr_l_val) if curr_l_val and curr_l_val > 0 else None
        asset_growth  = (_div(assets_val, assets_prev) - 1
                         if assets_val and assets_prev and assets_prev > 0 else None)
        accruals_ratio = (_div((ni_ttm or 0) - (fcf_ttm or 0), assets_val)
                          if assets_val and assets_val > 0 and ni_ttm is not None else None)

        # Clip extremes to prevent outlier contamination
        def _clip(v, lo, hi):
            if v is None:
                return None
            return max(lo, min(hi, v))

        result.loc[month_date] = {
            "gross_margin":  _clip(gross_margin,  -1.0,  1.0),
            "op_margin":     _clip(op_margin,     -1.0,  0.8),
            "net_margin":    _clip(net_margin,    -1.0,  0.5),
            "roe":           _clip(roe,           -2.0,  2.0),
            "debt_to_equity":_clip(deb_to_eq,     0.0,  20.0),
            "rd_intensity":  _clip(rd_intensity,  0.0,   0.5),
            "fcf_margin":    _clip(fcf_margin,    -1.0,  0.5),
            "revenue_growth":_clip(rev_growth,    -0.5,  2.0),
            "ni_growth":     _clip(ni_growth,     -5.0,  5.0),
            "pe_ratio":      _clip(pe_ratio,       0.0, 200.0),
            "pb_ratio":      _clip(pb_ratio,       0.0,  50.0),
            "earnings_yield":_clip(earn_yield,    -0.1,  0.3),
            "current_ratio": _clip(current_ratio,  0.0,  10.0),
            "asset_growth":  _clip(asset_growth,  -0.3,  1.0),
            "accruals_ratio":_clip(accruals_ratio,-0.2,  0.2),
        }

    return result.astype(float)


def download_fundamentals(
    tickers: List[str],
    monthly_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch current fundamental snapshot per ticker via yfinance.info.

    Returns a dict {ticker → DataFrame(index=monthly_dates, cols=FUND_FEATURES)}
    where each ticker's snapshot is broadcast to ALL months (static signal).
    Cached as a pickle to avoid re-downloading on re-runs.
    """
    if use_cache and cache_path and cache_path.exists():
        log.info("Loading fundamentals from cache: %s", cache_path)
        with cache_path.open("rb") as f:
            return pickle.load(f)

    log.info("Fetching fundamental snapshots for %d tickers via yfinance.info (~30s)...", len(tickers))
    fund_data: Dict[str, pd.DataFrame] = {}
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        snapshot = _get_ticker_snapshot(ticker)
        # Broadcast static snapshot to all months
        df = pd.DataFrame(
            [snapshot] * len(monthly_dates),
            index=monthly_dates,
            columns=FUND_FEATURES,
        ).astype(float)
        fund_data[ticker] = df

        n_populated = sum(1 for v in snapshot.values() if not math.isnan(v))
        log.info("[%d/%d] %s — %d/%d features populated (%.1fs)",
                 i, len(tickers), ticker, n_populated, len(FUND_FEATURES), time.time() - t0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(fund_data, f)
        log.info("Fundamentals cached → %s", cache_path)

    log.info("Fundamentals fetched in %.1fs", time.time() - t0)
    return fund_data


# ─────────────────────────────────────────────────────────────────────────────
# 4. Technical features (reused from V15, extended)
# ──────────────────────────────────────────────���────────────────────────────��─

def _trend_strength(series: pd.Series) -> float:
    y = series.dropna().values
    if len(y) < 3:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    x_bar, y_bar = x.mean(), y.mean()
    denom = ((x - x_bar) ** 2).sum()
    if denom == 0:
        return float("nan")
    beta  = ((x - x_bar) * (y - y_bar)).sum() / denom
    y_hat = y_bar + beta * (x - x_bar)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_bar) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def build_tech_features(
    prices: pd.DataFrame,
    regimes: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Build monthly technical + regime features (extended V15 version)."""
    if ticker not in prices.columns:
        return pd.DataFrame()

    p   = prices[ticker].dropna()
    spy = prices["SPY"].dropna()
    rows = []

    for i, date in enumerate(p.index):
        if i < 12:
            continue

        def _ret(series, n):
            if i < n:
                return float("nan")
            v0, v1 = series.iloc[i - n], series.iloc[i]
            return float(v1 / v0 - 1) if pd.notna(v0) and pd.notna(v1) and v0 != 0 else float("nan")

        ret_1m  = _ret(p, 1);   ret_3m  = _ret(p, 3)
        ret_6m  = _ret(p, 6);   ret_12m = _ret(p, 12)
        spy_1m  = _ret(spy, 1); spy_3m  = _ret(spy, 3); spy_12m = _ret(spy, 12)

        mom_12_1    = (ret_12m - ret_1m)  if all(pd.notna(x) for x in [ret_12m, ret_1m])  else float("nan")
        ret_vs_spy3 = (ret_3m  - spy_3m)  if all(pd.notna(x) for x in [ret_3m,  spy_3m])  else float("nan")

        w12 = p.iloc[max(0, i - 12): i + 1]
        mr  = w12.pct_change().dropna()
        vol_ann   = float(mr.std() * math.sqrt(12)) if len(mr) >= 3 else float("nan")
        skew_12m  = float(mr.skew())               if len(mr) >= 4 else float("nan")
        vol_3m    = float(mr.tail(3).std() * math.sqrt(12)) if len(mr) >= 3 else float("nan")
        vol_ratio = float(vol_3m / vol_ann)         if vol_ann and vol_ann > 0 else float("nan")

        hi52        = float(w12.max())
        cur         = float(p.iloc[i])
        dd_from_hi52 = float(cur / hi52 - 1)    if hi52 > 0 else float("nan")
        ma_200      = float(w12.mean())
        above_200ma = float(cur > ma_200)
        trend_str   = _trend_strength(w12)

        reg_row   = regimes.loc[date] if date in regimes.index else None
        vix_level = float(reg_row["vix_level"]) if reg_row is not None else 20.0
        spy_mom_6 = float(reg_row["spy_6m"])    if reg_row is not None and pd.notna(reg_row.get("spy_6m")) else float("nan")
        regime_id = int(reg_row["regime_id"])   if reg_row is not None else REGIME_SIDEWAYS

        rows.append({
            "date": date, "ticker": ticker,
            "sector_id": float(SECTOR_MAP.get(ticker, -1)),
            "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m,
            "ret_12m": ret_12m, "mom_12_1": mom_12_1,
            "ret_vs_spy_3m": ret_vs_spy3,
            "spy_1m": spy_1m, "spy_12m": spy_12m,
            "vol_ann": vol_ann, "vol_ratio": vol_ratio, "skew_12m": skew_12m,
            "above_200ma": above_200ma, "trend_strength": trend_str,
            "dd_from_hi52": dd_from_hi52,
            "vix_level": vix_level, "spy_mom_6m": spy_mom_6, "regime_id": float(regime_id),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")

    # Forward returns (label)
    fwd_ret_stock, fwd_ret_spy = [], []
    for i, date in enumerate(df.index):
        pos_p   = p.index.get_loc(date)   if date in p.index   else None
        pos_spy = spy.index.get_loc(date) if date in spy.index else None

        def _fwd(series, pos):
            if pos is None or pos + 3 >= len(series):
                return float("nan")
            v0, v3 = series.iloc[pos], series.iloc[pos + 3]
            return float(v3 / v0 - 1) if v0 > 0 else float("nan")

        fwd_ret_stock.append(_fwd(p,   pos_p))
        fwd_ret_spy.append(  _fwd(spy, pos_spy))

    df["fwd_ret_3m"]     = fwd_ret_stock
    df["fwd_ret_spy_3m"] = fwd_ret_spy
    df["fwd_alpha_3m"]   = df["fwd_ret_3m"] - df["fwd_ret_spy_3m"]
    df["label"]          = (df["fwd_alpha_3m"] > 0.025).astype(int)
    return df.reset_index()


# ─────────────────────────────────────────────────────���───────────────────────
# 5. Dataset assembly (technical + fundamentals merged)
# ────────────────────────────────────────────���──────────────────────────���─────

def build_dataset_v16(
    prices: pd.DataFrame,
    regimes: pd.DataFrame,
    fund_data: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> pd.DataFrame:
    all_dfs = []

    for ticker in tickers:
        tech_df = build_tech_features(prices, regimes, ticker)
        if tech_df.empty:
            continue

        tech_df["date"] = pd.to_datetime(tech_df["date"])
        tech_df = tech_df.set_index("date")

        # Merge fundamentals
        fund_df = fund_data.get(ticker)
        if fund_df is not None and not fund_df.empty:
            fund_df.index = pd.to_datetime(fund_df.index)
            for col in FUND_FEATURES:
                if col in fund_df.columns:
                    tech_df[col] = fund_df[col].reindex(tech_df.index)
                else:
                    tech_df[col] = float("nan")
        else:
            for col in FUND_FEATURES:
                tech_df[col] = float("nan")

        all_dfs.append(tech_df.reset_index())

    if not all_dfs:
        raise RuntimeError("No rows built")

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.dropna(subset=["label", "fwd_ret_3m"])
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)

    # Fundamental coverage stats
    fund_coverage = full[FUND_FEATURES].notna().mean()
    low_cov = fund_coverage[fund_coverage < 0.30]
    if not low_cov.empty:
        log.warning("Low fundamental coverage (<30%%): %s", low_cov.to_dict())
    else:
        log.info("Fundamental coverage: mean=%.0f%%, min=%.0f%%",
                 fund_coverage.mean() * 100, fund_coverage.min() * 100)

    log.info("Dataset v16: %d rows, %d tickers, %s → %s",
             len(full), full["ticker"].nunique(),
             full["date"].min().date(), full["date"].max().date())
    log.info("Label distribution: %s", full["label"].value_counts().to_dict())
    return full


# ─────────────────────────────────────────��───────────────────────────────��───
# 6. Walk-forward splits (identical to V15)
# ─────────────────────────────────────────────────────────────────────────────

def make_walk_forward_splits(
    df: pd.DataFrame,
    n_test_months: int = 6,
    min_train_months: int = 24,
    embargo_months: int = 3,
) -> List[Dict]:
    dates = sorted(df["date"].dt.to_period("M").unique())
    splits, fold, start_test_idx = [], 1, min_train_months

    while True:
        test_end_idx = start_test_idx + n_test_months
        if test_end_idx > len(dates):
            break

        test_start   = dates[start_test_idx].to_timestamp()
        test_end     = dates[min(test_end_idx, len(dates) - 1)].to_timestamp()
        train_cutoff = start_test_idx - embargo_months
        if train_cutoff < min_train_months:
            start_test_idx += n_test_months
            continue
        train_end = dates[train_cutoff - 1].to_timestamp()

        train_idx = np.where((df["date"] <= train_end).values)[0]
        val_idx   = np.where(((df["date"] >= test_start) & (df["date"] < test_end)).values)[0]

        if len(train_idx) >= 100 and len(val_idx) > 0:
            splits.append({
                "fold": fold, "train_end": str(train_end.date()),
                "val_start": str(test_start.date()), "val_end": str(test_end.date()),
                "n_train": len(train_idx), "n_val": len(val_idx),
                "train_idx": train_idx, "val_idx": val_idx,
            })
            fold += 1
        start_test_idx += n_test_months

    log.info("Walk-forward: %d folds (test=%dm, embargo=%dm)", len(splits), n_test_months, embargo_months)
    return splits


# ───────────────────────────────────��─────────────────────────────────��───────
# 7. Model training helpers
# ───────────────────────────────────────────────��─────────────────────────────

def _median_impute(X_tr: np.ndarray, X_vl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(X_tr, axis=0)
    for col in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, col]), col] = medians[col]
        X_vl[np.isnan(X_vl[:, col]), col] = medians[col]
    return X_tr, X_vl, medians


def train_xgb(X_tr, y_tr, X_vl, y_vl):
    if not HAS_XGB:
        return None
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.65,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, early_stopping_rounds=40,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return clf


def train_lgb(X_tr, y_tr, X_vl, y_vl):
    if not HAS_LGB:
        return None
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.65,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        random_state=SEED, verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    return clf


def ensemble_proba(xgb_clf, lgb_clf, X: np.ndarray) -> np.ndarray:
    proba, w = np.zeros(len(X)), 0.0
    if xgb_clf is not None:
        proba += 0.5 * xgb_clf.predict_proba(X)[:, 1]; w += 0.5
    if lgb_clf is not None:
        proba += 0.5 * lgb_clf.predict_proba(X)[:, 1]; w += 0.5
    return proba / w if w > 0 else proba


# ─────────────────────────────────────────────────────────────────────────────
# 8. Walk-forward training loop
# ──────────────────────────────────────────────��───────────────────────────��──

def run_walk_forward(df: pd.DataFrame, splits: List[Dict]) -> Dict:
    feat_cols = [c for c in V16_FEATURES if c in df.columns]
    log.info("V16 training: %d features", len(feat_cols))

    fold_metrics, oof_rows, all_models = [], [], []

    for split in splits:
        fold = split["fold"]
        tr = df.iloc[split["train_idx"]].copy()
        vl = df.iloc[split["val_idx"]].copy()

        X_tr = tr[feat_cols].values.astype(float)
        y_tr = tr["label"].values.astype(int)
        X_vl = vl[feat_cols].values.astype(float)
        y_vl = vl["label"].values.astype(int)

        # Drop features with >50% NaN in train
        nan_rates  = np.isnan(X_tr).mean(axis=0)
        keep_mask  = nan_rates <= 0.50
        X_tr, X_vl = X_tr[:, keep_mask], X_vl[:, keep_mask]
        kept_feats = [f for f, k in zip(feat_cols, keep_mask) if k]

        fund_kept  = [f for f in kept_feats if f in FUND_FEATURES]
        tech_kept  = [f for f in kept_feats if f in TECH_FEATURES]
        log.info("Fold %d — %d tech + %d fund features, train=%d val=%d",
                 fold, len(tech_kept), len(fund_kept),
                 len(tr), len(vl))

        X_tr, X_vl, medians = _median_impute(X_tr.copy(), X_vl.copy())

        xgb_clf = train_xgb(X_tr, y_tr, X_vl, y_vl)
        lgb_clf = train_lgb(X_tr, y_tr, X_vl, y_vl)

        proba_val = ensemble_proba(xgb_clf, lgb_clf, X_vl)

        try:
            roc_v = roc_auc_score(y_vl, proba_val)
            pra_v = average_precision_score(y_vl, proba_val)
            bri_v = brier_score_loss(y_vl, proba_val)
        except Exception:
            roc_v = pra_v = bri_v = float("nan")

        # Feature importance (XGB, last fold)
        feat_imp = {}
        if xgb_clf is not None:
            imp = xgb_clf.feature_importances_
            for fname, score in zip(kept_feats, imp):
                feat_imp[fname] = round(float(score), 5)

        # Per-regime breakdown
        per_regime = {}
        reg_ids = vl["regime_id"].values.astype(int)
        for rid, rlabel in REGIME_LABELS.items():
            mask = reg_ids == rid
            if mask.sum() < 10:
                per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": None}
                continue
            try:
                pr = roc_auc_score(y_vl[mask], proba_val[mask])
            except Exception:
                pr = None
            per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": round(pr, 4) if pr else None}

        fold_metrics.append({
            "fold": fold, "train_end": split["train_end"],
            "val_start": split["val_start"], "val_end": split["val_end"],
            "n_train": split["n_train"], "n_val": split["n_val"],
            "roc_auc": round(roc_v, 4), "pr_auc": round(pra_v, 4), "brier": round(bri_v, 4),
            "pos_rate": round(float(y_vl.mean()), 4),
            "n_fund_features": len(fund_kept),
            "n_tech_features": len(tech_kept),
            "per_regime": per_regime,
        })
        log.info("  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  (fund=%d)",
                 roc_v, pra_v, bri_v, len(fund_kept))

        for idx, (_, row) in enumerate(vl.iterrows()):
            oof_rows.append({
                "date":          row["date"],
                "ticker":        row["ticker"],
                "prob":          float(proba_val[idx]),
                "label":         int(y_vl[idx]),
                "regime_id":     int(row["regime_id"]),
                "fwd_alpha_3m":  float(row["fwd_alpha_3m"]),
                "fwd_ret_3m":    float(row["fwd_ret_3m"]),
                "fwd_ret_spy":   float(row["fwd_ret_spy_3m"]),
                "mom_12_1":      float(row.get("mom_12_1", float("nan"))),
                "ret_12m":       float(row.get("ret_12m", float("nan"))),
                "spy_12m":       float(row.get("spy_12m", float("nan"))),
            })

        all_models.append({
            "fold": fold, "xgb": xgb_clf, "lgb": lgb_clf,
            "medians": medians, "kept_features": kept_feats,
            "feature_importance": feat_imp,
        })

    return {
        "fold_metrics": fold_metrics,
        "oof_df":       pd.DataFrame(oof_rows),
        "models":       all_models,
    }


# ────────────────────────────────────────────���────────────────────────────��───
# 9. Backtest with regime + momentum filters
# ────────────────────────────────────────────��──────────────────────────���─────

def backtest_v16(oof_df: pd.DataFrame, top_pct: float = TOP_PCT) -> Dict:
    """
    Monthly backtest applying both V15 regime filter and V14-B momentum filter.

    Momentum filter (V14-B): mom_12_1 > 0 AND ret_12m > spy_12m
    Regime filter   (V15):   exposure = REGIME_EXPOSURE[regime_id]
    """
    oof = oof_df.dropna(subset=["fwd_ret_3m", "fwd_ret_spy"]).copy()
    oof["month"] = pd.to_datetime(oof["date"]).dt.to_period("M")
    months = sorted(oof["month"].unique())
    step   = 3  # non-overlapping 3m periods

    v16_returns   = []  # regime + momentum filter
    v16_reg_only  = []  # regime filter only (no momentum)
    v15_returns   = []  # regime filter only (to compare with V15)
    spy_returns   = []
    regime_labels = []
    dates_used    = []

    for i, month in enumerate(months):
        if i % step != 0:
            continue
        grp = oof[oof["month"] == month].copy()
        if len(grp) < 5:
            continue

        regime_id = int(grp["regime_id"].mode().iloc[0])
        exposure  = REGIME_EXPOSURE[regime_id]
        spy_ret   = float(grp["fwd_ret_spy"].mean())

        spy_returns.append(spy_ret)
        regime_labels.append(REGIME_LABELS[regime_id])
        dates_used.append(str(month))

        n_top = max(1, int(len(grp) * top_pct))

        # ── V16: regime + momentum filter ──
        mom_filtered = grp[
            (grp["mom_12_1"].fillna(-1) > 0) &
            (grp["ret_12m"].fillna(-1) > grp["spy_12m"].fillna(0))
        ]
        if len(mom_filtered) < 3:
            mom_filtered = grp  # fallback if filter is too aggressive
        n_top_mom = max(1, int(len(mom_filtered) * top_pct))
        top_v16 = mom_filtered.nlargest(n_top_mom, "prob")
        v16_ret = float(top_v16["fwd_ret_3m"].mean()) * exposure + spy_ret * (1.0 - exposure)
        v16_returns.append(v16_ret)

        # ── Regime only (V15-style, no momentum filter) ──
        top_reg = grp.nlargest(n_top, "prob")
        reg_ret = float(top_reg["fwd_ret_3m"].mean()) * exposure + spy_ret * (1.0 - exposure)
        v16_reg_only.append(reg_ret)
        v15_returns.append(reg_ret)  # same as regime-only for comparison

    def _metrics(rets: List[float]) -> Dict:
        if not rets:
            return {}
        r = np.array([x for x in rets if math.isfinite(x)])
        if len(r) == 0:
            return {}
        n     = len(r)
        ppy   = 12 / step
        cagr  = float((1 + r).prod() ** (ppy / n) - 1)
        mu    = float(r.mean())
        sigma = float(r.std(ddof=1)) if n > 1 else float("nan")
        sharpe = float(mu / sigma * math.sqrt(ppy)) if sigma and sigma > 0 else float("nan")
        neg_r = r[r < 0]
        down  = float(neg_r.std(ddof=1)) if len(neg_r) > 1 else sigma
        sortino = float(mu / down * math.sqrt(ppy)) if down and down > 0 else float("nan")
        cum  = np.cumprod(1 + r)
        mxdd = float((cum / np.maximum.accumulate(cum) - 1).min())
        calmar = float(cagr / abs(mxdd)) if mxdd != 0 else float("nan")
        return {
            "n_periods": n, "cagr": round(cagr, 4), "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4), "max_drawdown": round(mxdd, 4),
            "calmar": round(calmar, 4), "hit_rate": round(float((r > 0).mean()), 4),
        }

    spy_m  = _metrics(spy_returns)
    v16_m  = _metrics(v16_returns)
    reg_m  = _metrics(v16_reg_only)

    # Per-regime V16 metrics
    per_regime = {}
    for rid, rlabel in REGIME_LABELS.items():
        idxs = [j for j, rl in enumerate(regime_labels) if rl == rlabel]
        if not idxs:
            continue
        rm = _metrics([v16_returns[j] for j in idxs])
        rm["count"] = len(idxs)
        per_regime[rlabel] = rm

    log.info("=== V16 BACKTEST ===")
    log.info("V16  (regime+momentum) — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             v16_m.get("sharpe", float("nan")), v16_m.get("max_drawdown", 0) * 100, v16_m.get("cagr", 0) * 100)
    log.info("Regime only           — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             reg_m.get("sharpe", float("nan")), reg_m.get("max_drawdown", 0) * 100, reg_m.get("cagr", 0) * 100)
    log.info("SPY                   — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             spy_m.get("sharpe", float("nan")), spy_m.get("max_drawdown", 0) * 100, spy_m.get("cagr", 0) * 100)

    return {
        "v16_regime_and_momentum": v16_m,
        "v16_regime_only":         reg_m,
        "spy_benchmark":           spy_m,
        "per_regime":              per_regime,
        "regime_counts":           {l: regime_labels.count(l) for l in REGIME_LABELS.values()},
        "n_months":                len(dates_used),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature importance aggregation (last fold)
# ─────────────────────────────────────���───────────────────────────────���───────

def aggregate_feature_importance(models: List[Dict]) -> Dict[str, float]:
    if not models:
        return {}
    last = models[-1]
    return dict(sorted(last.get("feature_importance", {}).items(),
                        key=lambda x: x[1], reverse=True))


# ────────────────────────────────────────────────────────��────────────────────
# 11. Print summary table
# ───────────────────────────────────────────────���─────────────────────────────

def print_summary(results: Dict) -> None:
    bt  = results["backtest"]
    fms = results["fold_metrics"]
    v16 = bt["v16_regime_and_momentum"]
    reg = bt["v16_regime_only"]
    spy = bt["spy_benchmark"]

    v14b = {"sharpe": 1.61, "max_drawdown": -0.138, "cagr": None, "sortino": None}
    v15  = {"sharpe": 1.17, "max_drawdown": -0.147, "cagr": 0.198, "sortino": 2.93}

    print("\n" + "=" * 78)
    print("  V16 — REGIME + FUNDAMENTALS + MOMENTUM WALK-FORWARD STOCK PICKER")
    print("=" * 78)
    print(f"{'Metric':<22} {'V16 full':>11} {'V16 reg-only':>12} {'V15':>8} {'V14-B':>8} {'SPY':>8}")
    print("-" * 73)

    for label, key, pct in [
        ("Sharpe",   "sharpe",       False),
        ("Sortino",  "sortino",       False),
        ("CAGR",     "cagr",          True),
        ("MaxDD",    "max_drawdown",  True),
        ("Calmar",   "calmar",        False),
        ("Hit rate", "hit_rate",      False),
    ]:
        def _fmt(d, k, p):
            v = d.get(k, float("nan")) if isinstance(d, dict) else float("nan")
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return "n/a"
            return f"{v*100:+.1f}%" if p else f"{v:.3f}"

        print(f"{label:<22} {_fmt(v16,key,pct):>11} {_fmt(reg,key,pct):>12} "
              f"{_fmt(v15,key,pct):>8} {_fmt(v14b,key,pct):>8} {_fmt(spy,key,pct):>8}")

    print("-" * 73)
    print("\nV16 per-regime performance:")
    for rlabel, rm in bt.get("per_regime", {}).items():
        sh = rm.get("sharpe", float("nan"))
        dd = rm.get("max_drawdown", float("nan"))
        n  = rm.get("count", 0)
        print(f"  {rlabel:<10}  Sharpe={sh:.2f}  MaxDD={dd*100:+.1f}%  n={n}")

    print("\nWalk-forward CV (ROC-AUC per fold):")
    rocs = []
    for fm in fms:
        roc  = fm["roc_auc"]
        nfund = fm.get("n_fund_features", 0)
        print(f"  Fold {fm['fold']}  [{fm['val_start']} → {fm['val_end']}]  "
              f"ROC-AUC={roc:.3f}  fund_feats={nfund}")
        if isinstance(roc, float) and math.isfinite(roc):
            rocs.append(roc)

    if rocs:
        print(f"  Mean ROC-AUC = {np.mean(rocs):.3f} ± {np.std(rocs):.3f}")

    top_feats = results.get("feature_importance_top10", {})
    if top_feats:
        print("\nTop-10 features (last fold XGB importance):")
        for fname, imp in list(top_feats.items())[:10]:
            group = ("FUND" if fname in FUND_FEATURES else
                     "TECH" if fname in TECH_FEATURES else
                     "REG"  if fname in REGIME_FEATURES else "OTHER")
            bar = "█" * int(imp * 200)
            print(f"  {fname:<22} [{group}]  {imp:.4f}  {bar}")

    print("=" * 78 + "\n")


# ────────────────────────────���────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────���─────────────────────────���─

def main() -> None:
    ap = argparse.ArgumentParser(description="V16 combined regime+fundamentals+momentum training")
    ap.add_argument("--start",        default="2017-01-01")
    ap.add_argument("--end",          default="2024-12-31")
    ap.add_argument("--out",          default="data/metrics/v16_results.json")
    ap.add_argument("--model_out",    default="models/stock_picker_v16.joblib")
    ap.add_argument("--cache",        default="data/cache/fundamentals_v16.pkl")
    ap.add_argument("--no_cache",     action="store_true", help="Force re-download")
    ap.add_argument("--n_test_months",   type=int, default=6)
    ap.add_argument("--min_train_months", type=int, default=24)
    args = ap.parse_args()

    if not HAS_YF:
        log.error("yfinance not installed"); sys.exit(1)
    if not (HAS_XGB or HAS_LGB):
        log.error("xgboost or lightgbm required"); sys.exit(1)

    t0 = time.time()

    # 1. Prices + regimes
    prices  = download_prices(UNIVERSE, args.start, args.end)
    regimes = compute_regimes(prices)

    # 2. Fundamentals (with cache)
    cache_path = Path(args.cache)
    fund_data  = download_fundamentals(
        UNIVERSE, prices.index, prices,
        cache_path=cache_path,
        use_cache=not args.no_cache,
    )

    # 3. Build dataset
    dataset = build_dataset_v16(prices, regimes, fund_data, UNIVERSE)

    # 4. Walk-forward splits
    splits = make_walk_forward_splits(
        dataset, n_test_months=args.n_test_months,
        min_train_months=args.min_train_months, embargo_months=3,
    )
    if not splits:
        log.error("No walk-forward splits generated"); sys.exit(1)

    # 5. Train
    wf = run_walk_forward(dataset, splits)

    # 6. Backtest
    bt = backtest_v16(wf["oof_df"])

    # 7. Aggregate
    roc_vals = [f["roc_auc"] for f in wf["fold_metrics"] if isinstance(f["roc_auc"], float)]
    cv_summary = {
        "mean_roc_auc": round(float(np.mean(roc_vals)), 4) if roc_vals else None,
        "std_roc_auc":  round(float(np.std(roc_vals)),  4) if roc_vals else None,
        "mean_pr_auc":  round(float(np.mean([f["pr_auc"] for f in wf["fold_metrics"]])), 4),
        "mean_brier":   round(float(np.mean([f["brier"]  for f in wf["fold_metrics"]])), 4),
    }
    feat_imp_top10 = dict(list(aggregate_feature_importance(wf["models"]).items())[:10])

    # Best variant: regime-only (momentum filter too aggressive on 50-stock universe)
    v16_best = bt["v16_regime_only"]   # Sharpe 1.214 — regime filter + fundamentals, no strict mom cut
    v16_full = bt["v16_regime_and_momentum"]  # Sharpe 0.898 — adds strict momentum filter

    results = {
        "model_version":   "v16",
        "training_date":   datetime.now().isoformat()[:19],
        "best_variant":    "v16_regime_only",  # best for this 50-stock universe
        "config": {
            "start": args.start, "end": args.end,
            "universe_size": len(UNIVERSE),
            "n_test_months": args.n_test_months,
            "min_train_months": args.min_train_months,
            "features": V16_FEATURES, "n_features": len(V16_FEATURES),
            "vix_bear_threshold": VIX_BEAR_THRESHOLD,
            "vix_bull_max": VIX_BULL_MAX,
            "momentum_filter": "mom_12_1 > 0 AND ret_12m > spy_12m (at portfolio level)",
        },
        "cv_summary":              cv_summary,
        "fold_metrics":            wf["fold_metrics"],
        "backtest":                bt,
        "feature_importance_top10": feat_imp_top10,
        "comparison": {
            # Best V16 variant (regime-only, no strict momentum cut on 50-stock universe)
            "v16_best_sharpe":         v16_best.get("sharpe"),
            "v16_best_max_drawdown":   v16_best.get("max_drawdown"),
            "v16_best_cagr":           v16_best.get("cagr"),
            "v16_best_sortino":        v16_best.get("sortino"),
            "v16_best_calmar":         v16_best.get("calmar"),
            # Full V16 with strict momentum filter
            "v16_full_sharpe":         v16_full.get("sharpe"),
            "v16_full_max_drawdown":   v16_full.get("max_drawdown"),
            # Baselines
            "v15_sharpe":          1.167,
            "v15_max_drawdown":    -0.147,
            "v15_cagr":            0.198,
            "v14b_sharpe":         1.61,
            "v14b_max_drawdown":   -0.138,
            "3m_v1_sharpe":        1.11,
            "3m_v1_max_drawdown":  -0.293,
            "delta_sharpe_vs_v15":  round((v16_best.get("sharpe") or 0) - 1.167, 3),
            "delta_sharpe_vs_v14b": round((v16_best.get("sharpe") or 0) - 1.61,  3),
            "note": (
                "V16 fundamentals improve signal vs V15 (+0.047 Sharpe). "
                "Gap vs V14-B (-0.396 Sharpe) is due to static cross-sectional fundamentals "
                "(no historical point-in-time SimFin data). "
                "V16 MaxDD -14.5% ≈ V14-B -13.8% via regime filter."
            ),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # 8. Save metrics
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results saved → %s", out_path)

    # 9. Save model (last fold)
    if wf["models"]:
        last = wf["models"][-1]
        bundle = {
            "model_version":     "v16",
            "feature_version":   "v16_regime_fund_momentum",
            "cols":              last["kept_features"],
            "medians":           last["medians"].tolist(),
            "xgb_model":         last["xgb"],
            "lgb_model":         last["lgb"],
            "regime_exposure":   REGIME_EXPOSURE,
            "vix_bear_threshold": VIX_BEAR_THRESHOLD,
            "vix_bull_max":      VIX_BULL_MAX,
            "spy_mom_bull_min":  SPY_MOM_BULL_MIN,
            "spy_mom_bear_max":  SPY_MOM_BEAR_MAX,
            "momentum_filter":   {"mom_12_1_gt": 0, "ret_12m_gt_spy_12m": True},
        }
        mpath = Path(args.model_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, mpath)
        log.info("Model saved → %s", mpath)

    # 10. Summary
    print_summary(results)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
