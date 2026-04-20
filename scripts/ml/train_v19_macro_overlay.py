"""
scripts/ml/train_v19_macro_overlay.py
======================================
V19 — Macro Overlay: Yield Curve + Credit Spreads

Key improvement over V17 (best prior baseline, Sharpe 1.378):
  - Yield curve slope (10Y - 3M via ^TNX / ^IRX): leading recession indicator,
    inverted in early 2022 months before VIX crossed threshold
  - Credit spread (HYG 3m return - LQD 3m return): risk-appetite proxy
  - macro_factor [0,1]: composite exposure multiplier applied to regime allocation
      effective_exposure = REGIME_EXPOSURE[regime_id] × macro_factor
    → Early-warning reduction of bull/sideways exposure before VIX triggers
  - 5 macro features also fed into the ML model for cross-sectional stock selection
    (which individual stocks outperform in each macro environment)

Macro features (5 new, all market-wide → same for all stocks in a given month):
  yc_slope        : 10Y yield - 3M yield in % (positive = normal, negative = inverted)
  yc_slope_3m_chg : 3-month change in slope (negative = flattening/inverting)
  credit_spread_3m: HYG_3m_return - LQD_3m_return (negative = credit stress)
  credit_spread_6m: HYG_6m_return - LQD_6m_return
  macro_factor    : composite [0,1] — 1=healthy macro, 0=macro bear

macro_factor calibration:
  yc=+1.5%, cr=0%  → factor=1.00 (full regime exposure)
  yc=+0.0%, cr=0%  → factor=0.63 (flat curve = caution)
  yc=-0.5%, cr=-3% → factor=0.27 (inverted+stressed = strong caution, e.g. mid-2022)
  yc=-0.8%, cr=-6% → factor=0.00 (macro bear = 0% exposure)

Data:
  Prices:    yfinance monthly (same as V17)
  Yield:     ^TNX (10Y CBOE), ^IRX (13-week T-bill) — free, monthly
  Credit:    HYG, LQD ETF prices — free, monthly
  EDGAR:     reuses V17 cache (--cache data/cache/edgar_v17)

Feature set (38 total):
  Tech (14) + Regime (3) + Fund abs (15, SEC EDGAR) + Macro (5) + sector_id (1)

Target: fwd 3m alpha vs SPY > 2.5%
Goal:   beat V14-B (Sharpe 1.61, MaxDD -13.8%)

Usage:
  python scripts/ml/train_v19_macro_overlay.py \\
      --start 2017-01-01 --end 2024-12-31 \\
      --out data/metrics/v19_results.json \\
      --model_out models/stock_picker_v19.joblib \\
      --cache data/cache/edgar_v17
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

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
log = logging.getLogger("train_v19")

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Universe (same 50 stocks as V15–V18)
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
    "JPM",  "BAC",  "WFC",   "GS",   "MS",   "BRK-B","C",   "AXP",  "BLK",  "USB",
    "JNJ",  "UNH",  "PFE",   "ABBV", "MRK",  "LLY",  "TMO", "ABT",  "MDT",  "CVS",
    "WMT",  "PG",   "KO",    "PEP",  "MCD",  "COST", "NKE", "HD",   "TGT",  "LOW",
    "XOM",  "CVX",  "COP",   "SLB",  "CAT",  "HON",  "MMM", "GE",   "BA",   "UPS",
]

SECTOR_MAP = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0, "AMZN": 0,
    "NVDA": 0, "AMD":  0, "INTC":  0, "CSCO": 0, "ORCL": 0,
    "JPM":  1, "BAC":  1, "WFC":   1, "GS":   1, "MS":   1,
    "BRK-B":1, "C":    1, "AXP":   1, "BLK":  1, "USB":  1,
    "JNJ":  2, "UNH":  2, "PFE":   2, "ABBV": 2, "MRK":  2,
    "LLY":  2, "TMO":  2, "ABT":   2, "MDT":  2, "CVS":  2,
    "WMT":  3, "PG":   3, "KO":    3, "PEP":  3, "MCD":  3,
    "COST": 3, "NKE":  3, "HD":    3, "TGT":  3, "LOW":  3,
    "XOM":  4, "CVX":  4, "COP":   4, "SLB":  4, "CAT":  5,
    "HON":  5, "MMM":  5, "GE":    5, "BA":   5, "UPS":  5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Regime config (same thresholds as V15–V17)
# ─────────────────────────────────────────────────────────────────────────────
VIX_BEAR_THRESHOLD = 27.0
VIX_BULL_MAX       = 18.0
SPY_MOM_BULL_MIN   = 0.05
SPY_MOM_BEAR_MAX   = -0.05

REGIME_BULL     = 0
REGIME_SIDEWAYS = 1
REGIME_BEAR     = 2
REGIME_LABELS   = {REGIME_BULL: "bull", REGIME_SIDEWAYS: "sideways", REGIME_BEAR: "bear"}
REGIME_EXPOSURE = {REGIME_BULL: 1.0, REGIME_SIDEWAYS: 0.5, REGIME_BEAR: 0.0}

TOP_PCT = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Macro factor calibration
# ─────────────────────────────────────────────────────────────────────────────
# Yield curve (10Y - 3M in %): [-0.8, +1.2] → [0, 1]
_YC_LOW  = -0.8   # deeply inverted → yc_score = 0
_YC_HIGH = +1.2   # steep normal    → yc_score = 1

# Credit spread (HYG 3m - LQD 3m, decimal): [-0.06, 0] → [0, 1]
_CR_LOW  = -0.06  # severe stress   → cr_score = 0
_CR_HIGH =  0.00  # no stress       → cr_score = 1


def compute_macro_factor(yc_slope: float, credit_spread_3m: float) -> float:
    """
    Composite macro health factor in [0, 1].

    Calibration:
      yc=+1.5%, cr=0%  → 1.00 (full bull exposure)
      yc= 0.0%, cr=0%  → 0.63 (flat curve = mild caution)
      yc=-0.5%, cr=-3% → 0.27 (inverted + credit stress = strong caution)
      yc=-0.8%, cr=-6% → 0.00 (macro bear = zero exposure)
    """
    yc_score = float(np.clip((yc_slope        - _YC_LOW) / (_YC_HIGH - _YC_LOW), 0.0, 1.0))
    cr_score = float(np.clip((credit_spread_3m - _CR_LOW) / (_CR_HIGH - _CR_LOW), 0.0, 1.0))
    return float(np.sqrt(yc_score * cr_score))

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────────────────────────────────────────
TECH_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "ret_vs_spy_3m",
    "spy_1m", "spy_12m",
    "vol_ann", "vol_ratio", "skew_12m",
    "above_200ma", "trend_strength", "dd_from_hi52",
]
REGIME_FEATURES  = ["vix_level", "spy_mom_6m", "regime_id"]
FUND_FEATURES    = [
    "gross_margin", "op_margin", "net_margin", "roe",
    "debt_to_equity", "rd_intensity", "fcf_margin",
    "revenue_growth", "ni_growth",
    "pe_ratio", "pb_ratio", "earnings_yield",
    "current_ratio", "asset_growth", "accruals_ratio",
]
MACRO_FEATURES   = [
    "yc_slope",          # 10Y - 3M in % (inverted when < 0)
    "yc_slope_3m_chg",   # 3m change (negative = flattening)
    "credit_spread_3m",  # HYG_3m - LQD_3m (negative = credit stress)
    "credit_spread_6m",  # HYG_6m - LQD_6m
    "macro_factor",      # composite [0,1] health score
]
STATIC_FEATURES  = ["sector_id"]

V19_FEATURES = TECH_FEATURES + REGIME_FEATURES + FUND_FEATURES + MACRO_FEATURES + STATIC_FEATURES

_FUND_CLIPS: Dict[str, Tuple[float, float]] = {
    "gross_margin":   (-1.0, 1.0),  "op_margin":      (-1.0, 0.8),
    "net_margin":     (-1.0, 0.5),  "roe":            (-2.0, 2.0),
    "debt_to_equity": ( 0.0, 20.0), "rd_intensity":   ( 0.0, 0.5),
    "fcf_margin":     (-1.0, 0.5),  "revenue_growth": (-0.5, 2.0),
    "ni_growth":      (-5.0, 5.0),  "pe_ratio":       ( 0.0, 200.0),
    "pb_ratio":       ( 0.0, 50.0), "earnings_yield": (-0.1, 0.3),
    "current_ratio":  ( 0.0, 10.0), "asset_growth":   (-0.3, 1.0),
    "accruals_ratio": (-0.2, 0.2),
}

# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR config (identical to V17)
# ─────────────────────────────────────────────────────────────────────────────
_EDGAR_UA      = "Asymetra Research paul.nuyttens@gmail.com"
_EDGAR_HEADERS = {"User-Agent": _EDGAR_UA}
_EDGAR_SLEEP   = 0.15
_CIK_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL     = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

_MANUAL_CIK: Dict[str, int] = {
    "BRK-B": 1067983, "GOOGL": 1652044, "META": 1326801,
    "ABBV":  1551152,  "LLY":   59478,
}

_CONCEPTS: Dict[str, Dict] = {
    "revenue": {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":[
        "RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueGoodsNet","RevenuesNetOfInterestExpense"]},
    "gross_profit":     {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":["GrossProfit"]},
    "operating_income": {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":["OperatingIncomeLoss"]},
    "net_income":       {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":[
        "NetIncomeLoss","ProfitLoss","NetIncomeLossAvailableToCommonStockholdersDiluted"]},
    "rd_expense":       {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":[
        "ResearchAndDevelopmentExpense","ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]},
    "operating_cf":     {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":[
        "NetCashProvidedByUsedInOperatingActivities"]},
    "capex":            {"taxonomy":"us-gaap","unit":"USD","type":"flow","names":[
        "PaymentsToAcquirePropertyPlantAndEquipment","CapitalExpenditureContinuingOperations"]},
    "total_assets":     {"taxonomy":"us-gaap","unit":"USD","type":"stock","names":["Assets"]},
    "equity":           {"taxonomy":"us-gaap","unit":"USD","type":"stock","names":[
        "StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]},
    "current_assets":   {"taxonomy":"us-gaap","unit":"USD","type":"stock","names":["AssetsCurrent"]},
    "current_liabilities":{"taxonomy":"us-gaap","unit":"USD","type":"stock","names":["LiabilitiesCurrent"]},
    "long_term_debt":   {"taxonomy":"us-gaap","unit":"USD","type":"stock","names":["LongTermDebt","LongTermDebtNoncurrent"]},
    "short_term_debt":  {"taxonomy":"us-gaap","unit":"USD","type":"stock","names":["ShortTermBorrowings","DebtCurrent"]},
    "shares_dei":       {"taxonomy":"dei","unit":"shares","type":"stock","names":["EntityCommonStockSharesOutstanding"]},
    "shares_usgaap":    {"taxonomy":"us-gaap","unit":"shares","type":"stock","names":[
        "CommonStockSharesOutstanding","SharesOutstanding"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price + macro data download
# ─────────────────────────────────────────────────────────────────────────────
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    log.info("Downloading monthly prices (%d tickers + SPY + VIX)…", len(tickers))
    all_tickers = list(set(tickers + ["SPY", "^VIX"]))
    raw = yf.download(all_tickers, start=start, end=end, interval="1mo", auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index = prices.index.to_period("M").to_timestamp("M")
    log.info("Prices: %d months × %d tickers", len(prices), len(prices.columns))
    return prices


def download_macro_data(start: str, end: str) -> pd.DataFrame:
    """
    Download macro instruments monthly:
      ^TNX  — 10-Year Treasury Constant Maturity Yield (%)
      ^IRX  — 13-Week T-Bill Yield (%) — proxy for 3M rate
      HYG   — iShares High-Yield Bond ETF (price)
      LQD   — iShares Investment Grade Bond ETF (price)

    Returns DataFrame indexed by month-end timestamps with columns:
      tnx, irx, hyg, lqd
    """
    tickers = ["^TNX", "^IRX", "HYG", "LQD"]
    log.info("Downloading macro data (^TNX, ^IRX, HYG, LQD)…")
    raw = yf.download(tickers, start=start, end=end, interval="1mo", auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes.index = closes.index.to_period("M").to_timestamp("M")
    closes.columns = [c.lower().lstrip("^") for c in closes.columns]

    # Fill any missing yield data forward (yields rarely have full NaN runs)
    for col in ["tnx", "irx"]:
        if col in closes.columns:
            closes[col] = closes[col].ffill()

    log.info("Macro data: %d months, cols=%s", len(closes), list(closes.columns))
    return closes


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime detection (same VIX/SPY logic as V17)
# ─────────────────────────────────────────────────────────────────────────────
def compute_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    spy = prices["SPY"].dropna()
    vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()
    df  = pd.DataFrame(index=prices.index)
    df["spy_1m"]    = spy.pct_change(1)
    df["spy_6m"]    = spy.pct_change(6)
    df["spy_12m"]   = spy.pct_change(12)
    df["vix_level"] = vix.reindex(df.index).fillna(20.0)

    def _classify(row):
        v = row["vix_level"]; m = row["spy_6m"] if pd.notna(row["spy_6m"]) else 0.0
        if v >= VIX_BEAR_THRESHOLD or m <= SPY_MOM_BEAR_MAX: return REGIME_BEAR
        elif v <= VIX_BULL_MAX and m >= SPY_MOM_BULL_MIN:    return REGIME_BULL
        return REGIME_SIDEWAYS

    df["regime_id"]    = df.apply(_classify, axis=1)
    df["regime_label"] = df["regime_id"].map(REGIME_LABELS)
    rc = df["regime_id"].value_counts().sort_index()
    log.info("Regime distribution: %s", {REGIME_LABELS[k]: int(v) for k, v in rc.items()})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ★ NEW: Macro feature computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly macro features from ^TNX, ^IRX, HYG, LQD prices.

    Returns DataFrame(index=macro.index) with MACRO_FEATURES columns.
    """
    out = pd.DataFrame(index=macro.index)

    tnx = macro.get("tnx", pd.Series(dtype=float))
    irx = macro.get("irx", pd.Series(dtype=float))
    hyg = macro.get("hyg", pd.Series(dtype=float))
    lqd = macro.get("lqd", pd.Series(dtype=float))

    # ── Yield curve slope (10Y - 3M in %) ──
    # ^TNX and ^IRX are quoted as yield × 1 (e.g., 4.50 means 4.50%)
    out["yc_slope"] = (tnx - irx).reindex(out.index)

    # 3-month change in slope (negative = curve flattening/inverting)
    out["yc_slope_3m_chg"] = out["yc_slope"].diff(3)

    # ── Credit spread: HYG 3m return - LQD 3m return (decimal) ──
    hyg_ret3 = hyg.pct_change(3)
    lqd_ret3 = lqd.pct_change(3)
    out["credit_spread_3m"] = (hyg_ret3 - lqd_ret3).reindex(out.index)

    hyg_ret6 = hyg.pct_change(6)
    lqd_ret6 = lqd.pct_change(6)
    out["credit_spread_6m"] = (hyg_ret6 - lqd_ret6).reindex(out.index)

    # ── Composite macro factor ──
    def _macro_factor_row(row):
        yc  = row.get("yc_slope",         float("nan"))
        cr  = row.get("credit_spread_3m", float("nan"))
        if pd.isna(yc) or pd.isna(cr):
            return 1.0  # default to no adjustment if data missing
        return compute_macro_factor(float(yc), float(cr))

    out["macro_factor"] = out.apply(_macro_factor_row, axis=1)

    # Fill forward early months where pct_change returns NaN
    for col in ["yc_slope_3m_chg", "credit_spread_3m", "credit_spread_6m"]:
        out[col] = out[col].fillna(0.0)  # neutral (no change / no spread) for early months

    log.info("Macro features computed. yc_slope range: [%.2f, %.2f]",
             out["yc_slope"].min(), out["yc_slope"].max())
    log.info("macro_factor: mean=%.2f  min=%.2f  months_lt_05=%.0f%%",
             out["macro_factor"].mean(), out["macro_factor"].min(),
             (out["macro_factor"] < 0.5).mean() * 100)

    # Log 2022 snapshot for validation
    snap_2022 = out[out.index.year == 2022][["yc_slope", "credit_spread_3m", "macro_factor"]]
    if not snap_2022.empty:
        log.info("2022 macro snapshot:\n%s", snap_2022.to_string())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEC EDGAR helpers (identical to V17)
# ─────────────────────────────────────────────────────────────────────────────
def _get_raw_entries(facts: Dict, concept_key: str) -> List[Dict]:
    if facts is None: return []
    cfg = _CONCEPTS[concept_key]
    tax = facts.get("facts", {}).get(cfg["taxonomy"], {})
    for name in cfg["names"]:
        e = tax.get(name, {}).get("units", {}).get(cfg["unit"])
        if e: return list(e)
    return []


def _parse_entries(raw: List[Dict]) -> List[Dict]:
    out = []
    for e in raw:
        try:
            filed = pd.Timestamp(e["filed"]); end = pd.Timestamp(e["end"])
            s = e.get("start"); start = pd.Timestamp(s) if s else None
            out.append({**e, "_filed": filed, "_end": end, "_start": start,
                        "_dur": (end - start).days if start else 0})
        except Exception: continue
    return out


def _ttm_flow(entries: List[Dict], cutoff: pd.Timestamp) -> Optional[float]:
    avail = [e for e in entries if e["_filed"] <= cutoff]
    if not avail: return None
    qtrs = [e for e in avail if e["_start"] is not None and 75 <= e["_dur"] <= 110]
    if qtrs:
        pm: Dict[str, Dict] = {}
        for e in qtrs:
            k = e["_end"].strftime("%Y-%m")
            if k not in pm or e["_filed"] > pm[k]["_filed"]: pm[k] = e
        sq = sorted(pm.values(), key=lambda x: x["_end"], reverse=True)
        if len(sq) >= 4:
            t4 = sq[:4]
            if (t4[0]["_end"] - t4[3]["_end"]).days >= 270:
                return sum(float(e["val"]) for e in t4)
    annuals = [e for e in avail if e.get("fp")=="FY" and e["_start"] is not None and 340<=e["_dur"]<=395]
    if annuals: return float(max(annuals, key=lambda x: x["_end"])["val"])
    return None


def _latest_stock(entries: List[Dict], cutoff: pd.Timestamp) -> Optional[float]:
    avail = [e for e in entries if e["_filed"] <= cutoff]
    if not avail: return None
    v = float(max(avail, key=lambda x: (x["_end"], x["_filed"]))["val"])
    return v if math.isfinite(v) else None


def _clip(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None or not math.isfinite(v): return None
    return max(lo, min(hi, v))


def _div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or abs(d) < 1e-9: return None
    return float(n / d)


def compute_fund_monthly(
    ticker: str, facts: Optional[Dict],
    monthly_dates: pd.DatetimeIndex, prices_series: pd.Series
) -> pd.DataFrame:
    result = pd.DataFrame(float("nan"), index=monthly_dates, columns=FUND_FEATURES)
    if facts is None: return result
    parsed = {k: _parse_entries(_get_raw_entries(facts, k)) for k in _CONCEPTS}
    all_shares = parsed["shares_dei"] + parsed["shares_usgaap"]

    for month_date in monthly_dates:
        cutoff = month_date + pd.offsets.MonthEnd(0)
        cut_1y = cutoff - pd.DateOffset(months=12)

        rev_ttm  = _ttm_flow(parsed["revenue"],          cutoff)
        gp_ttm   = _ttm_flow(parsed["gross_profit"],     cutoff)
        oi_ttm   = _ttm_flow(parsed["operating_income"], cutoff)
        ni_ttm   = _ttm_flow(parsed["net_income"],       cutoff)
        rd_ttm   = _ttm_flow(parsed["rd_expense"],       cutoff)
        ocf_ttm  = _ttm_flow(parsed["operating_cf"],     cutoff)
        cap_ttm  = _ttm_flow(parsed["capex"],            cutoff)

        fcf_ttm: Optional[float] = None
        if ocf_ttm is not None:
            fcf_ttm = ocf_ttm - (abs(cap_ttm) if cap_ttm is not None else 0.0)

        rev_1y   = _ttm_flow(parsed["revenue"],    cut_1y)
        ni_1y    = _ttm_flow(parsed["net_income"], cut_1y)
        assets   = _latest_stock(parsed["total_assets"],       cutoff)
        equity   = _latest_stock(parsed["equity"],             cutoff)
        curr_a   = _latest_stock(parsed["current_assets"],     cutoff)
        curr_l   = _latest_stock(parsed["current_liabilities"],cutoff)
        ltd      = _latest_stock(parsed["long_term_debt"],     cutoff) or 0.0
        std      = _latest_stock(parsed["short_term_debt"],    cutoff) or 0.0
        debt     = (ltd + std) if (ltd or std) else None
        assets_1y= _latest_stock(parsed["total_assets"],       cut_1y)
        shares   = _latest_stock(all_shares,                   cutoff)

        mktcap: Optional[float] = None
        if shares and shares > 0 and month_date in prices_series.index:
            px = float(prices_series.loc[month_date])
            if pd.notna(px) and px > 0: mktcap = px * shares

        eq_pos = equity if (equity is not None and equity > 0) else None
        ni_pos = ni_ttm if (ni_ttm is not None and ni_ttm > 0)  else None
        rd_i: Optional[float] = None
        if rev_ttm is not None and rev_ttm > 0:
            rd_i = _clip(rd_ttm / rev_ttm if rd_ttm is not None else 0.0, 0.0, 0.5)
        pe = _clip(_div(mktcap, ni_pos), 0.0, 200.0)

        result.loc[month_date] = {
            "gross_margin":   _clip(_div(gp_ttm, rev_ttm),  -1.0, 1.0),
            "op_margin":      _clip(_div(oi_ttm, rev_ttm),  -1.0, 0.8),
            "net_margin":     _clip(_div(ni_ttm, rev_ttm),  -1.0, 0.5),
            "roe":            _clip(_div(ni_ttm, eq_pos),   -2.0, 2.0),
            "debt_to_equity": _clip(_div(debt, eq_pos),      0.0, 20.0),
            "rd_intensity":   rd_i,
            "fcf_margin":     _clip(_div(fcf_ttm, rev_ttm), -1.0, 0.5),
            "revenue_growth": _clip((_div(rev_ttm,rev_1y)-1 if rev_ttm and rev_1y and rev_1y>0 else None), -0.5, 2.0),
            "ni_growth":      _clip((_div(ni_ttm, ni_1y)-1 if ni_ttm is not None and ni_1y and ni_1y!=0 else None), -5.0, 5.0),
            "pe_ratio":       pe,
            "pb_ratio":       _clip(_div(mktcap, eq_pos),   0.0, 50.0),
            "earnings_yield": _clip(_div(1.0, pe) if pe and pe>0 else None, -0.1, 0.3),
            "current_ratio":  _clip(_div(curr_a, curr_l) if curr_l and curr_l>0 else None, 0.0, 10.0),
            "asset_growth":   _clip((_div(assets,assets_1y)-1 if assets and assets_1y and assets_1y>0 else None), -0.3, 1.0),
            "accruals_ratio": _clip(_div((ni_ttm or 0)-(fcf_ttm or 0),assets) if assets and assets>0 and ni_ttm is not None else None, -0.2, 0.2),
        }
    return result.astype(float)


def _build_cik_map(session: requests.Session) -> Dict[str, int]:
    try:
        r = session.get(_CIK_MAP_URL, headers=_EDGAR_HEADERS, timeout=30); r.raise_for_status()
        m: Dict[str, int] = {}
        for e in r.json().values():
            t=str(e.get("ticker","")).upper().replace(".","−"); c=e.get("cik_str")
            if t and c: m[t]=int(c)
        m.update(_MANUAL_CIK); log.info("CIK map: %d entries", len(m)); return m
    except Exception as exc:
        log.warning("CIK map failed: %s", exc); return dict(_MANUAL_CIK)


def _fetch_facts(cik: int, ticker: str, session: requests.Session) -> Optional[Dict]:
    try:
        r = session.get(_FACTS_URL.format(cik10=str(cik).zfill(10)), headers=_EDGAR_HEADERS, timeout=90)
        if r.status_code == 404: return None
        r.raise_for_status(); return r.json()
    except Exception: return None


def load_edgar_facts(
    tickers: List[str], cik_map: Dict[str, int],
    cache_dir: Path, session: requests.Session, use_cache: bool,
) -> Dict[str, Optional[Dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[Dict]] = {}; t0 = time.time()
    for i, ticker in enumerate(tickers, 1):
        safe=ticker.replace("-","_"); cpath=cache_dir/f"{safe}.json.gz"
        if use_cache and cpath.exists():
            try:
                with gzip.open(cpath,"rt",encoding="utf-8") as fh: results[ticker]=json.load(fh)
                if i%10==0 or i==len(tickers): log.info("[%d/%d] from cache (%.0fs)",i,len(tickers),time.time()-t0)
                continue
            except Exception: pass
        cik=cik_map.get(ticker.upper())
        if not cik: log.warning("[%s] no CIK",ticker); results[ticker]=None; continue
        time.sleep(_EDGAR_SLEEP); facts=_fetch_facts(cik, ticker, session)
        results[ticker]=facts
        if facts:
            try:
                with gzip.open(cpath,"wt",encoding="utf-8") as fh: json.dump(facts, fh)
            except Exception: pass
        log.info("[%d/%d] %s %s (%.0fs)",i,len(tickers),ticker,"✓" if facts else "✗",time.time()-t0)
    return results


def build_edgar_fundamentals(
    tickers: List[str], monthly_dates: pd.DatetimeIndex,
    prices: pd.DataFrame, cache_dir: Path, use_cache: bool,
) -> Dict[str, pd.DataFrame]:
    session = requests.Session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cik_cache = cache_dir / "cik_map.json"
    if use_cache and cik_cache.exists():
        with cik_cache.open() as fh: cik_map: Dict[str, int] = json.load(fh)
        log.info("CIK map from cache (%d entries)", len(cik_map))
    else:
        cik_map = _build_cik_map(session); cik_cache.write_text(json.dumps(cik_map))
    facts_dict = load_edgar_facts(tickers, cik_map, cache_dir/"facts", session, use_cache=use_cache)
    log.info("Computing fundamentals: %d tickers × %d months…", len(tickers), len(monthly_dates))
    fund_data: Dict[str, pd.DataFrame] = {}; t0 = time.time(); cov_list = []
    for i, ticker in enumerate(tickers, 1):
        facts = facts_dict.get(ticker)
        ps = prices[ticker] if ticker in prices.columns else pd.Series(dtype=float)
        df = compute_fund_monthly(ticker, facts, monthly_dates, ps)
        fund_data[ticker] = df; cov_list.append(float(df.notna().mean().mean()))
        if i%10==0 or i==len(tickers):
            log.info("[%d/%d] %s cov=%.0f%% (%.0fs)",i,len(tickers),ticker,cov_list[-1]*100,time.time()-t0)
    log.info("Fund coverage: mean=%.0f%%  min=%.0f%%", np.mean(cov_list)*100, np.min(cov_list)*100)
    return fund_data


# ─────────────────────────────────────────────────────────────────────────────
# 5. Technical features + macro merge (V17 tech + macro features per row)
# ─────────────────────────────────────────────────────────────────────────────
def _trend_strength(series: pd.Series) -> float:
    y=series.dropna().values
    if len(y)<3: return float("nan")
    x=np.arange(len(y),dtype=float); xb,yb=x.mean(),y.mean()
    d=((x-xb)**2).sum()
    if d==0: return float("nan")
    b=((x-xb)*(y-yb)).sum()/d; yh=yb+b*(x-xb)
    ss_r=((y-yh)**2).sum(); ss_t=((y-yb)**2).sum()
    return float(1.0-ss_r/ss_t) if ss_t>0 else float("nan")


def build_tech_features(
    prices: pd.DataFrame, regimes: pd.DataFrame,
    macro_feats: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """Build V17 technical + regime features, then merge monthly macro features."""
    if ticker not in prices.columns: return pd.DataFrame()
    p=prices[ticker].dropna(); spy=prices["SPY"].dropna(); rows=[]
    for i, date in enumerate(p.index):
        if i<12: continue
        def _ret(s,n):
            if i<n: return float("nan")
            v0,v1=s.iloc[i-n],s.iloc[i]
            return float(v1/v0-1) if pd.notna(v0) and pd.notna(v1) and v0!=0 else float("nan")
        r1=_ret(p,1); r3=_ret(p,3); r6=_ret(p,6); r12=_ret(p,12)
        s1=_ret(spy,1); s3=_ret(spy,3); s12=_ret(spy,12)
        mom=r12-r1 if all(pd.notna(x) for x in [r12,r1]) else float("nan")
        rspy3=r3-s3 if all(pd.notna(x) for x in [r3,s3]) else float("nan")
        w12=p.iloc[max(0,i-12):i+1]; mr=w12.pct_change().dropna()
        va=float(mr.std()*math.sqrt(12)) if len(mr)>=3 else float("nan")
        sk=float(mr.skew()) if len(mr)>=4 else float("nan")
        v3=float(mr.tail(3).std()*math.sqrt(12)) if len(mr)>=3 else float("nan")
        vr=float(v3/va) if va and va>0 else float("nan")
        hi=float(w12.max()); cur=float(p.iloc[i])
        dd=float(cur/hi-1) if hi>0 else float("nan")
        ma=float(w12.mean())
        rr=regimes.loc[date] if date in regimes.index else None
        vix=float(rr["vix_level"]) if rr is not None else 20.0
        sm6=float(rr["spy_6m"])    if rr is not None and pd.notna(rr.get("spy_6m")) else float("nan")
        rid=int(rr["regime_id"])   if rr is not None else REGIME_SIDEWAYS

        # Macro features (same for all tickers at this date)
        mrow = macro_feats.loc[date] if date in macro_feats.index else None
        ycs  = float(mrow["yc_slope"])          if mrow is not None and pd.notna(mrow.get("yc_slope"))         else float("nan")
        yc3c = float(mrow["yc_slope_3m_chg"])   if mrow is not None and pd.notna(mrow.get("yc_slope_3m_chg")) else float("nan")
        cr3  = float(mrow["credit_spread_3m"])  if mrow is not None and pd.notna(mrow.get("credit_spread_3m"))else 0.0
        cr6  = float(mrow["credit_spread_6m"])  if mrow is not None and pd.notna(mrow.get("credit_spread_6m"))else 0.0
        mf   = float(mrow["macro_factor"])       if mrow is not None and pd.notna(mrow.get("macro_factor"))    else 1.0

        rows.append({
            "date":date,"ticker":ticker,"sector_id":float(SECTOR_MAP.get(ticker,-1)),
            "ret_1m":r1,"ret_3m":r3,"ret_6m":r6,"ret_12m":r12,"mom_12_1":mom,"ret_vs_spy_3m":rspy3,
            "spy_1m":s1,"spy_12m":s12,"vol_ann":va,"vol_ratio":vr,"skew_12m":sk,
            "above_200ma":float(cur>ma),"trend_strength":_trend_strength(w12),"dd_from_hi52":dd,
            "vix_level":vix,"spy_mom_6m":sm6,"regime_id":float(rid),
            "yc_slope":ycs,"yc_slope_3m_chg":yc3c,"credit_spread_3m":cr3,
            "credit_spread_6m":cr6,"macro_factor":mf,
        })

    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows).set_index("date")
    p_i=p.copy(); spy_i=spy.copy()
    fws,fspy=[],[]
    for i2,date2 in enumerate(df.index):
        pp=p_i.index.get_loc(date2) if date2 in p_i.index else None
        ps2=spy_i.index.get_loc(date2) if date2 in spy_i.index else None
        def _fwd(s,pos):
            if pos is None or pos+3>=len(s): return float("nan")
            v0,v3=s.iloc[pos],s.iloc[pos+3]
            return float(v3/v0-1) if v0>0 else float("nan")
        fws.append(_fwd(p_i,pp)); fspy.append(_fwd(spy_i,ps2))
    df["fwd_ret_3m"]=fws; df["fwd_ret_spy_3m"]=fspy
    df["fwd_alpha_3m"]=df["fwd_ret_3m"]-df["fwd_ret_spy_3m"]
    df["label"]=(df["fwd_alpha_3m"]>0.025).astype(int)
    return df.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset_v19(
    prices: pd.DataFrame, regimes: pd.DataFrame,
    macro_feats: pd.DataFrame, fund_data: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> pd.DataFrame:
    all_dfs = []
    for ticker in tickers:
        tech_df = build_tech_features(prices, regimes, macro_feats, ticker)
        if tech_df.empty: continue
        tech_df["date"] = pd.to_datetime(tech_df["date"])
        tech_df = tech_df.set_index("date")
        fdf = fund_data.get(ticker)
        if fdf is not None and not fdf.empty:
            fdf.index = pd.to_datetime(fdf.index)
            for col in FUND_FEATURES:
                tech_df[col] = fdf[col].reindex(tech_df.index) if col in fdf.columns else float("nan")
        else:
            for col in FUND_FEATURES: tech_df[col] = float("nan")
        all_dfs.append(tech_df.reset_index())

    if not all_dfs: raise RuntimeError("No rows")
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.dropna(subset=["label","fwd_ret_3m"])
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)

    macro_cov = full[MACRO_FEATURES].notna().mean()
    log.info("Macro coverage: mean=%.0f%%  min=%.0f%%", macro_cov.mean()*100, macro_cov.min()*100)
    log.info("Dataset v19: %d rows, %d tickers, %s → %s",
             len(full), full["ticker"].nunique(), full["date"].min().date(), full["date"].max().date())
    return full


# ─────────────────────────────────────────────────────────────────────────────
# 7. Walk-forward splits (identical to V17)
# ─────────────────────────────────────────────────────────────────────────────
def make_walk_forward_splits(
    df: pd.DataFrame, n_test_months: int=6, min_train_months: int=24, embargo_months: int=3,
) -> List[Dict]:
    dates=sorted(df["date"].dt.to_period("M").unique()); splits,fold,si=[],1,min_train_months
    while True:
        ei=si+n_test_months
        if ei>len(dates): break
        tc=si-embargo_months
        if tc<min_train_months: si+=n_test_months; continue
        tri=np.where((df["date"]<=dates[tc-1].to_timestamp()).values)[0]
        vli=np.where(((df["date"]>=dates[si].to_timestamp())&(df["date"]<dates[min(ei,len(dates)-1)].to_timestamp())).values)[0]
        if len(tri)>=100 and len(vli)>0:
            splits.append({"fold":fold,"train_end":str(dates[tc-1].to_timestamp().date()),
                "val_start":str(dates[si].to_timestamp().date()),
                "val_end":str(dates[min(ei,len(dates)-1)].to_timestamp().date()),
                "n_train":len(tri),"n_val":len(vli),"train_idx":tri,"val_idx":vli})
            fold+=1
        si+=n_test_months
    log.info("Walk-forward: %d folds", len(splits)); return splits


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model training helpers (identical to V17)
# ─────────────────────────────────────────────────────────────────────────────
def _median_impute(X_tr,X_vl):
    m=np.nanmedian(X_tr,axis=0)
    for c in range(X_tr.shape[1]): X_tr[np.isnan(X_tr[:,c]),c]=m[c]; X_vl[np.isnan(X_vl[:,c]),c]=m[c]
    return X_tr,X_vl,m


def _train_xgb(X_tr,y_tr,X_vl,y_vl):
    if not HAS_XGB: return None
    spw=float((y_tr==0).sum())/max(float((y_tr==1).sum()),1.0)
    clf=xgb.XGBClassifier(n_estimators=500,learning_rate=0.04,max_depth=4,
        subsample=0.8,colsample_bytree=0.65,min_child_weight=5,reg_alpha=0.1,reg_lambda=1.0,
        scale_pos_weight=spw,early_stopping_rounds=40,eval_metric="logloss",random_state=SEED,verbosity=0)
    clf.fit(X_tr,y_tr,eval_set=[(X_vl,y_vl)],verbose=False); return clf


def _train_lgb(X_tr,y_tr,X_vl,y_vl):
    if not HAS_LGB: return None
    spw=float((y_tr==0).sum())/max(float((y_tr==1).sum()),1.0)
    clf=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.04,num_leaves=31,
        min_child_samples=20,subsample=0.8,colsample_bytree=0.65,
        reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=spw,random_state=SEED,verbose=-1)
    clf.fit(X_tr,y_tr,eval_set=[(X_vl,y_vl)],
            callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(-1)])
    return clf


def _ensemble(xc,lc,X):
    p,w=np.zeros(len(X)),0.0
    if xc: p+=0.5*xc.predict_proba(X)[:,1]; w+=0.5
    if lc: p+=0.5*lc.predict_proba(X)[:,1]; w+=0.5
    return p/w if w>0 else p


# ─────────────────────────────────────────────────────────────────────────────
# 9. Walk-forward training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(df: pd.DataFrame, splits: List[Dict]) -> Dict:
    feat_cols=[c for c in V19_FEATURES if c in df.columns]
    log.info("V19 training: %d features (%d macro)", len(feat_cols),
             sum(1 for f in feat_cols if f in MACRO_FEATURES))
    fold_metrics,oof_rows,all_models=[],[],[]

    for split in splits:
        fold=split["fold"]
        tr=df.iloc[split["train_idx"]].copy(); vl=df.iloc[split["val_idx"]].copy()
        X_tr=tr[feat_cols].values.astype(float); y_tr=tr["label"].values.astype(int)
        X_vl=vl[feat_cols].values.astype(float); y_vl=vl["label"].values.astype(int)

        nm=np.isnan(X_tr).mean(axis=0); km=nm<=0.50
        X_tr,X_vl=X_tr[:,km],X_vl[:,km]
        kept=[f for f,k in zip(feat_cols,km) if k]
        n_macro=sum(1 for f in kept if f in MACRO_FEATURES)
        log.info("Fold %d  feats=%d(macro=%d)  train=%d val=%d",fold,len(kept),n_macro,len(tr),len(vl))

        X_tr,X_vl,meds=_median_impute(X_tr.copy(),X_vl.copy())
        xc=_train_xgb(X_tr,y_tr,X_vl,y_vl); lc=_train_lgb(X_tr,y_tr,X_vl,y_vl)
        proba=_ensemble(xc,lc,X_vl)

        try: roc=roc_auc_score(y_vl,proba); pra=average_precision_score(y_vl,proba); bri=brier_score_loss(y_vl,proba)
        except Exception: roc=pra=bri=float("nan")

        fi={}
        if xc:
            for fn,sc in zip(kept,xc.feature_importances_): fi[fn]=round(float(sc),5)

        per_r={}
        rids=vl["regime_id"].values.astype(int)
        for rid2,rl in REGIME_LABELS.items():
            mask=rids==rid2
            if mask.sum()<10: per_r[rl]={"n":int(mask.sum()),"roc_auc":None}; continue
            try: pr2=roc_auc_score(y_vl[mask],proba[mask])
            except Exception: pr2=None
            per_r[rl]={"n":int(mask.sum()),"roc_auc":round(pr2,4) if pr2 else None}

        fold_metrics.append({"fold":fold,"train_end":split["train_end"],
            "val_start":split["val_start"],"val_end":split["val_end"],
            "n_train":split["n_train"],"n_val":split["n_val"],
            "roc_auc":round(roc,4),"pr_auc":round(pra,4),"brier":round(bri,4),
            "pos_rate":round(float(y_vl.mean()),4),"n_macro_features":n_macro,"per_regime":per_r})
        log.info("  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  macro_feats=%d",roc,pra,bri,n_macro)

        for idx,(_,row) in enumerate(vl.iterrows()):
            oof_rows.append({"date":row["date"],"ticker":row["ticker"],
                "prob":float(proba[idx]),"label":int(y_vl[idx]),
                "regime_id":int(row["regime_id"]),
                "fwd_alpha_3m":float(row["fwd_alpha_3m"]),"fwd_ret_3m":float(row["fwd_ret_3m"]),
                "fwd_ret_spy":float(row["fwd_ret_spy_3m"]),
                "mom_12_1":float(row.get("mom_12_1",float("nan"))),
                "ret_12m":float(row.get("ret_12m",float("nan"))),
                "spy_12m":float(row.get("spy_12m",float("nan"))),
                "macro_factor":float(row.get("macro_factor",1.0)),
            })
        all_models.append({"fold":fold,"xgb":xc,"lgb":lc,"medians":meds,
                            "kept_features":kept,"feature_importance":fi})

    return {"fold_metrics":fold_metrics,"oof_df":pd.DataFrame(oof_rows),"models":all_models}


# ─────────────────────────────────────────────────────────────────────────────
# 10. ★ Backtest with macro_factor exposure adjustment
# ─────────────────────────────────────────────────────────────────────────────
def backtest_v19(oof_df: pd.DataFrame, top_pct: float = TOP_PCT) -> Dict:
    """
    Two variants compared:
      v19_macro_adj  : effective_exposure = REGIME_EXPOSURE × macro_factor (V19 full)
      v19_regime_only: standard regime exposure (same as V17, no macro adjustment)
    """
    oof=oof_df.dropna(subset=["fwd_ret_3m","fwd_ret_spy"]).copy()
    oof["month"]=pd.to_datetime(oof["date"]).dt.to_period("M")
    months=sorted(oof["month"].unique()); step=3

    r_macro, r_regime, r_spy = [], [], []
    rlabels, mf_vals, dates_used = [], [], []

    for i, month in enumerate(months):
        if i%step!=0: continue
        grp=oof[oof["month"]==month].copy()
        if len(grp)<5: continue
        rid=int(grp["regime_id"].mode().iloc[0])
        base_exp=REGIME_EXPOSURE[rid]
        spy_r=float(grp["fwd_ret_spy"].mean())
        n_top=max(1,int(len(grp)*top_pct))
        mf=float(grp["macro_factor"].mean())   # macro_factor is same for all stocks in month

        r_spy.append(spy_r); rlabels.append(REGIME_LABELS[rid])
        mf_vals.append(mf); dates_used.append(str(month))

        top=grp.nlargest(n_top,"prob")

        # V19: regime × macro_factor (early warning exposure reduction)
        eff_exp=base_exp*mf
        r_macro.append(float(top["fwd_ret_3m"].mean())*eff_exp + spy_r*(1.0-eff_exp))

        # V17-style: regime only (baseline)
        r_regime.append(float(top["fwd_ret_3m"].mean())*base_exp + spy_r*(1.0-base_exp))

    def _m(rets):
        r=np.array([x for x in rets if math.isfinite(x)])
        if len(r)==0: return {}
        n=len(r); ppy=12/step; cagr=float((1+r).prod()**(ppy/n)-1)
        mu=float(r.mean()); sig=float(r.std(ddof=1)) if n>1 else float("nan")
        sh=float(mu/sig*math.sqrt(ppy)) if sig and sig>0 else float("nan")
        nr=r[r<0]; dn=float(nr.std(ddof=1)) if len(nr)>1 else sig
        so=float(mu/dn*math.sqrt(ppy)) if dn and dn>0 else float("nan")
        cum=np.cumprod(1+r); mxdd=float((cum/np.maximum.accumulate(cum)-1).min())
        cal=float(cagr/abs(mxdd)) if mxdd!=0 else float("nan")
        return {"n_periods":n,"cagr":round(cagr,4),"sharpe":round(sh,4),
                "sortino":round(so,4),"max_drawdown":round(mxdd,4),
                "calmar":round(cal,4),"hit_rate":round(float((r>0).mean()),4)}

    spy_m=_m(r_spy); macro_m=_m(r_macro); regime_m=_m(r_regime)

    # Per-regime breakdown (macro-adjusted variant)
    per_r: Dict={}
    for rid2,rl in REGIME_LABELS.items():
        ii=[j for j,l in enumerate(rlabels) if l==rl]
        if not ii: continue
        rm=_m([r_macro[j] for j in ii]); rm["count"]=len(ii)
        rm["avg_macro_factor"]=round(float(np.mean([mf_vals[j] for j in ii])),3)
        per_r[rl]=rm

    # macro_factor stats for understanding
    mf_arr=np.array(mf_vals)
    macro_stats={
        "mean":round(float(mf_arr.mean()),3),
        "min":round(float(mf_arr.min()),3),
        "pct_lt_05":round(float((mf_arr<0.5).mean()),3),
        "pct_lt_07":round(float((mf_arr<0.7).mean()),3),
    }

    log.info("=== V19 BACKTEST ===")
    log.info("Macro-adj  — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             macro_m.get("sharpe",float("nan")), macro_m.get("max_drawdown",0)*100, macro_m.get("cagr",0)*100)
    log.info("Regime-only— Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             regime_m.get("sharpe",float("nan")), regime_m.get("max_drawdown",0)*100, regime_m.get("cagr",0)*100)
    log.info("SPY        — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             spy_m.get("sharpe",float("nan")), spy_m.get("max_drawdown",0)*100, spy_m.get("cagr",0)*100)
    log.info("Macro factor stats: %s", macro_stats)
    return {"v19_macro_adjusted":macro_m,"v19_regime_only":regime_m,
            "spy_benchmark":spy_m,"per_regime":per_r,"macro_factor_stats":macro_stats,
            "n_months":len(dates_used)}


# ─────────────────────────────────────────────────────────────────────────────
# 11. Print summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: Dict) -> None:
    bt=results["backtest"]; fms=results["fold_metrics"]
    mac=bt["v19_macro_adjusted"]; reg=bt["v19_regime_only"]; spy=bt["spy_benchmark"]
    bl={"V14-B":{"sharpe":1.61,"max_drawdown":-0.138,"cagr":None},
        "V17":  {"sharpe":1.378,"max_drawdown":-0.138,"cagr":0.232}}

    print("\n"+"="*82)
    print("  V19 — MACRO OVERLAY: YIELD CURVE + CREDIT SPREADS")
    print("="*82)
    print(f"{'Metric':<22} {'V19 macro':>11} {'V19 reg-only':>12} {'V17':>8} {'V14-B':>8} {'SPY':>8}")
    print("-"*74)

    def _f(d,k,pct):
        if d is None: return "n/a"
        v=d.get(k,float("nan")) if isinstance(d,dict) else float("nan")
        if v is None or (isinstance(v,float) and not math.isfinite(v)): return "n/a"
        return f"{v*100:+.1f}%" if pct else f"{v:.3f}"

    for label,key,pct in [("Sharpe","sharpe",False),("Sortino","sortino",False),
                           ("CAGR","cagr",True),("MaxDD","max_drawdown",True),
                           ("Calmar","calmar",False),("Hit rate","hit_rate",False)]:
        print(f"{label:<22} {_f(mac,key,pct):>11} {_f(reg,key,pct):>12} "
              f"{_f(bl['V17'],key,pct):>8} {_f(bl['V14-B'],key,pct):>8} {_f(spy,key,pct):>8}")
    print("-"*74)

    mfst=bt.get("macro_factor_stats",{})
    print(f"\nMacro factor: mean={mfst.get('mean','?')}  min={mfst.get('min','?')}  "
          f"months<0.5={mfst.get('pct_lt_05',0)*100:.0f}%  months<0.7={mfst.get('pct_lt_07',0)*100:.0f}%")

    print("\nV19 per-regime (macro-adjusted):")
    for rl,rm in bt.get("per_regime",{}).items():
        print(f"  {rl:<10}  Sharpe={rm.get('sharpe',float('nan')):.2f}  "
              f"MaxDD={rm.get('max_drawdown',float('nan'))*100:+.1f}%  "
              f"n={rm.get('count',0)}  avg_macro_factor={rm.get('avg_macro_factor','?')}")

    print("\nWalk-forward CV:")
    rocs=[]
    for fm in fms:
        roc=fm["roc_auc"]
        print(f"  Fold {fm['fold']}  [{fm['val_start']} → {fm['val_end']}]  "
              f"ROC-AUC={roc:.3f}  macro_feats={fm.get('n_macro_features',0)}")
        if isinstance(roc,float) and math.isfinite(roc): rocs.append(roc)
    if rocs: print(f"  Mean ROC-AUC = {np.mean(rocs):.3f} ± {np.std(rocs):.3f}")

    top=results.get("feature_importance_top10",{})
    if top:
        print("\nTop-10 features (last-fold XGB):")
        for fname,imp in list(top.items())[:10]:
            g=("MACRO" if fname in MACRO_FEATURES else
               "FUND"  if fname in FUND_FEATURES  else
               "TECH"  if fname in TECH_FEATURES  else
               "REG"   if fname in REGIME_FEATURES else "OTHER")
            print(f"  {fname:<26} [{g}]  {imp:.4f}  {'█'*int(imp*200)}")
    print("="*82+"\n")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",            default="2017-01-01")
    ap.add_argument("--end",              default="2024-12-31")
    ap.add_argument("--out",              default="data/metrics/v19_results.json")
    ap.add_argument("--model_out",        default="models/stock_picker_v19.joblib")
    ap.add_argument("--cache",            default="data/cache/edgar_v17")
    ap.add_argument("--no_cache",         action="store_true")
    ap.add_argument("--n_test_months",    type=int, default=6)
    ap.add_argument("--min_train_months", type=int, default=24)
    args=ap.parse_args()

    if not HAS_YF:             log.error("yfinance missing"); sys.exit(1)
    if not (HAS_XGB or HAS_LGB): log.error("xgb/lgb missing"); sys.exit(1)

    t0=time.time()

    # 1. Prices + regimes
    prices  = download_prices(UNIVERSE, args.start, args.end)
    regimes = compute_regimes(prices)

    # 2. Macro data
    macro_raw  = download_macro_data(args.start, args.end)
    macro_feats = compute_macro_features(macro_raw)

    # 3. EDGAR fundamentals (reuse V17 cache)
    fund_data = build_edgar_fundamentals(
        UNIVERSE, prices.index, prices, Path(args.cache), use_cache=not args.no_cache)

    # 4. Dataset
    dataset = build_dataset_v19(prices, regimes, macro_feats, fund_data, UNIVERSE)

    # 5. Walk-forward
    splits = make_walk_forward_splits(
        dataset, n_test_months=args.n_test_months,
        min_train_months=args.min_train_months, embargo_months=3)
    if not splits: log.error("No splits"); sys.exit(1)

    # 6. Train
    wf = run_walk_forward(dataset, splits)

    # 7. Backtest (with and without macro factor)
    bt = backtest_v19(wf["oof_df"])

    # 8. Aggregate
    roc_vals=[f["roc_auc"] for f in wf["fold_metrics"] if isinstance(f["roc_auc"],float)]
    cv={"mean_roc_auc":round(float(np.mean(roc_vals)),4) if roc_vals else None,
        "std_roc_auc":round(float(np.std(roc_vals)),4) if roc_vals else None,
        "mean_pr_auc":round(float(np.mean([f["pr_auc"] for f in wf["fold_metrics"]])),4),
        "mean_brier":round(float(np.mean([f["brier"] for f in wf["fold_metrics"]])),4)}

    last_fi=wf["models"][-1].get("feature_importance",{}) if wf["models"] else {}
    feat_imp_top10=dict(list(sorted(last_fi.items(),key=lambda x:x[1],reverse=True)[:10]))

    # Best variant
    ms=bt["v19_macro_adjusted"].get("sharpe",0) or 0
    rs=bt["v19_regime_only"].get("sharpe",0) or 0
    best="v19_macro_adjusted" if ms>=rs else "v19_regime_only"
    best_bt=bt["v19_macro_adjusted"] if best=="v19_macro_adjusted" else bt["v19_regime_only"]

    results={
        "model_version":"v19","training_date":datetime.now().isoformat()[:19],
        "best_variant":best,
        "data_source":"SEC EDGAR XBRL + yield curve (^TNX/^IRX) + credit spread (HYG/LQD)",
        "config":{"start":args.start,"end":args.end,"universe_size":len(UNIVERSE),
                  "n_test_months":args.n_test_months,"min_train_months":args.min_train_months,
                  "n_features":len(V19_FEATURES),"features":V19_FEATURES,
                  "macro_features":MACRO_FEATURES,
                  "macro_factor_formula":"sqrt(yc_score × cr_score) where yc in [-0.8,+1.2]->0..1",
                  "effective_exposure":"REGIME_EXPOSURE[regime] × macro_factor"},
        "cv_summary":cv,"fold_metrics":wf["fold_metrics"],"backtest":bt,
        "feature_importance_top10":feat_imp_top10,
        "comparison":{
            "v19_best_sharpe":       best_bt.get("sharpe"),
            "v19_best_max_drawdown": best_bt.get("max_drawdown"),
            "v19_best_cagr":         best_bt.get("cagr"),
            "v19_best_sortino":      best_bt.get("sortino"),
            "v19_best_calmar":       best_bt.get("calmar"),
            "v17_sharpe":1.378,"v17_max_drawdown":-0.138,
            "v14b_sharpe":1.61,"v14b_max_drawdown":-0.138,
            "delta_sharpe_vs_v17":round((best_bt.get("sharpe") or 0)-1.378,3),
            "delta_sharpe_vs_v14b":round((best_bt.get("sharpe") or 0)-1.61,3),
        },
        "elapsed_seconds":round(time.time()-t0,1),
    }

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    log.info("Results → %s", out)

    if wf["models"]:
        last=wf["models"][-1]
        bundle={"model_version":"v19","feature_version":"v19_macro_overlay",
                "data_source":"SEC EDGAR + yield curve + credit spread",
                "cols":last["kept_features"],"medians":last["medians"].tolist(),
                "xgb_model":last["xgb"],"lgb_model":last["lgb"],
                "regime_exposure":REGIME_EXPOSURE,
                "vix_bear_threshold":VIX_BEAR_THRESHOLD,"vix_bull_max":VIX_BULL_MAX,
                "spy_mom_bull_min":SPY_MOM_BULL_MIN,"spy_mom_bear_max":SPY_MOM_BEAR_MAX,
                "macro_factor_params":{"yc_low":_YC_LOW,"yc_high":_YC_HIGH,
                                       "cr_low":_CR_LOW,"cr_high":_CR_HIGH},
                "macro_features":MACRO_FEATURES}
        mp=Path(args.model_out); mp.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(bundle,mp); log.info("Model → %s", mp)

    print_summary(results)
    log.info("Done in %.1fs", time.time()-t0)


if __name__ == "__main__":
    main()
