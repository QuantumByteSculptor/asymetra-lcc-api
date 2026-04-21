"""
scripts/ml/train_v20a_monthly.py
=================================
V20a — V17 exact, monthly backtest (step=1) for comparable Sharpe vs V14-B.

Key change vs V17: backtest uses 1-month forward returns + sqrt(12) annualisation
instead of 3-month + sqrt(4).  Everything else (universe, features, EDGAR cache,
regime logic) is identical to V17.  Reuses --cache data/cache/edgar_v17.

V14-B Sharpe 1.61 is measured monthly; V17 Sharpe 1.378 is measured quarterly.
On a comparable monthly scale, V17 should read ~2.0-2.4 if the underlying alpha
is genuinely better — this script reveals the true number.

Usage:
  python scripts/ml/train_v20a_monthly.py \\
      --start 2017-01-01 --end 2024-12-31 \\
      --out   data/metrics/v20a_results.json \\
      --model_out models/stock_picker_v20a.joblib \\
      --cache data/cache/edgar_v17

Key improvement over V16: true historical TTM fundamentals with zero look-ahead bias.
Each month M uses only EDGAR filings whose filed_date ≤ last_day_of_month(M).

SEC EDGAR API (free, no key required):
  companyfacts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json
  CIK map:      https://www.sec.gov/files/company_tickers.json
  User-Agent:   Asymetra Research paul.nuyttens@gmail.com
  Rate limit:   ≤ 10 req/sec

Point-in-time TTM logic:
  - Flow metrics (revenue, income, cash flow): sum of 4 most-recent standalone
    quarterly filings (duration ~90 days) with filed_date ≤ cutoff
  - Fallback: most recent annual (FY) filing if < 4 quarters available
  - Balance sheet: most recent filing's value with filed_date ≤ cutoff
  - PE/PB: price × shares_outstanding (EDGAR) / TTM_income / equity_MRQ

Feature set (33 — identical spec to V16):
  Tech (14) + Regime (3) + Fundamentals (15) + sector_id (1)

Target:  fwd 3m alpha vs SPY > 2.5%
Goal:    beat V14-B (Sharpe 1.61, MaxDD -13.8%)

Usage:
  python scripts/ml/train_v17_sec_fund.py \\
      --start 2017-01-01 --end 2024-12-31 \\
      --out   data/metrics/v17_results.json \\
      --model_out models/stock_picker_v17.joblib \\
      [--no_cache]
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
log = logging.getLogger("train_v20a")

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Universe (same 50 stocks as V15/V16)
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
# Regime config (identical to V15/V16)
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
# Feature columns
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
V17_FEATURES = TECH_FEATURES + REGIME_FEATURES + FUND_FEATURES + STATIC_FEATURES

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
# SEC EDGAR config
# ─────────────────────────────────────────────────────────────────────────────
_EDGAR_UA      = "Asymetra Research paul.nuyttens@gmail.com"
_EDGAR_HEADERS = {"User-Agent": _EDGAR_UA}
_EDGAR_SLEEP   = 0.15   # 6.7 req/sec — safely under the 10/sec limit
_CIK_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL     = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

_MANUAL_CIK: Dict[str, int] = {
    "BRK-B": 1067983,
    "GOOGL": 1652044,
    "META":  1326801,
    "ABBV":  1551152,
    "LLY":   59478,
}

# XBRL concept configs: {key: {taxonomy, unit, names[], type}}
_CONCEPTS: Dict[str, Dict] = {
    "revenue": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense",
        ],
    },
    "gross_profit": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": ["GrossProfit"],
    },
    "operating_income": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": ["OperatingIncomeLoss"],
    },
    "net_income": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersDiluted",
        ],
    },
    "rd_expense": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
    },
    "operating_cf": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": ["NetCashProvidedByUsedInOperatingActivities"],
    },
    "capex": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "flow",
        "names": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpenditureContinuingOperations",
        ],
    },
    "total_assets": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": ["Assets"],
    },
    "equity": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
    },
    "current_assets": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": ["AssetsCurrent"],
    },
    "current_liabilities": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": ["LiabilitiesCurrent"],
    },
    "long_term_debt": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": ["LongTermDebt", "LongTermDebtNoncurrent"],
    },
    "short_term_debt": {
        "taxonomy": "us-gaap", "unit": "USD", "type": "stock",
        "names": ["ShortTermBorrowings", "DebtCurrent"],
    },
    "shares_dei": {
        "taxonomy": "dei", "unit": "shares", "type": "stock",
        "names": ["EntityCommonStockSharesOutstanding"],
    },
    "shares_usgaap": {
        "taxonomy": "us-gaap", "unit": "shares", "type": "stock",
        "names": ["CommonStockSharesOutstanding", "SharesOutstanding"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price download (identical to V15/V16)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime detection (identical to V15/V16)
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
# 3. SEC EDGAR CIK map
# ─────────────────────────────────────────────────────────────────────────────
def _build_cik_map(session: requests.Session) -> Dict[str, int]:
    log.info("Downloading ticker→CIK map from SEC EDGAR...")
    try:
        r = session.get(_CIK_MAP_URL, headers=_EDGAR_HEADERS, timeout=30)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        log.warning("CIK map download failed (%s); using manual overrides only", exc)
        return dict(_MANUAL_CIK)

    cik_map: Dict[str, int] = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).upper().replace(".", "-")
        cik    = entry.get("cik_str")
        if ticker and cik:
            cik_map[ticker] = int(cik)

    cik_map.update(_MANUAL_CIK)
    log.info("CIK map: %d entries", len(cik_map))
    return cik_map


# ─────────────────────────────────────────────────────────────────────────────
# 4. companyfacts download + cache
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_facts(cik: int, ticker: str, session: requests.Session) -> Optional[Dict]:
    url = _FACTS_URL.format(cik10=str(cik).zfill(10))
    try:
        r = session.get(url, headers=_EDGAR_HEADERS, timeout=90)
        if r.status_code == 404:
            log.warning("[%s] CIK %d → 404", ticker, cik)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("[%s] EDGAR fetch failed: %s", ticker, exc)
        return None


def load_edgar_facts(
    tickers: List[str],
    cik_map: Dict[str, int],
    cache_dir: Path,
    session: requests.Session,
    use_cache: bool = True,
) -> Dict[str, Optional[Dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[Dict]] = {}
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        safe   = ticker.replace("-", "_")
        cpath  = cache_dir / f"{safe}.json.gz"

        if use_cache and cpath.exists():
            try:
                with gzip.open(cpath, "rt", encoding="utf-8") as fh:
                    results[ticker] = json.load(fh)
                if i % 10 == 0 or i == len(tickers):
                    log.info("[%d/%d] cache hits so far (%.0fs)", i, len(tickers), time.time() - t0)
                continue
            except Exception:
                pass  # cache corrupt → re-download

        cik = cik_map.get(ticker.upper())
        if cik is None:
            log.warning("[%s] no CIK", ticker)
            results[ticker] = None
            continue

        time.sleep(_EDGAR_SLEEP)
        facts = _fetch_facts(cik, ticker, session)
        results[ticker] = facts

        if facts is not None:
            try:
                with gzip.open(cpath, "wt", encoding="utf-8") as fh:
                    json.dump(facts, fh)
            except Exception:
                pass

        status = "✓" if facts else "✗"
        log.info("[%d/%d] %s CIK=%d %s (%.0fs)", i, len(tickers), ticker, cik, status, time.time() - t0)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. XBRL entry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_raw_entries(facts: Dict, concept_key: str) -> List[Dict]:
    """Return raw unit entries for the first matching concept name."""
    if facts is None:
        return []
    cfg      = _CONCEPTS[concept_key]
    tax_data = facts.get("facts", {}).get(cfg["taxonomy"], {})
    unit     = cfg["unit"]
    for name in cfg["names"]:
        entries = tax_data.get(name, {}).get("units", {}).get(unit)
        if entries:
            return list(entries)
    return []


def _parse_entries(raw: List[Dict]) -> List[Dict]:
    """Attach parsed Timestamps and duration_days to each entry."""
    out = []
    for e in raw:
        try:
            filed    = pd.Timestamp(e["filed"])
            end      = pd.Timestamp(e["end"])
            start_s  = e.get("start")
            start    = pd.Timestamp(start_s) if start_s else None
            duration = (end - start).days if start is not None else 0
            out.append({**e, "_filed": filed, "_end": end, "_start": start, "_dur": duration})
        except Exception:
            continue
    return out


def _ttm_flow(entries: List[Dict], cutoff: pd.Timestamp) -> Optional[float]:
    """
    TTM for a flow metric (income, revenue, cash flow) as of cutoff.

    Primary: sum of 4 non-overlapping standalone quarterly filings (duration 75–110 days).
    Fallback: most recent annual FY filing (duration 340–395 days).
    """
    avail = [e for e in entries if e["_filed"] <= cutoff]
    if not avail:
        return None

    # ── Standalone quarters (duration 75–110 days) ──
    qtrs = [e for e in avail if e["_start"] is not None and 75 <= e["_dur"] <= 110]
    if qtrs:
        # Deduplicate by period-end-month: keep latest-filed for each
        pm: Dict[str, Dict] = {}
        for e in qtrs:
            key = e["_end"].strftime("%Y-%m")
            if key not in pm or e["_filed"] > pm[key]["_filed"]:
                pm[key] = e
        sorted_q = sorted(pm.values(), key=lambda x: x["_end"], reverse=True)
        if len(sorted_q) >= 4:
            top4     = sorted_q[:4]
            coverage = (top4[0]["_end"] - top4[3]["_end"]).days
            if coverage >= 270:  # at least ~9 months span
                return sum(float(e["val"]) for e in top4)

    # ── Fallback: annual FY ──
    annuals = [
        e for e in avail
        if e.get("fp") == "FY"
        and e["_start"] is not None
        and 340 <= e["_dur"] <= 395
    ]
    if annuals:
        best = max(annuals, key=lambda x: x["_end"])
        return float(best["val"])

    return None


def _latest_stock(entries: List[Dict], cutoff: pd.Timestamp) -> Optional[float]:
    """Most recent balance-sheet value filed ≤ cutoff."""
    avail = [e for e in entries if e["_filed"] <= cutoff]
    if not avail:
        return None
    best = max(avail, key=lambda x: (x["_end"], x["_filed"]))
    v = float(best["val"])
    return v if math.isfinite(v) else None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Point-in-time fundamentals per (ticker, month)
# ─────────────────────────────────────────────────────────────────────────────
def _clip(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None or not math.isfinite(v):
        return None
    return max(lo, min(hi, v))


def _div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or abs(denom) < 1e-9:
        return None
    return float(num / denom)


def compute_fund_monthly(
    ticker: str,
    facts: Optional[Dict],
    monthly_dates: pd.DatetimeIndex,
    prices_series: pd.Series,
) -> pd.DataFrame:
    """
    Return DataFrame(index=monthly_dates, columns=FUND_FEATURES) with
    point-in-time fundamentals for one ticker.
    """
    result = pd.DataFrame(float("nan"), index=monthly_dates, columns=FUND_FEATURES)
    if facts is None:
        return result

    # Pre-parse all concepts once
    parsed: Dict[str, List[Dict]] = {}
    for key in _CONCEPTS:
        parsed[key] = _parse_entries(_get_raw_entries(facts, key))
    all_shares = parsed["shares_dei"] + parsed["shares_usgaap"]

    for month_date in monthly_dates:
        cutoff   = month_date + pd.offsets.MonthEnd(0)
        cut_1y   = cutoff - pd.DateOffset(months=12)

        # ── Flow TTM ──
        rev_ttm  = _ttm_flow(parsed["revenue"],         cutoff)
        gp_ttm   = _ttm_flow(parsed["gross_profit"],    cutoff)
        oi_ttm   = _ttm_flow(parsed["operating_income"],cutoff)
        ni_ttm   = _ttm_flow(parsed["net_income"],      cutoff)
        rd_ttm   = _ttm_flow(parsed["rd_expense"],      cutoff)
        ocf_ttm  = _ttm_flow(parsed["operating_cf"],    cutoff)
        cap_ttm  = _ttm_flow(parsed["capex"],           cutoff)

        fcf_ttm: Optional[float] = None
        if ocf_ttm is not None:
            cap_abs = abs(cap_ttm) if cap_ttm is not None else 0.0
            fcf_ttm = ocf_ttm - cap_abs

        # Year-ago TTM for growth
        rev_1y   = _ttm_flow(parsed["revenue"],   cut_1y)
        ni_1y    = _ttm_flow(parsed["net_income"], cut_1y)

        # ── Balance sheet (MRQ) ──
        assets   = _latest_stock(parsed["total_assets"],       cutoff)
        equity   = _latest_stock(parsed["equity"],             cutoff)
        curr_a   = _latest_stock(parsed["current_assets"],     cutoff)
        curr_l   = _latest_stock(parsed["current_liabilities"],cutoff)
        ltd      = _latest_stock(parsed["long_term_debt"],     cutoff) or 0.0
        std      = _latest_stock(parsed["short_term_debt"],    cutoff) or 0.0
        debt     = (ltd + std) if (ltd or std) else None
        assets_1y= _latest_stock(parsed["total_assets"],       cut_1y)
        shares   = _latest_stock(all_shares,                   cutoff)

        # ── Market cap ──
        mktcap: Optional[float] = None
        if shares and shares > 0 and month_date in prices_series.index:
            px = float(prices_series.loc[month_date])
            if pd.notna(px) and px > 0:
                mktcap = px * shares

        # ── Ratios ──
        eq_pos = equity if (equity is not None and equity > 0) else None

        gross_margin   = _clip(_div(gp_ttm,  rev_ttm),    -1.0, 1.0)
        op_margin      = _clip(_div(oi_ttm,  rev_ttm),    -1.0, 0.8)
        net_margin     = _clip(_div(ni_ttm,  rev_ttm),    -1.0, 0.5)
        roe            = _clip(_div(ni_ttm,  eq_pos),     -2.0, 2.0)
        deb_to_eq      = _clip(_div(debt,    eq_pos),      0.0, 20.0)
        fcf_margin     = _clip(_div(fcf_ttm, rev_ttm),    -1.0, 0.5)

        # R&D intensity: 0 if no R&D (retailers/financials), NaN if no revenue
        if rev_ttm is not None and rev_ttm > 0:
            rd_intensity = _clip(
                (rd_ttm / rev_ttm if rd_ttm is not None else 0.0),
                0.0, 0.5
            )
        else:
            rd_intensity = None

        rev_growth = _clip(
            (_div(rev_ttm, rev_1y) - 1 if rev_ttm and rev_1y and rev_1y > 0 else None),
            -0.5, 2.0,
        )
        ni_growth  = _clip(
            (_div(ni_ttm, ni_1y) - 1 if ni_ttm is not None and ni_1y and ni_1y != 0 else None),
            -5.0, 5.0,
        )

        ni_pos = ni_ttm if (ni_ttm is not None and ni_ttm > 0) else None
        pe_ratio      = _clip(_div(mktcap, ni_pos),    0.0, 200.0)
        pb_ratio      = _clip(_div(mktcap, eq_pos),    0.0,  50.0)
        earn_yield    = _clip(_div(1.0, pe_ratio) if pe_ratio and pe_ratio > 0 else None, -0.1, 0.3)
        current_ratio = _clip(_div(curr_a, curr_l) if curr_l and curr_l > 0 else None, 0.0, 10.0)
        asset_growth  = _clip(
            (_div(assets, assets_1y) - 1 if assets and assets_1y and assets_1y > 0 else None),
            -0.3, 1.0,
        )
        accruals_ratio = _clip(
            _div(
                (ni_ttm or 0) - (fcf_ttm or 0),
                assets,
            ) if assets and assets > 0 and ni_ttm is not None else None,
            -0.2, 0.2,
        )

        result.loc[month_date] = {
            "gross_margin":   gross_margin,
            "op_margin":      op_margin,
            "net_margin":     net_margin,
            "roe":            roe,
            "debt_to_equity": deb_to_eq,
            "rd_intensity":   rd_intensity,
            "fcf_margin":     fcf_margin,
            "revenue_growth": rev_growth,
            "ni_growth":      ni_growth,
            "pe_ratio":       pe_ratio,
            "pb_ratio":       pb_ratio,
            "earnings_yield": earn_yield,
            "current_ratio":  current_ratio,
            "asset_growth":   asset_growth,
            "accruals_ratio": accruals_ratio,
        }

    return result.astype(float)


def build_edgar_fundamentals(
    tickers: List[str],
    monthly_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    cache_dir: Path,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrate CIK map + EDGAR download + point-in-time fundamentals.
    Returns {ticker: DataFrame(monthly_dates × FUND_FEATURES)}.
    """
    session = requests.Session()

    # CIK map (cached as JSON)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cik_cache = cache_dir / "cik_map.json"
    if use_cache and cik_cache.exists():
        with cik_cache.open() as fh:
            cik_map: Dict[str, int] = json.load(fh)
        log.info("CIK map: loaded from cache (%d entries)", len(cik_map))
    else:
        cik_map = _build_cik_map(session)
        cik_cache.write_text(json.dumps(cik_map))

    # Raw companyfacts JSONs (compressed per ticker)
    facts_dir  = cache_dir / "facts"
    facts_dict = load_edgar_facts(tickers, cik_map, facts_dir, session, use_cache=use_cache)

    # Compute point-in-time fundamentals
    log.info("Computing point-in-time fundamentals: %d tickers × %d months …",
             len(tickers), len(monthly_dates))
    fund_data: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    cov_list = []

    for i, ticker in enumerate(tickers, 1):
        facts    = facts_dict.get(ticker)
        p_series = prices[ticker] if ticker in prices.columns else pd.Series(dtype=float)
        df = compute_fund_monthly(ticker, facts, monthly_dates, p_series)
        fund_data[ticker] = df
        cov = float(df.notna().mean().mean())
        cov_list.append(cov)
        if i % 10 == 0 or i == len(tickers):
            log.info("[%d/%d] %s  cov=%.0f%%  (%.0fs)", i, len(tickers), ticker, cov * 100, time.time() - t0)

    mean_cov = float(np.mean(cov_list)) if cov_list else 0.0
    log.info("Fundamental coverage: mean=%.0f%%  min=%.0f%%",
             mean_cov * 100, float(np.min(cov_list)) * 100 if cov_list else 0.0)
    return fund_data


# ─────────────────────────────────────────────────────────────────────────────
# 7. Technical features (identical to V16)
# ─────────────────────────────────────────────────────────────────────────────
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
    prices: pd.DataFrame, regimes: pd.DataFrame, ticker: str
) -> pd.DataFrame:
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

        ret_1m  = _ret(p, 1);  ret_3m  = _ret(p, 3)
        ret_6m  = _ret(p, 6);  ret_12m = _ret(p, 12)
        spy_1m  = _ret(spy, 1); spy_3m  = _ret(spy, 3); spy_12m = _ret(spy, 12)

        mom_12_1    = (ret_12m - ret_1m) if all(pd.notna(x) for x in [ret_12m, ret_1m]) else float("nan")
        ret_vs_spy3 = (ret_3m  - spy_3m) if all(pd.notna(x) for x in [ret_3m,  spy_3m]) else float("nan")

        w12 = p.iloc[max(0, i - 12): i + 1]
        mr  = w12.pct_change().dropna()
        vol_ann   = float(mr.std() * math.sqrt(12)) if len(mr) >= 3 else float("nan")
        skew_12m  = float(mr.skew())               if len(mr) >= 4 else float("nan")
        vol_3m    = float(mr.tail(3).std() * math.sqrt(12)) if len(mr) >= 3 else float("nan")
        vol_ratio = float(vol_3m / vol_ann)         if vol_ann and vol_ann > 0 else float("nan")

        hi52       = float(w12.max())
        cur        = float(p.iloc[i])
        dd_from_hi52 = float(cur / hi52 - 1) if hi52 > 0 else float("nan")
        ma_200     = float(w12.mean())
        above_200ma = float(cur > ma_200)
        trend_str  = _trend_strength(w12)

        reg_row   = regimes.loc[date] if date in regimes.index else None
        vix_level = float(reg_row["vix_level"]) if reg_row is not None else 20.0
        spy_mom_6 = float(reg_row["spy_6m"])    if reg_row is not None and pd.notna(reg_row.get("spy_6m")) else float("nan")
        regime_id = int(reg_row["regime_id"])   if reg_row is not None else REGIME_SIDEWAYS

        rows.append({
            "date": date, "ticker": ticker,
            "sector_id": float(SECTOR_MAP.get(ticker, -1)),
            "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m, "ret_12m": ret_12m,
            "mom_12_1": mom_12_1, "ret_vs_spy_3m": ret_vs_spy3,
            "spy_1m": spy_1m, "spy_12m": spy_12m,
            "vol_ann": vol_ann, "vol_ratio": vol_ratio, "skew_12m": skew_12m,
            "above_200ma": above_200ma, "trend_strength": trend_str, "dd_from_hi52": dd_from_hi52,
            "vix_level": vix_level, "spy_mom_6m": spy_mom_6, "regime_id": float(regime_id),
        })

    if not rows:
        return pd.DataFrame()

    df  = pd.DataFrame(rows).set_index("date")
    p_i = p.copy(); spy_i = spy.copy()

    fwd_stock, fwd_spy_list = [], []
    fwd_stock_1m, fwd_spy_1m_list = [], []
    for i, date in enumerate(df.index):
        pos_p   = p_i.index.get_loc(date)   if date in p_i.index   else None
        pos_spy = spy_i.index.get_loc(date) if date in spy_i.index else None

        def _fwd(series, pos, h=3):
            if pos is None or pos + h >= len(series):
                return float("nan")
            v0, vh = series.iloc[pos], series.iloc[pos + h]
            return float(vh / v0 - 1) if v0 > 0 else float("nan")

        fwd_stock.append(_fwd(p_i,   pos_p,   h=3))
        fwd_spy_list.append(_fwd(spy_i, pos_spy, h=3))
        fwd_stock_1m.append(_fwd(p_i,   pos_p,   h=1))
        fwd_spy_1m_list.append(_fwd(spy_i, pos_spy, h=1))

    df["fwd_ret_3m"]     = fwd_stock
    df["fwd_ret_spy_3m"] = fwd_spy_list
    df["fwd_ret_1m"]     = fwd_stock_1m
    df["fwd_ret_spy_1m"] = fwd_spy_1m_list
    df["fwd_alpha_3m"]   = df["fwd_ret_3m"] - df["fwd_ret_spy_3m"]
    df["label"]          = (df["fwd_alpha_3m"] > 0.025).astype(int)
    return df.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset_v17(
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

        fund_df = fund_data.get(ticker)
        if fund_df is not None and not fund_df.empty:
            fund_df.index = pd.to_datetime(fund_df.index)
            for col in FUND_FEATURES:
                tech_df[col] = fund_df[col].reindex(tech_df.index) if col in fund_df.columns else float("nan")
        else:
            for col in FUND_FEATURES:
                tech_df[col] = float("nan")

        all_dfs.append(tech_df.reset_index())

    if not all_dfs:
        raise RuntimeError("No rows built")

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.dropna(subset=["label", "fwd_ret_3m", "fwd_ret_1m"])
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)

    fund_cov = full[FUND_FEATURES].notna().mean()
    log.info("Fund coverage: mean=%.0f%%  min=%.0f%%",
             fund_cov.mean() * 100, fund_cov.min() * 100)
    low_cov = fund_cov[fund_cov < 0.20]
    if not low_cov.empty:
        log.warning("Low coverage (<20%%): %s", low_cov.to_dict())

    log.info("Dataset v20a: %d rows, %d tickers, %s → %s",
             len(full), full["ticker"].nunique(),
             full["date"].min().date(), full["date"].max().date())
    return full


# ─────────────────────────────────────────────────────────────────────────────
# 9. Walk-forward splits (identical to V15/V16)
# ─────────────────────────────────────────────────────────────────────────────
def make_walk_forward_splits(
    df: pd.DataFrame,
    n_test_months: int = 6,
    min_train_months: int = 24,
    embargo_months: int = 3,
) -> List[Dict]:
    dates = sorted(df["date"].dt.to_period("M").unique())
    splits, fold, start_idx = [], 1, min_train_months

    while True:
        end_idx = start_idx + n_test_months
        if end_idx > len(dates):
            break
        test_start   = dates[start_idx].to_timestamp()
        test_end     = dates[min(end_idx, len(dates) - 1)].to_timestamp()
        train_cutoff = start_idx - embargo_months
        if train_cutoff < min_train_months:
            start_idx += n_test_months
            continue
        train_end    = dates[train_cutoff - 1].to_timestamp()
        train_idx    = np.where((df["date"] <= train_end).values)[0]
        val_idx      = np.where(((df["date"] >= test_start) & (df["date"] < test_end)).values)[0]
        if len(train_idx) >= 100 and len(val_idx) > 0:
            splits.append({
                "fold": fold, "train_end": str(train_end.date()),
                "val_start": str(test_start.date()), "val_end": str(test_end.date()),
                "n_train": len(train_idx), "n_val": len(val_idx),
                "train_idx": train_idx, "val_idx": val_idx,
            })
            fold += 1
        start_idx += n_test_months

    log.info("Walk-forward: %d folds (test=%dm, embargo=%dm)", len(splits), n_test_months, embargo_months)
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 10. Model training helpers (identical to V16)
# ─────────────────────────────────────────────────────────────────────────────
def _median_impute(
    X_tr: np.ndarray, X_vl: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(X_tr, axis=0)
    for col in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, col]), col] = medians[col]
        X_vl[np.isnan(X_vl[:, col]), col] = medians[col]
    return X_tr, X_vl, medians


def _train_xgb(X_tr, y_tr, X_vl, y_vl):
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


def _train_lgb(X_tr, y_tr, X_vl, y_vl):
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


def _ensemble_proba(xgb_clf, lgb_clf, X: np.ndarray) -> np.ndarray:
    proba, w = np.zeros(len(X)), 0.0
    if xgb_clf is not None:
        proba += 0.5 * xgb_clf.predict_proba(X)[:, 1]; w += 0.5
    if lgb_clf is not None:
        proba += 0.5 * lgb_clf.predict_proba(X)[:, 1]; w += 0.5
    return proba / w if w > 0 else proba


# ─────────────────────────────────────────────────────────────────────────────
# 11. Walk-forward training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(df: pd.DataFrame, splits: List[Dict]) -> Dict:
    feat_cols = [c for c in V17_FEATURES if c in df.columns]
    log.info("V20a training: %d features", len(feat_cols))

    fold_metrics, oof_rows, all_models = [], [], []

    for split in splits:
        fold = split["fold"]
        tr   = df.iloc[split["train_idx"]].copy()
        vl   = df.iloc[split["val_idx"]].copy()

        X_tr = tr[feat_cols].values.astype(float)
        y_tr = tr["label"].values.astype(int)
        X_vl = vl[feat_cols].values.astype(float)
        y_vl = vl["label"].values.astype(int)

        # Drop features with >50% NaN in train
        nan_rates  = np.isnan(X_tr).mean(axis=0)
        keep_mask  = nan_rates <= 0.50
        X_tr, X_vl = X_tr[:, keep_mask], X_vl[:, keep_mask]
        kept       = [f for f, k in zip(feat_cols, keep_mask) if k]

        n_fund = sum(1 for f in kept if f in FUND_FEATURES)
        n_tech = sum(1 for f in kept if f in TECH_FEATURES)
        log.info("Fold %d  tech=%d fund=%d  train=%d val=%d",
                 fold, n_tech, n_fund, len(tr), len(vl))

        X_tr, X_vl, medians = _median_impute(X_tr.copy(), X_vl.copy())

        xgb_clf = _train_xgb(X_tr, y_tr, X_vl, y_vl)
        lgb_clf = _train_lgb(X_tr, y_tr, X_vl, y_vl)
        proba   = _ensemble_proba(xgb_clf, lgb_clf, X_vl)

        try:
            roc = roc_auc_score(y_vl, proba)
            pra = average_precision_score(y_vl, proba)
            bri = brier_score_loss(y_vl, proba)
        except Exception:
            roc = pra = bri = float("nan")

        feat_imp: Dict[str, float] = {}
        if xgb_clf is not None:
            for fname, score in zip(kept, xgb_clf.feature_importances_):
                feat_imp[fname] = round(float(score), 5)

        per_regime: Dict = {}
        reg_ids = vl["regime_id"].values.astype(int)
        for rid, rlabel in REGIME_LABELS.items():
            mask = reg_ids == rid
            if mask.sum() < 10:
                per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": None}
                continue
            try:
                pr = roc_auc_score(y_vl[mask], proba[mask])
            except Exception:
                pr = None
            per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": round(pr, 4) if pr else None}

        fold_metrics.append({
            "fold": fold, "train_end": split["train_end"],
            "val_start": split["val_start"], "val_end": split["val_end"],
            "n_train": split["n_train"], "n_val": split["n_val"],
            "roc_auc": round(roc, 4), "pr_auc": round(pra, 4), "brier": round(bri, 4),
            "pos_rate": round(float(y_vl.mean()), 4),
            "n_fund_features": n_fund, "n_tech_features": n_tech,
            "per_regime": per_regime,
        })
        log.info("  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  fund=%d", roc, pra, bri, n_fund)

        for idx, (_, row) in enumerate(vl.iterrows()):
            oof_rows.append({
                "date":           row["date"],
                "ticker":         row["ticker"],
                "prob":           float(proba[idx]),
                "label":          int(y_vl[idx]),
                "regime_id":      int(row["regime_id"]),
                "fwd_alpha_3m":   float(row["fwd_alpha_3m"]),
                "fwd_ret_3m":     float(row["fwd_ret_3m"]),
                "fwd_ret_spy":    float(row["fwd_ret_spy_3m"]),
                "fwd_ret_1m":     float(row.get("fwd_ret_1m",    float("nan"))),
                "fwd_ret_spy_1m": float(row.get("fwd_ret_spy_1m", float("nan"))),
                "mom_12_1":       float(row.get("mom_12_1", float("nan"))),
                "ret_12m":        float(row.get("ret_12m",  float("nan"))),
                "spy_12m":        float(row.get("spy_12m",  float("nan"))),
            })

        all_models.append({
            "fold": fold, "xgb": xgb_clf, "lgb": lgb_clf,
            "medians": medians, "kept_features": kept,
            "feature_importance": feat_imp,
        })

    return {"fold_metrics": fold_metrics, "oof_df": pd.DataFrame(oof_rows), "models": all_models}


# ─────────────────────────────────────────────────────────────────────────────
# 12. Backtest with regime filter (+ optional momentum)
# ─────────────────────────────────────────────────────────────────────────────
def backtest_v20a(oof_df: pd.DataFrame, top_pct: float = TOP_PCT) -> Dict:
    """Monthly-step backtest (step=1, ppy=12) using 1m forward returns.
    Produces Sharpe on same scale as V14-B (monthly sqrt(12)) for direct comparison."""
    oof   = oof_df.dropna(subset=["fwd_ret_1m", "fwd_ret_spy_1m"]).copy()
    oof["month"] = pd.to_datetime(oof["date"]).dt.to_period("M")
    months = sorted(oof["month"].unique())
    step   = 1  # monthly — comparable to V14-B's monthly Sharpe

    ret_reg_only: List[float] = []
    ret_reg_mom:  List[float] = []
    ret_spy:      List[float] = []
    regime_labels: List[str] = []
    dates_used:   List[str] = []

    for i, month in enumerate(months):
        if i % step != 0:
            continue
        grp = oof[oof["month"] == month].copy()
        if len(grp) < 5:
            continue

        regime_id = int(grp["regime_id"].mode().iloc[0])
        exposure  = REGIME_EXPOSURE[regime_id]
        spy_ret   = float(grp["fwd_ret_spy_1m"].mean())
        n_top     = max(1, int(len(grp) * top_pct))

        ret_spy.append(spy_ret)
        regime_labels.append(REGIME_LABELS[regime_id])
        dates_used.append(str(month))

        # Regime-only
        top_reg = grp.nlargest(n_top, "prob")
        reg_ret = float(top_reg["fwd_ret_1m"].mean()) * exposure + spy_ret * (1.0 - exposure)
        ret_reg_only.append(reg_ret)

        # Regime + momentum
        mom_filtered = grp[
            (grp["mom_12_1"].fillna(-1) > 0) &
            (grp["ret_12m"].fillna(-1) > grp["spy_12m"].fillna(0))
        ]
        if len(mom_filtered) < 3:
            mom_filtered = grp
        n_top_m = max(1, int(len(mom_filtered) * top_pct))
        top_m   = mom_filtered.nlargest(n_top_m, "prob")
        mom_ret = float(top_m["fwd_ret_1m"].mean()) * exposure + spy_ret * (1.0 - exposure)
        ret_reg_mom.append(mom_ret)

    def _metrics(rets: List[float]) -> Dict:
        r = np.array([x for x in rets if math.isfinite(x)])
        if len(r) == 0:
            return {}
        n    = len(r)
        ppy  = 12  # monthly observations → annualise with sqrt(12)
        cagr = float((1 + r).prod() ** (ppy / n) - 1)
        mu   = float(r.mean())
        sig  = float(r.std(ddof=1)) if n > 1 else float("nan")
        sharpe  = float(mu / sig * math.sqrt(ppy)) if sig and sig > 0 else float("nan")
        neg_r   = r[r < 0]
        down    = float(neg_r.std(ddof=1)) if len(neg_r) > 1 else sig
        sortino = float(mu / down * math.sqrt(ppy)) if down and down > 0 else float("nan")
        cum     = np.cumprod(1 + r)
        mxdd    = float((cum / np.maximum.accumulate(cum) - 1).min())
        calmar  = float(cagr / abs(mxdd)) if mxdd != 0 else float("nan")
        return {
            "n_periods": n, "cagr": round(cagr, 4), "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4), "max_drawdown": round(mxdd, 4),
            "calmar": round(calmar, 4), "hit_rate": round(float((r > 0).mean()), 4),
        }

    spy_m = _metrics(ret_spy)
    reg_m = _metrics(ret_reg_only)
    mom_m = _metrics(ret_reg_mom)

    per_regime: Dict = {}
    for rid, rlabel in REGIME_LABELS.items():
        idxs = [j for j, rl in enumerate(regime_labels) if rl == rlabel]
        if not idxs:
            continue
        rm = _metrics([ret_reg_only[j] for j in idxs])
        rm["count"] = len(idxs)
        per_regime[rlabel] = rm

    log.info("=== V17 BACKTEST ===")
    log.info("Regime-only  — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             reg_m.get("sharpe", float("nan")), reg_m.get("max_drawdown", 0) * 100, reg_m.get("cagr", 0) * 100)
    log.info("Regime+mom   — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             mom_m.get("sharpe", float("nan")), mom_m.get("max_drawdown", 0) * 100, mom_m.get("cagr", 0) * 100)
    log.info("SPY          — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             spy_m.get("sharpe", float("nan")), spy_m.get("max_drawdown", 0) * 100, spy_m.get("cagr", 0) * 100)

    return {
        "v20a_regime_only":         reg_m,
        "v20a_regime_and_momentum": mom_m,
        "spy_benchmark":           spy_m,
        "per_regime":              per_regime,
        "regime_counts":           {l: regime_labels.count(l) for l in REGIME_LABELS.values()},
        "n_months":                len(dates_used),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 13. Feature importance aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_feature_importance(models: List[Dict]) -> Dict[str, float]:
    if not models:
        return {}
    last = models[-1]
    return dict(sorted(last.get("feature_importance", {}).items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# 14. Print summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: Dict) -> None:
    bt   = results["backtest"]
    fms  = results["fold_metrics"]
    reg  = bt["v20a_regime_only"]
    mom  = bt["v20a_regime_and_momentum"]
    spy  = bt["spy_benchmark"]

    baselines = {
        "V14-B": {"sharpe": 1.61,  "max_drawdown": -0.138, "cagr": None},
        "V17 (Q)": {"sharpe": 1.378, "max_drawdown": -0.138, "cagr": 0.232},
    }

    print("\n" + "=" * 78)
    print("  V20a — MONTHLY BACKTEST (step=1, sqrt(12)) — comparable to V14-B monthly")
    print("=" * 78)
    print(f"{'Metric':<22} {'V20a reg':>10} {'V20a mom':>10} {'V17 (Q)':>8} {'V14-B':>8} {'SPY':>8}")
    print("-" * 70)

    def _fmt(d, k, pct):
        if d is None:
            return "n/a"
        v = d.get(k, float("nan")) if isinstance(d, dict) else float("nan")
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "n/a"
        return f"{v*100:+.1f}%" if pct else f"{v:.3f}"

    for label, key, pct in [
        ("Sharpe (monthly)",   "sharpe",      False),
        ("Sortino",            "sortino",     False),
        ("CAGR",               "cagr",        True),
        ("MaxDD",              "max_drawdown",True),
        ("Calmar",             "calmar",      False),
        ("Hit rate",           "hit_rate",    False),
    ]:
        print(f"{label:<22} {_fmt(reg,key,pct):>10} {_fmt(mom,key,pct):>10} "
              f"{_fmt(baselines['V17 (Q)'],key,pct):>8} {_fmt(baselines['V14-B'],key,pct):>8} "
              f"{_fmt(spy,key,pct):>8}")

    print("-" * 70)
    print("\nV17 per-regime performance (regime-only variant):")
    for rlabel, rm in bt.get("per_regime", {}).items():
        sh = rm.get("sharpe", float("nan"))
        dd = rm.get("max_drawdown", float("nan"))
        n  = rm.get("count", 0)
        print(f"  {rlabel:<10}  Sharpe={sh:.2f}  MaxDD={dd*100:+.1f}%  n={n}")

    print("\nWalk-forward CV (ROC-AUC per fold):")
    rocs = []
    for fm in fms:
        roc   = fm["roc_auc"]
        nfund = fm.get("n_fund_features", 0)
        print(f"  Fold {fm['fold']}  [{fm['val_start']} → {fm['val_end']}]  "
              f"ROC-AUC={roc:.3f}  fund_feats={nfund}")
        if isinstance(roc, float) and math.isfinite(roc):
            rocs.append(roc)
    if rocs:
        print(f"  Mean ROC-AUC = {np.mean(rocs):.3f} ± {np.std(rocs):.3f}")

    top_feats = results.get("feature_importance_top10", {})
    if top_feats:
        print("\nTop-10 features (last-fold XGB importance):")
        for fname, imp in list(top_feats.items())[:10]:
            group = ("FUND" if fname in FUND_FEATURES else
                     "TECH" if fname in TECH_FEATURES else
                     "REG"  if fname in REGIME_FEATURES else "OTHER")
            bar = "█" * int(imp * 200)
            print(f"  {fname:<22} [{group}]  {imp:.4f}  {bar}")
    print("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="V20a monthly-step backtest (V17 + step=1)")
    ap.add_argument("--start",             default="2017-01-01")
    ap.add_argument("--end",               default="2024-12-31")
    ap.add_argument("--out",               default="data/metrics/v20a_results.json")
    ap.add_argument("--model_out",         default="models/stock_picker_v20a.joblib")
    ap.add_argument("--cache",             default="data/cache/edgar_v17")
    ap.add_argument("--no_cache",          action="store_true")
    ap.add_argument("--n_test_months",     type=int, default=6)
    ap.add_argument("--min_train_months",  type=int, default=24)
    args = ap.parse_args()

    if not HAS_YF:
        log.error("yfinance not installed"); sys.exit(1)
    if not (HAS_XGB or HAS_LGB):
        log.error("xgboost or lightgbm required"); sys.exit(1)

    t0 = time.time()

    # 1. Prices + regimes
    prices  = download_prices(UNIVERSE, args.start, args.end)
    regimes = compute_regimes(prices)

    # 2. SEC EDGAR fundamentals (with per-ticker JSON.gz cache)
    cache_dir = Path(args.cache)
    fund_data = build_edgar_fundamentals(
        UNIVERSE, prices.index, prices, cache_dir,
        use_cache=not args.no_cache,
    )

    # 3. Build dataset
    dataset = build_dataset_v17(prices, regimes, fund_data, UNIVERSE)

    # 4. Walk-forward splits
    splits = make_walk_forward_splits(
        dataset,
        n_test_months=args.n_test_months,
        min_train_months=args.min_train_months,
        embargo_months=3,
    )
    if not splits:
        log.error("No walk-forward splits generated"); sys.exit(1)

    # 5. Train
    wf = run_walk_forward(dataset, splits)

    # 6. Backtest
    bt = backtest_v20a(wf["oof_df"])

    # 7. Aggregate metrics
    roc_vals = [f["roc_auc"] for f in wf["fold_metrics"] if isinstance(f["roc_auc"], float)]
    cv_summary = {
        "mean_roc_auc": round(float(np.mean(roc_vals)), 4) if roc_vals else None,
        "std_roc_auc":  round(float(np.std(roc_vals)),  4) if roc_vals else None,
        "mean_pr_auc":  round(float(np.mean([f["pr_auc"]  for f in wf["fold_metrics"]])), 4),
        "mean_brier":   round(float(np.mean([f["brier"]    for f in wf["fold_metrics"]])), 4),
    }
    feat_imp_top10 = dict(list(aggregate_feature_importance(wf["models"]).items())[:10])

    # Best variant determined after inspection (log both)
    reg_sharpe = bt["v20a_regime_only"].get("sharpe", 0) or 0
    mom_sharpe = bt["v20a_regime_and_momentum"].get("sharpe", 0) or 0
    best_variant = "v20a_regime_only" if reg_sharpe >= mom_sharpe else "v20a_regime_and_momentum"
    best_bt      = bt["v20a_regime_only"] if best_variant == "v20a_regime_only" else bt["v20a_regime_and_momentum"]

    results = {
        "model_version":   "v20a",
        "training_date":   datetime.now().isoformat()[:19],
        "best_variant":    best_variant,
        "data_source":     "SEC EDGAR XBRL companyfacts API (point-in-time, no look-ahead)",
        "backtest_note":   "monthly step=1 / sqrt(12) — comparable scale to V14-B (1.61 monthly)",
        "config": {
            "start": args.start, "end": args.end,
            "universe_size": len(UNIVERSE),
            "n_test_months": args.n_test_months,
            "min_train_months": args.min_train_months,
            "features": V17_FEATURES, "n_features": len(V17_FEATURES),
            "backtest_step": 1,
            "backtest_ppy":  12,
            "vix_bear_threshold": VIX_BEAR_THRESHOLD,
            "vix_bull_max": VIX_BULL_MAX,
            "point_in_time_rule": "filed_date <= last_day_of_month(M)",
        },
        "cv_summary":               cv_summary,
        "fold_metrics":             wf["fold_metrics"],
        "backtest":                 bt,
        "feature_importance_top10": feat_imp_top10,
        "comparison": {
            "v20a_best_sharpe":      best_bt.get("sharpe"),
            "v20a_best_max_drawdown":best_bt.get("max_drawdown"),
            "v20a_best_cagr":        best_bt.get("cagr"),
            "v20a_best_sortino":     best_bt.get("sortino"),
            "v20a_best_calmar":      best_bt.get("calmar"),
            "v17_sharpe_quarterly":  1.378,
            "v14b_sharpe_monthly":   1.61,
            "v14b_max_drawdown":     -0.138,
            "note_scale":            "all Sharpe here are monthly (sqrt(12)); V17 1.378 was quarterly (sqrt(4))",
            "delta_sharpe_vs_v14b":  round((best_bt.get("sharpe") or 0) - 1.61, 3),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # 8. Save metrics
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results → %s", out_path)

    # 9. Save model (last fold)
    if wf["models"]:
        last = wf["models"][-1]
        bundle = {
            "model_version":      "v20a",
            "feature_version":    "v20a_edgar_regime_monthly",
            "data_source":        "SEC EDGAR XBRL companyfacts",
            "cols":               last["kept_features"],
            "medians":            last["medians"].tolist(),
            "xgb_model":          last["xgb"],
            "lgb_model":          last["lgb"],
            "regime_exposure":    REGIME_EXPOSURE,
            "vix_bear_threshold": VIX_BEAR_THRESHOLD,
            "vix_bull_max":       VIX_BULL_MAX,
            "spy_mom_bull_min":   SPY_MOM_BULL_MIN,
            "spy_mom_bear_max":   SPY_MOM_BEAR_MAX,
        }
        mpath = Path(args.model_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, mpath)
        log.info("Model → %s", mpath)

    # 10. Print summary
    print_summary(results)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
