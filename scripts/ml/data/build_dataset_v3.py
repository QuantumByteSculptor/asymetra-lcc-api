"""
scripts/ml/data/build_dataset_v3.py
=====================================
Dataset v3 pipeline — temporally rigorous, multi-asset, multi-horizon.

Key improvements over v2 (build_dataset_daily.py):
  - Mandatory timestamps on every record:
      window_start_date / window_end_date / label_start_date / label_end_date
  - Multi-horizon labels: 5d, 10d, 20d, 60d
  - Full asset-type support: equity, etf, fx, commodity, crypto, index
  - =X / =F / ^ tickers no longer skipped — routed to correct provider
  - Explicit PROVIDER_MAP per asset_type (yfinance / stooq / fallback)
  - Expanding-window CV compatible (cv_meta block per record)
  - Zero future leakage (features strictly from [window_start, window_end])
  - Macro context features from FRED (VIX, 10Y, 2Y, HY spread, IG spread)
  - Robust retry with exponential back-off
  - Clean structured logging + optional --log_file
  - Never crashes globally on ticker failure

Usage:
  python scripts/ml/data/build_dataset_v3.py \\
      --universe data/universe.json \\
      --out data/training/train_v3_all.jsonl \\
      --workers 4

Constraints:
  - No impact on API / prod code
  - All writes go to data/training/
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Repo-root bootstrap (script lives 3 levels deep: scripts/ml/data/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent  # scripts/ml/data → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_utils import (  # noqa: E402
    compute_downside_dev,
    compute_semivariance,
    compute_vol_of_vol,
    compute_worst_rolling_return,
    compute_autocorr,
    compute_ewma_vol_ann,
    compute_dd_duration_recovery,
    compute_stress_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging — set up in main() once args are parsed
# ---------------------------------------------------------------------------
log = logging.getLogger("build_dataset_v3")


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)s │ %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOOKBACK_DAYS: int = 252        # feature window length (trading days)
_HORIZON_PRIMARY: int = 20       # primary label horizon for warn/block classification
_HORIZON_MAX: int = 60           # max label horizon (needed ahead of window end)
_STEP_DAYS: int = 10             # stride between rolling windows
_MAX_PER_TICKER: int = 80        # hard cap on windows per ticker
_MIN_HISTORY: int = _LOOKBACK_DAYS + _HORIZON_MAX + 60  # 372 trading days minimum

# Label thresholds (same as v2 for backward compat)
_WARN_DD: float = -0.07
_BLOCK_DD: float = -0.12
_WARN_VOL_RATIO: float = 1.8
_BLOCK_VOL_RATIO: float = 2.5

# ---------------------------------------------------------------------------
# Provider mapping — explicit by asset_type
# "yf_first": yfinance tried first, stooq as fallback
# "stooq_first": stooq tried first, yfinance as fallback
# "yf_only": stooq doesn't support this format
# ---------------------------------------------------------------------------
PROVIDER_MAP: Dict[str, str] = {
    "equity":    "stooq_first",
    "etf":       "stooq_first",
    "fx":        "yf_first",      # =X format, stooq unreliable
    "commodity": "yf_first",      # =F futures; ETF proxies work on stooq too but yf simpler
    "crypto":    "yf_only",       # -USD format, stooq has no crypto
    "index":     "yf_first",      # ^GSPC / ^NDX — stooq has some but yf is more reliable
    "rate":      "yf_first",      # ^TNX, ^IRX — same as index
}

# Ticker-format overrides (stronger signal than asset_type)
def _provider_for_ticker(ticker: str, asset_type: str) -> str:
    t = ticker.strip()
    if t.endswith("-USD"):       # crypto
        return "yf_only"
    if t.endswith("=X"):        # FX pair
        return "yf_first"
    if t.endswith("=F"):        # futures
        return "yf_first"
    if t.startswith("^"):       # index
        return "yf_first"
    return PROVIDER_MAP.get(asset_type.lower(), "stooq_first")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
def _with_retry(fn, max_tries: int = 3, base_sleep: float = 0.8):
    """Call fn(*args, **kwargs) up to max_tries times with exponential back-off."""
    def wrapper(*args, **kwargs):
        last_err: Exception = RuntimeError("no attempts")
        for attempt in range(max_tries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_err = exc
                if attempt < max_tries - 1:
                    time.sleep(base_sleep * (1.5 ** attempt))
        raise last_err
    return wrapper


# ---------------------------------------------------------------------------
# Yahoo Finance v8 direct API (no yfinance library — avoids YFTzMissingError)
# ---------------------------------------------------------------------------
def _yahoo_direct_download(ticker: str, start: str) -> pd.Series:
    """Download via Yahoo Finance chart API v8 (no yfinance dep). Works for all ticker types."""
    from datetime import timezone as _tz
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=_tz.utc).timestamp())
    end_ts   = int(datetime.now(tz=_tz.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    })
    r.raise_for_status()
    j = r.json()
    result = (j.get("chart") or {}).get("result") or []
    if not result:
        err = (j.get("chart") or {}).get("error")
        raise RuntimeError(f"Yahoo v8 no result for {ticker}: {err}")
    timestamps = result[0].get("timestamp") or []
    quotes = result[0].get("indicators", {}).get("quote", [{}])
    closes = (quotes[0] if quotes else {}).get("close") or []
    if not timestamps or not closes:
        raise RuntimeError(f"Yahoo v8 empty data for {ticker}")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    s = pd.Series(
        [float(c) if c is not None else float("nan") for c in closes],
        index=dates,
    ).dropna()
    if s.empty:
        raise RuntimeError(f"Yahoo v8 all-NaN for {ticker}")
    return s


# ---------------------------------------------------------------------------
# Disk cache — atomic parquet per (ticker, start_date)
# ---------------------------------------------------------------------------
_CACHE_DIR: Optional[Path] = None
_SKIP_STOOQ: bool = False


def _cache_path(ticker: str, start: str) -> Optional[Path]:
    if _CACHE_DIR is None:
        return None
    safe = (
        ticker.replace("/", "_").replace("=", "_EQ_")
               .replace("^", "_HAT_").replace("-", "_DASH_")
               .replace(".", "_DOT_")
    )
    return _CACHE_DIR / f"{safe}__{start.replace('-', '')}.parquet"


def _cache_load(ticker: str, start: str) -> Optional[pd.Series]:
    p = _cache_path(ticker, start)
    if p is None or not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        s = pd.Series(df["close"], index=pd.DatetimeIndex(df.index)).dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _cache_save(ticker: str, start: str, s: pd.Series) -> None:
    p = _cache_path(ticker, start)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        s.to_frame(name="close").to_parquet(tmp)
        tmp.replace(p)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Price download helpers
# ---------------------------------------------------------------------------

def _as_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract a clean Close series from a yfinance DataFrame."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            return pd.Series(df[("Close", ticker)]).dropna()
        if "Close" in df.columns.get_level_values(0):
            close_df = df["Close"]
            if isinstance(close_df, pd.DataFrame):
                if ticker in close_df.columns:
                    return close_df[ticker].dropna()
                return close_df.iloc[:, 0].dropna()
            return pd.Series(close_df).dropna()
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            return s[ticker].dropna() if ticker in s.columns else s.iloc[:, 0].dropna()
        return pd.Series(s).dropna()
    return pd.Series(dtype=float)


def _stooq_candidates(ticker: str, market: str) -> List[str]:
    t = ticker.strip()
    m = (market or "").strip().upper()
    cands: List[str] = []
    if "." not in t and "=" not in t and not t.startswith("^"):
        if m == "US":
            cands.append(f"{t}.US")
        cands.append(t)
        if m != "US":
            cands.append(f"{t}.US")
    else:
        cands.append(t)
    # De-dup
    seen: set = set()
    out: List[str] = []
    for c in cands:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _stooq_download_raw(ticker: str, market: str) -> pd.Series:
    """Download from Stooq (no retry — caller wraps with _with_retry)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }
    last_err: Exception = RuntimeError("no candidates")
    for sym in _stooq_candidates(ticker, market):
        try:
            url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt or txt.lower().startswith("no data"):
                last_err = RuntimeError(f"stooq: no data for {sym}")
                continue
            first = txt.splitlines()[0].strip()
            if not first.lower().startswith("date,open,high,low,close"):
                last_err = RuntimeError(f"stooq: bad header for {sym}: {first[:60]}")
                continue
            df = pd.read_csv(io.StringIO(txt))
            if df is None or df.empty or "Close" not in df.columns:
                last_err = RuntimeError(f"stooq: missing Close for {sym}")
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
            close = pd.Series(
                df["Close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(df["Date"].to_numpy()),
            ).dropna()
            if not close.empty:
                return close
            last_err = RuntimeError(f"stooq: empty close for {sym}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"stooq failed for {ticker}: {last_err}")


def _yf_download_raw(ticker: str, start: str, max_tries: int = 3) -> pd.Series:
    """
    Download from Yahoo Finance.
    Order: (1) Yahoo v8 direct API, (2) yf.Ticker.history(), (3) yf.download() fallback.
    Avoids YFTzMissingError that affects yf.download() on many tickers.
    """
    # 1. Yahoo direct v8 API — fastest, no yfinance dependency
    try:
        s = _yahoo_direct_download(ticker, start)
        if s is not None and not s.empty:
            return s
    except Exception:
        pass

    import yfinance as yf  # lazy import

    last_err: Exception = RuntimeError("no attempts")
    for attempt in range(max_tries):
        try:
            # 2. yf.Ticker.history() — avoids YFTzMissingError
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(start=start, interval="1d", auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                s = pd.Series(df["Close"]).dropna()
                if not s.empty:
                    return s
            last_err = RuntimeError("Ticker.history: empty close")
        except Exception as exc:
            last_err = exc
        if attempt < max_tries - 1:
            time.sleep(0.8 * (1.5 ** attempt))

    # 3. Last resort: yf.download() (may hit YFTzMissingError but try anyway)
    try:
        df = yf.download(ticker, start=start, interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        close = _as_close_series(df, ticker)
        s = pd.Series(close).dropna()
        if not s.empty:
            return s
    except Exception as e:
        last_err = e

    raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")


def download_close(
    ticker: str,
    market: str,
    asset_type: str,
    start: str,
    max_tries: int = 3,
) -> Tuple[pd.Series, str]:
    """
    Download close prices with provider routing, disk cache, and fallback.
    Returns (close_series, source_name).
    Never crashes — raises RuntimeError with all error details.
    """
    # Check disk cache first
    cached = _cache_load(ticker, start)
    if cached is not None:
        return cached, "cache"

    strategy = _provider_for_ticker(ticker, asset_type)
    errs: List[str] = []

    def _try_stooq() -> Optional[pd.Series]:
        if _SKIP_STOOQ:
            return None
        try:
            s = _with_retry(_stooq_download_raw, max_tries=max_tries)(ticker, market)
            return s if s is not None and not s.empty else None
        except Exception as e:
            errs.append(f"stooq({e})")
            return None

    def _try_yf() -> Optional[pd.Series]:
        try:
            s = _yf_download_raw(ticker, start, max_tries=max_tries)
            return s if s is not None and not s.empty else None
        except Exception as e:
            errs.append(f"yf({e})")
            return None

    result: Optional[pd.Series] = None
    source: str = ""

    if strategy == "stooq_first":
        s = _try_stooq()
        if s is not None:
            result, source = s, "stooq"
        else:
            s = _try_yf()
            if s is not None:
                result, source = s, "yfinance"
    elif strategy == "yf_only":
        s = _try_yf()
        if s is not None:
            result, source = s, "yfinance"
    else:  # yf_first
        s = _try_yf()
        if s is not None:
            result, source = s, "yfinance"
        else:
            s = _try_stooq()
            if s is not None:
                result, source = s, "stooq"

    if result is not None:
        _cache_save(ticker, start, result)
        return result, source

    raise RuntimeError(f"all providers failed for {ticker}: {'; '.join(errs)}")


# ---------------------------------------------------------------------------
# Macro data from FRED
# ---------------------------------------------------------------------------

def _fred_series(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col]).set_index(date_col).sort_index()
        return df[val_col]
    except Exception as e:
        log.warning("FRED %s failed: %s", series_id, e)
        return pd.Series(dtype=float)


def download_macro_data(start: str) -> Dict[str, pd.Series]:
    """
    Download key macro indicators from FRED (no API key).
    Returns {name: daily-ffilled pd.Series}.
    """
    series_map = {
        "vix":              "VIXCLS",
        "rate_10y":         "DGS10",
        "rate_2y":          "DGS2",
        "fed_funds":        "DFF",
        "credit_spread_hy": "BAMLH0A0HYM2",
        "credit_spread_ig": "BAMLC0A0CM",
    }
    macro: Dict[str, pd.Series] = {}
    for name, fred_id in series_map.items():
        log.info("FRED %-20s (%s)...", name, fred_id)
        s = _fred_series(fred_id, start=start)
        time.sleep(0.5)
        if not s.empty:
            bday = pd.date_range(s.index.min(), s.index.max(), freq="B")
            s = s.reindex(bday).ffill().bfill()
            macro[name] = s
            log.info("  → %d pts  [%s → %s]", len(s),
                     s.index.min().date(), s.index.max().date())
        else:
            log.warning("  → EMPTY — NaN fallback for %s", name)
    return macro


def download_spy_returns(start: str, cache_dir: Optional[Path] = None) -> Tuple[pd.Series, str]:
    """
    Download SPY (or proxy) daily returns with multi-provider fallback.

    Cascade:
      1. Cache hit  (data/cache/spy_returns_<start>.parquet)
      2. Stooq      spy.us
      3. Yahoo v8   SPY
      4. Stooq      ^spx  (S&P 500 index)
      5. Yahoo v8   ^GSPC

    Returns (series, proxy_name) where proxy_name ∈ {SPY, ^GSPC, NONE}.
    Records are cached locally after a successful download.
    """
    # ── 1. Cache hit ──────────────────────────────────────────────────────────
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"spy_returns_{start.replace('-', '')}.parquet"
        if cache_path.exists():
            try:
                df_c = pd.read_parquet(cache_path)
                s = df_c["ret"]
                log.info("SPY cache hit: %d points from %s", len(s), cache_path.name)
                proxy = df_c.attrs.get("proxy", "SPY") if hasattr(df_c, "attrs") else "SPY"
                return s, proxy
            except Exception as e:
                log.warning("SPY cache read failed: %s", e)

    def _try(name: str, fn) -> Optional[pd.Series]:
        try:
            s = fn()
            if s is not None and not s.empty:
                log.info("SPY source OK: %s  (%d pts)", name, len(s))
                return s
        except Exception as ex:
            log.warning("SPY source FAIL %s: %s", name, ex)
        return None

    # ── 2. Stooq spy.us ───────────────────────────────────────────────────────
    def _stooq_spy():
        url = f"https://stooq.com/q/d/l/?s=spy.us&i=d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        r.raise_for_status()
        first = r.text[:80]
        if "Date" not in first:
            raise RuntimeError(f"stooq spy.us bad header: {first}")
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            raise RuntimeError("stooq spy.us missing Close column")
        s = df["Close"].dropna()
        if s.empty:
            raise RuntimeError("stooq spy.us empty")
        return s.pct_change().dropna()

    # ── 3. Yahoo v8 SPY ───────────────────────────────────────────────────────
    def _yahoo_spy():
        s = _yahoo_direct_download("SPY", start)
        return s.pct_change().dropna()

    # ── 4. Stooq ^spx ─────────────────────────────────────────────────────────
    def _stooq_spx():
        url = "https://stooq.com/q/d/l/?s=%5espx&i=d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        r.raise_for_status()
        first = r.text[:80]
        if "Date" not in first:
            raise RuntimeError(f"stooq ^spx bad header: {first}")
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            raise RuntimeError("stooq ^spx missing Close column")
        s = df["Close"].dropna()
        if s.empty:
            raise RuntimeError("stooq ^spx empty")
        return s.pct_change().dropna()

    # ── 5. Yahoo v8 ^GSPC ────────────────────────────────────────────────────
    def _yahoo_gspc():
        s = _yahoo_direct_download("%5EGSPC", start)
        return s.pct_change().dropna()

    candidates = [
        ("SPY",   _stooq_spy),
        ("SPY",   _yahoo_spy),
        ("^GSPC", _stooq_spx),
        ("^GSPC", _yahoo_gspc),
    ]

    for proxy_name, fn in candidates:
        s = _try(proxy_name, fn)
        if s is not None and len(s) >= 100:
            # Normalize index to date-only (midnight) so it aligns with ticker close indexes.
            # Yahoo v8 returns intraday timestamps (e.g., 14:30 UTC) that won't intersect
            # with date-only stooq/yfinance close indexes.
            if s.index.dtype != "datetime64[ns]" or s.index[0].hour != 0:
                s = s.copy()
                s.index = pd.DatetimeIndex(s.index.normalize())
            # Deduplicate after normalization (shouldn't happen but guard)
            s = s[~s.index.duplicated(keep="last")]
            # filter to [start, today]
            s = s[s.index >= pd.Timestamp(start)]
            if len(s) >= 50:
                # ── Save to cache ──────────────────────────────────────────────
                if cache_path is not None:
                    try:
                        df_save = pd.DataFrame({"ret": s})
                        df_save.attrs["proxy"] = proxy_name
                        df_save.to_parquet(cache_path)
                        log.info("SPY cached → %s", cache_path.name)
                    except Exception as e:
                        log.warning("SPY cache write failed: %s", e)
                return s, proxy_name

    log.warning("All SPY sources failed — cross-asset features will be NaN")
    return pd.Series(dtype=float), "NONE"


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _safe(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _winsorise(x: np.ndarray, lo: float = 0.005, hi: float = 0.995) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if len(finite) < 20:
        return x
    q_lo, q_hi = np.quantile(finite, [lo, hi])
    return np.clip(x, q_lo, q_hi)


def _skew_kurtosis(x: np.ndarray) -> Tuple[float, float]:
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
    kurt = float(np.mean(c ** 4) / (s2 ** 2 + 1e-12) - 3.0)
    return skew, kurt


def _rsi(prices: pd.Series, period: int = 14) -> float:
    x = prices.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def _max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    return float((prices / (peak + 1e-12) - 1.0).min())


def _bollinger_distance(prices: pd.Series, window: int = 20) -> Optional[float]:
    if len(prices) < window + 5:
        return None
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std(ddof=1)
    width = 4 * std.iloc[-1]
    if width < 1e-12:
        return 0.0
    return float((prices.iloc[-1] - mid.iloc[-1]) / width)


def _macd_hist(prices: pd.Series) -> Optional[float]:
    if len(prices) < 35:
        return None
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    px = float(prices.iloc[-1])
    return float((macd - signal).iloc[-1] / (px + 1e-12))


def _sma_slope(prices: pd.Series, window: int) -> Optional[float]:
    if len(prices) < window + 5:
        return None
    sma = prices.rolling(window).mean()
    if abs(sma.iloc[-1]) < 1e-12:
        return 0.0
    return float((sma.iloc[-1] - sma.iloc[-window]) / (sma.iloc[-1] + 1e-12) * 252 / window)


def _jump_indicator(ret: np.ndarray, sigma: float = 3.0) -> Optional[float]:
    r = ret[np.isfinite(ret)]
    if len(r) < 20:
        return None
    s = np.std(r, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(np.abs(r) > sigma * s))


def _hill_estimator(ret: np.ndarray, k: int = 20) -> Optional[float]:
    losses = -ret[np.isfinite(ret)]
    losses = losses[losses > 0]
    k = min(k, max(5, len(losses) // 20))
    if len(losses) < k + 5:
        return None
    top = np.sort(losses)[::-1][:k]
    if top[-1] < 1e-12:
        return None
    return float(1.0 / (np.mean(np.log(top / top[-1])) + 1e-12))


def _ratio(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        fa, fb = float(a), float(b)
        if not np.isfinite(fa) or not np.isfinite(fb) or abs(fb) < 1e-12:
            return None
        return float(fa / fb)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature builder (per window)
# ---------------------------------------------------------------------------

def build_features_v3(
    ticker: str,
    asset_type: str,
    market: str,
    closes: pd.Series,
    returns: pd.Series,
    macro: Dict[str, pd.Series],
    spy_returns: pd.Series,
    window_end_date: pd.Timestamp,
    market_proxy: str = "NONE",
) -> Dict[str, Any]:
    """
    Build v3 feature vector from a single backward-looking window.
    Returns {} if insufficient data.
    All features are computed strictly from [window_start, window_end] — zero leakage.
    """
    ret = returns.to_numpy(dtype=float)
    ret = ret[np.isfinite(ret)]
    px = closes.to_numpy(dtype=float)

    if len(ret) < 60 or len(px) < 60:
        return {}

    ret_w = _winsorise(ret)

    # ---- Base volatility
    vol_ann   = float(np.std(ret_w, ddof=1) * np.sqrt(252)) if len(ret_w) > 10 else None
    vol_20d   = float(np.std(ret_w[-20:], ddof=1))              if len(ret_w) >= 20  else None
    vol_60d   = float(np.std(ret_w[-60:], ddof=1) * np.sqrt(252)) if len(ret_w) >= 60  else None
    vol_120d  = float(np.std(ret_w[-120:], ddof=1) * np.sqrt(252)) if len(ret_w) >= 120 else None

    # ---- VaR / ES (positive-loss convention)
    losses = -ret_w
    var95 = float(np.quantile(losses, 0.95)) if len(losses) >= 30 else None
    var99 = float(np.quantile(losses, 0.99)) if len(losses) >= 30 else None
    tail95 = losses[losses >= (var95 or 1e9)] if var95 is not None else np.array([])
    tail99 = losses[losses >= (var99 or 1e9)] if var99 is not None else np.array([])
    es95 = float(tail95.mean()) if len(tail95) > 0 else var95
    es99 = float(tail99.mean()) if len(tail99) > 0 else var99
    tail_obs_99 = int(np.sum(losses >= (var99 or 1e9))) if var99 is not None else 0

    mdd = _max_drawdown(closes)
    n_used = int(len(ret))
    missing_pct = float(max(0.0, 1.0 - n_used / 252.0))

    # ---- Distribution shape
    skew, kurt = _skew_kurtosis(ret)

    # ---- Momentum / trend
    rsi_val      = _rsi(closes) if len(closes) >= 20 else None
    bb_dist      = _bollinger_distance(closes)
    macd_h       = _macd_hist(closes)
    sma_slope_20 = _sma_slope(closes, 20)
    sma_slope_60 = _sma_slope(closes, 60)

    # ---- Downside risk (feature_utils)
    downside_dev = compute_downside_dev(ret)
    semivariance = compute_semivariance(ret)
    vol_of_vol   = compute_vol_of_vol(ret)
    worst_5d     = compute_worst_rolling_return(ret, 5)
    worst_10d    = compute_worst_rolling_return(ret, 10)
    worst_20d    = compute_worst_rolling_return(ret, 20)
    autocorr_1   = compute_autocorr(ret, lag=1)
    autocorr_5   = compute_autocorr(ret, lag=5)
    vol_ewma     = compute_ewma_vol_ann(ret)
    dd_dur, rec  = compute_dd_duration_recovery(px)
    stress       = compute_stress_features(ret, base_var99=var99)

    # ---- Tail metrics
    jump_ind = _jump_indicator(ret)
    hill_est = _hill_estimator(ret)

    # ---- Cross-asset (SPY correlation / beta)
    corr_spy = beta_mkt = corr_vix = None
    if not spy_returns.empty:
        common = closes.index.intersection(spy_returns.index)
        if len(common) >= 60:
            a = returns.reindex(common).dropna()
            s = spy_returns.reindex(a.index).dropna()
            common2 = a.index.intersection(s.index)
            if len(common2) >= 30:
                av = a.reindex(common2).to_numpy(dtype=float)[-252:]
                sv = s.reindex(common2).to_numpy(dtype=float)[-252:]
                mask = np.isfinite(av) & np.isfinite(sv)
                av, sv = av[mask], sv[mask]
                if len(av) >= 30:
                    corr_spy = float(np.corrcoef(av, sv)[0, 1])
                    var_s = float(np.var(sv, ddof=1))
                    if var_s > 1e-12:
                        beta_mkt = float(np.cov(av, sv, ddof=1)[0, 1] / var_s)

    vix_series = macro.get("vix")
    if vix_series is not None and not vix_series.empty:
        common = closes.index.intersection(vix_series.index)
        if len(common) >= 60:
            a = returns.reindex(common).dropna()
            vr = vix_series.reindex(a.index).pct_change().dropna()
            common2 = a.index.intersection(vr.index)
            if len(common2) >= 30:
                av = a.reindex(common2).to_numpy(dtype=float)[-120:]
                vv = vr.reindex(common2).to_numpy(dtype=float)[-120:]
                mask = np.isfinite(av) & np.isfinite(vv)
                av, vv = av[mask], vv[mask]
                if len(av) >= 30:
                    corr_vix = float(np.corrcoef(av, vv)[0, 1])

    # ---- Macro snapshot as of window_end_date (no leakage)
    def _macro_val(name: str) -> Optional[float]:
        series = macro.get(name)
        if series is None or series.empty:
            return None
        valid = series[series.index <= window_end_date]
        return float(valid.iloc[-1]) if not valid.empty else None

    def _macro_pct(name: str, lb: int = 60) -> Optional[float]:
        series = macro.get(name)
        if series is None or series.empty:
            return None
        valid = series[series.index <= window_end_date]
        if len(valid) < lb:
            return None
        return float((valid.iloc[-lb:] <= float(valid.iloc[-1])).mean())

    vix_level       = _macro_val("vix")
    vix_pct_60d     = _macro_pct("vix", 60)
    rate_10y        = _macro_val("rate_10y")
    rate_2y         = _macro_val("rate_2y")
    credit_spread_hy = _macro_val("credit_spread_hy")
    credit_spread_ig = _macro_val("credit_spread_ig")
    term_spread = _ratio(rate_10y and rate_10y - (rate_2y or 0), 1) if (
        rate_10y is not None and rate_2y is not None
    ) else None
    if rate_10y is not None and rate_2y is not None:
        term_spread = float(rate_10y - rate_2y)

    vol_regime: Optional[int] = None
    if vix_pct_60d is not None:
        vol_regime = 0 if vix_pct_60d < 0.33 else (1 if vix_pct_60d < 0.67 else 2)

    # ---- Assemble feature dict
    feats: Dict[str, Any] = {
        # Identity (not used as ML features, used for splits/filtering)
        "asset_type":   (asset_type or "").strip().lower(),
        "market":       (market or "").strip().upper(),
        "ticker":       (ticker or "").strip(),

        # Base risk
        "vol_ann":      _safe(vol_ann),
        "vol_20d":      _safe(vol_20d),
        "vol_60d":      _safe(vol_60d),
        "vol_120d":     _safe(vol_120d),
        "var95":        _safe(var95),
        "var99":        _safe(var99),
        "es95":         _safe(es95),
        "es99":         _safe(es99),
        "max_dd":       _safe(mdd),
        "max_drawdown": _safe(mdd),     # alias for API compat
        "n_used":       int(n_used),
        "missing_pct":  float(missing_pct),
        "tuw_pct":      95.0,           # placeholder, computed separately if needed
        "tail_obs_99":  int(tail_obs_99),

        # Distribution
        "skew":             _safe(skew),
        "kurtosis_excess":  _safe(kurt),

        # Momentum / trend
        "rsi":          _safe(rsi_val),
        "rsi_centered": _safe((rsi_val - 50.0) / 50.0 if rsi_val is not None else None),
        "bb_distance":  _safe(bb_dist),
        "macd_hist":    _safe(macd_h),
        "sma_slope_20": _safe(sma_slope_20),
        "sma_slope_60": _safe(sma_slope_60),

        # Downside risk
        "downside_dev": _safe(downside_dev),
        "semivariance": _safe(semivariance),

        # Volatility dynamics
        "vol_of_vol":   _safe(vol_of_vol),
        "vol_ewma_ann": _safe(vol_ewma),

        # Worst rolling returns
        "worst_5d_ret":  _safe(worst_5d),
        "worst_10d_ret": _safe(worst_10d),
        "worst_20d_ret": _safe(worst_20d),

        # Serial correlation
        "autocorr_1": _safe(autocorr_1),
        "autocorr_5": _safe(autocorr_5),

        # Drawdown dynamics
        # recovery_defined=1 means the drawdown recovered within the observation window.
        # When undefined (price didn't recover), sentinel value -1 is used so the model
        # can learn from the absence of recovery — no information leakage, no NaN.
        "dd_duration":      float(dd_dur) if dd_dur and dd_dur > 0 else -1.0,
        "recovery_defined": 1.0 if rec and rec > 0 else 0.0,
        "recovery_days":    float(rec)    if rec and rec > 0    else -1.0,

        # Stress
        "stress_var99":        _safe(stress.get("stress_var99")),
        "stress_multiplier":   _safe(stress.get("stress_multiplier")),

        # Tail metrics (v3 new)
        "jump_indicator":  _safe(jump_ind),
        "hill_tail_index": _safe(hill_est),

        # Cross-asset (v3 new)
        "corr_spy":     _safe(corr_spy),
        "corr_vix":     _safe(corr_vix),
        "beta_market":  _safe(beta_mkt),
        "abs_corr_mkt": _safe(abs(corr_spy) if corr_spy is not None else None),
        "market_proxy": market_proxy,   # string metadata: "SPY" | "^GSPC" | "NONE"

        # Macro (v3 new)
        "vix_level":         _safe(vix_level),
        "vix_pct_60d":       _safe(vix_pct_60d),
        "rate_10y":          _safe(rate_10y),
        "rate_2y":           _safe(rate_2y),
        "term_spread":       _safe(term_spread),
        "credit_spread_hy":  _safe(credit_spread_hy),
        "credit_spread_ig":  _safe(credit_spread_ig),
        "vol_regime":        vol_regime,

        # Derived ratios
        "var99_var95":    _safe(_ratio(var99, var95)),
        "es99_es95":      _safe(_ratio(es99, es95)),
        "es95_var95":     _safe(_ratio(es95, var95)),
        "es99_var99":     _safe(_ratio(es99, var99)),
        "vol20_vol_ann":  _safe(_ratio(vol_20d, vol_ann / np.sqrt(252) if vol_ann else None)),
        "vol60_vol_ann":  _safe(_ratio(vol_60d, vol_ann)),
        "vol120_vol_ann": _safe(_ratio(vol_120d, vol_ann)),
        "vol20_vol60":    _safe(_ratio(vol_20d, vol_60d / np.sqrt(252) if vol_60d else None)),
        "dd_to_var99":    _safe(_ratio(abs(mdd) if mdd else None, var99)),
        "log_n_used":     _safe(float(np.log1p(n_used))),
        "downside_div_vol":   _safe(_ratio(downside_dev, vol_ann)),
        "worst_5d_vs_var99":  _safe(_ratio(abs(worst_5d) if worst_5d else None, var99)),
        "dd_duration_per_n":  _safe(_ratio(dd_dur, n_used) if dd_dur else None),
        # sentinel -1 when recovery is undefined (consistent with recovery_days)
        "recovery_per_dd":    _safe(_ratio(rec, dd_dur)) if rec and dd_dur else -1.0,
    }

    return feats


# ---------------------------------------------------------------------------
# Label computation — multi-horizon, strictly future
# ---------------------------------------------------------------------------

def compute_labels_v3(
    ret_past: pd.Series,
    px_future: pd.Series,
    ret_future: pd.Series,
) -> Dict[str, Any]:
    """
    Multi-horizon labels computed from future data only.
    Horizons: 5d, 10d, 20d (primary), 60d.
    Primary classification (20d) uses same thresholds as v2.
    """
    result: Dict[str, Any] = {}
    ret_fut = ret_future.to_numpy(dtype=float)
    px_fut  = px_future.to_numpy(dtype=float)

    # Forward returns per horizon
    for h in (5, 10, 20, 60):
        if len(ret_fut) >= h:
            cum = float(np.prod(1.0 + np.clip(ret_fut[:h], -0.9, 10.0)) - 1.0)
            result[f"forward_return_{h}d"] = round(cum, 6)
        else:
            result[f"forward_return_{h}d"] = None

    # Primary classification: 20d drawdown + volatility ratio
    label = future_dd = vol_ratio = None
    if len(ret_fut) >= 20 and len(px_fut) >= 20:
        px20 = pd.Series(px_fut[:20])
        future_dd = float((px20 / (px20.cummax() + 1e-12) - 1.0).min())

        v_past = float(np.std(ret_past.to_numpy(dtype=float), ddof=1) * np.sqrt(252))
        v_fut  = float(np.std(ret_fut[:20], ddof=1) * np.sqrt(252))
        vol_ratio = v_fut / (v_past + 1e-12) if np.isfinite(v_past) and v_past > 1e-12 else float("inf")

        if future_dd <= _BLOCK_DD or vol_ratio >= _BLOCK_VOL_RATIO:
            label = "block"
        elif future_dd <= _WARN_DD or vol_ratio >= _WARN_VOL_RATIO:
            label = "warn"
        else:
            label = "ok"

    result["label"]            = label
    result["target_non_ok"]    = (0 if label == "ok" else 1) if label else None
    result["future_dd_20d"]    = round(float(future_dd), 6) if future_dd is not None else None
    result["future_vol_ratio"] = round(float(vol_ratio), 4) if (
        vol_ratio is not None and np.isfinite(vol_ratio)
    ) else None

    return result


# ---------------------------------------------------------------------------
# Module-level shared macro (picklable for multiprocessing.Pool on macOS)
# ---------------------------------------------------------------------------
_SHARED_MACRO: Dict[str, pd.Series] = {}
_SHARED_SPY: pd.Series = pd.Series(dtype=float)
_SHARED_MARKET_PROXY: str = "NONE"


def _init_worker(
    macro_dict: Dict[str, pd.Series],
    spy_ret: pd.Series,
    cache_dir: Optional[Path],
    skip_stooq: bool,
    market_proxy: str = "NONE",
) -> None:
    global _SHARED_MACRO, _SHARED_SPY, _CACHE_DIR, _SKIP_STOOQ, _SHARED_MARKET_PROXY
    _SHARED_MACRO         = macro_dict
    _SHARED_SPY           = spy_ret
    _CACHE_DIR            = cache_dir
    _SKIP_STOOQ           = skip_stooq
    _SHARED_MARKET_PROXY  = market_proxy


# ---------------------------------------------------------------------------
# Per-ticker worker
# ---------------------------------------------------------------------------

def process_ticker(task: Dict[str, Any]) -> Tuple[List[str], str, Optional[str]]:
    """
    Process one ticker into v3 JSONL records.
    Returns (json_lines, ticker, error_msg | None).
    Never raises — all exceptions are caught and returned as error_msg.
    """
    ticker     = task["ticker"]
    asset_type = task["asset_type"]
    market     = task["market"]
    start      = task["start"]
    lookback   = int(task.get("lookback_days", _LOOKBACK_DAYS))
    step       = int(task.get("step_days",     _STEP_DAYS))
    max_wins   = int(task.get("max_per_ticker", _MAX_PER_TICKER))

    try:
        close, source = download_close(
            ticker=ticker, market=market, asset_type=asset_type, start=start,
        )
        close = pd.Series(close).dropna()
        if close.empty:
            return [], ticker, "empty close series"

        ret = close.pct_change()

        # Need room for: lookback window + 60d forward + 10d warmup margin
        min_len = lookback + _HORIZON_MAX + 10
        if len(close) < min_len:
            return [], ticker, f"too short {len(close)}<{min_len} [{source}]"

        # Window end indices (integer positions in the series)
        min_end = lookback
        max_end = len(close) - _HORIZON_MAX - 2   # ensure 60d forward fits
        if max_end <= min_end + 5:
            return [], ticker, f"insufficient window room [{source}]"

        end_ixs = list(range(min_end, max_end + 1, step))
        if len(end_ixs) > max_wins:
            idx = np.linspace(0, len(end_ixs) - 1, max_wins, dtype=int)
            end_ixs = [end_ixs[i] for i in idx]

        lines: List[str] = []
        for end_ix in end_ixs:
            # Feature window: strictly backward-looking
            px_past  = close.iloc[max(0, end_ix - lookback): end_ix]
            ret_past = ret.iloc[max(0, end_ix - lookback): end_ix].dropna()

            # Label window: strictly forward-looking
            px_fut  = close.iloc[end_ix: end_ix + _HORIZON_MAX + 2]
            ret_fut = ret.iloc[end_ix: end_ix + _HORIZON_MAX + 2].dropna()

            if len(ret_past) < 60 or len(ret_fut) < 20:
                continue

            # --- Timestamps (mandatory)
            window_start_date = str(px_past.index[0].date())
            window_end_date   = str(px_past.index[-1].date())
            label_start_date  = str(px_fut.index[0].date()) if len(px_fut) > 0 else None
            # label_end_date = end of primary 20d horizon
            label_end_date    = (
                str(px_fut.index[min(19, len(px_fut) - 1)].date())
                if len(px_fut) > 0 else None
            )
            # label_end_date for 60d horizon
            label_end_date_60d = (
                str(px_fut.index[min(59, len(px_fut) - 1)].date())
                if len(px_fut) >= 20 else None
            )

            # --- Features (zero leakage)
            feats = build_features_v3(
                ticker=ticker,
                asset_type=asset_type,
                market=market,
                closes=px_past,
                returns=ret_past,
                macro=_SHARED_MACRO,
                spy_returns=_SHARED_SPY,
                window_end_date=px_past.index[-1],
                market_proxy=_SHARED_MARKET_PROXY,
            )
            if not feats:
                continue

            # --- Labels
            labels = compute_labels_v3(ret_past, px_fut, ret_fut)
            if labels.get("label") is None:
                continue  # skip if 20d window incomplete

            # --- CV metadata (expanding-window compatible)
            cv_meta = {
                "cv_window_type": "expanding",
                "window_end_date": window_end_date,
            }

            # --- Final record
            rec: Dict[str, Any] = {
                "version":             "v3",
                # Primary target
                "label":               labels["label"],
                "label_v2":            labels["label"],         # alias for API compat
                "target_non_ok":       labels["target_non_ok"],
                # Mandatory timestamps
                "window_start_date":   window_start_date,
                "window_end_date":     window_end_date,
                "label_start_date":    label_start_date,
                "label_end_date":      label_end_date,          # 20d horizon
                "label_end_date_60d":  label_end_date_60d,      # 60d horizon
                # Multi-horizon forward returns (5d/10d/20d/60d)
                "forward_return_5d":   labels.get("forward_return_5d"),
                "forward_return_10d":  labels.get("forward_return_10d"),
                "forward_return_20d":  labels.get("forward_return_20d"),
                "forward_return_60d":  labels.get("forward_return_60d"),
                # Auxiliary label info
                "future_dd_20d":       labels.get("future_dd_20d"),
                "future_vol_ratio":    labels.get("future_vol_ratio"),
                # Provenance
                "source":              source,
                "cv_meta":             cv_meta,
                # Feature vector
                "features":            feats,
            }
            lines.append(json.dumps(rec, ensure_ascii=False))

        return lines, ticker, None

    except Exception as exc:
        return [], ticker, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dataset v3 — temporally rigorous, multi-asset, multi-horizon"
    )
    ap.add_argument("--universe",       default="data/universe.json")
    ap.add_argument("--out",            default="data/training/train_v3_all.jsonl")
    ap.add_argument("--start",          default=None,
                    help="Override start date (YYYY-MM-DD). Default: lookback_years ago.")
    ap.add_argument("--lookback_years", type=int, default=7)
    ap.add_argument("--end",            default=None)
    ap.add_argument("--lookback_days",  type=int, default=_LOOKBACK_DAYS)
    ap.add_argument("--step_days",      type=int, default=_STEP_DAYS)
    ap.add_argument("--max_per_ticker", type=int, default=_MAX_PER_TICKER)
    ap.add_argument("--workers",        type=int, default=1)
    ap.add_argument("--sleep_ticker",   type=float, default=0.0)
    ap.add_argument("--asset_types",    default=None,
                    help="Comma-separated list of asset types to include (e.g. equity,etf). "
                         "Default: all.")
    ap.add_argument("--skip_macro",     action="store_true",
                    help="Skip FRED macro download (faster, useful for testing).")
    ap.add_argument("--skip_stooq",     action="store_true",
                    help="Skip stooq entirely (use when rate-limited).")
    ap.add_argument("--max_retries",    type=int, default=3,
                    help="Max provider retries per ticker (default 3).")
    ap.add_argument("--cache_dir",      default="data/cache",
                    help="Disk cache dir for downloaded price series (default data/cache).")
    ap.add_argument("--log_level",      default="INFO")
    ap.add_argument("--log_file",       default=None,
                    help="Optional log file path (in addition to stdout).")
    args = ap.parse_args()

    _setup_logging(args.log_level, args.log_file)

    # Resolve start date
    if args.start:
        start_date = args.start
    else:
        cutoff = datetime.today() - timedelta(days=args.lookback_years * 365 + 300)
        start_date = cutoff.strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("BUILD DATASET V3")
    log.info("start_date=%s  lookback_days=%d  step=%dd  max_per_ticker=%d",
             start_date, args.lookback_days, args.step_days, args.max_per_ticker)
    log.info("workers=%d  skip_macro=%s  skip_stooq=%s  max_retries=%d  cache_dir=%s",
             args.workers, args.skip_macro, args.skip_stooq, args.max_retries, args.cache_dir)
    log.info("=" * 60)

    # Load universe
    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe not found: {uni_path}")
    uni: List[Dict[str, str]] = json.loads(uni_path.read_text(encoding="utf-8"))
    log.info("Universe loaded: %d tickers from %s", len(uni), uni_path)

    # Optional filter by asset_type
    allowed_types: Optional[set] = None
    if args.asset_types:
        allowed_types = {t.strip().lower() for t in args.asset_types.split(",") if t.strip()}
        log.info("Filtering to asset_types: %s", allowed_types)

    # Build task list — v3 does NOT skip =X, =F, ^ tickers
    tasks: List[Dict[str, Any]] = []
    skipped_type = 0
    for item in uni:
        ticker = str(item.get("ticker", "")).strip()
        if not ticker:
            continue
        asset_type = str(item.get("asset_type", "")).strip().lower()
        if allowed_types and asset_type not in allowed_types:
            skipped_type += 1
            continue
        tasks.append({
            "ticker":       ticker,
            "asset_type":   asset_type,
            "market":       str(item.get("market", "")).strip(),
            "start":        start_date,
            "lookback_days": args.lookback_days,
            "step_days":    args.step_days,
            "max_per_ticker": args.max_per_ticker,
        })

    log.info("Tasks: %d tickers to process (%d skipped by asset_type filter)",
             len(tasks), skipped_type)

    # Download macro & SPY data (once, shared across workers)
    if args.skip_macro:
        log.info("Skipping macro data (--skip_macro flag set)")
        macro: Dict[str, pd.Series] = {}
        spy_ret = pd.Series(dtype=float)
    else:
        log.info("Downloading macro data from FRED...")
        macro = download_macro_data(start=start_date)
        log.info("Downloading SPY/market returns (with stooq/yahoo fallback)...")
        _cache = Path(args.cache_dir) if args.cache_dir else Path("data/cache")
        spy_ret, market_proxy = download_spy_returns(start=start_date, cache_dir=_cache)
        log.info("SPY: %d daily return points  proxy=%s", len(spy_ret), market_proxy)

    # Set globals for sequential mode and cache/stooq settings
    global _CACHE_DIR, _SKIP_STOOQ
    _CACHE_DIR  = Path(args.cache_dir) if args.cache_dir else None
    _SKIP_STOOQ = args.skip_stooq
    if _CACHE_DIR:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Cache dir: %s", _CACHE_DIR)

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failed_path = out_path.parent / "tickers_failed.txt"
    failed_file = failed_path.open("w", encoding="utf-8")

    counts:       Dict[str, int] = {"ok": 0, "warn": 0, "block": 0}
    n_ok = n_fail = total_windows = 0
    workers = max(1, args.workers)

    with out_path.open("w", encoding="utf-8") as out_file:

        def _handle(lines: List[str], ticker: str, err: Optional[str]) -> None:
            nonlocal n_ok, n_fail, total_windows
            if err:
                n_fail += 1
                log.warning("FAIL %-20s %s", ticker, err)
                failed_file.write(f"{ticker}\t{err}\n")
                failed_file.flush()
                return
            n_ok += 1
            for ln in lines:
                out_file.write(ln + "\n")
                lab = json.loads(ln).get("label", "ok") or "ok"
                counts[lab] = counts.get(lab, 0) + 1
                total_windows += 1
            if lines:
                log.info("OK   %-20s %d windows", ticker, len(lines))

        if workers > 1:
            with multiprocessing.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(macro, spy_ret, _CACHE_DIR, _SKIP_STOOQ, market_proxy),
            ) as pool:
                for result in pool.imap_unordered(process_ticker, tasks, chunksize=1):
                    _handle(*result)
        else:
            global _SHARED_MACRO, _SHARED_SPY, _SHARED_MARKET_PROXY
            _SHARED_MACRO        = macro
            _SHARED_SPY          = spy_ret
            _SHARED_MARKET_PROXY = market_proxy
            for task in tasks:
                _handle(*process_ticker(task))
                if args.sleep_ticker > 0:
                    time.sleep(args.sleep_ticker)

    failed_file.close()

    # Summary
    log.info("=" * 60)
    log.info("DONE")
    log.info("Output:  %s", out_path)
    log.info("Tickers: %d ok  /  %d failed  /  %d total",
             n_ok, n_fail, len(tasks))
    log.info("Windows: %d total", total_windows)
    log.info("Labels:  %s", counts)
    if total_windows > 0:
        dist = {k: f"{100 * v / total_windows:.1f}%" for k, v in counts.items()}
        log.info("Dist:    %s", dist)
        log.info("Estimated samples: ~%d", total_windows)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
