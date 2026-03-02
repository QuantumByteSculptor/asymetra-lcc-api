# build_dataset_daily.py
from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path as _Path
_REPO_ROOT = str(_Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from feature_utils import compute_all_v2_features  # type: ignore


# ============================================================
# Price adapters
# ============================================================
def _as_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance can return:
      - single-level columns ("Close")
      - MultiIndex columns (("Close","AAPL"), ...)
    Always returns a clean Close series.
    """
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
        if isinstance(s, pd.Series):
            return s.dropna()
        if isinstance(s, pd.DataFrame):
            if ticker in s.columns:
                return s[ticker].dropna()
            return s.iloc[:, 0].dropna()

    return pd.Series(dtype=float)


# ============================================================
# Stats
# ============================================================
def rsi(series: pd.Series, period: int = 14) -> float:
    x = series.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd = prices / (roll_max + 1e-12) - 1.0
    return float(dd.min())


def realized_vol_ann(returns: pd.Series) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(252))


def var_es(returns: pd.Series, q: float) -> Tuple[float, float]:
    """
    Convention: VaR/ES as positive loss magnitude (loss = -return).
    """
    losses = (-returns).dropna().to_numpy(dtype=float)
    if len(losses) < 30:
        return (np.nan, np.nan)
    v = float(np.quantile(losses, q))
    tail = losses[losses >= v]
    es = float(tail.mean()) if len(tail) else v
    return v, es


# ============================================================
# Labels (future-based)
# ============================================================
@dataclass
class LabelRules:
    horizon_days: int = 20
    warn_dd: float = -0.07
    block_dd: float = -0.12
    warn_vol_ratio: float = 1.8
    block_vol_ratio: float = 2.5


def label_from_future(
    px_past: pd.Series,
    ret_past: pd.Series,
    px_future: pd.Series,
    ret_future: pd.Series,
    rules: LabelRules,
) -> str:
    fut_dd = max_drawdown(px_future)

    v_past = realized_vol_ann(ret_past)
    v_fut = realized_vol_ann(ret_future)
    ratio = (v_fut / (v_past + 1e-12)) if np.isfinite(v_past) else np.inf

    if fut_dd <= rules.block_dd or ratio >= rules.block_vol_ratio:
        return "block"
    if fut_dd <= rules.warn_dd or ratio >= rules.warn_vol_ratio:
        return "warn"
    return "ok"


# ============================================================
# Unsupervised bundle -> inject z scores
# ============================================================
def load_unsup_bundle(path: str) -> Dict[str, Any]:
    b = joblib.load(path)
    models = b.get("models") or {}
    cols = b.get("columns") or []
    score_norm = b.get("score_norm") or {}

    if not cols or "iforest" not in models or "lof" not in models:
        raise ValueError("Invalid unsup bundle: need models.iforest/lof + columns + score_norm.")
    if "if" not in score_norm or "lof" not in score_norm:
        raise ValueError("Invalid score_norm keys (expected 'if' and 'lof').")

    return b


def add_unsup_zscores_inplace(feats: Dict[str, Any], unsup: Dict[str, Any]) -> None:
    """
    Adds:
      - raw_if, raw_lof
      - z_if, z_lof
      - z_gap_if_lof

    Single-row safe:
      - missing/non-finite values are coerced to 0.0
      - numpy array only (no DataFrame)
    """
    cols: List[str] = list(unsup["columns"])
    iforest = unsup["models"]["iforest"]
    lof = unsup["models"]["lof"]
    score_norm = unsup["score_norm"]

    row: List[float] = []
    for c in cols:
        v = feats.get(c)

        if v is None and c == "max_dd":
            v = feats.get("max_drawdown")
        if v is None and c == "max_drawdown":
            v = feats.get("max_dd")

        try:
            fv = float(v)
            if not np.isfinite(fv):
                fv = 0.0
        except Exception:
            fv = 0.0

        row.append(fv)

    X = np.asarray([row], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    raw_if = float(np.asarray(iforest.score_samples(X), dtype=float)[0])
    raw_lof = float(np.asarray(lof.score_samples(X), dtype=float)[0])

    mu_if = float(score_norm["if"]["mu"])
    sd_if = float(score_norm["if"]["sigma"] or 1e-12)
    mu_lof = float(score_norm["lof"]["mu"])
    sd_lof = float(score_norm["lof"]["sigma"] or 1e-12)

    z_if = (raw_if - mu_if) / (sd_if + 1e-12)
    z_lof = (raw_lof - mu_lof) / (sd_lof + 1e-12)

    feats["raw_if"] = raw_if
    feats["raw_lof"] = raw_lof
    feats["z_if"] = float(z_if)
    feats["z_lof"] = float(z_lof)
    feats["z_gap_if_lof"] = float(z_if - z_lof)


# ============================================================
# Feature engineering (per window)
# ============================================================
def _skew_kurtosis_np(x: np.ndarray):
    """Pure-numpy skew and excess kurtosis (matches api/main.py helper)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
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


def build_features(
    ticker: str,
    asset_type: str,
    market: str,
    closes: pd.Series,
    returns: pd.Series,
) -> Dict[str, Any]:
    ret20 = returns.tail(20).dropna()
    ret60 = returns.tail(60).dropna()
    ret120 = returns.tail(120).dropna()
    ret252 = returns.tail(252).dropna()
    if len(ret20) < 10 or len(ret252) < 60:
        return {}

    vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1))
    vol_60d = float(np.std(ret60.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret60) >= 20 else float("nan")
    vol_120d = float(np.std(ret120.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret120) >= 40 else float("nan")
    vol_ann = realized_vol_ann(ret252)
    mdd = max_drawdown(closes)

    v95, e95 = var_es(ret252, 0.95)
    v99, e99 = var_es(ret252, 0.99)

    n_used = int(len(ret252))
    missing_pct = float(1.0 - (n_used / 252.0))
    missing_pct = float(max(0.0, min(1.0, missing_pct)))

    corr_mkt = 0.0
    max_dd = float(mdd)

    tuw_pct = 95.0

    r = ret252.to_numpy(dtype=float)
    px = closes.to_numpy(dtype=float)
    skew, kurt_excess = _skew_kurtosis_np(r)

    v2 = compute_all_v2_features(r, px, base_var99=(float(v99) if np.isfinite(v99) else None))

    def _f(x) -> Optional[float]:
        if x is None:
            return None
        try:
            fv = float(x)
            return fv if np.isfinite(fv) else None
        except Exception:
            return None

    return {
        "asset_type": (asset_type or "").strip().lower(),
        "market": (market or "").strip().upper(),
        "ticker": (ticker or "").strip(),
        "vol_ann": _f(vol_ann),
        "vol_20d": _f(vol_20d),
        "vol_60d": _f(vol_60d),
        "vol_120d": _f(vol_120d),
        "max_drawdown": float(max_dd),
        "max_dd": float(max_dd),
        "corr_mkt": float(corr_mkt),
        "var95": _f(v95),
        "var99": _f(v99),
        "es95": _f(e95),
        "es99": _f(e99),
        "n_used": int(n_used),
        "missing_pct": float(missing_pct),
        "tuw_pct": float(tuw_pct),
        "tail_obs_99": int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9)))),
        "rsi": float(rsi(closes)) if len(closes) >= 20 else None,
        "skew": _f(skew),
        "kurtosis_excess": _f(kurt_excess),
        # v2 features
        "downside_dev": _f(v2.get("downside_dev")),
        "semivariance": _f(v2.get("semivariance")),
        "vol_of_vol": _f(v2.get("vol_of_vol")),
        "worst_5d_ret": _f(v2.get("worst_5d_ret")),
        "worst_20d_ret": _f(v2.get("worst_20d_ret")),
        "autocorr_1": _f(v2.get("autocorr_1")),
        "vol_ewma_ann": _f(v2.get("vol_ewma_ann")),
        "stress_var99": _f(v2.get("stress_var99")),
        "stress_multiplier": _f(v2.get("stress_multiplier")),
        "dd_duration": v2.get("dd_duration"),
        "recovery_days": v2.get("recovery_days"),
    }


# ============================================================
# Provider logic (stooq-first; yfinance fallback)
# ============================================================
def should_skip_ticker_for_training(ticker: str) -> Optional[str]:
    """
    Excludes instruments that often trigger provider quirks:
      - indices: ^GSPC, ^NDX, ...
      - FX pairs: EURUSD=X, ...
      - futures: GC=F, CL=F, ...
    """
    t = (ticker or "").strip()
    if not t:
        return "empty"
    if t.startswith("^"):
        return "index"
    if t.endswith("=X"):
        return "fx_pair"
    if t.endswith("=F"):
        return "future"
    return None


def _stooq_symbol_candidates(ticker: str, market: str) -> List[str]:
    """
    Stooq is picky about symbols.
    For US-listed tickers, 'TICKER.US' is often the reliable form,
    even when your universe tags them as GLOBAL/EU by "asset exposure".
    """
    t = (ticker or "").strip()
    m = (market or "").strip().upper()

    cands: List[str] = []

    # market-based first guess
    if "." not in t and m == "US":
        cands.append(f"{t}.US")
    else:
        cands.append(t)

    # pragmatic fallback: try US suffix if ticker has no suffix
    if "." not in t:
        cands.append(f"{t}.US")

    # de-dup, keep order
    out: List[str] = []
    seen = set()
    for s in cands:
        s2 = s.strip()
        if not s2:
            continue
        k = s2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s2)
    return out


def _download_stooq_close(ticker: str, market: str) -> pd.Series:
    """
    Stooq CSV downloader with a browser-like UA.
    Tries multiple symbol candidates (ex: EFA -> EFA.US).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }

    last_err: Optional[Exception] = None
    for sym in _stooq_symbol_candidates(ticker, market):
        try:
            url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()

            txt = (r.text or "").strip()
            if not txt:
                raise RuntimeError("stooq empty body")

            first_line = txt.splitlines()[0].strip()
            if first_line.lower().startswith("no data"):
                raise RuntimeError("stooq No data")

            if not first_line.lower().startswith("date,open,high,low,close"):
                sample = txt[:200].replace("\n", "\\n")
                raise RuntimeError(f"stooq non-csv payload first_line='{first_line[:80]}' sample='{sample}'")

            df = pd.read_csv(io.StringIO(txt))
            if df is None or df.empty or "Close" not in df.columns or "Date" not in df.columns:
                raise RuntimeError(f"stooq missing columns: {list(df.columns) if df is not None else None}")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

            idx = pd.DatetimeIndex(df["Date"].to_numpy())
            close = pd.Series(df["Close"].to_numpy(dtype=float), index=idx).dropna()

            if close.empty:
                raise RuntimeError("stooq empty close after parse")

            return close

        except Exception as e:
            last_err = e

    raise RuntimeError(f"stooq failed for {ticker} (cands={_stooq_symbol_candidates(ticker, market)}): {last_err}")


def _download_yf_close(ticker: str, start: str, end: Optional[str], max_tries: int, sleep_try: float) -> pd.Series:
    """
    yfinance fallback (best-effort).
    """
    last_err: Optional[Exception] = None
    t = (ticker or "").strip()

    for k in range(max_tries):
        try:
            df = yf.download(
                t,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = _as_close_series(df, t)
            close = pd.Series(close).dropna()
            if not close.empty:
                return close
            last_err = RuntimeError("empty close series (start/end)")
        except Exception as e:
            last_err = e
        time.sleep(sleep_try * (1.4**k))

    for k in range(max_tries):
        try:
            df = yf.download(
                t,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = _as_close_series(df, t)
            close = pd.Series(close).dropna()
            if not close.empty:
                return close
            last_err = RuntimeError("empty close series (period=max)")
        except Exception as e:
            last_err = e
        time.sleep(sleep_try * (1.4**k))

    raise RuntimeError(f"yfinance failed for {t}: {last_err}")


def download_close_with_fallback(
    ticker: str,
    market: str,
    start: str,
    end: Optional[str],
    max_tries: int,
    sleep_try: float,
    prefer: str = "stooq",
) -> Tuple[pd.Series, str]:
    """
    Returns (close_series, source).
    Default: stooq-first (your yfinance tz metadata is broken right now).
    """
    prefer = (prefer or "stooq").strip().lower()
    if prefer not in ("stooq", "yfinance"):
        prefer = "stooq"

    errs: List[str] = []

    def _try_stooq() -> Optional[pd.Series]:
        try:
            s = _download_stooq_close(ticker, market=market)
            return s if s is not None and not s.empty else None
        except Exception as e:
            errs.append(f"stooq failed ({type(e).__name__}: {e})")
            return None

    def _try_yf() -> Optional[pd.Series]:
        try:
            s = _download_yf_close(ticker, start, end, max_tries=max_tries, sleep_try=sleep_try)
            return s if s is not None and not s.empty else None
        except Exception as e:
            errs.append(f"yf failed ({type(e).__name__}: {e})")
            return None

    if prefer == "stooq":
        s = _try_stooq()
        if s is not None:
            return s, "stooq"
        s = _try_yf()
        if s is not None:
            return s, "yfinance"
    else:
        s = _try_yf()
        if s is not None:
            return s, "yfinance"
        s = _try_stooq()
        if s is not None:
            return s, "stooq"

    raise RuntimeError(" + ".join(errs) if errs else "no provider returned data")


# ============================================================
# Multiprocessing worker (module-level = picklable on macOS)
# ============================================================
def _worker(task: Dict[str, Any]) -> Tuple[List[str], str, Optional[str]]:
    """
    Process one ticker.  Returns (json_lines, ticker, error_msg|None).
    All params come through the `task` dict so Pool.imap works without closures.
    """
    ticker = task["ticker"]
    asset_type = task["asset_type"]
    market = task["market"]
    start = task["start"]
    end = task["end"]
    lookback_days = task["lookback_days"]
    horizon_days = task["horizon_days"]
    step_days = task["step_days"]
    max_per_ticker = task["max_per_ticker"]
    max_tries = task["max_tries"]
    sleep_try = task["sleep_try"]
    prefer = task["prefer"]
    rules = LabelRules(horizon_days=horizon_days)

    try:
        close, src = download_close_with_fallback(
            ticker=ticker,
            market=market,
            start=start,
            end=end,
            max_tries=max_tries,
            sleep_try=sleep_try,
            prefer=prefer,
        )

        close = pd.Series(close).dropna()
        if close.empty:
            return [], ticker, "empty close series"

        ret = close.pct_change()

        min_len = lookback_days + horizon_days + 30
        if len(close) < min_len:
            return [], ticker, f"too short len={len(close)} need={min_len} [{src}]"

        min_end = lookback_days
        max_end = len(close) - horizon_days - 1
        available = max_end - min_end
        if available <= 10:
            return [], ticker, f"not enough window room [{src}]"

        end_ixs = list(range(min_end, max_end + 1, step_days))
        if len(end_ixs) > max_per_ticker:
            # Evenly subsample if too many
            idx = np.linspace(0, len(end_ixs) - 1, max_per_ticker, dtype=int)
            end_ixs = [end_ixs[i] for i in idx]

        lines: List[str] = []
        for end_ix in end_ixs:
            past_slice = slice(end_ix - lookback_days, end_ix)
            fut_slice = slice(end_ix, end_ix + horizon_days)

            px_past = close.iloc[past_slice]
            ret_past = ret.iloc[past_slice]
            px_fut = close.iloc[fut_slice]
            ret_fut = ret.iloc[fut_slice]

            feats = build_features(ticker, asset_type, market, px_past, ret_past)
            if not feats:
                continue

            lab = label_from_future(px_past, ret_past, px_fut, ret_fut, rules)
            feats["label_v2"] = lab

            rec = {"label": lab, "label_v2": lab, "features": feats}
            lines.append(json.dumps(rec, ensure_ascii=False))

        return lines, ticker, None

    except Exception as exc:
        return [], ticker, str(exc)


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/universe.json")
    # New: date range via lookback_years instead of --start
    ap.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD). Defaults to lookback_years ago.")
    ap.add_argument("--lookback_years", type=int, default=5, help="Years of history to fetch (default 5).")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    # New: step_days replaces windows_per_ticker as the primary windowing param
    ap.add_argument("--step_days", type=int, default=10, help="Step between rolling windows (days). Default 10.")
    ap.add_argument("--windows_per_ticker", type=int, default=0, help="Legacy: max windows. 0 = use step_days logic.")
    ap.add_argument("--max_per_ticker", type=int, default=60, help="Hard cap on windows per ticker. Default 60.")
    # New: single output file (v2 format, label embedded in features)
    ap.add_argument("--out", default=None, help="Single output JSONL (label_v2 embedded). If set, --out_dir is ignored.")
    ap.add_argument("--out_dir", default="data/training", help="Output dir (3-file format, legacy).")
    # New: multiprocessing
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (multiprocessing). Default 1 (sequential).")
    ap.add_argument("--unsup_bundle", default=None, help="Optional unsup_bundle.joblib to add z_if/z_lof/z_gap_if_lof (single-worker only).")
    ap.add_argument("--sleep_ticker", type=float, default=0.0, help="Sleep between tickers (seconds, single-worker only).")
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--sleep_try", type=float, default=0.8)
    ap.add_argument("--prefer_provider", default="stooq", choices=["stooq", "yfinance"])

    args = ap.parse_args()

    # Compute start date from lookback_years if --start not given
    start_date: str
    if args.start:
        start_date = args.start
    else:
        cutoff = datetime.today() - timedelta(days=int(args.lookback_years) * 365 + 300)
        start_date = cutoff.strftime("%Y-%m-%d")

    end_date: Optional[str] = args.end

    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe file not found: {uni_path}")

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    if not isinstance(uni, list):
        raise ValueError("Universe must be a JSON list of {ticker, asset_type, market} objects.")

    # Determine windowing mode
    step_days = int(args.step_days)
    max_per_ticker = int(args.max_per_ticker)
    if args.windows_per_ticker and int(args.windows_per_ticker) > 0:
        # Legacy mode: use windows_per_ticker as the cap
        max_per_ticker = int(args.windows_per_ticker)

    workers = max(1, int(args.workers))

    # Build task list (skip FX/futures/indices)
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    for item in uni:
        ticker = str(item.get("ticker", "")).strip()
        asset_type = str(item.get("asset_type", "")).strip()
        market = str(item.get("market", "")).strip()
        reason = should_skip_ticker_for_training(ticker)
        if reason:
            skipped += 1
            continue
        tasks.append({
            "ticker": ticker,
            "asset_type": asset_type,
            "market": market,
            "start": start_date,
            "end": end_date,
            "lookback_days": int(args.lookback_days),
            "horizon_days": int(args.horizon_days),
            "step_days": step_days,
            "max_per_ticker": max_per_ticker,
            "max_tries": int(args.max_tries),
            "sleep_try": float(args.sleep_try),
            "prefer": str(args.prefer_provider),
        })

    print(f"Universe: {len(uni)} tickers, {skipped} skipped, {len(tasks)} to process")
    print(f"History: start={start_date}, end={end_date or 'today'}")
    print(f"Windows: step={step_days}d, max_per_ticker={max_per_ticker}")
    print(f"Workers: {workers}")

    # Open output
    counts: Dict[str, int] = {"ok": 0, "warn": 0, "block": 0}
    fails = 0

    if args.out:
        # Single-file v2 format
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")
        out_ok = out_warn = out_block = None
    else:
        # Legacy 3-file format
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = None
        out_ok = (out_dir / "train_ok.jsonl").open("w", encoding="utf-8")
        out_warn = (out_dir / "train_warn.jsonl").open("w", encoding="utf-8")
        out_block = (out_dir / "train_block.jsonl").open("w", encoding="utf-8")

    # Load unsup bundle for z-score injection (single-worker mode only)
    unsup: Optional[Dict[str, Any]] = None
    if args.unsup_bundle and workers == 1:
        unsup = load_unsup_bundle(args.unsup_bundle)
        print(f"✅ loaded unsup bundle: {args.unsup_bundle} (cols={len(unsup['columns'])})")
    elif args.unsup_bundle and workers > 1:
        print("⚠️  --unsup_bundle ignored in multi-worker mode (not picklable). Run z-score injection separately.")

    def _write_lines(lines: List[str], label: str) -> None:
        if out_file is not None:
            for ln in lines:
                out_file.write(ln + "\n")
        else:
            f = out_ok if label == "ok" else (out_warn if label == "warn" else out_block)
            for ln in lines:
                f.write(ln + "\n")  # type: ignore[union-attr]

    def _handle_result(result: Tuple[List[str], str, Optional[str]]) -> None:
        nonlocal fails
        lines, ticker, err = result
        if err:
            fails += 1
            print(f"⚠️  {ticker}: {err}")
            return
        for ln in lines:
            rec = json.loads(ln)
            lab = rec.get("label", "ok")
            counts[lab] = counts.get(lab, 0) + 1
        _write_lines(lines, "")  # label-split done inside _worker already via out_file
        if lines:
            print(f"✅ {ticker} done (windows={len(lines)})")

    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                _handle_result(result)
    else:
        # Sequential (allows sleep + unsup z-score injection)
        for task in tasks:
            lines, ticker, err = _worker(task)
            if err:
                fails += 1
                print(f"⚠️  {ticker}: {err}")
            else:
                # Inject unsup z-scores if bundle loaded
                if unsup is not None and lines:
                    enriched: List[str] = []
                    for ln in lines:
                        rec = json.loads(ln)
                        feats = rec.get("features", {})
                        add_unsup_zscores_inplace(feats, unsup)
                        rec["features"] = feats
                        enriched.append(json.dumps(rec, ensure_ascii=False))
                    lines = enriched

                for ln in lines:
                    rec = json.loads(ln)
                    lab = rec.get("label", "ok")
                    counts[lab] = counts.get(lab, 0) + 1
                    _write_lines([ln], lab)

                if lines:
                    print(f"✅ {ticker} done (windows={len(lines)})")

            if float(args.sleep_ticker) > 0:
                time.sleep(float(args.sleep_ticker))

    # Close outputs
    if out_file is not None:
        out_file.close()
    if out_ok is not None:
        out_ok.close()
        out_warn.close()  # type: ignore[union-attr]
        out_block.close()  # type: ignore[union-attr]

    total = sum(counts.values())
    print("\n--- DONE ---")
    print(f"Total records: {total}")
    print("counts:", counts)
    print(f"fails: {fails} / {len(tasks)}")
    print(f"skipped (filter): {skipped}")
    if args.out:
        print(f"output: {args.out}")
    else:
        print(f"output dir: {args.out_dir}")


if __name__ == "__main__":
    main()







