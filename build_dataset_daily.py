# build_dataset_daily.py
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf


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
def build_features(
    ticker: str,
    asset_type: str,
    market: str,
    closes: pd.Series,
    returns: pd.Series,
) -> Dict[str, Any]:
    ret20 = returns.tail(20).dropna()
    ret252 = returns.tail(252).dropna()
    if len(ret20) < 10 or len(ret252) < 60:
        return {}

    vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1))
    vol_ann = realized_vol_ann(ret252)
    mdd = max_drawdown(closes)

    v95, e95 = var_es(ret252, 0.95)
    v99, e99 = var_es(ret252, 0.99)

    n_used = int(len(ret252))
    missing_pct = float(1.0 - (n_used / 252.0))
    missing_pct = float(max(0.0, min(1.0, missing_pct)))

    corr_mkt = 0.0
    max_dd = float(mdd)

    dd_to_var99: Optional[float] = None
    if np.isfinite(v99) and float(v99) > 0:
        dd_to_var99 = float(abs(max_dd) / (float(v99) + 1e-12))

    tuw_pct = 95.0
    tuw_per_dd = float(tuw_pct / (abs(max_dd) + 1e-6))

    return {
        "asset_type": (asset_type or "").strip().lower(),
        "market": (market or "").strip().upper(),
        "ticker": (ticker or "").strip(),
        "vol_ann": float(vol_ann) if np.isfinite(vol_ann) else None,
        "vol_20d": float(vol_20d) if np.isfinite(vol_20d) else None,
        "max_drawdown": float(max_dd),
        "max_dd": float(max_dd),
        "corr_mkt": float(corr_mkt),
        "var95": float(v95) if np.isfinite(v95) else None,
        "var99": float(v99) if np.isfinite(v99) else None,
        "es95": float(e95) if np.isfinite(e95) else None,
        "es99": float(e99) if np.isfinite(e99) else None,
        "n_used": int(n_used),
        "missing_pct": float(missing_pct),
        "tuw_pct": float(tuw_pct),
        "tail_obs_99": int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9)))),
        "rsi": float(rsi(closes)) if len(closes) >= 20 else None,
        "dd_to_var99": dd_to_var99,
        "tuw_per_dd": tuw_per_dd,
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
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--windows_per_ticker", type=int, default=120)
    ap.add_argument("--out_dir", default="data/training")

    ap.add_argument("--unsup_bundle", default=None, help="Optional unsup_bundle.joblib to add z_if/z_lof/z_gap_if_lof")

    ap.add_argument("--sleep_ticker", type=float, default=0.0, help="Sleep between tickers (seconds)")
    ap.add_argument("--max_tries", type=int, default=3, help="Retry count for yfinance calls")
    ap.add_argument("--sleep_try", type=float, default=0.8, help="Base sleep between retries (seconds)")

    ap.add_argument("--prefer_provider", default="stooq", choices=["stooq", "yfinance"])

    args = ap.parse_args()

    rules = LabelRules(horizon_days=int(args.horizon_days))

    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe file not found: {uni_path}")

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    if not isinstance(uni, list):
        raise ValueError("Universe must be a JSON list of {ticker, asset_type, market} objects.")

    unsup: Optional[Dict[str, Any]] = None
    if args.unsup_bundle:
        unsup = load_unsup_bundle(args.unsup_bundle)
        print(f"✅ loaded unsup bundle: {args.unsup_bundle} (cols={len(unsup['columns'])})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ok = (out_dir / "train_ok.jsonl").open("w", encoding="utf-8")
    out_warn = (out_dir / "train_warn.jsonl").open("w", encoding="utf-8")
    out_block = (out_dir / "train_block.jsonl").open("w", encoding="utf-8")

    counts = {"ok": 0, "warn": 0, "block": 0}
    fails = 0
    skipped = 0

    for item in uni:
        ticker = str(item.get("ticker", "")).strip()
        asset_type = str(item.get("asset_type", "")).strip()
        market = str(item.get("market", "")).strip()

        skip_reason = should_skip_ticker_for_training(ticker)
        if skip_reason:
            skipped += 1
            print(f"↪︎ skip {ticker} ({skip_reason})")
            continue

        try:
            close, src = download_close_with_fallback(
                ticker=ticker,
                market=market,
                start=str(args.start),
                end=str(args.end) if args.end else None,
                max_tries=int(args.max_tries),
                sleep_try=float(args.sleep_try),
                prefer=str(args.prefer_provider),
            )

            close = pd.Series(close).dropna()
            if close.empty:
                raise RuntimeError("empty close series after download")

            ret = close.pct_change()

            min_len = int(args.lookback_days) + int(args.horizon_days) + 30
            if len(close) < min_len:
                print(f"⚠️ {ticker} too short len={len(close)} (need {min_len}) [{src}]")
                fails += 1
                continue

            min_end = int(args.lookback_days)
            max_end = len(close) - int(args.horizon_days) - 1
            available = max_end - min_end
            if available <= 10:
                print(f"⚠️ {ticker} not enough window room (len={len(close)}) [{src}]")
                fails += 1
                continue

            end_ixs = np.linspace(
                min_end,
                max_end,
                num=min(int(args.windows_per_ticker), available),
                dtype=int,
            )

            wrote = 0
            for end_ix in end_ixs:
                past_slice = slice(end_ix - int(args.lookback_days), end_ix)
                fut_slice = slice(end_ix, end_ix + int(args.horizon_days))

                px_past = close.iloc[past_slice]
                ret_past = ret.iloc[past_slice]
                px_fut = close.iloc[fut_slice]
                ret_fut = ret.iloc[fut_slice]

                feats = build_features(ticker, asset_type, market, px_past, ret_past)
                if not feats:
                    continue

                if unsup is not None:
                    add_unsup_zscores_inplace(feats, unsup)

                lab = label_from_future(px_past, ret_past, px_fut, ret_fut, rules)

                rec = {"label": lab, "features": feats}
                line = json.dumps(rec, ensure_ascii=False)

                if lab == "ok":
                    out_ok.write(line + "\n")
                elif lab == "warn":
                    out_warn.write(line + "\n")
                else:
                    out_block.write(line + "\n")

                counts[lab] += 1
                wrote += 1

            print(f"✅ {ticker} done (windows={wrote}) [{src}]")

            if float(args.sleep_ticker) > 0:
                time.sleep(float(args.sleep_ticker))

        except Exception as e:
            fails += 1
            print(f"⚠️ {ticker} failed: {e}")
            if float(args.sleep_ticker) > 0:
                time.sleep(float(args.sleep_ticker))

    out_ok.close()
    out_warn.close()
    out_block.close()

    print("\n--- DONE ---")
    print("counts:", counts)
    print("fails:", fails)
    print("skipped:", skipped)
    print(f"written to: {out_dir}")


if __name__ == "__main__":
    main()







