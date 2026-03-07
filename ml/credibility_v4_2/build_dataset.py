"""
ml/credibility_v4_2/build_dataset.py
─────────────────────────────────────
Construit dataset_raw.jsonl pour Credibility v4.2.

Ajouts vs build_dataset_daily.py:
  - window_end_date   : date ISO de la dernière observation du lookback
  - corr_spy          : corrélation glissante 252j des rendements vs SPY
  - beta_market       : beta glissant 252j vs SPY (cov/var)
  - vix_level         : niveau ^VIX au window_end_date

Tous les records incluent run_id (injecté en argument).

Usage:
    python ml/credibility_v4_2/build_dataset.py \
        --run_id v42_20260307_1cccfa1_abc12345 \
        --out_dir artifacts/credibility_v4_2/v42_20260307_1cccfa1_abc12345 \
        --start 2010-01-01 \
        --end   2025-12-31 \
        --universe data/universe.json \
        --step_days 20 \
        --max_per_ticker 200
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from feature_utils import compute_all_v2_features  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# Price download (stooq-first, yfinance fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_STOOQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}


def _stooq_download(sym: str) -> pd.Series:
    """Download daily close from stooq. Returns Series indexed by date."""
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
    r = requests.get(url, headers=_STOOQ_HEADERS, timeout=30)
    r.raise_for_status()
    txt = (r.text or "").strip()
    if not txt or txt.splitlines()[0].lower().startswith("no data"):
        raise RuntimeError(f"stooq no data for {sym}")
    first = txt.splitlines()[0].strip().lower()
    if not first.startswith("date,open,high,low,close"):
        raise RuntimeError(f"stooq unexpected header for {sym}: {first[:60]}")
    df = pd.read_csv(io.StringIO(txt))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    idx = pd.DatetimeIndex(df["Date"].to_numpy())
    s = pd.Series(df["Close"].to_numpy(dtype=float), index=idx).dropna()
    if s.empty:
        raise RuntimeError(f"stooq empty close after parse: {sym}")
    return s


def _yf_download(sym: str, start: str, end: Optional[str]) -> pd.Series:
    """Download daily close from yfinance."""
    import yfinance as yf

    kwargs: Dict[str, Any] = dict(
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = "max"

    df = yf.download(sym, **kwargs)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {sym}")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            cl = df["Close"]
            if isinstance(cl, pd.DataFrame):
                col = sym if sym in cl.columns else cl.columns[0]
                s = cl[col]
            else:
                s = cl
        else:
            raise RuntimeError(f"yfinance no Close column: {sym}")
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        raise RuntimeError(f"yfinance no Close column: {sym}")

    s = pd.Series(s).dropna()
    if s.empty:
        raise RuntimeError(f"yfinance empty close: {sym}")
    return s


def download_close(
    ticker: str,
    market: str,
    start: str,
    end: Optional[str],
    max_tries: int = 3,
    sleep_s: float = 0.8,
) -> Tuple[pd.Series, str]:
    """stooq-first, yfinance fallback. Returns (series, source)."""
    errors: List[str] = []

    # stooq candidates
    cands = [ticker]
    if "." not in ticker:
        cands.append(f"{ticker}.US")
    if market == "UK" and not ticker.endswith(".UK"):
        cands.append(f"{ticker}.UK")

    for sym in cands:
        try:
            return _stooq_download(sym), "stooq"
        except Exception as e:
            errors.append(f"stooq[{sym}]: {e}")

    for attempt in range(max_tries):
        try:
            return _yf_download(ticker, start, end), "yfinance"
        except Exception as e:
            errors.append(f"yf[attempt={attempt}]: {e}")
            time.sleep(sleep_s * (1.4 ** attempt))

    raise RuntimeError(" | ".join(errors))


# ═══════════════════════════════════════════════════════════════════════════════
# Reference series: SPY (returns) and VIX (levels)
# ═══════════════════════════════════════════════════════════════════════════════

def load_spy_returns(start: str, end: Optional[str]) -> pd.Series:
    """Returns SPY daily log-returns indexed by date (DatetimeIndex)."""
    print("  [ref] downloading SPY …")
    for attempt in range(4):
        try:
            close, src = download_close("SPY", "US", start, end)
            print(f"  [ref] SPY ok ({len(close)} days, src={src})")
            ret = close.pct_change().dropna()
            return ret
        except Exception as e:
            print(f"  [ref] SPY attempt {attempt}: {e}")
            time.sleep(2.0)
    raise RuntimeError("Cannot download SPY reference series")


def load_vix_levels(start: str, end: Optional[str]) -> pd.Series:
    """Returns VIX daily close indexed by date."""
    print("  [ref] downloading VIX …")
    for sym, provider in [("VIX.US", "stooq"), ("^VIX", "yfinance")]:
        try:
            if provider == "stooq":
                close = _stooq_download(sym)
            else:
                close = _yf_download("^VIX", start, end)
            print(f"  [ref] VIX ok ({len(close)} days, src={provider})")
            return close
        except Exception as e:
            print(f"  [ref] VIX {sym}: {e}")
    raise RuntimeError("Cannot download VIX reference series")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, period: int = 14) -> float:
    x = series.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd = prices / (roll_max + 1e-12) - 1.0
    return float(dd.min())


def _var_es(returns: pd.Series, q: float) -> Tuple[float, float]:
    losses = (-returns).dropna().to_numpy(dtype=float)
    if len(losses) < 30:
        return float("nan"), float("nan")
    v = float(np.quantile(losses, q))
    tail = losses[losses >= v]
    es = float(tail.mean()) if len(tail) else v
    return v, es


def _realized_vol_ann(returns: pd.Series) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(252))


def _skew_kurt(x: np.ndarray) -> Tuple[float, float]:
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
    kurt = float(np.mean(c ** 4) / (s2 ** 2 + 1e-12) - 3.0)
    return skew, kurt


def compute_corr_beta(
    ret_ticker: pd.Series,
    spy_returns: pd.Series,
) -> Tuple[float, float]:
    """
    corr_spy  : Pearson correlation of ticker returns with SPY returns.
    beta_market: cov(ticker, SPY) / var(SPY).
    Both computed on the intersection of dates.
    """
    if ret_ticker.empty or spy_returns.empty:
        return float("nan"), float("nan")

    # Align on common dates
    common = ret_ticker.index.intersection(spy_returns.index)
    if len(common) < 30:
        return float("nan"), float("nan")

    r_t = ret_ticker.loc[common].to_numpy(dtype=float)
    r_s = spy_returns.loc[common].to_numpy(dtype=float)

    # Remove any non-finite
    mask = np.isfinite(r_t) & np.isfinite(r_s)
    r_t, r_s = r_t[mask], r_s[mask]
    n = len(r_t)
    if n < 20:
        return float("nan"), float("nan")

    mu_t, mu_s = r_t.mean(), r_s.mean()
    cov = float(np.mean((r_t - mu_t) * (r_s - mu_s)))
    var_s = float(np.var(r_s, ddof=0))
    std_t = float(np.std(r_t, ddof=0))
    std_s = float(np.sqrt(var_s)) if var_s > 0 else 0.0

    corr = float(cov / (std_t * std_s + 1e-12)) if (std_t > 0 and std_s > 0) else float("nan")
    corr = float(np.clip(corr, -1.0, 1.0))
    beta = float(cov / (var_s + 1e-12)) if var_s > 1e-14 else float("nan")

    return corr, beta


def lookup_vix(vix_series: pd.Series, window_end_date: pd.Timestamp) -> float:
    """Return VIX close on or before window_end_date."""
    if vix_series.empty:
        return float("nan")
    idx = vix_series.index[vix_series.index <= window_end_date]
    if idx.empty:
        return float("nan")
    return float(vix_series.loc[idx[-1]])


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Label
# ═══════════════════════════════════════════════════════════════════════════════

_HORIZON = 20
_WARN_DD = -0.07
_BLOCK_DD = -0.12
_WARN_VOL_RATIO = 1.8
_BLOCK_VOL_RATIO = 2.5


def label_from_future(
    px_past: pd.Series,
    ret_past: pd.Series,
    px_future: pd.Series,
    ret_future: pd.Series,
) -> str:
    fut_dd = _max_drawdown(px_future)
    v_past = _realized_vol_ann(ret_past)
    v_fut = _realized_vol_ann(ret_future)
    ratio = (v_fut / (v_past + 1e-12)) if np.isfinite(v_past) else np.inf

    if fut_dd <= _BLOCK_DD or ratio >= _BLOCK_VOL_RATIO:
        return "block"
    if fut_dd <= _WARN_DD or ratio >= _WARN_VOL_RATIO:
        return "warn"
    return "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Per-ticker processing
# ═══════════════════════════════════════════════════════════════════════════════

def process_ticker(
    ticker: str,
    asset_type: str,
    market: str,
    start: str,
    end: Optional[str],
    spy_returns: pd.Series,
    vix_series: pd.Series,
    lookback_days: int,
    horizon_days: int,
    step_days: int,
    max_per_ticker: int,
    run_id: str,
    max_tries: int = 3,
    sleep_s: float = 0.8,
) -> Tuple[List[str], Optional[str]]:
    """
    Process one ticker. Returns (json_lines, error_msg|None).
    """
    try:
        close, src = download_close(ticker, market, start, end, max_tries, sleep_s)
    except Exception as e:
        return [], str(e)

    close = pd.Series(close).dropna()
    if close.empty:
        return [], "empty close series"

    # Normalize timezone-naive
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    ret = close.pct_change()

    min_len = lookback_days + horizon_days + 30
    if len(close) < min_len:
        return [], f"too short len={len(close)} need={min_len} [{src}]"

    min_end = lookback_days
    max_end = len(close) - horizon_days - 1
    available = max_end - min_end
    if available <= 0:
        return [], f"no window room [{src}]"

    end_ixs = list(range(min_end, max_end + 1, step_days))
    if len(end_ixs) > max_per_ticker:
        idx = np.linspace(0, len(end_ixs) - 1, max_per_ticker, dtype=int)
        end_ixs = [end_ixs[i] for i in idx]

    lines: List[str] = []
    for end_ix in end_ixs:
        past_sl = slice(end_ix - lookback_days, end_ix)
        fut_sl = slice(end_ix, end_ix + horizon_days)

        px_past = close.iloc[past_sl]
        ret_past = ret.iloc[past_sl]
        px_fut = close.iloc[fut_sl]
        ret_fut = ret.iloc[fut_sl]

        if len(ret_past.dropna()) < 60 or len(ret_past.dropna()) < 20:
            continue

        # Window end date = last date of lookback
        window_end_ts = close.index[end_ix - 1]
        window_end_date = str(pd.Timestamp(window_end_ts).date())

        r = ret_past.dropna().to_numpy(dtype=float)
        px = px_past.to_numpy(dtype=float)

        ret252 = ret_past.dropna()
        ret20 = ret_past.tail(20).dropna()
        ret60 = ret_past.tail(60).dropna()
        ret120 = ret_past.tail(120).dropna()

        vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1)) if len(ret20) >= 10 else float("nan")
        vol_60d = float(np.std(ret60.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret60) >= 20 else float("nan")
        vol_120d = float(np.std(ret120.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret120) >= 40 else float("nan")
        vol_ann = _realized_vol_ann(ret252)

        v95, e95 = _var_es(ret252, 0.95)
        v99, e99 = _var_es(ret252, 0.99)

        max_dd = _max_drawdown(px_past)
        n_used = int(len(ret252))
        missing_pct = float(max(0.0, min(1.0, 1.0 - n_used / 252.0)))
        tuw_pct = float(np.mean(px_past.to_numpy() <= px_past.cummax().to_numpy()) * 100.0)
        tail_obs_99 = int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9))))
        skew, kurt_excess = _skew_kurt(r)
        rsi_val = _rsi(px_past) if len(px_past) >= 20 else float("nan")

        # v2 features
        v2 = compute_all_v2_features(r, px, base_var99=(_safe_float(v99)))

        # === NEW: corr_spy, beta_market, vix_level ===
        spy_window = spy_returns.loc[
            (spy_returns.index >= px_past.index[0]) & (spy_returns.index <= px_past.index[-1])
        ] if not spy_returns.empty else pd.Series(dtype=float)
        corr_spy, beta_market = compute_corr_beta(ret_past, spy_window)

        vix_ts = pd.Timestamp(window_end_ts)
        if vix_ts.tzinfo is not None:
            vix_ts = vix_ts.tz_localize(None)
        vix_level = lookup_vix(vix_series, vix_ts)
        # === END NEW ===

        feats: Dict[str, Any] = {
            "run_id": run_id,
            "window_end_date": window_end_date,
            "ticker": ticker.strip(),
            "asset_type": asset_type.strip().lower(),
            "market": market.strip().upper(),
            "vol_ann": _safe_float(vol_ann),
            "vol_20d": _safe_float(vol_20d),
            "vol_60d": _safe_float(vol_60d),
            "vol_120d": _safe_float(vol_120d),
            "max_drawdown": float(max_dd),
            "max_dd": float(max_dd),
            "corr_mkt": _safe_float(corr_spy),   # backward compat alias
            "corr_spy": _safe_float(corr_spy),    # explicit name
            "beta_market": _safe_float(beta_market),
            "vix_level": _safe_float(vix_level),
            "var95": _safe_float(v95),
            "var99": _safe_float(v99),
            "es95": _safe_float(e95),
            "es99": _safe_float(e99),
            "n_used": n_used,
            "missing_pct": missing_pct,
            "tuw_pct": tuw_pct,
            "tail_obs_99": tail_obs_99,
            "rsi": _safe_float(rsi_val),
            "skew": _safe_float(skew),
            "kurtosis_excess": _safe_float(kurt_excess),
            "downside_dev": _safe_float(v2.get("downside_dev")),
            "semivariance": _safe_float(v2.get("semivariance")),
            "vol_of_vol": _safe_float(v2.get("vol_of_vol")),
            "worst_5d_ret": _safe_float(v2.get("worst_5d_ret")),
            "worst_20d_ret": _safe_float(v2.get("worst_20d_ret")),
            "autocorr_1": _safe_float(v2.get("autocorr_1")),
            "vol_ewma_ann": _safe_float(v2.get("vol_ewma_ann")),
            "stress_var99": _safe_float(v2.get("stress_var99")),
            "stress_multiplier": _safe_float(v2.get("stress_multiplier")),
            "dd_duration": v2.get("dd_duration"),
            "recovery_days": v2.get("recovery_days"),
        }

        label = label_from_future(px_past, ret_past, px_fut, ret_fut)
        feats["label_v2"] = label

        rec = {
            "run_id": run_id,
            "label": label,
            "label_v2": label,
            "features": feats,
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    return lines, None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Build Credibility v4.2 dataset")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=20,
                    help="Step between rolling windows (trading days). Default=20 (~monthly).")
    ap.add_argument("--max_per_ticker", type=int, default=200)
    ap.add_argument("--sleep_ticker", type=float, default=0.3,
                    help="Sleep between tickers (seconds). Avoids rate-limiting.")
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--sleep_try", type=float, default=0.8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_raw.jsonl"

    print(f"=== build_dataset.py — run_id={args.run_id} ===")
    print(f"  output  : {out_path}")
    print(f"  period  : {args.start} → {args.end}")
    print(f"  lookback: {args.lookback_days}d  horizon: {args.horizon_days}d  step: {args.step_days}d")

    # ── Load universe ────────────────────────────────────────────────────────
    uni_path = Path(args.universe)
    if not uni_path.exists():
        sys.exit(f"Universe not found: {uni_path}")
    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    print(f"  universe: {len(uni)} tickers")

    # Filter out un-downloadable instruments
    _SKIP_PATTERNS = (
        lambda t: t.startswith("^"),
        lambda t: t.endswith("=X"),
        lambda t: t.endswith("=F"),
    )
    tasks = [
        u for u in uni
        if not any(p(u.get("ticker", "")) for p in _SKIP_PATTERNS)
    ]
    skipped = len(uni) - len(tasks)
    print(f"  tasks   : {len(tasks)} (skipped {skipped})")

    # ── Load reference series ────────────────────────────────────────────────
    spy_returns = load_spy_returns(args.start, args.end)
    vix_series = load_vix_levels(args.start, args.end)

    # Normalize index timezones
    if spy_returns.index.tz is not None:
        spy_returns.index = spy_returns.index.tz_localize(None)
    if vix_series.index.tz is not None:
        vix_series.index = vix_series.index.tz_localize(None)

    # ── Process tickers ──────────────────────────────────────────────────────
    counts: Dict[str, int] = {"ok": 0, "warn": 0, "block": 0}
    fails = 0
    total_written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(tasks):
            ticker = str(item.get("ticker", "")).strip()
            asset_type = str(item.get("asset_type", "")).strip()
            market = str(item.get("market", "")).strip()

            lines, err = process_ticker(
                ticker=ticker,
                asset_type=asset_type,
                market=market,
                start=args.start,
                end=args.end,
                spy_returns=spy_returns,
                vix_series=vix_series,
                lookback_days=args.lookback_days,
                horizon_days=args.horizon_days,
                step_days=args.step_days,
                max_per_ticker=args.max_per_ticker,
                run_id=args.run_id,
                max_tries=args.max_tries,
                sleep_s=args.sleep_try,
            )

            if err:
                fails += 1
                print(f"  ⚠  [{i+1}/{len(tasks)}] {ticker}: {err}")
            else:
                for ln in lines:
                    rec = json.loads(ln)
                    lab = rec.get("label", "ok")
                    counts[lab] = counts.get(lab, 0) + 1
                    fout.write(ln + "\n")
                total_written += len(lines)
                if lines:
                    print(f"  ✓  [{i+1}/{len(tasks)}] {ticker} windows={len(lines)}")

            if args.sleep_ticker > 0:
                time.sleep(args.sleep_ticker)

    total = sum(counts.values())
    print("\n=== build_dataset DONE ===")
    print(f"  Total records : {total}")
    print(f"  Label counts  : {counts}")
    print(f"  Fails         : {fails} / {len(tasks)}")
    print(f"  Output        : {out_path}")

    if total < 500:
        print(
            "  ⚠  WARNING: very few records produced. "
            "Check network connectivity and universe file."
        )


if __name__ == "__main__":
    main()
