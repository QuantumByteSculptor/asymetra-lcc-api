# build_dataset_daily.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Helpers
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna()
    if len(s) < period + 2:
        return float("nan")
    x = s.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2:
        return float("nan")
    roll_max = p.cummax()
    dd = p / (roll_max + 1e-12) - 1.0
    return float(dd.min())


def realized_vol(returns: pd.Series) -> float:
    # annualized vol from daily returns (robust)
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    v = np.std(r, ddof=1) * np.sqrt(252)
    return float(v) if np.isfinite(v) else float("nan")


def var_es(returns: pd.Series, q: float) -> Tuple[float, float]:
    """
    VaR/ES computed on losses = -returns.
    Convention: positive = bad (loss magnitude).
    """
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 30:
        return (float("nan"), float("nan"))

    losses = -r
    losses = losses[np.isfinite(losses)]
    if len(losses) < 30:
        return (float("nan"), float("nan"))

    v = float(np.quantile(losses, q))
    if not np.isfinite(v):
        return (float("nan"), float("nan"))

    tail = losses[losses >= v]
    es = float(tail.mean()) if len(tail) else v
    if not np.isfinite(es):
        es = v
    return v, es


@dataclass
class LabelRules:
    horizon_days: int = 20
    # thresholds on FUTURE worst drawdown (negative)
    warn_dd: float = -0.07
    block_dd: float = -0.12
    # vol explosion ratio (future vol / past vol)
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

    v_past = realized_vol(ret_past)
    v_fut = realized_vol(ret_future)

    # Robust ratio
    ratio = float("inf")
    if np.isfinite(v_past) and v_past > 0 and np.isfinite(v_fut):
        ratio = float(v_fut / (v_past + 1e-12))

    # If fut_dd is NaN (weird data), fall back to vol signal only
    dd_ok = np.isfinite(fut_dd)

    # BLOCK if either is severe
    if (dd_ok and fut_dd <= rules.block_dd) or (np.isfinite(ratio) and ratio >= rules.block_vol_ratio):
        return "block"
    if (dd_ok and fut_dd <= rules.warn_dd) or (np.isfinite(ratio) and ratio >= rules.warn_vol_ratio):
        return "warn"
    return "ok"


# -----------------------------
# Feature building
# -----------------------------
def build_features(
    ticker: str,
    asset_type: str,
    market: str,
    closes: pd.Series,
    returns: pd.Series,
) -> Dict[str, Any]:
    closes = closes.dropna()
    returns = returns.dropna()

    # Use last point of window
    ret20 = returns.tail(20).dropna()
    ret252 = returns.tail(252).dropna()

    if len(ret20) < 10 or len(ret252) < 60 or len(closes) < 60:
        return {}

    vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1)) if len(ret20) >= 2 else float("nan")
    vol_ann = realized_vol(ret252)
    mdd = max_drawdown(closes)

    v95, e95 = var_es(ret252, 0.95)
    v99, e99 = var_es(ret252, 0.99)

    # n_used on daily window
    n_used = int(len(ret252))
    missing_pct = float(1.0 - (n_used / 252.0))
    missing_pct = float(max(0.0, min(1.0, missing_pct)))

    # Placeholder corr_mkt: v1 neutral; can be upgraded later using local index proxy
    corr_mkt = 0.0

    feats = {
        "asset_type": asset_type,
        "market": market,
        "vol_ann": float(vol_ann) if np.isfinite(vol_ann) else None,
        "vol_20d": float(vol_20d) if np.isfinite(vol_20d) else None,
        "max_drawdown": float(mdd) if np.isfinite(mdd) else None,
        "corr_mkt": float(corr_mkt),
        # VaR/ES in positive loss magnitude
        "var95": float(v95) if np.isfinite(v95) else None,
        "var99": float(v99) if np.isfinite(v99) else None,
        "es95": float(e95) if np.isfinite(e95) else None,
        "es99": float(e99) if np.isfinite(e99) else None,
        "n_used": n_used,
        "missing_pct": missing_pct,
        "tuw_pct": 95.0,  # schema compatibility
        "tail_obs_99": int(
            max(
                0,
                np.sum(
                    (-ret252.to_numpy(dtype=float))
                    >= (v99 if np.isfinite(v99) else 1e9)
                ),
            )
        ),
        "rsi": float(rsi(closes)) if np.isfinite(rsi(closes)) else None,
        "ticker": ticker,
    }
    return feats


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--windows_per_ticker", type=int, default=120)
    ap.add_argument("--out_dir", default="data/training")
    args = ap.parse_args()

    rules = LabelRules(horizon_days=args.horizon_days)

    uni = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ok = (out_dir / "train_ok.jsonl").open("w", encoding="utf-8")
    out_warn = (out_dir / "train_warn.jsonl").open("w", encoding="utf-8")
    out_block = (out_dir / "train_block.jsonl").open("w", encoding="utf-8")

    counts = {"ok": 0, "warn": 0, "block": 0}
    fails = 0

    for item in uni:
        ticker = item["ticker"]
        asset_type = item["asset_type"]
        market = item["market"]

        try:
            df = yf.download(
                ticker,
                start=args.start,
                end=args.end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty or "Close" not in df:
                print(f"⚠️ no data for {ticker}")
                fails += 1
                continue

            close = df["Close"].dropna()
            ret = close.pct_change()

            min_len = args.lookback_days + args.horizon_days + 30
            if len(close) < min_len:
                print(f"⚠️ too short {ticker} len={len(close)}")
                fails += 1
                continue

            # Choose end indices for windows (spread across history)
            max_end = len(close) - args.horizon_days - 1
            if max_end <= args.lookback_days + 1:
                print(f"⚠️ not enough room for windows {ticker}")
                fails += 1
                continue

            end_ixs = np.linspace(
                args.lookback_days,
                max_end,
                num=min(args.windows_per_ticker, max_end - args.lookback_days),
                dtype=int,
            )

            for end_ix in end_ixs:
                past_slice = slice(end_ix - args.lookback_days, end_ix)
                fut_slice = slice(end_ix, end_ix + args.horizon_days)

                px_past = close.iloc[past_slice]
                ret_past = ret.iloc[past_slice]
                px_fut = close.iloc[fut_slice]
                ret_fut = ret.iloc[fut_slice]

                feats = build_features(ticker, asset_type, market, px_past, ret_past)
                if not feats:
                    continue

                lab = label_from_future(px_past, ret_past, px_fut, ret_fut, rules)

                # ✅ add stable timestamp + id
                asof = str(close.index[end_ix - 1].date())
                rec_id = f"{ticker}|{asof}"

                rec = {
                    "id": rec_id,
                    "asof": asof,
                    "label": lab,
                    "features": feats,
                }

                line = json.dumps(rec, ensure_ascii=False)
                if lab == "ok":
                    out_ok.write(line + "\n")
                elif lab == "warn":
                    out_warn.write(line + "\n")
                else:
                    out_block.write(line + "\n")

                counts[lab] += 1

            print(f"✅ {ticker} done")

        except Exception as e:
            print(f"⚠️ {ticker} failed: {e}")
            fails += 1

    out_ok.close()
    out_warn.close()
    out_block.close()

    print("\n--- DONE ---")
    print("counts:", counts)
    print("fails:", fails)
    print(f"written to: {out_dir}")


if __name__ == "__main__":
    main()