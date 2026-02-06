# build_dataset_daily.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import joblib


# -----------------------------
# Helpers
# -----------------------------
def _as_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance peut renvoyer:
      - colonnes simples (Close)
      - MultiIndex (('Close','AAPL'), ...)
    On retourne toujours une Series de close.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # MultiIndex: df["Close"] => DataFrame avec colonne ticker
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            return df[("Close", ticker)].dropna()

        if "Close" in df.columns.get_level_values(0):
            close_df = df["Close"]
            if isinstance(close_df, pd.DataFrame):
                if ticker in close_df.columns:
                    return close_df[ticker].dropna()
                return close_df.iloc[:, 0].dropna()
            return pd.Series(close_df).dropna()

    # colonnes simples
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.Series):
            return s.dropna()
        if isinstance(s, pd.DataFrame):
            if ticker in s.columns:
                return s[ticker].dropna()
            return s.iloc[:, 0].dropna()

    return pd.Series(dtype=float)


def rsi(series: pd.Series, period: int = 14) -> float:
    x = series.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    val = 100 - (100 / (1 + rs.iloc[-1]))
    return float(val)


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
    returns: simple returns
    convention: VaR/ES en "loss magnitude" positive (loss=-ret)
    """
    losses = (-returns).dropna().to_numpy(dtype=float)
    if len(losses) < 30:
        return (np.nan, np.nan)
    v = float(np.quantile(losses, q))
    tail = losses[losses >= v]
    es = float(tail.mean()) if len(tail) else v
    return v, es


# -----------------------------
# Label rules (future-based)
# -----------------------------
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


# -----------------------------
# Unsupervised z-scores
# -----------------------------
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
    Ajoute:
      - raw_if, raw_lof
      - z_if, z_lof
      - z_gap_if_lof

    PATCH IMPORTANT:
      - On passe un numpy array aux modèles (pas DataFrame) => pas de warning sklearn
      - Imputation robuste colonne par colonne:
          * si une colonne est 100% NaN -> 0.0 (sans warning)
          * sinon -> médiane
    """
    cols: List[str] = unsup["columns"]
    iforest = unsup["models"]["iforest"]
    lof = unsup["models"]["lof"]
    score_norm = unsup["score_norm"]

    # row dans l'ordre exact des colonnes du bundle
    row: List[float] = []
    for c in cols:
        v = feats.get(c)

        # alias de compat
        if v is None and c == "max_dd":
            v = feats.get("max_drawdown")
        if v is None and c == "max_drawdown":
            v = feats.get("max_dd")

        row.append(np.nan if v is None else float(v))

    X = np.asarray([row], dtype=float)

    # ---- PATCH (robuste): imputation colonne par colonne ----
    if np.isnan(X).any():
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.all(np.isnan(col)):
                X[:, j] = 0.0
            else:
                med = np.nanmedian(col)
                X[np.isnan(col), j] = med
    # --------------------------------------------------------

    # score_samples renvoie (n,)
    raw_if = float(np.asarray(iforest.score_samples(X))[0])
    raw_lof = float(np.asarray(lof.score_samples(X))[0])

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


# -----------------------------
# Feature building (daily)
# -----------------------------
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

    # v1: neutre (tu peux faire corr vs index proxy ensuite)
    corr_mkt = 0.0

    # compat pipeline: max_dd + max_drawdown
    max_dd = float(mdd)

    # features dérivées
    dd_to_var99 = None
    if np.isfinite(v99):
        dd_to_var99 = float(abs(max_dd) / (float(v99) + 1e-12))

    tuw_per_dd = float(95.0 / (abs(max_dd) + 1e-6))

    feats: Dict[str, Any] = {
        "asset_type": asset_type,
        "market": market,
        "ticker": ticker,

        "vol_ann": float(vol_ann),
        "vol_20d": float(vol_20d),

        "max_drawdown": float(max_dd),
        "max_dd": float(max_dd),

        "corr_mkt": float(corr_mkt),

        "var95": float(v95) if np.isfinite(v95) else None,
        "var99": float(v99) if np.isfinite(v99) else None,
        "es95": float(e95) if np.isfinite(e95) else None,
        "es99": float(e99) if np.isfinite(e99) else None,

        "n_used": int(n_used),
        "missing_pct": float(missing_pct),

        "tuw_pct": 95.0,
        "tail_obs_99": int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9)))),
        "rsi": float(rsi(closes)),

        "dd_to_var99": dd_to_var99,
        "tuw_per_dd": tuw_per_dd,
    }
    return feats


# -----------------------------
# Download with retries
# -----------------------------
def download_daily(
    ticker: str,
    start: str,
    end: Optional[str],
    max_tries: int,
    sleep_try: float,
) -> pd.DataFrame:
    last_err = None
    for k in range(max_tries):
        try:
            return yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:
            last_err = e
            time.sleep(sleep_try * (1.4 ** k))
    raise RuntimeError(f"download failed for {ticker}: {last_err}")


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

    # add z features
    ap.add_argument("--unsup_bundle", default=None, help="Optional unsup_bundle.joblib to add z_if/z_lof/z_gap_if_lof")

    # throttling / retry
    ap.add_argument("--sleep_ticker", type=float, default=0.0, help="sleep between tickers (seconds)")
    ap.add_argument("--max_tries", type=int, default=3, help="download retry count")
    ap.add_argument("--sleep_try", type=float, default=0.8, help="base sleep between retries (seconds)")

    args = ap.parse_args()

    rules = LabelRules(horizon_days=args.horizon_days)

    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe file not found: {uni_path}")

    uni = json.loads(uni_path.read_text(encoding="utf-8"))

    unsup = None
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

    for item in uni:
        ticker = item["ticker"]
        asset_type = item["asset_type"]
        market = item["market"]

        try:
            df = download_daily(
                ticker,
                start=args.start,
                end=args.end,
                max_tries=args.max_tries,
                sleep_try=args.sleep_try,
            )

            if df is None or df.empty:
                print(f"⚠️ no data for {ticker}")
                fails += 1
                continue

            close = _as_close_series(df, ticker)
            if close.empty:
                print(f"⚠️ no close series for {ticker}")
                fails += 1
                continue

            ret = close.pct_change()

            min_len = args.lookback_days + args.horizon_days + 30
            if len(close) < min_len:
                print(f"⚠️ too short {ticker} len={len(close)} (need {min_len})")
                fails += 1
                continue

            min_end = args.lookback_days
            max_end = len(close) - args.horizon_days - 1
            available = max_end - min_end
            if available <= 10:
                print(f"⚠️ not enough window room for {ticker} (len={len(close)})")
                fails += 1
                continue

            end_ixs = np.linspace(
                min_end,
                max_end,
                num=min(args.windows_per_ticker, available),
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

            print(f"✅ {ticker} done")

            if args.sleep_ticker > 0:
                time.sleep(args.sleep_ticker)

        except Exception as e:
            print(f"⚠️ {ticker} failed: {e}")
            fails += 1
            if args.sleep_ticker > 0:
                time.sleep(args.sleep_ticker)

    out_ok.close()
    out_warn.close()
    out_block.close()

    print("\n--- DONE ---")
    print("counts:", counts)
    print("fails:", fails)
    print(f"written to: {out_dir}")


if __name__ == "__main__":
    main()