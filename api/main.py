# api/main.py
from __future__ import annotations

import os
import time
import json
import sqlite3
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Reuse your existing feature builder (from your ML LCC project)
from features import DEFAULT_CONFIG, features_to_row, vector_columns


# =============================
# Config
# =============================
UNSUP_BUNDLE_PATH = os.getenv("UNSUP_BUNDLE_PATH", "models/unsup_bundle.joblib")
SUP_BUNDLE_PATH = os.getenv("SUP_BUNDLE_PATH", "models/sup_bundle.joblib")
API_KEY_ENV = os.getenv("API_KEY", "")

XGB_SHADOW_ENABLED = os.getenv("XGB_SHADOW_ENABLED", "1").strip() not in ("0", "false", "False")

# Oracle throttling (yfinance can rate-limit)
ORACLE_MAX_TRIES = int(os.getenv("ORACLE_MAX_TRIES", "4"))
ORACLE_SLEEP_TRY = float(os.getenv("ORACLE_SLEEP_TRY", "0.8"))

# Cache DB path
ORACLE_CACHE_DB = os.getenv("ORACLE_CACHE_DB", "data/oracle_cache.sqlite3")

# WARN gating knobs
WARN_MARGIN = float(os.getenv("ORACLE_WARN_MARGIN", "0.08"))  # in "ensemble units"
XGB_OK_PBLOCK_MAX = float(os.getenv("ORACLE_XGB_OK_PBLOCK_MAX", "0.20"))

# ✅ UNSUP coverage gating knobs
UNSUP_MAX_MISSING_RATIO = float(os.getenv("UNSUP_MAX_MISSING_RATIO", "0.25"))  # e.g., 25%
UNSUP_MAX_MISSING_COUNT = int(os.getenv("UNSUP_MAX_MISSING_COUNT", "0"))  # 0 = ignore count gate


# =============================
# FastAPI
# =============================
app = FastAPI(title="Asymetra LCC API", version="1.4-oracle-stooq-fallback+cache-source")


# =============================
# Pydantic Models
# =============================
class ScoreRequest(BaseModel):
    asset_type: str = Field(..., examples=["equity", "etf", "fx", "commodity", "index"])
    market: str = Field(..., examples=["US", "EU", "ASIA", "OCE", "GLOBAL"])
    ticker: Optional[str] = None

    # engineered stats (Lovable may send some or all)
    vol_ann: Optional[float] = None
    vol_20d: Optional[float] = None
    max_drawdown: Optional[float] = None
    corr_mkt: Optional[float] = None
    var95: Optional[float] = None
    var99: Optional[float] = None
    es95: Optional[float] = None
    es99: Optional[float] = None
    n_used: Optional[float] = None
    missing_pct: Optional[float] = None
    tuw_pct: Optional[float] = None
    tail_obs_99: Optional[float] = None
    rsi: Optional[float] = None

    model_config = {"extra": "allow"}  # accept extra fields


class OracleRequest(BaseModel):
    asset_type: str = Field(..., examples=["equity", "etf", "fx", "commodity", "index"])
    market: str = Field(..., examples=["US", "EU", "ASIA", "OCE", "GLOBAL"])
    ticker: Optional[str] = Field(default=None, description="If provided, Oracle can download data via market data source.")
    closes: Optional[List[float]] = None
    dates: Optional[List[str]] = None
    lookback_days: int = 252


class ScoreOracleRequest(BaseModel):
    lovable: ScoreRequest
    closes: Optional[List[float]] = None
    dates: Optional[List[str]] = None
    force_oracle: bool = False
    lookback_days: int = 252


# =============================
# Bundles (cached)
# =============================
_UNSUP: Optional[Dict[str, Any]] = None
_SUP: Optional[Dict[str, Any]] = None


def _load_unsup() -> Dict[str, Any]:
    global _UNSUP
    if _UNSUP is None:
        p = Path(UNSUP_BUNDLE_PATH)
        if not p.exists():
            raise RuntimeError(f"Missing unsup bundle: {p}")
        _UNSUP = joblib.load(p)
    return _UNSUP


def _load_sup() -> Dict[str, Any]:
    global _SUP
    if _SUP is None:
        p = Path(SUP_BUNDLE_PATH)
        if not p.exists():
            _SUP = {}
        else:
            _SUP = joblib.load(p)
    return _SUP


# =============================
# Security
# =============================
def _require_api_key(x_api_key: Optional[str]) -> None:
    if not API_KEY_ENV:
        return
    if not x_api_key or x_api_key.strip() != API_KEY_ENV:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =============================
# Small utils
# =============================
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _features_dict(req: ScoreRequest) -> Dict[str, Any]:
    d = dict(req.model_dump())
    d["asset_type"] = (d.get("asset_type") or "").strip().lower()
    d["market"] = (d.get("market") or "").strip().upper()
    return d


def _utc_now() -> int:
    return int(time.time())


# =============================
# OracleCache (SQLite)
# =============================
@dataclass
class OracleCacheRow:
    cache_key: str
    asset_type: str
    market: str
    ticker: str
    lookback_days: int
    source: str
    fetched_at_utc: int
    expires_at_utc: int
    row_json: str


class OracleCache:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS oracle_cache (
                    cache_key TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    market TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    lookback_days INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    fetched_at_utc INTEGER NOT NULL,
                    expires_at_utc INTEGER NOT NULL,
                    row_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_oracle_cache_exp ON oracle_cache(expires_at_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_oracle_cache_ticker ON oracle_cache(market, ticker);")

    @staticmethod
    def make_key(asset_type: str, market: str, ticker: str, lookback_days: int, source: str) -> str:
        return f"{asset_type}|{market}|{ticker}|{lookback_days}|{source}".lower()

    def get(self, key: str, now_utc: Optional[int] = None) -> Optional[OracleCacheRow]:
        now_utc = _utc_now() if now_utc is None else int(now_utc)
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT cache_key, asset_type, market, ticker, lookback_days, source, fetched_at_utc, expires_at_utc, row_json "
                "FROM oracle_cache WHERE cache_key=? LIMIT 1;",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            r = OracleCacheRow(*row)
            if r.expires_at_utc <= now_utc:
                conn.execute("DELETE FROM oracle_cache WHERE cache_key=?;", (key,))
                return None
            return r

    def set(self, row: OracleCacheRow) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO oracle_cache
                (cache_key, asset_type, market, ticker, lookback_days, source, fetched_at_utc, expires_at_utc, row_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    row.cache_key,
                    row.asset_type,
                    row.market,
                    row.ticker,
                    int(row.lookback_days),
                    row.source,
                    int(row.fetched_at_utc),
                    int(row.expires_at_utc),
                    row.row_json,
                ),
            )

    def columns(self) -> List[str]:
        with self._conn() as conn:
            cur = conn.execute("PRAGMA table_info(oracle_cache);")
            return [r[1] for r in cur.fetchall()]

    def recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT cache_key, market, ticker, fetched_at_utc, expires_at_utc "
                "FROM oracle_cache ORDER BY fetched_at_utc DESC LIMIT ?;",
                (int(limit),),
            )
            out = []
            for ck, m, t, f, e in cur.fetchall():
                out.append({"cache_key": ck, "market": m, "ticker": t, "fetched_at_utc": f, "expires_at_utc": e})
            return out


_ORACLE_CACHE = OracleCache(ORACLE_CACHE_DB)


def _market_ttl_seconds(market: str) -> int:
    m = (market or "").upper()
    if m in ("US", "EU"):
        return int(os.getenv("ORACLE_TTL_US_EU", "28800"))  # 8h
    if m in ("ASIA", "OCE"):
        return int(os.getenv("ORACLE_TTL_ASIA_OCE", "43200"))  # 12h
    return int(os.getenv("ORACLE_TTL_GLOBAL", "21600"))  # 6h


# =============================
# yfinance helpers (MultiIndex-safe)
# =============================
def _as_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
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


def _download_daily_yf(ticker: str, lookback_days: int, max_tries: int, sleep_try: float) -> pd.Series:
    """
    Yahoo via yfinance. On cloud providers, this can return empty (blocked/throttled).
    We do retries + start/end and period=max attempts.
    """
    last_err: Optional[Exception] = None
    t = (ticker or "").strip()

    period_days = int(max(lookback_days * 3, lookback_days + 120))
    end = pd.Timestamp.utcnow().normalize()
    start = (end - pd.Timedelta(days=period_days)).date().isoformat()
    end_s = end.date().isoformat()

    for k in range(max_tries):
        try:
            df = yf.download(
                t,
                start=start,
                end=end_s,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = _as_close_series(df, t)
            close = pd.Series(close).dropna()
            if len(close) >= lookback_days + 2:
                return close.iloc[-(lookback_days + 2):]
            last_err = RuntimeError(f"insufficient closes via start/end len={len(close)} df_empty={df is None or df.empty}")
        except Exception as e:
            last_err = e
        time.sleep(sleep_try * (1.6 ** k))

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
            if len(close) >= lookback_days + 2:
                return close.iloc[-(lookback_days + 2):]
            last_err = RuntimeError(f"insufficient closes via period=max len={len(close)} df_empty={df is None or df.empty}")
        except Exception as e:
            last_err = e
        time.sleep(sleep_try * (1.6 ** k))

    raise RuntimeError(f"yfinance download failed or insufficient data for {t}: {last_err}")


def _download_daily_stooq(ticker: str, lookback_days: int) -> pd.Series:
    """
    Fallback data source when Yahoo/yfinance returns empty from cloud providers.
    Stooq US tickers are like AAPL.US (case-insensitive).
    """
    t = (ticker or "").strip()
    if not t:
        return pd.Series(dtype=float)

    # If user passed plain ticker for US equities, add .US
    t_stooq = t if "." in t else f"{t}.US"
    url = f"https://stooq.com/q/d/l/?s={t_stooq.lower()}&i=d"

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Close" not in df.columns or "Date" not in df.columns:
        return pd.Series(dtype=float)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    close = pd.Series(df["Close"].to_numpy(dtype=float), index=df["Date"]).dropna()
    if len(close) >= lookback_days + 2:
        return close.iloc[-(lookback_days + 2):]
    return close


# =============================
# Oracle computations (stats)
# =============================
def _max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd = prices / (roll_max + 1e-12) - 1.0
    return float(dd.min())


def _realized_vol_ann(returns: pd.Series) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(252))


def _var_es(returns: pd.Series, q: float) -> Tuple[float, float]:
    losses = (-returns).dropna().to_numpy(dtype=float)
    if len(losses) < 30:
        return (np.nan, np.nan)
    v = float(np.quantile(losses, q))
    tail = losses[losses >= v]
    es = float(tail.mean()) if len(tail) else v
    return v, es


def _rsi(series: pd.Series, period: int = 14) -> float:
    x = series.diff()
    up = x.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-x.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def _oracle_compute_from_closes(
    asset_type: str,
    market: str,
    ticker: Optional[str],
    closes: pd.Series,
    lookback_days: int,
) -> Dict[str, Any]:
    closes = pd.Series(closes).dropna()
    if len(closes) < lookback_days + 2:
        raise ValueError("not enough closes")

    closes = closes.iloc[-(lookback_days + 2):]
    rets = closes.pct_change().dropna()

    ret20 = rets.tail(20)
    ret252 = rets.tail(lookback_days)

    vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1)) if len(ret20) >= 10 else float("nan")
    vol_ann = _realized_vol_ann(ret252)
    mdd = _max_drawdown(closes)

    v95, e95 = _var_es(ret252, 0.95)
    v99, e99 = _var_es(ret252, 0.99)

    n_used = int(len(ret252))
    missing_pct = float(max(0.0, min(1.0, 1.0 - (n_used / float(lookback_days)))))

    tail_obs_99 = int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9))))

    return {
        "asset_type": asset_type,
        "market": market,
        "ticker": ticker,
        "vol_ann": float(vol_ann) if np.isfinite(vol_ann) else None,
        "vol_20d": float(vol_20d) if np.isfinite(vol_20d) else None,
        "max_drawdown": float(mdd),
        "var95": float(v95) if np.isfinite(v95) else None,
        "var99": float(v99) if np.isfinite(v99) else None,
        "es95": float(e95) if np.isfinite(e95) else None,
        "es99": float(e99) if np.isfinite(e99) else None,
        "n_used": n_used,
        "missing_pct": missing_pct,
        "tuw_pct": 95.0,
        "tail_obs_99": tail_obs_99,
        "rsi": float(_rsi(closes)) if len(closes) >= 20 else None,
        "corr_mkt": 0.0,
    }


def _oracle_analyze(req: OracleRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    asset_type = (req.asset_type or "").strip().lower()
    market = (req.market or "").strip().upper()
    ticker = (req.ticker or "").strip()

    if req.closes:
        closes = pd.Series(req.closes, dtype=float)
        feats = _oracle_compute_from_closes(asset_type, market, req.ticker, closes, req.lookback_days)
        meta = {"oracle_source": "provided_closes", "oracle_cache_hit": False}
        return feats, meta

    if not ticker:
        raise ValueError("ticker required when closes not provided")

    # We decide the source BEFORE caching (because cache key must include source)
    now = _utc_now()

    # Try yfinance first
    try:
        close = _download_daily_yf(ticker, req.lookback_days, ORACLE_MAX_TRIES, ORACLE_SLEEP_TRY)
        source = "yfinance"
    except Exception as yf_err:
        # Fallback stooq
        close = _download_daily_stooq(ticker, req.lookback_days)
        source = "stooq"
        if close is None or len(close) < req.lookback_days + 2:
            raise RuntimeError(f"no usable price data (yfinance failed: {yf_err})")

    key = OracleCache.make_key(asset_type, market, ticker, int(req.lookback_days), source)

    hit = _ORACLE_CACHE.get(key, now_utc=now)
    if hit is not None:
        feats = json.loads(hit.row_json)
        meta = {
            "oracle_source": source,
            "oracle_cache_hit": True,
            "oracle_cache_expires_at_utc": hit.expires_at_utc,
            "oracle_cache_ttl_seconds": max(0, hit.expires_at_utc - now),
        }
        return feats, meta

    feats = _oracle_compute_from_closes(asset_type, market, ticker, close, req.lookback_days)

    ttl = _market_ttl_seconds(market)
    expires = now + int(ttl)

    _ORACLE_CACHE.set(
        OracleCacheRow(
            cache_key=key,
            asset_type=asset_type,
            market=market,
            ticker=ticker,
            lookback_days=int(req.lookback_days),
            source=source,
            fetched_at_utc=now,
            expires_at_utc=expires,
            row_json=json.dumps(feats, ensure_ascii=False),
        )
    )

    meta = {
        "oracle_source": source,
        "oracle_cache_hit": False,
        "oracle_cache_ttl_seconds": ttl,
        "oracle_cache_expires_at_utc": expires,
    }
    return feats, meta


# =============================
# Unsupervised scoring (IF + LOF) ✅ robust (no NaN) + coverage
# =============================
def _unsup_vector_numpy(feats: Dict[str, Any], cfg: Dict[str, Any], cols: list[str]) -> np.ndarray:
    row_dict = features_to_row(feats, cfg=cfg)
    row = []

    for c in cols:
        v = row_dict.get(c, None)

        if v is None and c == "max_dd":
            v = row_dict.get("max_drawdown", None)
        if v is None and c == "max_drawdown":
            v = row_dict.get("max_dd", None)

        try:
            fv = float(v)
            if not np.isfinite(fv):
                fv = np.nan
        except Exception:
            fv = np.nan

        row.append(fv)

    X = np.asarray([row], dtype=float)

    if np.isnan(X).any():
        meds = np.nanmedian(X, axis=0)
        meds = np.where(np.isnan(meds), 0.0, meds)
        ii, jj = np.where(np.isnan(X))
        X[ii, jj] = meds[jj]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _unsup_missing_coverage(feats: Dict[str, Any], cfg: Dict[str, Any], cols: list[str]) -> Tuple[int, float, List[str]]:
    row_dict = features_to_row(feats, cfg=cfg)

    missing_cols: List[str] = []
    for c in cols:
        v = row_dict.get(c, None)

        if v is None and c == "max_dd":
            v = row_dict.get("max_drawdown", None)
        if v is None and c == "max_drawdown":
            v = row_dict.get("max_dd", None)

        fv = _safe_float(v)
        if fv is None:
            missing_cols.append(c)

    total = max(1, len(cols))
    missing_count = len(missing_cols)
    missing_ratio = float(missing_count / total)
    sample = missing_cols[:25]
    return missing_count, missing_ratio, sample


def _unsup_score(feats: Dict[str, Any]) -> Dict[str, Any]:
    b = _load_unsup()
    cfg = b.get("config", DEFAULT_CONFIG)
    cols = b.get("columns") or vector_columns(cfg)

    missing_count, missing_ratio, missing_sample = _unsup_missing_coverage(feats, cfg, cols)

    models = b.get("models") or {}
    iforest = models.get("iforest")
    lof = models.get("lof")
    if iforest is None or lof is None:
        raise RuntimeError("unsup bundle missing models.iforest / models.lof")

    score_norm = b.get("score_norm") or {}
    norm_if = score_norm.get("if") or {}
    norm_lof = score_norm.get("lof") or {}

    thr_global = b.get("thresholds_global") or {}
    thr_per_asset = b.get("thresholds_per_asset_type") or {}

    w = b.get("ensemble_weights") or {"if": 0.5, "lof": 0.5}
    w_if = float(w.get("if", 0.5))
    w_lof = float(w.get("lof", 0.5))

    X = _unsup_vector_numpy(feats, cfg, cols)

    raw_if = float(np.asarray(iforest.score_samples(X), dtype=float)[0])
    raw_lof = float(np.asarray(lof.score_samples(X), dtype=float)[0])

    mu_if = float(norm_if.get("mu", 0.0))
    sg_if = float(norm_if.get("sigma", 1.0)) or 1.0
    mu_lof = float(norm_lof.get("mu", 0.0))
    sg_lof = float(norm_lof.get("sigma", 1.0)) or 1.0

    z_if = (raw_if - mu_if) / (sg_if + 1e-12)
    z_lof = (raw_lof - mu_lof) / (sg_lof + 1e-12)

    ensemble_normal = (w_if * z_if) + (w_lof * z_lof)
    anomaly_score = float(-ensemble_normal)

    asset_type = (feats.get("asset_type") or "").strip().lower()
    thr = (thr_per_asset.get(asset_type) or thr_global) or {}
    warn_thr = float(thr.get("warn", thr.get("WARN", 0.0)))
    block_thr = float(thr.get("block", thr.get("BLOCK", 1.0)))

    status = "OK"
    if anomaly_score >= block_thr:
        status = "BLOCK"
    elif anomaly_score >= warn_thr:
        status = "WARN"

    return {
        "raw_if": raw_if,
        "raw_lof": raw_lof,
        "z_if": float(z_if),
        "z_lof": float(z_lof),
        "ensemble": anomaly_score,
        "status": status,
        "thresholds": {"warn": warn_thr, "block": block_thr},
        "asset_thresholds_used": asset_type in (thr_per_asset or {}),
        "missing_count": int(missing_count),
        "missing_ratio": float(missing_ratio),
        "missing_cols_sample": missing_sample,
        "n_cols": int(len(cols)),
    }


# =============================
# XGB shadow (optional)
# =============================
def _xgb_shadow_score(feats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not XGB_SHADOW_ENABLED:
        return None

    sup = _load_sup()
    if not sup or "models" not in sup:
        return None

    model = (sup.get("models") or {}).get("xgb")
    if model is None:
        return None

    prep = sup.get("prep") or {}
    numeric_cols = list(prep.get("numeric_cols") or [])
    feature_cols = list(prep.get("feature_columns") or [])
    medians = dict(prep.get("medians") or {})

    cfg = sup.get("config", DEFAULT_CONFIG)

    row = features_to_row(feats, cfg=cfg)
    base = {c: row.get(c, None) for c in numeric_cols}
    base["_asset_type"] = (feats.get("asset_type") or "").strip().lower()
    base["_market"] = (feats.get("market") or "").strip().upper()

    df = pd.DataFrame([base])

    for c in numeric_cols:
        v = _safe_float(df.at[0, c])
        df.at[0, c] = float(medians.get(c, 0.0)) if v is None else float(v)

    X = pd.get_dummies(df, columns=["_asset_type", "_market"], prefix=["asset", "mkt"], dummy_na=False)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols].to_numpy(dtype=float)

    proba = model.predict_proba(X)[0]
    pred_i = int(np.argmax(proba))

    inv = (sup.get("labels") or {}).get("inv") or {0: "ok", 1: "warn", 2: "block"}
    pred = str(inv.get(pred_i, "ok")).upper()

    return {
        "pred": pred,
        "probs": {"OK": float(proba[0]), "WARN": float(proba[1]), "BLOCK": float(proba[2])},
    }


# =============================
# Integrity mini-checks (simple & safe)
# =============================
def _integrity_flags(feats: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    flags: List[str] = []
    critical: List[str] = []

    v95 = _safe_float(feats.get("var95"))
    v99 = _safe_float(feats.get("var99"))
    e95 = _safe_float(feats.get("es95"))
    e99 = _safe_float(feats.get("es99"))

    if v95 is not None and v99 is not None and v99 < v95:
        flags.append("VAR99_LT_VAR95")
        critical.append("VAR99_LT_VAR95")

    if v95 is not None and e95 is not None and e95 < v95:
        flags.append("ES95_LT_VAR95")
        critical.append("ES95_LT_VAR95")

    if v99 is not None and e99 is not None and e99 < v99:
        flags.append("ES99_LT_VAR99")
        critical.append("ES99_LT_VAR99")

    return flags, critical


# =============================
# Endpoints
# =============================
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "app": "api.main",
        "version": app.version,
        "has_oracle": True,
        "oracle_cache_db": ORACLE_CACHE_DB,
        "oracle_cache_columns": _ORACLE_CACHE.columns(),
        "oracle_cache_recent": _ORACLE_CACHE.recent(limit=5),
        "unsup_coverage": {
            "max_missing_ratio": UNSUP_MAX_MISSING_RATIO,
            "max_missing_count": UNSUP_MAX_MISSING_COUNT,
        },
    }


@app.post("/oracle/analyze")
def oracle_endpoint(req: OracleRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    try:
        feats, meta = _oracle_analyze(req)
        return {"features": feats, "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"compute failed: {e}")


@app.post("/score")
def score(req: ScoreRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")) -> Dict[str, Any]:
    _require_api_key(x_api_key)

    feats = _features_dict(req)
    u = _unsup_score(feats)

    resp: Dict[str, Any] = {
        "ensemble": float(u["ensemble"]),
        "status": u["status"],
        "thresholds": u["thresholds"],
        "debug_unsup": {
            "raw_if": u.get("raw_if"),
            "raw_lof": u.get("raw_lof"),
            "z_if": u.get("z_if"),
            "z_lof": u.get("z_lof"),
            "missing_count": u.get("missing_count"),
            "missing_ratio": u.get("missing_ratio"),
            "missing_cols_sample": u.get("missing_cols_sample"),
            "n_cols": u.get("n_cols"),
        },
    }

    xgb = _xgb_shadow_score(feats)
    if xgb is not None:
        resp["shadow"] = {"xgb": xgb}

    return resp


# NOTE: score_oracle endpoint continues below in your file unchanged.
# (You already pasted it; keep it as-is.)











