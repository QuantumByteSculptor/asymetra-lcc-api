# =========================
# api/main.py  (PARTIE 1/2)
# =========================
from __future__ import annotations

import logging
import os
import time
import json
import sqlite3
import io
import sys
import traceback
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------------
# Ensure repo root is on sys.path so we import local features.py
# (avoid collision with pip package named "features")
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse your existing feature builder (from your ML LCC project)
from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore

# Expert scoring layer (lazy-loaded per-asset bundles, graceful if dir missing)
# Controlled by EXPERTS_ENABLED env var (default "0" = OFF)
try:
    from api.scoring import (  # type: ignore
        score_expert, list_loaded_experts,
        preload_experts as _preload_experts,
        is_experts_enabled, experts_health,
    )
    _EXPERTS_AVAILABLE = True
except Exception as _scoring_import_err:
    logger.warning("api/scoring.py not importable — expert scoring disabled: %s", _scoring_import_err)
    _EXPERTS_AVAILABLE = False
    def score_expert(*a, **kw): return None  # type: ignore
    def list_loaded_experts(): return []  # type: ignore
    def _preload_experts(): pass  # type: ignore
    def is_experts_enabled(): return False  # type: ignore
    def experts_health(): return {"enabled": False, "error": "scoring module not importable"}  # type: ignore

from feature_utils import (  # type: ignore
    compute_dd_duration_recovery,
    compute_downside_dev,
    compute_semivariance,
    compute_vol_of_vol,
    compute_worst_rolling_return,
    compute_autocorr,
)


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

# Oracle provider selection: yfinance | stooq | auto (default auto behaves like before)
ORACLE_PROVIDER = os.getenv("ORACLE_PROVIDER", "auto").strip().lower()
if ORACLE_PROVIDER not in ("yfinance", "stooq", "auto"):
    ORACLE_PROVIDER = "auto"

# WARN gating knobs
WARN_MARGIN = float(os.getenv("ORACLE_WARN_MARGIN", "0.08"))  # in "ensemble units"
XGB_OK_PBLOCK_MAX = float(os.getenv("ORACLE_XGB_OK_PBLOCK_MAX", "0.20"))

# ✅ UNSUP coverage gating knobs
UNSUP_MAX_MISSING_RATIO = float(os.getenv("UNSUP_MAX_MISSING_RATIO", "0.25"))  # e.g., 25%
UNSUP_MAX_MISSING_COUNT = int(os.getenv("UNSUP_MAX_MISSING_COUNT", "0"))  # 0 = ignore count gate

# ✅ BIN calibrated decision knobs (api.decision)
BIN_ENABLED = os.getenv("BIN_ENABLED", "1").strip() not in ("0", "false", "False")
BIN_BUNDLE_PATH = os.getenv("BIN_BUNDLE_PATH", "models/bin_sigmoid.joblib")
BIN_THRESHOLDS_PATH = os.getenv("BIN_THRESHOLDS_PATH", "models/threshold_sigmoid.json")
BIN_T_HI_DEFAULT = float(os.getenv("BIN_T_HI_DEFAULT", "0.85"))

# ✅ 3m stock-picking model
MODEL_3M_PATH = os.getenv("MODEL_3M_PATH", str(REPO_ROOT / "models/bin_sigmoid_return_simfin_3m.joblib"))
MODEL_3M_ENABLED = os.getenv("MODEL_3M_ENABLED", "1").strip() not in ("0", "false", "False")

# Validated backtest metrics for /metrics endpoint
_MODEL_3M_BACKTEST = {
    "model_version":          "3m_v1",
    "backtest_cagr":          0.327,
    "backtest_sharpe":        1.11,
    "backtest_sortino":       1.99,
    "backtest_max_drawdown":  -0.293,
    "alpha_vs_spy":           0.178,
    "test_period":            "2019-2023",
    "oos_2024_sharpe":        2.33,
    "feature_count":          30,
    "universe_size":          301,
}

# V15 — regime-aware walk-forward model (2017-2024 OOF backtest)
_MODEL_V15_BACKTEST = {
    "model_version":          "v15",
    "backtest_sharpe":        1.167,
    "backtest_sortino":       2.933,
    "backtest_cagr":          0.198,
    "backtest_max_drawdown":  -0.147,
    "calmar":                 1.345,
    "test_period":            "2020-2024 (walk-forward OOF)",
    "walk_forward_folds":     8,
    "mean_roc_auc_cv":        0.513,
    "feature_count":          18,
    "universe_size":          50,
    "regime_innovation":      "Bear→0% / Sideways→50% / Bull→100% exposure",
    "v14b_sharpe_ref":        1.61,
    "v14b_max_drawdown_ref":  -0.138,
}

# V16 — regime + fundamentals + momentum walk-forward (2020-2024 OOF backtest)
_MODEL_V16_BACKTEST = {
    "model_version":              "v16",
    "backtest_sharpe":            1.214,   # regime-filter variant (best)
    "backtest_sortino":           5.455,
    "backtest_cagr":              0.222,
    "backtest_max_drawdown":     -0.145,
    "calmar":                     1.529,
    "test_period":                "2020-2024 (walk-forward OOF)",
    "walk_forward_folds":         8,
    "mean_roc_auc_cv":            0.520,
    "feature_count":              33,
    "n_tech_features":            14,
    "n_fundamental_features":     12,
    "n_regime_features":          3,
    "universe_size":              50,
    "innovations_vs_v15":         "fundamentals from yfinance.info (cross-sectional)",
    "momentum_filter":            "mom_12_1>0 AND ret_12m>spy_12m (applied at portfolio level)",
    "delta_sharpe_vs_v15":        0.047,
    "delta_sharpe_vs_v14b":      -0.396,
    "fund_data_note":             "static cross-sectional snapshot; prod upgrade: SimFin point-in-time",
}

# Optional debug toggles (control verbosity / extra debug fields)
DEBUG_RESPONSE = os.getenv("DEBUG_RESPONSE", "0").strip() in ("1", "true", "True")


# =============================
# FastAPI
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload expert bundles and 3m model."""
    if _EXPERTS_AVAILABLE:
        try:
            _preload_experts()
            logger.info("startup: expert bundles preloaded — %s", list_loaded_experts())
        except Exception as exc:
            logger.warning("startup: preload_experts failed — %s", exc)
    try:
        m = _load_3m_model()
        if m:
            logger.info("startup: 3m model loaded — %d features", len(m.get("cols", [])))
        else:
            logger.warning("startup: 3m model not loaded (missing or disabled)")
    except Exception as exc:
        logger.warning("startup: 3m model load failed — %s", exc)
    yield


app = FastAPI(title="Asymetra LCC API", version="2.0.0", lifespan=lifespan)


# =============================
# In-memory metrics (thread-safe)
# =============================
_METRICS_LOCK = threading.Lock()
_METRICS: Dict[str, Any] = {
    "calls_score": 0,
    "calls_score_oracle": 0,
    "statuses": {"OK": 0, "WARN": 0, "BLOCK": 0},
    "scores": [],          # kept in memory (capped at 10000)
    "expert_non_null": 0,
}
_METRICS_SCORES_CAP = 10_000


def _record_metric(endpoint: str, status: str, score: Optional[float], has_expert: bool) -> None:
    with _METRICS_LOCK:
        if endpoint == "score":
            _METRICS["calls_score"] += 1
        else:
            _METRICS["calls_score_oracle"] += 1
        s = (status or "OK").upper()
        if s in _METRICS["statuses"]:
            _METRICS["statuses"][s] += 1
        if score is not None and len(_METRICS["scores"]) < _METRICS_SCORES_CAP:
            _METRICS["scores"].append(float(score))
        if has_expert:
            _METRICS["expert_non_null"] += 1


# ✅ UN SEUL exception handler global (avec error_id)
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    import uuid

    error_id = str(uuid.uuid4())[:8]
    logger.exception(f"[UNHANDLED:{error_id}] {request.method} {request.url} -> {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error_id": error_id})


# =============================
# Pydantic Models
# =============================
_VALID_ASSET_TYPES = {"equity", "etf", "fx", "crypto", "commodity", "global", "index"}
import re as _re
_TICKER_RE = _re.compile(r"^[A-Z0-9.\-\^]{1,20}$")


class ScoreRequest(BaseModel):
    asset_type: str = Field(..., examples=["equity", "etf", "fx", "commodity", "index"], max_length=32)
    market: str = Field(..., examples=["US", "EU", "ASIA", "OCE", "GLOBAL"], max_length=32)
    ticker: Optional[str] = Field(default=None, max_length=20)

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

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in _VALID_ASSET_TYPES:
            raise ValueError(
                f"asset_type '{v}' invalide. Valeurs acceptées: {sorted(_VALID_ASSET_TYPES)}"
            )
        return v

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        upper = v.strip().upper()
        if not _TICKER_RE.match(upper):
            raise ValueError(
                f"ticker '{v}' invalide. Format attendu: ^[A-Z0-9.\\-\\^]{{1,20}}$"
            )
        return v


class OracleRequest(BaseModel):
    asset_type: str = Field(..., examples=["equity", "etf", "fx", "commodity", "index"], max_length=32)
    market: str = Field(..., examples=["US", "EU", "ASIA", "OCE", "GLOBAL"], max_length=32)
    ticker: Optional[str] = Field(default=None, description="If provided, Oracle can download data via yfinance/stooq.", max_length=20)
    closes: Optional[List[float]] = Field(default=None, max_length=2000)
    dates: Optional[List[str]] = Field(default=None, max_length=2000)
    lookback_days: int = Field(default=252, ge=10, le=756)

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in _VALID_ASSET_TYPES:
            raise ValueError(
                f"asset_type '{v}' invalide. Valeurs acceptées: {sorted(_VALID_ASSET_TYPES)}"
            )
        return v

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        upper = v.strip().upper()
        if not _TICKER_RE.match(upper):
            raise ValueError(
                f"ticker '{v}' invalide. Format attendu: ^[A-Z0-9.\\-\\^]{{1,20}}$"
            )
        return v


class ScoreOracleRequest(BaseModel):
    lovable: ScoreRequest
    closes: Optional[List[float]] = Field(default=None, max_length=2000)
    dates: Optional[List[str]] = Field(default=None, max_length=2000)
    force_oracle: bool = False
    lookback_days: int = Field(default=252, ge=10, le=756)


# =============================
# Bundles (cached)
# =============================
_UNSUP: Optional[Dict[str, Any]] = None
_SUP: Optional[Dict[str, Any]] = None
_MODEL_3M: Optional[Dict[str, Any]] = None


def _load_unsup() -> Dict[str, Any]:
    """
    Loads unsupervised bundle.
    NEVER crashes the API.
    If missing or corrupted, unsup is disabled.
    """
    global _UNSUP

    if _UNSUP is not None:
        return _UNSUP

    p = Path(UNSUP_BUNDLE_PATH)

    if not p.exists():
        logger.error(f"[UNSUP] Missing unsup bundle: {p} -> unsup disabled")
        _UNSUP = {}
        return _UNSUP

    try:
        _UNSUP = joblib.load(p)
    except Exception as e:
        logger.exception(f"[UNSUP] Failed to load bundle: {type(e).__name__}: {e} -> unsup disabled")
        _UNSUP = {}

    return _UNSUP


def _load_sup() -> Dict[str, Any]:
    """
    IMPORTANT: must NEVER crash the API.
    The sup bundle may fail to unpickle if xgboost isn't installed on the runtime image.
    In that case we disable XGB shadow by returning {}.
    """
    global _SUP
    if _SUP is None:
        p = Path(SUP_BUNDLE_PATH)
        if not p.exists():
            _SUP = {}
            return _SUP

        try:
            _SUP = joblib.load(p)
        except Exception as e:
            logger.error("[SUP] failed to load %s: %s: %s -> disabling XGB shadow", p, type(e).__name__, e)
            _SUP = {}
    return _SUP


_MODEL_3M_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1",
    "ret_12m_vs_spy", "vol_ann", "vol_ratio", "dd_from_hi52",
    "above_200ma", "trend_strength",
    "gross_margin", "op_margin", "net_margin", "roe",
    "debt_to_equity", "rd_intensity", "fcf_margin",
    "revenue_growth", "ni_growth",
    "pe_ratio", "pb_ratio", "earnings_yield", "ev_to_revenue",
    "accruals_ratio", "asset_growth", "current_ratio",
    "ret_1m_lag", "skew_6m", "sector_id",
]


def _load_3m_model() -> Dict[str, Any]:
    """Lazy-load the 3m stock-picking model. Never crashes the API."""
    global _MODEL_3M
    if _MODEL_3M is not None:
        return _MODEL_3M
    if not MODEL_3M_ENABLED:
        _MODEL_3M = {}
        return _MODEL_3M
    p = Path(MODEL_3M_PATH)
    if not p.exists():
        logger.warning("[3M] Model file not found: %s — /score_3m will return error", p)
        _MODEL_3M = {}
        return _MODEL_3M
    try:
        _MODEL_3M = joblib.load(p)
        logger.info("[3M] Loaded 3m model: %d features, version=%s",
                    len(_MODEL_3M.get("cols", [])), _MODEL_3M.get("feature_version", "?"))
    except Exception as e:
        logger.exception("[3M] Failed to load model: %s -> disabled", e)
        _MODEL_3M = {}
    return _MODEL_3M


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


def _err500(where: str, exc: Exception):
    import uuid

    error_id = str(uuid.uuid4())[:8]
    logger.exception(f"[ERR500:{error_id}] {where} -> {type(exc).__name__}: {exc}")
    raise HTTPException(status_code=500, detail={"message": "Internal Server Error", "error_id": error_id})


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
                "SELECT cache_key, market, ticker, fetched_at_utc, expires_at_utc, source "
                "FROM oracle_cache ORDER BY fetched_at_utc DESC LIMIT ?;",
                (int(limit),),
            )
            out = []
            for ck, m, t, f, e, s in cur.fetchall():
                out.append(
                    {"cache_key": ck, "market": m, "ticker": t, "fetched_at_utc": f, "expires_at_utc": e, "source": s}
                )
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
# yfinance helpers (MultiIndex-safe) + Stooq fallback
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


def _download_daily_stooq(
    ticker: str,
    lookback_days: int,
    market: str,
    max_tries: int = 3,
    base_sleep: float = 2.0,
) -> pd.Series:
    """
    Robust Stooq downloader with exponential backoff on rate-limit errors.
    Fixes cloud/provider blocks where Stooq returns HTML/anti-bot instead of CSV.
    """
    t = (ticker or "").strip()
    if not t:
        return pd.Series(dtype=float)

    m = (market or "").upper()

    # Auto-add .US only for US market when user provides plain ticker
    if "." not in t and m == "US":
        t_stooq = f"{t}.US"
    else:
        t_stooq = t

    url = f"https://stooq.com/q/d/l/?s={t_stooq.lower()}&i=d"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_tries):
        try:
            r = requests.get(url, headers=headers, timeout=25)

            if r.status_code in (429, 503):
                raise RuntimeError(f"stooq rate limit (HTTP {r.status_code})")

            r.raise_for_status()

            txt = (r.text or "").strip()
            if not txt:
                raise RuntimeError("stooq returned empty body")

            first_line = txt.splitlines()[0].strip()
            if not first_line.lower().startswith("date,open,high,low,close"):
                sample = txt[:300].replace("\n", "\\n")
                raise RuntimeError(f"stooq non-csv response (first_line='{first_line[:80]}') sample='{sample}'")

            df = pd.read_csv(io.StringIO(txt))
            if df is None or df.empty or "Close" not in df.columns or "Date" not in df.columns:
                raise RuntimeError(f"stooq parsed but missing columns: cols={list(df.columns) if df is not None else None}")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

            close = pd.Series(df["Close"].to_numpy(dtype=float), index=df["Date"]).dropna()
            if len(close) >= lookback_days + 2:
                return close.iloc[-(lookback_days + 2):]
            return close

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(k in msg for k in ("rate limit", "429", "503", "timeout", "connection", "reset"))
            if not transient or attempt == max_tries - 1:
                break
            logger.warning("stooq rate limit for %s attempt %d/%d — retrying in %.1fs",
                           t, attempt + 1, max_tries, base_sleep * (2 ** attempt))
            time.sleep(base_sleep * (2 ** attempt))

    raise last_err or RuntimeError(f"stooq failed for {t}")


def _download_daily_yf(ticker: str, lookback_days: int, max_tries: int, sleep_try: float) -> pd.Series:
    last_err: Optional[Exception] = None
    t = (ticker or "").strip()

    period_days = int(max(lookback_days * 3, lookback_days + 120))
    end = pd.Timestamp.utcnow().normalize()
    start_s = (end - pd.Timedelta(days=period_days)).date().isoformat()
    end_s = end.date().isoformat()

    for k in range(max_tries):
        try:
            df = yf.download(
                t,
                start=start_s,
                end=end_s,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = _as_close_series(df, t)
            close = pd.Series(close).dropna()

            if len(close) >= lookback_days + 2:
                return close.iloc[-(lookback_days + 2) :]

            last_err = RuntimeError(f"insufficient closes via start/end len={len(close)} df_empty={df is None or df.empty}")
        except Exception as e:
            last_err = e

        time.sleep(sleep_try * (1.6**k))

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
                return close.iloc[-(lookback_days + 2) :]

            last_err = RuntimeError(f"insufficient closes via period=max len={len(close)} df_empty={df is None or df.empty}")
        except Exception as e:
            last_err = e

        time.sleep(sleep_try * (1.6**k))

    raise RuntimeError(f"yfinance download failed or insufficient data for {t}: {last_err}")


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


def _skew_kurtosis(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return (float("nan"), float("nan"))

    m = x.mean()
    c = x - m
    s2 = float(np.mean(c * c))
    if s2 <= 1e-18:
        return (0.0, -3.0)

    s = np.sqrt(s2)
    m3 = float(np.mean(c**3))
    m4 = float(np.mean(c**4))

    skew = float(m3 / (s**3 + 1e-12))
    kurt_excess = float(m4 / (s2**2 + 1e-12) - 3.0)
    return skew, kurt_excess


def _ewma_vol_ann(returns: np.ndarray, lam: float = 0.94, ann: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return float("nan")

    var = float(np.var(r, ddof=1))
    for x in r:
        var = lam * var + (1.0 - lam) * (x * x)

    vol = np.sqrt(max(var, 0.0))
    return float(vol * np.sqrt(ann))


def _garch_vol_ann(returns: np.ndarray, ann: int = 252) -> Optional[float]:
    try:
        from arch import arch_model  # type: ignore
    except Exception:
        return None

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 200:
        return None

    rp = 100.0 * r
    try:
        am = arch_model(rp, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
        res = am.fit(disp="off")
        cond_vol_daily_pct = float(res.conditional_volatility.iloc[-1])
        cond_vol_daily = cond_vol_daily_pct / 100.0
        return float(cond_vol_daily * np.sqrt(ann))
    except Exception:
        return None


def _stress_var(returns: np.ndarray, base_var99: Optional[float], window: int = 20, q: float = 0.99) -> Dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < max(60, window + 10):
        return {"stress_var99": None, "stress_multiplier": None}

    worst_i = None
    worst_cum = 1e9
    for i in range(0, n - window + 1):
        w = r[i : i + window]
        cum = float(np.prod(1.0 + w) - 1.0)
        if cum < worst_cum:
            worst_cum = cum
            worst_i = i

    if worst_i is None:
        return {"stress_var99": None, "stress_multiplier": None}

    w = r[worst_i : worst_i + window]
    losses = -w
    v = float(np.quantile(losses, q))
    stress_var = v if np.isfinite(v) else float("nan")

    mult = None
    if base_var99 is not None and np.isfinite(base_var99) and base_var99 > 1e-12 and np.isfinite(stress_var):
        mult = float(stress_var / base_var99)

    return {
        "stress_window_start_idx": int(worst_i),
        "stress_window_end_idx": int(worst_i + window - 1),
        "stress_cumret": float(worst_cum),
        "stress_var99": float(stress_var) if np.isfinite(stress_var) else None,
        "stress_multiplier": float(mult) if mult is not None and np.isfinite(mult) else None,
        "stress_window_days": int(window),
    }


# =========================
# api/main.py  (PARTIE 2/2)
# =========================

# ---------------------------------------------------------------------------
# Market context: corr_spy, beta_market, vix_level (live via yfinance, cached 1h)
# ---------------------------------------------------------------------------
_MARKET_CONTEXT_CACHE = {}
_MARKET_CONTEXT_CACHE_TS = 0.0
_MARKET_CONTEXT_CACHE_TTL = 3600.0


def _compute_market_context(closes, lookback=60):
    """Return corr_spy, beta_market, vix_level from live SPY/VIX data (cached 1h)."""
    import time
    global _MARKET_CONTEXT_CACHE, _MARKET_CONTEXT_CACHE_TS
    out = {'corr_spy': None, 'beta_market': None, 'vix_level': None}
    try:
        now = time.time()
        if now - _MARKET_CONTEXT_CACHE_TS > _MARKET_CONTEXT_CACHE_TTL or not _MARKET_CONTEXT_CACHE:
            spy_raw = yf.download('SPY', period='6mo', auto_adjust=True, progress=False)['Close'].squeeze()
            vix_raw = yf.download('^VIX', period='6mo', auto_adjust=True, progress=False)['Close'].squeeze()
            _MARKET_CONTEXT_CACHE.update({'spy': spy_raw, 'vix': vix_raw})
            _MARKET_CONTEXT_CACHE_TS = now
            logger.info('market_context: refreshed SPY/VIX cache (%d bars)', len(spy_raw))
        spy = _MARKET_CONTEXT_CACHE.get('spy', pd.Series(dtype=float))
        vix = _MARKET_CONTEXT_CACHE.get('vix', pd.Series(dtype=float))
        if len(vix) > 0:
            out['vix_level'] = float(vix.iloc[-1])
        if len(spy) < lookback or len(closes) < lookback:
            return out
        spy_ret = spy.pct_change().dropna()
        asset_ret = pd.Series(closes).pct_change().dropna()
        aligned = pd.DataFrame({'spy': spy_ret, 'asset': asset_ret}).dropna().tail(lookback)
        if len(aligned) < 20:
            return out
        corr = float(aligned['asset'].corr(aligned['spy']))
        cov = float(aligned['asset'].cov(aligned['spy']))
        var_spy = float(aligned['spy'].var())
        beta = cov / var_spy if var_spy > 1e-12 else None
        out['corr_spy'] = corr if np.isfinite(corr) else None
        out['beta_market'] = float(beta) if beta is not None and np.isfinite(beta) else None
    except Exception as exc:
        logger.warning('market_context: failed (%s: %s)', type(exc).__name__, exc)
    return out


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

    closes = closes.iloc[-(lookback_days + 2) :]
    rets = closes.pct_change().dropna()

    ret20 = rets.tail(20)
    ret60 = rets.tail(60)
    ret120 = rets.tail(120)
    ret252 = rets.tail(lookback_days)

    vol_20d = float(np.std(ret20.to_numpy(dtype=float), ddof=1)) if len(ret20) >= 10 else float("nan")
    vol_60d = float(np.std(ret60.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret60) >= 20 else float("nan")
    vol_120d = float(np.std(ret120.to_numpy(dtype=float), ddof=1) * np.sqrt(252)) if len(ret120) >= 40 else float("nan")
    vol_ann = _realized_vol_ann(ret252)
    mdd = _max_drawdown(closes)

    v95, e95 = _var_es(ret252, 0.95)
    v99, e99 = _var_es(ret252, 0.99)

    n_used = int(len(ret252))
    missing_pct = float(max(0.0, min(1.0, 1.0 - (n_used / float(lookback_days)))))

    tail_obs_99 = int(max(0, np.sum((-ret252).to_numpy(dtype=float) >= (v99 if np.isfinite(v99) else 1e9))))

    r = ret252.to_numpy(dtype=float)
    px = closes.to_numpy(dtype=float)
    skew, kurt_excess = _skew_kurtosis(r)
    vol_ewma_ann = _ewma_vol_ann(r, lam=0.94, ann=252)
    vol_garch_ann = _garch_vol_ann(r, ann=252)
    stress = _stress_var(r, base_var99=(v99 if np.isfinite(v99) else None), window=20, q=0.99)

    # v2 features
    dd_duration, recovery_days = compute_dd_duration_recovery(px)
    downside_dev = compute_downside_dev(r)
    semivariance = compute_semivariance(r)
    vol_of_vol = compute_vol_of_vol(r)
    worst_5d_ret = compute_worst_rolling_return(r, 5)
    worst_20d_ret = compute_worst_rolling_return(r, 20)
    autocorr_1 = compute_autocorr(r)

    def _f(x: float) -> Optional[float]:
        return float(x) if np.isfinite(x) else None

    feats: Dict[str, Any] = {
        "asset_type": asset_type,
        "market": market,
        "ticker": ticker,
        "vol_ann": _f(vol_ann),
        "vol_20d": _f(vol_20d),
        "vol_60d": _f(vol_60d),
        "vol_120d": _f(vol_120d),
        "max_drawdown": float(mdd),
        "max_dd": float(mdd),
        "var95": _f(v95),
        "var99": _f(v99),
        "es95": _f(e95),
        "es99": _f(e99),
        "n_used": n_used,
        "missing_pct": missing_pct,
        "tuw_pct": 95.0,
        "tail_obs_99": tail_obs_99,
        "rsi": (lambda v: float(v) if np.isfinite(v) else None)(_rsi(closes)) if len(closes) >= 20 else None,
        "corr_mkt": 0.0,
        "skew": _f(skew),
        "kurtosis_excess": _f(kurt_excess),
        "vol_ewma_ann": _f(vol_ewma_ann),
        "vol_garch_ann": float(vol_garch_ann) if vol_garch_ann is not None and np.isfinite(vol_garch_ann) else None,
        "stress_var99": stress.get("stress_var99"),
        "stress_multiplier": stress.get("stress_multiplier"),
        "stress_window_days": stress.get("stress_window_days"),
        "stress_cumret": stress.get("stress_cumret"),
        # v2
        "dd_duration": dd_duration if dd_duration > 0 else None,
        "recovery_days": recovery_days if recovery_days > 0 else None,
        "downside_dev": _f(downside_dev),
        "semivariance": _f(semivariance),
        "vol_of_vol": _f(vol_of_vol),
        "worst_5d_ret": _f(worst_5d_ret),
        "worst_20d_ret": _f(worst_20d_ret),
        "autocorr_1": _f(autocorr_1),
    }
    # Inject live market context (corr_spy, beta_market, vix_level)
    try:
        mkt = _compute_market_context(closes)
        feats.update({k: v for k, v in mkt.items() if v is not None})
    except Exception as _mkt_exc:
        logger.debug("market_context inject skipped: %s", _mkt_exc)
    return feats


def _oracle_analyze(req: OracleRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    asset_type = (req.asset_type or "").strip().lower()
    market = (req.market or "").strip().upper()
    ticker = (req.ticker or "").strip() if req.ticker else ""

    # 1) Rescue mode (frontend closes)
    if req.closes:
        closes = pd.Series(req.closes, dtype=float)

        feats = _oracle_compute_from_closes(
            asset_type=asset_type,
            market=market,
            ticker=ticker or None,
            closes=closes,
            lookback_days=req.lookback_days,
        )

        meta: Dict[str, Any] = {
            "oracle_source": "provided_closes",
            "oracle_cache_hit": False,
            "oracle_cache_ttl_seconds": 0,
            "oracle_cache_expires_at_utc": None,
            "oracle_note": "rescued from frontend closes; no external provider used",
        }
        return feats, meta

    if not ticker:
        raise ValueError("ticker required when closes are not provided")

    now = _utc_now()
    ttl = _market_ttl_seconds(market)

    def _try_provider(provider: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        provider: 'yfinance' or 'stooq'
        Applies cache per provider.
        """
        key = OracleCache.make_key(
            asset_type=asset_type,
            market=market,
            ticker=ticker,
            lookback_days=int(req.lookback_days),
            source=provider,
        )

        hit = _ORACLE_CACHE.get(key, now_utc=now)
        if hit is not None:
            feats_hit = json.loads(hit.row_json)
            meta_hit = {
                "oracle_source": provider,
                "oracle_cache_hit": True,
                "oracle_cache_expires_at_utc": hit.expires_at_utc,
                "oracle_cache_ttl_seconds": max(0, hit.expires_at_utc - now),
            }
            return feats_hit, meta_hit

        if provider == "yfinance":
            close = _download_daily_yf(
                ticker=ticker,
                lookback_days=req.lookback_days,
                max_tries=ORACLE_MAX_TRIES,
                sleep_try=ORACLE_SLEEP_TRY,
            )
        elif provider == "stooq":
            close = _download_daily_stooq(
                ticker=ticker,
                lookback_days=req.lookback_days,
                market=market,
            )
            if len(close) < req.lookback_days + 2:
                raise RuntimeError(f"stooq insufficient closes len={len(close)}")
        else:
            raise ValueError(f"unknown provider: {provider}")

        feats_new = _oracle_compute_from_closes(
            asset_type=asset_type,
            market=market,
            ticker=ticker,
            closes=close,
            lookback_days=req.lookback_days,
        )

        expires = now + int(ttl)
        _ORACLE_CACHE.set(
            OracleCacheRow(
                cache_key=key,
                asset_type=asset_type,
                market=market,
                ticker=ticker,
                lookback_days=int(req.lookback_days),
                source=provider,
                fetched_at_utc=now,
                expires_at_utc=expires,
                row_json=json.dumps(feats_new, ensure_ascii=False),
            )
        )

        meta_new = {
            "oracle_source": provider,
            "oracle_cache_hit": False,
            "oracle_cache_ttl_seconds": ttl,
            "oracle_cache_expires_at_utc": expires,
        }
        return feats_new, meta_new

    # Deterministic provider selection
    if ORACLE_PROVIDER == "yfinance":
        return _try_provider("yfinance")
    if ORACLE_PROVIDER == "stooq":
        return _try_provider("stooq")

    # auto: yfinance then stooq (fallback)
    try:
        return _try_provider("yfinance")
    except Exception as yf_err:
        feats2, meta2 = _try_provider("stooq")
        meta2["oracle_yfinance_error"] = str(yf_err)
        return feats2, meta2


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
                fv = float("nan")  # keep NaN so bundle SimpleImputer applies training medians
        except Exception:
            fv = float("nan")  # keep NaN so bundle SimpleImputer applies training medians

        row.append(fv)

    # Do NOT nan_to_num here — imputer.transform() handles NaN with training-set medians.
    # Fallback to 0-fill / nanmedian happens below (lines ~988-996) only if imputer absent/fails.
    X = np.asarray([row], dtype=float)
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
    if not b:
        return {
            "raw_if": None,
            "raw_lof": None,
            "z_if": None,
            "z_lof": None,
            "ensemble": None,
            "status": "DISABLED",
            "thresholds": {"warn": None, "block": None},
            "asset_thresholds_used": False,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "missing_cols_sample": [],
            "n_cols": 0,
            "debug": {"reason": "missing_unsup_bundle"},
        }

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

    imputer_block = b.get("imputer", {}) or {}
    imputer = imputer_block.get("object")

    if imputer is not None:
        try:
            X = imputer.transform(X)
        except Exception as e:
            logger.warning(f"[UNSUP] imputer.transform failed -> fallback training-medians. err={e!r}")
            # Use training-set medians stored in bundle, not raw-data nanmedian (single-row = useless)
            stats = np.asarray(imputer_block.get("statistics") or [], dtype=float)
            if len(stats) == X.shape[1] and np.isnan(X).any():
                for j in range(X.shape[1]):
                    if np.isnan(X[0, j]):
                        X[0, j] = stats[j] if np.isfinite(stats[j]) else 0.0
            elif np.isnan(X).any():
                X = np.where(np.isnan(X), 0.0, X)
    else:
        if np.isnan(X).any():
            X = np.where(np.isnan(X), 0.0, X)

    raw_if = float(np.asarray(iforest.score_samples(X), dtype=float)[0])
    raw_lof = float(np.asarray(lof.score_samples(X), dtype=float)[0])

    mu_if = float(norm_if.get("mu", 0.0))
    sg_if = float(norm_if.get("sigma", 1.0)) or 1.0
    mu_lof = float(norm_lof.get("mu", 0.0))
    sg_lof = float(norm_lof.get("sigma", 1.0)) or 1.0

    z_if = (raw_if - mu_if) / (sg_if + 1e-12)
    z_lof = (raw_lof - mu_lof) / (sg_lof + 1e-12)

    anomaly_score = float((w_if * z_if) + (w_lof * z_lof))

    asset_type = (feats.get("asset_type") or "").strip().lower()
    thr = (thr_per_asset.get(asset_type) or thr_global) or {}
    warn_thr = float(thr.get("warn", thr.get("WARN", 0.0)))
    block_thr = float(thr.get("block", thr.get("BLOCK", 1.0)))

    status = "OK"
    if anomaly_score >= block_thr:
        status = "BLOCK"
    elif anomaly_score >= warn_thr:
        status = "WARN"

    meta = b.get("meta", {}) or {}
    logger.info(
        f"[UNSUP] raw_if={raw_if:.6f} raw_lof={raw_lof:.6f} "
        f"z_if={float(z_if):.4f} z_lof={float(z_lof):.4f} "
        f"w_if={w_if:.3f} w_lof={w_lof:.3f} ensemble={anomaly_score:.4f} status={status} "
        f"thr_warn={warn_thr:.4f} thr_block={block_thr:.4f} "
        f"missing_ratio={missing_ratio:.3f} n_cols={len(cols)}"
    )

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
        "debug": {
            "bundle_meta": meta,
            "score_norm": score_norm,
            "weights": {"if": w_if, "lof": w_lof},
            "thr_source": "per_asset" if asset_type in (thr_per_asset or {}) else "global",
        },
    }


# =============================
# XGB shadow (optional)  ✅ NEVER CRASH
# =============================
def _xgb_shadow_score(feats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not XGB_SHADOW_ENABLED:
        return None

    p = Path(SUP_BUNDLE_PATH)
    if not p.exists():
        return None

    sup = _load_sup()
    if not sup or "models" not in sup:
        return None

    model = (sup.get("models") or {}).get("xgb")
    if model is None:
        return None

    try:
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

        if len(proba) < 3:
            return None  # model not trained as 3-class; skip shadow score
        return {"pred": pred, "probs": {"OK": float(proba[0]), "WARN": float(proba[1]), "BLOCK": float(proba[2])}}

    except Exception as e:
        logger.warning("[XGB] shadow error: %s: %s", type(e).__name__, e)
        return None


# =============================
# Integrity mini-checks
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


def _bin_calibrated_decision(feats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Safe wrapper around api.decision.decide().
    - Never raises
    - Always returns JSON-serializable content
    """
    if not BIN_ENABLED:
        return None

    try:
        from api.decision import decide  # import local, avoids circular issues

        out = decide(feats)

        # sécurité JSON (numpy → float/int)
        if isinstance(out, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in out.items():
                try:
                    if hasattr(v, "item"):  # numpy scalar
                        cleaned[k] = v.item()
                    else:
                        cleaned[k] = v
                except Exception:
                    cleaned[k] = None
            return cleaned

        return out  # type: ignore[return-value]

    except Exception as e:
        # ⚠️ jamais faire tomber l'API
        return {
            "p_non_ok": None,
            "decision": "OK",
            "thresholds": {"t_lo": 0.5, "t_hi": BIN_T_HI_DEFAULT, "alpha": None},
            "thresholds_source": {"path": BIN_THRESHOLDS_PATH, "fallback_used": True},
            "debug": {"error": f"{type(e).__name__}: {e}"},
        }


# =============================
# Endpoints
# =============================
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "asymetra-lcc-api",
        "version": app.version,
        "hint": "Use /health, /metrics, /score, /score_oracle, /oracle/analyze, /score_3m",
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Statistiques agrégées des appels depuis le démarrage du serveur."""
    with _METRICS_LOCK:
        calls_score = _METRICS["calls_score"]
        calls_oracle = _METRICS["calls_score_oracle"]
        total = calls_score + calls_oracle
        statuses = dict(_METRICS["statuses"])
        scores = list(_METRICS["scores"])
        expert_non_null = _METRICS["expert_non_null"]

    status_pct: Dict[str, Any] = {}
    if total > 0:
        status_pct = {k: round(v / total * 100, 1) for k, v in statuses.items()}

    score_stats: Dict[str, Any] = {}
    if scores:
        arr = np.array(scores, dtype=float)
        score_stats = {
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "n": len(scores),
        }

    return {
        "calls_score": calls_score,
        "calls_score_oracle": calls_oracle,
        "calls_total": total,
        "status_counts": statuses,
        "status_pct": status_pct,
        "score_stats": score_stats,
        "expert_non_null_calls": expert_non_null,
        "model_3m": _MODEL_3M_BACKTEST,
        "model_v15": _MODEL_V15_BACKTEST,
        "model_v16": _MODEL_V16_BACKTEST,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    # ----- BIN CALIBRATED STATUS -----
    bin_status: Dict[str, Any] = {
        "enabled": BIN_ENABLED,
        "bundle_path": BIN_BUNDLE_PATH,
        "thresholds_path": BIN_THRESHOLDS_PATH,
        "t_hi_default": BIN_T_HI_DEFAULT,
        "bundle_loaded": False,
        "thresholds_loaded": False,
        "thresholds": None,
    }

    if BIN_ENABLED:
        try:
            # internal loaders in api.decision
            from api.decision import _load_bin_bundle, _load_thresholds  # type: ignore

            b = _load_bin_bundle()
            bin_status["bundle_loaded"] = True
            bin_status["bundle_calibrated"] = bool(b.get("calibrated", False))
            bin_status["calib_method"] = b.get("calib_method")
            bin_status["n_cols"] = len(b.get("cols") or b.get("columns") or [])

            thr = _load_thresholds()
            bin_status["thresholds_loaded"] = True

            t_lo = float(thr.get("t_lo", 0.5))
            t_hi = float(thr.get("t_hi", BIN_T_HI_DEFAULT))

            out_thr: Dict[str, Any] = {"t_lo": t_lo, "t_hi": t_hi}
            raw = thr.get("raw")
            if isinstance(raw, dict) and "alpha" in raw:
                out_thr["alpha"] = raw["alpha"]

            bin_status["thresholds"] = out_thr
            bin_status["thresholds_fallback_used"] = bool(thr.get("fallback_used", False))
            bin_status["thresholds_source_path"] = thr.get("path", BIN_THRESHOLDS_PATH)

        except Exception as e:
            bin_status["error"] = f"{type(e).__name__}: {e}"

    unsup_path_abs = str(Path(UNSUP_BUNDLE_PATH).resolve())
    unsup_exists = Path(UNSUP_BUNDLE_PATH).exists()

    return {
        "ok": True,
        "app": "api.main",
        "version": app.version,
        "paths_hint": ["/", "/health", "/oracle/analyze", "/score", "/score_oracle"],
        "oracle_cache_db": ORACLE_CACHE_DB,
        "oracle_cache_columns": _ORACLE_CACHE.columns(),
        "oracle_cache_recent": _ORACLE_CACHE.recent(limit=5),
        "unsup_coverage": {"max_missing_ratio": UNSUP_MAX_MISSING_RATIO, "max_missing_count": UNSUP_MAX_MISSING_COUNT},
        "bin_calibrated": bin_status,
        # ✅ observabilité bundle
        "unsup_bundle_path": unsup_path_abs,
        "unsup_bundle_exists": bool(unsup_exists),
        # Expert scoring layer (structured)
        "experts": experts_health(),
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

    try:
        u = _unsup_score(feats)
    except Exception as e:
        _err500("unsup_score", e)  # raises

    resp: Dict[str, Any] = {
        "ensemble": float(u["ensemble"]) if u.get("ensemble") is not None else None,
        "status": u.get("status"),
        "thresholds": u.get("thresholds"),
        "debug_unsup": {
            "raw_if": u.get("raw_if"),
            "raw_lof": u.get("raw_lof"),
            "z_if": u.get("z_if"),
            "z_lof": u.get("z_lof"),
            "missing_count": u.get("missing_count"),
            "missing_ratio": u.get("missing_ratio"),
            "missing_cols_sample": u.get("missing_cols_sample"),
            "n_cols": u.get("n_cols"),
            "reason": (u.get("debug") or {}).get("reason"),
        },
    }

    shadow_obj: Dict[str, Any] = {}
    try:
        shadow_obj["bin_calibrated"] = _bin_calibrated_decision(feats)
    except Exception as e:
        shadow_obj["bin_calibrated_error"] = f"{type(e).__name__}: {e}"

    try:
        xgb = _xgb_shadow_score(feats)
        if xgb is not None:
            shadow_obj["xgb"] = xgb
    except Exception as e:
        shadow_obj["xgb_error"] = f"{type(e).__name__}: {e}"

    if shadow_obj:
        resp["shadow"] = shadow_obj

    # Expert scoring (additive — does not replace existing fields)
    # Guarded by EXPERTS_ENABLED env var (default OFF)
    if is_experts_enabled():
        try:
            asset_type = str(req.asset_type or "").strip().lower()
            expert_result = score_expert(feats, asset_type)
            resp["expert_decision"] = expert_result
            resp["expert_loaded"] = expert_result is not None
        except Exception as _e:
            error_id = uuid.uuid4().hex[:8]
            logger.error("expert scoring error [%s]: %s", error_id, _e, exc_info=True)
            resp["expert_decision"] = None
            resp["expert_loaded"] = False
            resp["expert_decision_error"] = f"{type(_e).__name__}: {_e}"
            resp["expert_error_id"] = error_id
    else:
        resp["expert_decision"] = None
        resp["expert_loaded"] = False

    _record_metric(
        endpoint="score",
        status=str(u.get("status") or "OK"),
        score=u.get("ensemble"),
        has_expert=bool(resp.get("expert_decision") is not None),
    )
    return jsonable_encoder(resp)


# -----------------------------
# PATCH helpers (score_oracle)
# -----------------------------
def _oracle_error_code(err: Exception) -> str:
    msg = (str(err) or "").lower()

    if "no data" in msg or "notreal" in msg:
        return "NO_DATA"
    if "insufficient closes" in msg or "not enough closes" in msg:
        return "NO_DATA"

    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return "RATE_LIMIT"

    if "timeout" in msg or "timed out" in msg or "connection" in msg or "dns" in msg:
        return "NETWORK"

    if "non-csv response" in msg or "jsondecodeerror" in msg or "parse" in msg:
        return "PARSE"

    return "UNKNOWN"


def _unsup_skip_reason(
    missing_ratio: float,
    missing_count: int,
    max_missing_ratio: float,
    max_missing_count: int,
) -> Optional[str]:
    if missing_ratio > max_missing_ratio:
        return f"MISSING_RATIO>{max_missing_ratio:.3f}"
    if max_missing_count > 0 and missing_count > max_missing_count:
        return f"MISSING_COUNT>{max_missing_count}"
    return None


@app.post(
    "/score_oracle",
    description=(
        "Pipeline fiabilité:\n"
        "- Reçoit les stats calculées par Lovable\n"
        "- Score LCC (unsup) pour détecter incohérences / risques\n"
        "- Si suspect (ou forcé), lance Oracle (avec cache)\n"
        "- Renvoie features_final (à afficher) + oracle_used"
    ),
)
def score_oracle(
    req: ScoreOracleRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict[str, Any]:
    _require_api_key(x_api_key)

    lovable_feats = _features_dict(req.lovable)
    ticker = lovable_feats.get("ticker")

    has_closes = bool(req.closes)
    n_closes = len(req.closes) if req.closes else 0
    has_dates = bool(req.dates)
    n_dates = len(req.dates) if req.dates else 0

    effective_lookback = int(lovable_feats.get("lookback_days") or req.lookback_days or 252)

    # --- Integrity + UNSUP (pré-oracle) ---
    integrity_flags, integrity_critical = _integrity_flags(lovable_feats)
    unsup_pre = _unsup_score(lovable_feats)

    unsup_missing_ratio_pre = float(unsup_pre.get("missing_ratio", 0.0) or 0.0)
    unsup_missing_count_pre = int(unsup_pre.get("missing_count", 0) or 0)
    unsup_n_cols_pre = int(unsup_pre.get("n_cols", 0) or 0)

    unsup_skip_reason_pre = _unsup_skip_reason(
        missing_ratio=unsup_missing_ratio_pre,
        missing_count=unsup_missing_count_pre,
        max_missing_ratio=UNSUP_MAX_MISSING_RATIO,
        max_missing_count=UNSUP_MAX_MISSING_COUNT,
    )
    unsup_skip_pre = unsup_skip_reason_pre is not None
    unsup_status_pre = "SKIP" if unsup_skip_pre else str(unsup_pre.get("status"))

    # --- Missing critical (pré-oracle) ---
    critical = ["var95", "var99", "es95", "vol_ann", "max_drawdown"]
    missing_critical_before = [k for k in critical if _safe_float(lovable_feats.get(k)) is None]

    # --- XGB shadow + BIN calibrated (pré-oracle) ---
    xgb = _xgb_shadow_score(lovable_feats)
    bin_before_oracle = _bin_calibrated_decision(lovable_feats)

    xgb_ok_relaxes = False
    if xgb and isinstance(xgb.get("probs"), dict):
        pb = xgb["probs"].get("BLOCK")
        if xgb.get("pred") == "OK" and isinstance(pb, (int, float)) and float(pb) <= XGB_OK_PBLOCK_MAX:
            xgb_ok_relaxes = True

    warn_thr = float((unsup_pre.get("thresholds") or {}).get("warn") or 0.0)
    ensemble = float(unsup_pre.get("ensemble") or 0.0)
    warn_is_shallow = (unsup_status_pre == "WARN") and (ensemble < (warn_thr + WARN_MARGIN))

    # --- Should Oracle? ---
    should_oracle = False
    reasons: Dict[str, Any] = {
        "force_oracle": bool(req.force_oracle),
        "missing_critical": bool(missing_critical_before),
        "integrity_critical": bool(integrity_critical),
        "unsup_skip": bool(unsup_skip_pre),
        "unsup_skip_reason": unsup_skip_reason_pre,
        "unsup_status_pre": unsup_status_pre,
        "effective_lookback": int(effective_lookback),
        "bin_before_oracle": bin_before_oracle,
    }

    if req.force_oracle:
        should_oracle = True
    elif integrity_critical:
        should_oracle = True
    elif missing_critical_before:
        should_oracle = True
    elif unsup_skip_pre:
        should_oracle = True
    elif unsup_status_pre == "BLOCK":
        should_oracle = True
    elif unsup_status_pre == "WARN":
        if (not warn_is_shallow) and (not xgb_ok_relaxes):
            should_oracle = True

    # --- Oracle execution ---
    oracle_used = False
    oracle_succeeded = False
    oracle_failed = False
    oracle_error = None
    oracle_error_code = None

    oracle_feats: Optional[Dict[str, Any]] = None
    oracle_meta: Optional[Dict[str, Any]] = None
    oracle_mode = "none"  # none | rescue | recompute

    if should_oracle:
        oracle_used = True

        enough_closes = has_closes and (n_closes >= (effective_lookback + 2))
        aligned_dates = (not has_dates) or (n_dates == n_closes)
        oracle_mode = "rescue" if (enough_closes and aligned_dates) else "recompute"

        try:
            oreq = OracleRequest(
                asset_type=lovable_feats.get("asset_type") or "equity",
                market=lovable_feats.get("market") or "US",
                ticker=ticker,
                closes=req.closes if oracle_mode == "rescue" else None,
                dates=req.dates if oracle_mode == "rescue" else None,
                lookback_days=effective_lookback,
            )
            oracle_feats, oracle_meta = _oracle_analyze(oreq)
            oracle_succeeded = True
        except Exception as e:
            oracle_failed = True
            oracle_error = str(e)
            oracle_error_code = _oracle_error_code(e)

    # --- Features final ---
    features_final = oracle_feats if oracle_feats is not None else lovable_feats
    missing_critical_final = [k for k in critical if _safe_float(features_final.get(k)) is None]

    # --- UNSUP final (post-oracle si oracle réussi, sinon on garde le pré) ---
    unsup_final = None
    if oracle_succeeded and oracle_feats is not None:
        unsup_final = _unsup_score(features_final)

    unsup_status_final: str = unsup_status_pre
    unsup_skip_reason_final: Optional[str] = unsup_skip_reason_pre

    if unsup_final is not None:
        miss_ratio_f = float(unsup_final.get("missing_ratio", 0.0) or 0.0)
        miss_count_f = int(unsup_final.get("missing_count", 0) or 0)
        unsup_skip_reason_final = _unsup_skip_reason(
            missing_ratio=miss_ratio_f,
            missing_count=miss_count_f,
            max_missing_ratio=UNSUP_MAX_MISSING_RATIO,
            max_missing_count=UNSUP_MAX_MISSING_COUNT,
        )
        if unsup_skip_reason_final is not None:
            unsup_status_final = "SKIP"
        else:
            unsup_status_final = str(unsup_final.get("status"))

    # ✅ BIN calibrated final (sur features_final)
    bin_final = _bin_calibrated_decision(features_final)

    out: Dict[str, Any] = {
        "features_final": features_final,
        "oracle_used": oracle_used,
        "oracle_succeeded": oracle_succeeded,
        "oracle_failed": oracle_failed,
        "oracle_error": oracle_error,
        "oracle_error_code": oracle_error_code,
        "oracle_mode": oracle_mode,
        "unsup_status_pre_oracle": unsup_status_pre,
        "unsup_skip_reason_pre_oracle": unsup_skip_reason_pre,
        "unsup_status_final": unsup_status_final,
        "unsup_skip_reason_final": unsup_skip_reason_final,
        "unsup_status": unsup_status_final,
        "missing_critical": missing_critical_final,
        "integrity_flags": integrity_flags,
        "integrity_critical_flags": integrity_critical,
        "debug_unsup": {
            "raw_if": unsup_pre.get("raw_if"),
            "raw_lof": unsup_pre.get("raw_lof"),
            "z_if": unsup_pre.get("z_if"),
            "z_lof": unsup_pre.get("z_lof"),
            "ensemble": unsup_pre.get("ensemble"),
            "thresholds": unsup_pre.get("thresholds"),
            "missing_count": unsup_pre.get("missing_count"),
            "missing_ratio": unsup_pre.get("missing_ratio"),
            "missing_cols_sample": unsup_pre.get("missing_cols_sample"),
            "n_cols": unsup_pre.get("n_cols"),
        },
        "debug_unsup_after_oracle": unsup_final,
        "xgb_shadow": xgb,
        "bin_calibrated_before_oracle": bin_before_oracle,
        "bin_calibrated_final": bin_final,

        # ✅ AJOUT: métriques stables (pour éviter tes null dans jq)
        "unsup_pre_metrics": {
            "missing_ratio": unsup_missing_ratio_pre,
            "missing_count": unsup_missing_count_pre,
            "n_cols": unsup_n_cols_pre,
        },
        "unsup_final_metrics": {
            "missing_ratio": (float(unsup_final.get("missing_ratio", 0.0)) if unsup_final else None),
            "missing_count": (int(unsup_final.get("missing_count", 0)) if unsup_final else None),
            "n_cols": (int(unsup_final.get("n_cols", 0)) if unsup_final else None),
        },

        "decision_trace": {
            "should_oracle": should_oracle,
            "oracle_mode": oracle_mode,
            "reasons": reasons,
            "warn_is_shallow": warn_is_shallow,
            "xgb_ok_relaxes": xgb_ok_relaxes,
            "unsup_status_pre_oracle": unsup_status_pre,
            "unsup_status_final": unsup_status_final,
            "unsup_skip_reason_pre_oracle": unsup_skip_reason_pre,
            "unsup_skip_reason_final": unsup_skip_reason_final,
            "missing_critical_before": missing_critical_before,
            "missing_critical_final": missing_critical_final,
            "integrity_critical": bool(integrity_critical),
            "has_closes": has_closes,
            "closes_len": n_closes,
            "has_dates": has_dates,
            "dates_len": n_dates,
            "effective_lookback": int(effective_lookback),
            "oracle_used": oracle_used,
            "oracle_succeeded": oracle_succeeded,
            "oracle_failed": oracle_failed,
            "oracle_error_code": oracle_error_code,
        },
        "gating_debug": {
            "should_oracle": should_oracle,
            "reasons": reasons,
            "warn_is_shallow": warn_is_shallow,
            "xgb_ok_relaxes": xgb_ok_relaxes,
            "warn_thr": warn_thr,
            "ensemble": ensemble,
            "warn_margin": WARN_MARGIN,
            "xgb_ok_pblock_max": XGB_OK_PBLOCK_MAX,
            "unsup_missing_ratio_pre": unsup_missing_ratio_pre,
            "unsup_missing_count_pre": unsup_missing_count_pre,
            "unsup_n_cols_pre": unsup_n_cols_pre,
            "unsup_max_missing_ratio": UNSUP_MAX_MISSING_RATIO,
            "unsup_max_missing_count": UNSUP_MAX_MISSING_COUNT,
        },
        "oracle_input_debug": {
            "has_closes": has_closes,
            "n_closes": n_closes,
            "has_dates": has_dates,
            "n_dates": n_dates,
            "lookback_days_req": int(req.lookback_days),
            "lookback_days_effective": int(effective_lookback),
        },
    }

    if oracle_meta is not None:
        out["oracle_meta"] = oracle_meta

    # Expert scoring (additive — does not replace existing fields)
    # Guarded by EXPERTS_ENABLED env var (default OFF)
    if is_experts_enabled():
        try:
            at = str(features_final.get("asset_type") or lovable_feats.get("asset_type") or "").strip().lower()
            expert_result = score_expert(features_final, at)
            out["expert_decision"] = expert_result
            out["expert_loaded"] = expert_result is not None
        except Exception as _e:
            error_id = uuid.uuid4().hex[:8]
            logger.error("expert scoring error [%s]: %s", error_id, _e, exc_info=True)
            out["expert_decision"] = None
            out["expert_loaded"] = False
            out["expert_decision_error"] = f"{type(_e).__name__}: {_e}"
            out["expert_error_id"] = error_id
    else:
        out["expert_decision"] = None
        out["expert_loaded"] = False

    _record_metric(
        endpoint="score_oracle",
        status=str(unsup_status_final or "OK"),
        score=(unsup_final or unsup_pre).get("ensemble"),
        has_expert=bool(out.get("expert_decision") is not None),
    )
    return jsonable_encoder(out)






# =============================================================================
# 3m Stock-Picking Model — /score_3m
# =============================================================================

class Score3mItem(BaseModel):
    """Features for one stock to score with the 3m model."""
    ticker: Optional[str] = Field(default=None, max_length=20)

    # Price / momentum features
    ret_1m:         Optional[float] = None
    ret_3m:         Optional[float] = None
    ret_6m:         Optional[float] = None
    ret_12m:        Optional[float] = None
    mom_12_1:       Optional[float] = None
    ret_12m_vs_spy: Optional[float] = None
    vol_ann:        Optional[float] = None
    vol_ratio:      Optional[float] = None
    dd_from_hi52:   Optional[float] = None
    above_200ma:    Optional[float] = None
    trend_strength: Optional[float] = None

    # Fundamental features V1
    gross_margin:    Optional[float] = None
    op_margin:       Optional[float] = None
    net_margin:      Optional[float] = None
    roe:             Optional[float] = None
    debt_to_equity:  Optional[float] = None
    rd_intensity:    Optional[float] = None
    fcf_margin:      Optional[float] = None
    revenue_growth:  Optional[float] = None
    ni_growth:       Optional[float] = None

    # Fundamental features V2
    pe_ratio:        Optional[float] = None
    pb_ratio:        Optional[float] = None
    earnings_yield:  Optional[float] = None
    ev_to_revenue:   Optional[float] = None
    accruals_ratio:  Optional[float] = None
    asset_growth:    Optional[float] = None
    current_ratio:   Optional[float] = None

    # Technical V2
    ret_1m_lag:  Optional[float] = None
    skew_6m:     Optional[float] = None

    # Sector
    sector_id: Optional[float] = None

    model_config = {"extra": "allow"}


class Score3mRequest(BaseModel):
    stocks: List[Score3mItem] = Field(..., min_length=1, max_length=2000)
    top_pct: float = Field(default=0.10, ge=0.01, le=0.50,
                           description="Fraction of stocks to flag as top picks (default 10%)")


def _predict_3m(items: List[Score3mItem], top_pct: float) -> Dict[str, Any]:
    """Score stocks with 3m model. Returns per-stock probs + top picks."""
    bundle = _load_3m_model()
    if not bundle:
        raise HTTPException(status_code=503, detail="3m model not loaded (check MODEL_3M_PATH)")

    xgb_model = bundle.get("xgb_model")
    lgb_model  = bundle.get("lgb_model")
    medians    = bundle.get("medians", {})
    cols       = bundle.get("cols", _MODEL_3M_FEATURES)

    if xgb_model is None or lgb_model is None:
        raise HTTPException(status_code=503, detail="3m model bundle missing xgb_model or lgb_model")

    # Build feature matrix
    rows = []
    for item in items:
        row = {}
        item_dict = item.model_dump()
        for col in cols:
            val = item_dict.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = medians.get(col, 0.0)
            row[col] = float(val)
        rows.append(row)

    X = pd.DataFrame(rows, columns=cols)

    # Predict ensemble
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    lgb_probs  = lgb_model.predict_proba(X)[:, 1]
    probs      = 0.5 * xgb_probs + 0.5 * lgb_probs

    # Determine top picks
    n_top    = max(1, int(len(items) * top_pct))
    top_idx  = set(np.argsort(probs)[::-1][:n_top].tolist())

    scored = []
    for i, (item, prob) in enumerate(zip(items, probs)):
        scored.append({
            "ticker":           item.ticker,
            "prob_beat_spy_3m": round(float(prob), 4),
            "top_pick":         i in top_idx,
        })

    # Summary stats
    probs_arr = np.array([s["prob_beat_spy_3m"] for s in scored])
    return {
        "model_version": "3m_v1",
        "n_stocks":      len(scored),
        "n_top_picks":   n_top,
        "top_pct":       top_pct,
        "scores":        scored,
        "summary": {
            "prob_mean":   round(float(probs_arr.mean()), 4),
            "prob_median": round(float(np.median(probs_arr)), 4),
            "prob_p90":    round(float(np.percentile(probs_arr, 90)), 4),
            "prob_min":    round(float(probs_arr.min()), 4),
            "prob_max":    round(float(probs_arr.max()), 4),
        },
    }


@app.post("/score_3m")
def score_3m(
    req: Score3mRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict[str, Any]:
    """Score a list of stocks with the 3m stock-picking model.

    Returns prob_beat_spy_3m (probability of beating SPY by >2.5% over 3 months)
    and marks the top N% as top_pick=true.

    Validated performance (walk-forward 2019-2023):
    CAGR +32.7% | Sharpe 1.11 | Sortino 1.99 | Alpha +17.8%/yr
    """
    _require_api_key(x_api_key)
    return _predict_3m(req.stocks, req.top_pct)
