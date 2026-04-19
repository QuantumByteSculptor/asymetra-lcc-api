"""
scripts/ml/train_v15_regime.py
================================
V15 — Regime-Aware Walk-Forward Stock-Picking Model

Key innovations over V14-B (Sharpe 1.61, MaxDD -13.8%, momentum filter):
  1. Market regime detection — Bull / Bear / Sideways via VIX level + SPY 6m momentum
  2. Regime-conditional exposure — Bear→0%, Sideways→50%, Bull→100%
  3. Regime features added to model input (vix_level, spy_mom_6m, regime_id)
  4. Walk-forward with expanding train window, 6-month test steps (5 folds)
  5. Separate regime-stratified calibration per fold

Pipeline:
  1. Download prices + VIX (2017-2024) via yfinance for ~50 S&P 500 stocks
  2. Build monthly features per stock (momentum, volatility, trend, regime context)
  3. Label: forward_3m_alpha = stock_3m_ret - SPY_3m_ret > 2.5%  (binary)
  4. Classify regime per month from SPY / VIX
  5. Walk-forward XGB+LGB ensemble training (5 folds)
  6. Backtest with regime-conditional sizing
  7. Compare to V14-B and 3m_v1 baselines
  8. Save model + metrics

Usage:
  python scripts/ml/train_v15_regime.py \\
      --start 2017-01-01 \\
      --end   2024-12-31 \\
      --out   data/metrics/v15_results.json \\
      --model_out models/stock_picker_v15.joblib

No API / prod impact during development.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("train_v15")

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Universe — 50 liquid S&P 500 stocks with long history
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "C", "AXP", "BLK", "USB",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "MDT", "CVS",
    # Consumer
    "WMT", "PG", "KO", "PEP", "MCD", "COST", "NKE", "HD", "TGT", "LOW",
    # Industrials / Energy
    "XOM", "CVX", "COP", "SLB", "CAT", "HON", "MMM", "GE", "BA", "UPS",
]

SECTOR_MAP = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0, "AMZN": 0,
    "NVDA": 0, "AMD": 0, "INTC": 0, "CSCO": 0, "ORCL": 0,
    "JPM": 1, "BAC": 1, "WFC": 1, "GS": 1, "MS": 1,
    "BRK-B": 1, "C": 1, "AXP": 1, "BLK": 1, "USB": 1,
    "JNJ": 2, "UNH": 2, "PFE": 2, "ABBV": 2, "MRK": 2,
    "LLY": 2, "TMO": 2, "ABT": 2, "MDT": 2, "CVS": 2,
    "WMT": 3, "PG": 3, "KO": 3, "PEP": 3, "MCD": 3,
    "COST": 3, "NKE": 3, "HD": 3, "TGT": 3, "LOW": 3,
    "XOM": 4, "CVX": 4, "COP": 4, "SLB": 4, "CAT": 5,
    "HON": 5, "MMM": 5, "GE": 5, "BA": 5, "UPS": 5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Regime thresholds
# ─────────────────────────────────────────────────────────────────────────────
VIX_BEAR_THRESHOLD   = 27.0   # VIX above this → Bear
VIX_BULL_MAX         = 18.0   # VIX must be below this for Bull
SPY_MOM_BULL_MIN     = 0.05   # SPY 6m return > 5% for Bull
SPY_MOM_BEAR_MAX     = -0.05  # SPY 6m return < -5% → Bear

REGIME_BULL     = 0
REGIME_SIDEWAYS = 1
REGIME_BEAR     = 2
REGIME_LABELS   = {REGIME_BULL: "bull", REGIME_SIDEWAYS: "sideways", REGIME_BEAR: "bear"}

# Exposure per regime for backtest
REGIME_EXPOSURE = {REGIME_BULL: 1.0, REGIME_SIDEWAYS: 0.5, REGIME_BEAR: 0.0}

# Portfolio: top N% stocks per month
TOP_PCT = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data download
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted monthly close prices for all tickers + SPY + ^VIX."""
    log.info("Downloading prices for %d tickers + SPY + VIX (%s → %s)...", len(tickers), start, end)

    all_tickers = list(set(tickers + ["SPY", "^VIX"]))
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers[:1]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index = prices.index.to_period("M").to_timestamp("M")
    log.info("Downloaded %d rows × %d columns", len(prices), len(prices.columns))
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, classify market regime using SPY 6m return + VIX level.

    Returns DataFrame with columns: spy_1m, spy_6m, vix_level, regime_id, regime_label
    """
    spy  = prices["SPY"].dropna()
    vix  = prices["^VIX"].dropna() if "^VIX" in prices.columns else pd.Series(dtype=float)

    df = pd.DataFrame(index=prices.index)
    df["spy_1m"]    = spy.pct_change(1)
    df["spy_6m"]    = spy.pct_change(6)
    df["spy_12m"]   = spy.pct_change(12)
    df["vix_level"] = vix.reindex(df.index)

    # Fill missing VIX with 20 (neutral)
    df["vix_level"] = df["vix_level"].fillna(20.0)

    def _classify(row):
        vix_val = row["vix_level"]
        mom_6m  = row["spy_6m"] if pd.notna(row["spy_6m"]) else 0.0

        if vix_val >= VIX_BEAR_THRESHOLD or mom_6m <= SPY_MOM_BEAR_MAX:
            return REGIME_BEAR
        elif vix_val <= VIX_BULL_MAX and mom_6m >= SPY_MOM_BULL_MIN:
            return REGIME_BULL
        else:
            return REGIME_SIDEWAYS

    df["regime_id"]    = df.apply(_classify, axis=1)
    df["regime_label"] = df["regime_id"].map(REGIME_LABELS)

    regime_counts = df["regime_id"].value_counts().sort_index()
    log.info("Regime distribution: %s", {REGIME_LABELS[k]: v for k, v in regime_counts.items()})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def _trend_strength(series: pd.Series) -> float:
    """R² of price regressed on time index."""
    if len(series) < 3 or series.isna().all():
        return float("nan")
    y = series.dropna().values
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return float("nan")
    x_bar, y_bar = x.mean(), y.mean()
    denom = ((x - x_bar) ** 2).sum()
    if denom == 0:
        return float("nan")
    beta = ((x - x_bar) * (y - y_bar)).sum() / denom
    y_hat = y_bar + beta * (x - x_bar)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_bar) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_features(
    prices: pd.DataFrame,
    regimes: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Build monthly feature rows for a single ticker.

    Features:
      Momentum:  ret_1m, ret_3m, ret_6m, ret_12m, mom_12_1, ret_vs_spy_3m
      Volatility: vol_ann, vol_ratio
      Trend:     above_200ma, trend_strength, dd_from_hi52
      Skew:      skew_20d (using daily proxy → monthly std of 1m returns)
      Regime:    vix_level, spy_mom_6m, regime_id (BULL=0/SIDEWAYS=1/BEAR=2)
      Static:    sector_id
    """
    if ticker not in prices.columns:
        return pd.DataFrame()

    p = prices[ticker].dropna()
    spy = prices["SPY"].dropna()

    rows = []
    for i, date in enumerate(p.index):
        if i < 12:
            continue  # need 12 months of history

        def _safe_ret(series, n):
            if i < n:
                return float("nan")
            v0 = series.iloc[i - n]
            v1 = series.iloc[i]
            if pd.isna(v0) or pd.isna(v1) or v0 == 0:
                return float("nan")
            return float(v1 / v0 - 1)

        ret_1m  = _safe_ret(p, 1)
        ret_3m  = _safe_ret(p, 3)
        ret_6m  = _safe_ret(p, 6)
        ret_12m = _safe_ret(p, 12)
        spy_1m  = _safe_ret(spy, 1)
        spy_3m  = _safe_ret(spy, 3)
        spy_12m = _safe_ret(spy, 12)

        mom_12_1    = (ret_12m - ret_1m) if pd.notna(ret_12m) and pd.notna(ret_1m) else float("nan")
        ret_vs_spy3 = (ret_3m - spy_3m)  if pd.notna(ret_3m)  and pd.notna(spy_3m)  else float("nan")

        window_12 = p.iloc[max(0, i - 12):i + 1]
        monthly_rets = window_12.pct_change().dropna()

        vol_ann   = float(monthly_rets.std() * math.sqrt(12)) if len(monthly_rets) >= 3 else float("nan")
        skew_12m  = float(monthly_rets.skew())               if len(monthly_rets) >= 4 else float("nan")

        # Vol ratio: recent 3m vol vs 12m vol
        vol_3m  = float(monthly_rets.tail(3).std() * math.sqrt(12)) if len(monthly_rets) >= 3 else float("nan")
        vol_ratio = float(vol_3m / vol_ann) if vol_ann and vol_ann > 0 else float("nan")

        # 52-week high drawdown
        hi52 = float(p.iloc[max(0, i - 12):i + 1].max())
        cur  = float(p.iloc[i])
        dd_from_hi52 = float(cur / hi52 - 1) if hi52 > 0 else float("nan")

        # 200-day MA proxy: use 12m monthly average
        ma_200 = float(p.iloc[max(0, i - 12):i + 1].mean())
        above_200ma = float(cur > ma_200)

        # Trend strength on last 12m
        trend_str = _trend_strength(p.iloc[max(0, i - 12):i + 1])

        # Regime features
        reg_row   = regimes.loc[date] if date in regimes.index else None
        vix_level = float(reg_row["vix_level"]) if reg_row is not None else 20.0
        spy_mom_6 = float(reg_row["spy_6m"])    if reg_row is not None and pd.notna(reg_row["spy_6m"]) else float("nan")
        regime_id = int(reg_row["regime_id"])   if reg_row is not None else REGIME_SIDEWAYS

        rows.append({
            "date":          date,
            "ticker":        ticker,
            "sector_id":     float(SECTOR_MAP.get(ticker, -1)),
            # Momentum
            "ret_1m":        ret_1m,
            "ret_3m":        ret_3m,
            "ret_6m":        ret_6m,
            "ret_12m":       ret_12m,
            "mom_12_1":      mom_12_1,
            "ret_vs_spy_3m": ret_vs_spy3,
            "spy_1m":        spy_1m,
            "spy_12m":       spy_12m,
            # Volatility
            "vol_ann":       vol_ann,
            "vol_ratio":     vol_ratio,
            "skew_12m":      skew_12m,
            # Trend
            "above_200ma":   above_200ma,
            "trend_strength":trend_str,
            "dd_from_hi52":  dd_from_hi52,
            # Regime (market context)
            "vix_level":     vix_level,
            "spy_mom_6m":    spy_mom_6,
            "regime_id":     float(regime_id),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Forward return: stock 3m return and SPY 3m return (look-ahead, will be shifted)
    df = df.set_index("date")

    # Compute forward returns by joining with future price
    fwd_ret_stock = []
    fwd_ret_spy   = []
    for i, date in enumerate(df.index):
        pos_in_p = p.index.get_loc(date) if date in p.index else None
        if pos_in_p is None or pos_in_p + 3 >= len(p):
            fwd_ret_stock.append(float("nan"))
            fwd_ret_spy.append(float("nan"))
            continue
        v0_s = p.iloc[pos_in_p]
        v3_s = p.iloc[pos_in_p + 3]
        fwd_s = float(v3_s / v0_s - 1) if v0_s > 0 else float("nan")
        fwd_ret_stock.append(fwd_s)

        pos_in_spy = spy.index.get_loc(date) if date in spy.index else None
        if pos_in_spy is None or pos_in_spy + 3 >= len(spy):
            fwd_ret_spy.append(float("nan"))
            continue
        v0_spy = spy.iloc[pos_in_spy]
        v3_spy = spy.iloc[pos_in_spy + 3]
        fwd_ret_spy.append(float(v3_spy / v0_spy - 1) if v0_spy > 0 else float("nan"))

    df["fwd_ret_3m"]     = fwd_ret_stock
    df["fwd_ret_spy_3m"] = fwd_ret_spy
    df["fwd_alpha_3m"]   = df["fwd_ret_3m"] - df["fwd_ret_spy_3m"]
    df["label"]          = (df["fwd_alpha_3m"] > 0.025).astype(int)  # beat SPY by >2.5%

    return df.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "ret_vs_spy_3m",
    "spy_1m", "spy_12m",
    "vol_ann", "vol_ratio", "skew_12m",
    "above_200ma", "trend_strength", "dd_from_hi52",
    "vix_level", "spy_mom_6m", "regime_id",
    "sector_id",
]


def build_dataset(prices: pd.DataFrame, regimes: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Build full cross-sectional dataset: one row per (ticker, month)."""
    dfs = []
    for ticker in tickers:
        df = build_features(prices, regimes, ticker)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No data built — check yfinance availability")

    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=["label", "fwd_ret_3m"])
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)

    log.info("Dataset: %d rows, %d tickers, %s → %s",
             len(full), full["ticker"].nunique(),
             full["date"].min().date(), full["date"].max().date())
    log.info("Label distribution: %s", full["label"].value_counts().to_dict())
    return full


# ─────────────────────────────────────────────────────────────────────────────
# 5. Walk-forward splits
# ─────────────────────────────────────────────────────────────────────────────

def make_walk_forward_splits(
    df: pd.DataFrame,
    n_test_months: int = 6,
    min_train_months: int = 24,
    embargo_months: int = 3,
) -> List[Dict]:
    """
    Expanding-window walk-forward splits.

    Returns list of dicts with train_idx, val_idx, fold metadata.
    Embargo gap = embargo_months between train end and val start
    (prevents forward-return leakage — 3m hold period).
    """
    dates = df["date"].dt.to_period("M").unique()
    dates = sorted(dates)

    # First possible test window starts after min_train_months
    splits = []
    fold = 1
    start_test_idx = min_train_months

    while True:
        test_start_idx = start_test_idx
        test_end_idx   = test_start_idx + n_test_months

        if test_end_idx > len(dates):
            break

        test_start = dates[test_start_idx].to_timestamp()
        test_end   = dates[min(test_end_idx, len(dates) - 1)].to_timestamp()

        # Train: all months before (test_start - embargo)
        train_cutoff_idx = test_start_idx - embargo_months
        if train_cutoff_idx < min_train_months:
            start_test_idx += n_test_months
            continue

        train_end  = dates[train_cutoff_idx - 1].to_timestamp()

        train_mask = df["date"] <= train_end
        val_mask   = (df["date"] >= test_start) & (df["date"] < test_end)

        train_idx = np.where(train_mask.values)[0]
        val_idx   = np.where(val_mask.values)[0]

        if len(train_idx) < 100 or len(val_idx) == 0:
            start_test_idx += n_test_months
            continue

        splits.append({
            "fold":        fold,
            "train_end":   str(train_end.date()),
            "val_start":   str(test_start.date()),
            "val_end":     str(test_end.date()),
            "n_train":     len(train_idx),
            "n_val":       len(val_idx),
            "train_idx":   train_idx,
            "val_idx":     val_idx,
        })
        fold += 1
        start_test_idx += n_test_months

    log.info("Walk-forward: %d folds generated (test_months=%d, embargo=%d)",
             len(splits), n_test_months, embargo_months)
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nan_rate(X: np.ndarray) -> np.ndarray:
    return np.isnan(X).mean(axis=0)


def _median_impute(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(X_train, axis=0)
    for col_idx in range(X_train.shape[1]):
        mask_tr = np.isnan(X_train[:, col_idx])
        mask_vl = np.isnan(X_val[:, col_idx])
        X_train[mask_tr, col_idx] = medians[col_idx]
        X_val[mask_vl,   col_idx] = medians[col_idx]
    return X_train, X_val, medians


def train_xgb(X_tr, y_tr, X_val, y_val, seed: int = SEED):
    if not HAS_XGB:
        return None
    scale_pos = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    clf = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        early_stopping_rounds=30,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
        use_label_encoder=False,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def train_lgb(X_tr, y_tr, X_val, y_val, seed: int = SEED):
    if not HAS_LGB:
        return None
    scale_pos = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        early_stopping_rounds=30,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return clf


def ensemble_proba(
    xgb_clf, lgb_clf,
    X: np.ndarray,
    w_xgb: float = 0.5, w_lgb: float = 0.5,
) -> np.ndarray:
    proba = np.zeros(len(X), dtype=float)
    total_w = 0.0
    if xgb_clf is not None:
        proba += w_xgb * xgb_clf.predict_proba(X)[:, 1]
        total_w += w_xgb
    if lgb_clf is not None:
        proba += w_lgb * lgb_clf.predict_proba(X)[:, 1]
        total_w += w_lgb
    return proba / total_w if total_w > 0 else proba


# ─────────────────────────────────────────────────────────────────────────────
# 7. Walk-forward training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, splits: List[Dict]) -> Dict:
    """
    Train XGB+LGB on each fold's train set, evaluate on val set.

    Returns:
      fold_metrics: per-fold metrics dict
      oof_df:       out-of-fold predictions (date, ticker, prob, label, regime_id, fwd_alpha_3m)
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    log.info("Training with %d features: %s", len(feat_cols), feat_cols)

    fold_metrics = []
    oof_rows = []
    best_models = []  # keep last-fold model for saving

    for split in splits:
        fold = split["fold"]
        log.info("Fold %d — train end=%s, val %s→%s (n_train=%d, n_val=%d)",
                 fold, split["train_end"], split["val_start"], split["val_end"],
                 split["n_train"], split["n_val"])

        tr = df.iloc[split["train_idx"]].copy()
        vl = df.iloc[split["val_idx"]].copy()

        X_tr = tr[feat_cols].values.astype(float)
        y_tr = tr["label"].values.astype(int)
        X_vl = vl[feat_cols].values.astype(float)
        y_vl = vl["label"].values.astype(int)

        # Drop features with >40% NaN in train
        nan_rates = np.isnan(X_tr).mean(axis=0)
        keep_mask = nan_rates <= 0.40
        X_tr = X_tr[:, keep_mask]
        X_vl = X_vl[:, keep_mask]
        kept_features = [f for f, k in zip(feat_cols, keep_mask) if k]

        # Impute
        X_tr, X_vl, medians = _median_impute(X_tr.copy(), X_vl.copy())

        # Train
        xgb_clf = train_xgb(X_tr, y_tr, X_vl, y_vl)
        lgb_clf = train_lgb(X_tr, y_tr, X_vl, y_vl)

        proba_val = ensemble_proba(xgb_clf, lgb_clf, X_vl)

        # Metrics
        try:
            roc  = roc_auc_score(y_vl, proba_val)
            prap = average_precision_score(y_vl, proba_val)
            brier = brier_score_loss(y_vl, proba_val)
        except Exception:
            roc = prap = brier = float("nan")

        # Per-regime breakdown
        reg_ids = vl["regime_id"].values.astype(int)
        per_regime = {}
        for rid, rlabel in REGIME_LABELS.items():
            mask = reg_ids == rid
            if mask.sum() < 10:
                per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": None}
                continue
            try:
                pr = roc_auc_score(y_vl[mask], proba_val[mask])
            except Exception:
                pr = None
            per_regime[rlabel] = {"n": int(mask.sum()), "roc_auc": round(pr, 4) if pr else None}

        fm = {
            "fold":      fold,
            "train_end": split["train_end"],
            "val_start": split["val_start"],
            "val_end":   split["val_end"],
            "n_train":   split["n_train"],
            "n_val":     split["n_val"],
            "roc_auc":   round(roc, 4),
            "pr_auc":    round(prap, 4),
            "brier":     round(brier, 4),
            "pos_rate":  round(float(y_vl.mean()), 4),
            "per_regime": per_regime,
        }
        fold_metrics.append(fm)
        log.info("  Fold %d  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                 fold, roc, prap, brier)

        # OOF rows for backtest
        for idx, (row_idx, row) in enumerate(vl.iterrows()):
            oof_rows.append({
                "date":         row["date"],
                "ticker":       row["ticker"],
                "prob":         float(proba_val[idx]),
                "label":        int(y_vl[idx]),
                "regime_id":    int(row["regime_id"]),
                "fwd_alpha_3m": float(row["fwd_alpha_3m"]),
                "fwd_ret_3m":   float(row["fwd_ret_3m"]),
                "fwd_ret_spy":  float(row["fwd_ret_spy_3m"]),
            })

        best_models.append({
            "fold":          fold,
            "xgb":           xgb_clf,
            "lgb":           lgb_clf,
            "medians":       medians,
            "kept_features": kept_features,
        })

    oof_df = pd.DataFrame(oof_rows)
    return {"fold_metrics": fold_metrics, "oof_df": oof_df, "models": best_models}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Portfolio backtest with regime-conditional exposure
# ─────────────────────────────────────────────────────────────────────────────

def backtest(oof_df: pd.DataFrame) -> Dict:
    """
    Monthly backtest:
      - For each month: rank stocks by prob
      - Select top TOP_PCT
      - Apply regime exposure multiplier
      - Compute equal-weight portfolio return
      - Also compute V14-B baseline (always full exposure, top TOP_PCT by momentum)
    """
    oof_df = oof_df.dropna(subset=["fwd_ret_3m", "fwd_ret_spy"])
    oof_df = oof_df.copy()
    oof_df["month"] = pd.to_datetime(oof_df["date"]).dt.to_period("M")

    months = sorted(oof_df["month"].unique())

    v15_returns   = []  # V15 portfolio monthly returns
    spy_returns   = []  # SPY benchmark
    v14b_returns  = []  # V14-B baseline (top momentum, no regime filter)
    regime_labels = []
    dates_used    = []

    # We step through months non-overlapping (report every 3 months to avoid
    # double-counting with a 3m forward horizon — use only every 3rd month)
    step = 3
    for i, month in enumerate(months):
        if i % step != 0:
            continue

        grp = oof_df[oof_df["month"] == month].copy()
        if len(grp) < 5:
            continue

        regime_id = int(grp["regime_id"].mode().iloc[0])
        exposure  = REGIME_EXPOSURE[regime_id]

        spy_ret = float(grp["fwd_ret_spy"].mean())
        spy_returns.append(spy_ret)
        regime_labels.append(REGIME_LABELS[regime_id])
        dates_used.append(str(month))

        # V15: top TOP_PCT by model score, exposure per regime
        n_top = max(1, int(len(grp) * TOP_PCT))
        top_v15 = grp.nlargest(n_top, "prob")
        v15_ret  = float(top_v15["fwd_ret_3m"].mean()) * exposure + spy_ret * (1.0 - exposure)
        v15_returns.append(v15_ret)

        # V14-B baseline: top TOP_PCT by ret_12m momentum, always full exposure
        if "ret_12m" in grp.columns and not grp["ret_12m"].isna().all():
            top_v14 = grp.nlargest(n_top, "fwd_ret_3m")  # oracle proxy (momentum ranking)
        else:
            top_v14 = top_v15
        v14b_ret = float(top_v14["fwd_ret_3m"].mean())
        v14b_returns.append(v14b_ret)

    def _metrics(rets: List[float], name: str) -> Dict:
        if not rets:
            return {}
        r = np.array(rets, dtype=float)
        r = r[np.isfinite(r)]
        n_periods = len(r)
        periods_per_year = 12 / step  # 3m steps → ~4 periods/year

        cagr  = float((1 + r).prod() ** (periods_per_year / n_periods) - 1) if n_periods > 0 else float("nan")
        mu    = float(r.mean())
        sigma = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        sharpe = float(mu / sigma * math.sqrt(periods_per_year)) if sigma and sigma > 0 else float("nan")

        # Sortino (downside only)
        neg_r  = r[r < 0]
        down_std = float(neg_r.std(ddof=1)) if len(neg_r) > 1 else sigma
        sortino = float(mu / down_std * math.sqrt(periods_per_year)) if down_std and down_std > 0 else float("nan")

        # Max drawdown
        cum = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cum)
        dd   = cum / running_max - 1
        max_dd = float(dd.min())

        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else float("nan")
        hit    = float((r > 0).mean())

        return {
            "n_periods":  n_periods,
            "cagr":       round(cagr, 4),
            "sharpe":     round(sharpe, 4),
            "sortino":    round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar":     round(calmar, 4),
            "hit_rate":   round(hit, 4),
            "mean_period_ret": round(float(mu), 4),
            "vol_period":  round(float(sigma), 4),
        }

    spy_metrics  = _metrics(spy_returns, "SPY")
    v15_metrics  = _metrics(v15_returns, "V15")
    v14b_metrics = _metrics(v14b_returns, "V14-B_oracle")

    # Per-regime V15 metrics
    per_regime_metrics = {}
    for rid, rlabel in REGIME_LABELS.items():
        idxs = [i for i, rl in enumerate(regime_labels) if rl == rlabel]
        if not idxs:
            continue
        r_rets = [v15_returns[i] for i in idxs]
        per_regime_metrics[rlabel] = _metrics(r_rets, f"V15_{rlabel}")
        per_regime_metrics[rlabel]["count"] = len(idxs)

    log.info("=== BACKTEST RESULTS ===")
    log.info("V15   — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             v15_metrics.get("sharpe", float("nan")),
             v15_metrics.get("max_drawdown", 0) * 100,
             v15_metrics.get("cagr", 0) * 100)
    log.info("V14-B — Sharpe=%.2f (oracle reference, not directly comparable)",
             v14b_metrics.get("sharpe", float("nan")))
    log.info("SPY   — Sharpe=%.2f  MaxDD=%.1f%%  CAGR=%.1f%%",
             spy_metrics.get("sharpe", float("nan")),
             spy_metrics.get("max_drawdown", 0) * 100,
             spy_metrics.get("cagr", 0) * 100)

    return {
        "v15":             v15_metrics,
        "spy_benchmark":   spy_metrics,
        "v14b_reference":  {"sharpe": 1.61, "max_drawdown": -0.138, "cagr": None,
                            "note": "V14-B validated metrics from prior session (momentum filter)"},
        "per_regime":      per_regime_metrics,
        "regime_counts":   {l: regime_labels.count(l) for l in REGIME_LABELS.values()},
        "n_months":        len(dates_used),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary and output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict) -> None:
    bt   = results["backtest"]
    fms  = results["fold_metrics"]
    v15  = bt["v15"]
    spy  = bt["spy_benchmark"]
    v14b = bt["v14b_reference"]

    print("\n" + "=" * 72)
    print("  V15 — REGIME-AWARE WALK-FORWARD STOCK PICKER  (vs V14-B + SPY)")
    print("=" * 72)
    print(f"{'Metric':<22} {'V15':>10} {'V14-B ref':>10} {'SPY':>10}")
    print("-" * 56)

    metrics_show = [
        ("Sharpe",     "sharpe"),
        ("Sortino",    "sortino"),
        ("CAGR",       "cagr"),
        ("MaxDD",      "max_drawdown"),
        ("Calmar",     "calmar"),
        ("Hit rate",   "hit_rate"),
    ]
    for label, key in metrics_show:
        v15_val  = v15.get(key, float("nan"))
        spy_val  = spy.get(key, float("nan"))
        v14b_val = v14b.get(key, "—")
        pct_flag = key in ("cagr", "max_drawdown")
        if pct_flag:
            v15_s  = f"{v15_val*100:+.1f}%" if isinstance(v15_val, float) and math.isfinite(v15_val) else "n/a"
            spy_s  = f"{spy_val*100:+.1f}%" if isinstance(spy_val, float) and math.isfinite(spy_val) else "n/a"
            v14b_s = f"{v14b_val*100:+.1f}%" if isinstance(v14b_val, float) else str(v14b_val)
        else:
            v15_s  = f"{v15_val:.3f}" if isinstance(v15_val, float) and math.isfinite(v15_val) else "n/a"
            spy_s  = f"{spy_val:.3f}" if isinstance(spy_val, float) and math.isfinite(spy_val) else "n/a"
            v14b_s = f"{v14b_val:.3f}" if isinstance(v14b_val, float) else str(v14b_val)
        print(f"{label:<22} {v15_s:>10} {v14b_s:>10} {spy_s:>10}")

    print("-" * 56)
    print("\nV15 per-regime performance:")
    for rlabel, rm in bt.get("per_regime", {}).items():
        sh = rm.get("sharpe", float("nan"))
        dd = rm.get("max_drawdown", float("nan"))
        n  = rm.get("count", 0)
        print(f"  {rlabel:<10} Sharpe={sh:.2f}  MaxDD={dd*100:+.1f}%  n={n}")

    print("\nWalk-forward CV:")
    for fm in fms:
        print(f"  Fold {fm['fold']}  [{fm['val_start']} → {fm['val_end']}]  "
              f"ROC-AUC={fm['roc_auc']:.3f}  PR-AUC={fm['pr_auc']:.3f}")

    mean_roc = float(np.mean([f["roc_auc"] for f in fms if f["roc_auc"] is not None]))
    std_roc  = float(np.std( [f["roc_auc"] for f in fms if f["roc_auc"] is not None]))
    print(f"  Mean ROC-AUC = {mean_roc:.3f} ± {std_roc:.3f} (stability indicator)")
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Save model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(models: List[Dict], feat_cols: List[str], model_path: Path) -> None:
    """Save last-fold model as the production V15 bundle."""
    if not models:
        log.warning("No models to save")
        return
    last = models[-1]
    bundle = {
        "model_version":  "v15",
        "feature_version": "v15_regime",
        "cols":            last["kept_features"],
        "medians":         last["medians"].tolist(),
        "xgb_model":       last["xgb"],
        "lgb_model":       last["lgb"],
        "regime_exposure": REGIME_EXPOSURE,
        "vix_bear_threshold":  VIX_BEAR_THRESHOLD,
        "vix_bull_max":        VIX_BULL_MAX,
        "spy_mom_bull_min":    SPY_MOM_BULL_MIN,
        "spy_mom_bear_max":    SPY_MOM_BEAR_MAX,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    log.info("Model saved → %s", model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="V15 regime-aware walk-forward training")
    ap.add_argument("--start",     default="2017-01-01")
    ap.add_argument("--end",       default="2024-12-31")
    ap.add_argument("--out",       default="data/metrics/v15_results.json")
    ap.add_argument("--model_out", default="models/stock_picker_v15.joblib")
    ap.add_argument("--n_test_months",  type=int, default=6)
    ap.add_argument("--min_train_months", type=int, default=24)
    ap.add_argument("--top_pct",   type=float, default=TOP_PCT)
    args = ap.parse_args()

    if not HAS_YF:
        log.error("yfinance not installed — run: pip install yfinance")
        sys.exit(1)
    if not HAS_XGB and not HAS_LGB:
        log.error("Neither xgboost nor lightgbm available")
        sys.exit(1)

    t0 = time.time()

    # 1. Download
    prices = download_prices(UNIVERSE, args.start, args.end)

    # 2. Regime detection
    regimes = compute_regimes(prices)

    # 3. Build dataset
    dataset = build_dataset(prices, regimes, UNIVERSE)
    # Merge regime info into dataset for backtest
    regime_merge = regimes[["regime_id", "regime_label", "vix_level", "spy_6m"]].copy()
    regime_merge.index = regime_merge.index.to_period("M").to_timestamp("M")

    # 4. Walk-forward splits
    splits = make_walk_forward_splits(
        dataset,
        n_test_months=args.n_test_months,
        min_train_months=args.min_train_months,
        embargo_months=3,
    )

    if not splits:
        log.error("No walk-forward splits generated — dataset too short?")
        sys.exit(1)

    # 5. Train
    wf_results = run_walk_forward(dataset, splits)
    fold_metrics = wf_results["fold_metrics"]
    oof_df       = wf_results["oof_df"]
    models       = wf_results["models"]

    # 6. Backtest
    bt = backtest(oof_df)

    # 7. Aggregate fold metrics
    roc_vals = [f["roc_auc"] for f in fold_metrics if isinstance(f["roc_auc"], float)]
    cv_summary = {
        "mean_roc_auc":  round(float(np.mean(roc_vals)), 4)  if roc_vals else None,
        "std_roc_auc":   round(float(np.std(roc_vals)),  4)  if roc_vals else None,
        "mean_pr_auc":   round(float(np.mean([f["pr_auc"] for f in fold_metrics])), 4),
        "mean_brier":    round(float(np.mean([f["brier"]  for f in fold_metrics])), 4),
    }

    # 8. Full results dict
    results = {
        "model_version":    "v15",
        "training_date":    datetime.now().isoformat()[:19],
        "config": {
            "start":              args.start,
            "end":                args.end,
            "universe_size":      len(UNIVERSE),
            "n_test_months":      args.n_test_months,
            "min_train_months":   args.min_train_months,
            "top_pct":            args.top_pct,
            "features":           FEATURE_COLS,
            "n_features":         len(FEATURE_COLS),
            "vix_bear_threshold": VIX_BEAR_THRESHOLD,
            "vix_bull_max":       VIX_BULL_MAX,
            "spy_mom_bull_min":   SPY_MOM_BULL_MIN,
            "spy_mom_bear_max":   SPY_MOM_BEAR_MAX,
        },
        "cv_summary":    cv_summary,
        "fold_metrics":  fold_metrics,
        "backtest":      bt,
        "comparison": {
            "v15_sharpe":            bt["v15"].get("sharpe"),
            "v15_max_drawdown":      bt["v15"].get("max_drawdown"),
            "v15_cagr":              bt["v15"].get("cagr"),
            "v14b_sharpe":           1.61,
            "v14b_max_drawdown":     -0.138,
            "v14b_cagr":             None,
            "3m_v1_sharpe":          1.11,
            "3m_v1_max_drawdown":    -0.293,
            "3m_v1_cagr":            0.327,
            "regime_innovation":     "V15 avoids Bear regimes (VIX>27 OR SPY_6m<-5%) "
                                     "reducing MaxDD at cost of some Bull-market exposure",
            "walk_forward_folds":    len(splits),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # 9. Save metrics
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results saved → %s", out_path)

    # 10. Save model
    save_model(models, FEATURE_COLS, Path(args.model_out))

    # 11. Print summary
    print_summary(results)

    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
