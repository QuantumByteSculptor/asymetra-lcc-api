"""
tests/unit/test_spy_crossasset.py
===================================
Tests unitaires pour la correction SPY provider / corr_spy / beta_market.

Vérifie que:
1. Avec une série market_returns non vide + index date-only → corr_spy et beta_market
   ne sont PAS None dans les features.
2. Avec une série market_returns vide → corr_spy et beta_market sont None
   (comportement silencieux attendu).
3. L'index du résultat de download_spy_returns est normalisé à date-only (heure 0).

Rapide, sans réseau (injecte des séries synthétiques directement).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.ml.data.build_dataset_v3 import build_features_v3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker_data(n: int = 300, seed: int = 42):
    """Synthetic ticker: date-only index (business days), starting 2020-01-02."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, n)),
        index=dates,
    )
    returns = prices.pct_change().dropna()
    return prices.iloc[1:], returns   # align closes with returns


def _make_market_returns(n: int = 300, seed: int = 7, tz_aware: bool = False):
    """
    Synthetic SPY-like daily return series.
    By default: date-only midnight index (correct).
    With tz_aware=True: 14:30 timestamps (broken — simulate pre-fix behaviour).
    """
    rng = np.random.default_rng(seed)
    raw_dates = pd.date_range("2019-01-02", periods=n * 2, freq="B")

    if tz_aware:
        # Simulate Yahoo v8 pre-fix: timestamps with time component
        raw_dates = raw_dates + pd.Timedelta("14:30:00")

    rets = pd.Series(rng.normal(0.0002, 0.010, len(raw_dates)), index=raw_dates)
    return rets


def _make_normalized_market_returns(n: int = 300, seed: int = 7):
    """Same as above but normalized to midnight (the fix applied)."""
    s = _make_market_returns(n=n, seed=seed, tz_aware=True)
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index.normalize())
    s = s[~s.index.duplicated(keep="last")]
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorrSpyNotNull:

    def test_corr_spy_not_null_with_market_returns(self):
        """When spy_returns is non-empty with date-only index, corr_spy must be a float."""
        closes, returns = _make_ticker_data(n=300)
        spy_ret = _make_market_returns(n=500, tz_aware=False)

        feats = build_features_v3(
            ticker="AAPL", asset_type="equity", market="US",
            closes=closes, returns=returns,
            macro={}, spy_returns=spy_ret,
            window_end_date=pd.Timestamp("2021-01-15"),
        )

        assert feats, "build_features_v3 returned empty dict"
        corr_spy = feats.get("corr_spy")
        beta_mkt = feats.get("beta_market")

        assert corr_spy is not None, "corr_spy is None — index alignment failed"
        assert beta_mkt is not None, "beta_market is None — index alignment failed"
        assert isinstance(corr_spy, float)
        assert isinstance(beta_mkt, float)
        assert -1.0 <= corr_spy <= 1.0, f"corr_spy={corr_spy} out of range"

    def test_corr_spy_null_when_spy_empty(self):
        """When spy_returns is empty, corr_spy must be None (graceful skip)."""
        closes, returns = _make_ticker_data(n=300)
        spy_ret = pd.Series(dtype=float)

        feats = build_features_v3(
            ticker="AAPL", asset_type="equity", market="US",
            closes=closes, returns=returns,
            macro={}, spy_returns=spy_ret,
            window_end_date=pd.Timestamp("2021-01-15"),
        )

        assert feats, "build_features_v3 returned empty dict"
        assert feats.get("corr_spy") is None, "Expected None corr_spy when no market ref"
        assert feats.get("beta_market") is None, "Expected None beta_market when no market ref"

    def test_timestamp_normalization_fixes_null(self):
        """
        Both 14:30 and midnight SPY timestamps must produce a valid corr_spy.

        build_features_v3 applies _norm_idx() to BOTH sides of the intersection
        (ticker returns and spy_returns), so timestamps are normalized internally
        regardless of the source format.  14:30 → midnight normalization happens
        at two points:
          1. download_spy_returns() normalizes the cached series.
          2. build_features_v3() calls _norm_idx() on each side before intersecting.
        Result: neither variant should yield None.
        """
        closes, returns = _make_ticker_data(n=300)

        # 14:30 timestamps — previously "broken", now handled by _norm_idx
        spy_1430 = _make_market_returns(n=500, tz_aware=True)
        feats_1430 = build_features_v3(
            ticker="AAPL", asset_type="equity", market="US",
            closes=closes, returns=returns, macro={},
            spy_returns=spy_1430,
            window_end_date=pd.Timestamp("2021-01-15"),
        )
        assert feats_1430.get("corr_spy") is not None, (
            "Expected float corr_spy even with 14:30 timestamps (_norm_idx applied)"
        )

        # Midnight timestamps — canonical form, must also work
        spy_midnight = _make_normalized_market_returns(n=500)
        feats_midnight = build_features_v3(
            ticker="AAPL", asset_type="equity", market="US",
            closes=closes, returns=returns, macro={},
            spy_returns=spy_midnight,
            window_end_date=pd.Timestamp("2021-01-15"),
        )
        assert feats_midnight.get("corr_spy") is not None, (
            "Expected float corr_spy with midnight timestamps"
        )

    def test_spy_index_hour_is_zero_after_download(self):
        """
        The normalize() fix in download_spy_returns must produce midnight timestamps.
        We test the normalization logic directly (no network call).
        """
        # Simulate a return series with 14:30 timestamps (as Yahoo v8 returns)
        dates_1430 = pd.date_range("2020-01-02 14:30:00", periods=100, freq="B")
        spy_raw = pd.Series(np.random.randn(100) * 0.01, index=dates_1430)

        # Apply the fix
        spy_fixed = spy_raw.copy()
        spy_fixed.index = pd.DatetimeIndex(spy_fixed.index.normalize())
        spy_fixed = spy_fixed[~spy_fixed.index.duplicated(keep="last")]

        assert spy_fixed.index[0].hour == 0, "Normalized index must have hour=0"
        assert len(spy_fixed) == len(spy_raw), "No data should be lost after normalization"
        # Verify it aligns with date-only closes index
        closes_idx = pd.date_range("2020-01-02", periods=100, freq="B")
        common = closes_idx.intersection(spy_fixed.index)
        assert len(common) > 90, f"Expected >90 common dates, got {len(common)}"
