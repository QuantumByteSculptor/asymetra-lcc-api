# tests/conftest.py
"""Shared fixtures for unit and integration tests."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def base_feats() -> Dict[str, Any]:
    """Minimal valid features dict (v1-compatible)."""
    return {
        "asset_type": "equity",
        "market": "US",
        "vol_ann": 0.18,
        "vol_20d": 0.015,
        "var95": 0.016,
        "var99": 0.025,
        "es95": 0.022,
        "es99": 0.031,
        "max_dd": -0.12,
        "max_drawdown": -0.12,
        "tuw_pct": 95.0,
        "n_used": 252,
        "missing_pct": 0.0,
        "tail_obs_99": 3,
        "corr_mkt": 0.1,
        "rsi": 48.0,
        "raw_if": -0.1,
        "raw_lof": -0.15,
        "z_if": 0.3,
        "z_lof": 0.2,
        "z_gap_if_lof": 0.1,
    }


@pytest.fixture
def v2_feats(base_feats: Dict[str, Any]) -> Dict[str, Any]:
    """Features dict with all v2 fields populated."""
    v2 = dict(base_feats)
    v2.update({
        "vol_60d": 0.17,
        "vol_120d": 0.16,
        "skew": -0.3,
        "kurtosis_excess": 1.2,
        "dd_duration": 45,
        "recovery_days": 30,
        "downside_dev": 0.14,
        "semivariance": 0.02,
        "vol_of_vol": 0.25,
        "worst_5d_ret": -0.04,
        "worst_20d_ret": -0.09,
        "autocorr_1": 0.05,
        "vol_ewma_ann": 0.19,
        "stress_var99": 0.035,
        "stress_multiplier": 1.4,
    })
    return v2


@pytest.fixture
def sample_returns() -> np.ndarray:
    np.random.seed(42)
    return np.random.normal(0, 0.01, 260)


@pytest.fixture
def sample_prices(sample_returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + sample_returns) * 100.0
