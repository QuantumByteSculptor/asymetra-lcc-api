# tests/unit/test_features.py
"""Unit tests for features.py feature pipeline."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from features import (
    DEFAULT_CONFIG,
    DERIVED,
    NUMERIC_BASE,
    compute_derived,
    features_to_row,
    vector_columns,
)


# ---------------------------------------------------------------------------
# Column counts
# ---------------------------------------------------------------------------

class TestColumnCounts:
    def test_numeric_base_count(self):
        assert len(NUMERIC_BASE) == 33

    def test_derived_count(self):
        assert len(DERIVED) == 22

    def test_total_columns(self):
        cols = vector_columns(DEFAULT_CONFIG)
        # 33 numeric + 22 derived + 6 asset_types + 30 markets = 91
        assert len(cols) == 91

    def test_version_v2(self):
        assert DEFAULT_CONFIG.version == "v2"

    def test_new_features_in_numeric_base(self):
        new_v2 = [
            "downside_dev", "semivariance", "vol_of_vol",
            "worst_5d_ret", "worst_20d_ret", "autocorr_1",
            "vol_ewma_ann", "stress_var99", "stress_multiplier",
        ]
        for f in new_v2:
            assert f in NUMERIC_BASE, f"{f} missing from NUMERIC_BASE"

    def test_new_derived_features(self):
        assert "downside_div_vol" in DERIVED
        assert "worst_5d_vs_var99" in DERIVED


# ---------------------------------------------------------------------------
# features_to_row
# ---------------------------------------------------------------------------

class TestFeaturesToRow:
    def test_returns_all_columns(self, base_feats):
        cols = vector_columns(DEFAULT_CONFIG)
        row = features_to_row(base_feats, DEFAULT_CONFIG)
        assert set(cols) == set(row.keys()), "row must have exactly the expected columns"

    def test_alias_max_drawdown(self):
        feats = {
            "asset_type": "etf", "market": "US",
            "max_drawdown": -0.10,
            "var99": 0.02,
        }
        row = features_to_row(feats, DEFAULT_CONFIG)
        assert math.isfinite(row["max_dd"])
        assert abs(row["max_dd"] - (-0.10)) < 1e-9

    def test_one_hot_asset_type(self, base_feats):
        row = features_to_row(base_feats, DEFAULT_CONFIG)
        assert row["asset_type__equity"] == 1.0
        assert row["asset_type__etf"] == 0.0

    def test_one_hot_market(self, base_feats):
        row = features_to_row(base_feats, DEFAULT_CONFIG)
        assert row["market__US"] == 1.0
        assert row["market__EU"] == 0.0

    def test_nan_on_missing_numeric(self, base_feats):
        feats = dict(base_feats)
        feats.pop("vol_ann", None)
        row = features_to_row(feats, DEFAULT_CONFIG)
        assert not math.isfinite(row["vol_ann"])

    def test_v2_features_in_row(self, v2_feats):
        row = features_to_row(v2_feats, DEFAULT_CONFIG)
        assert math.isfinite(row["downside_dev"])
        assert math.isfinite(row["worst_5d_ret"])
        assert math.isfinite(row["vol_ewma_ann"])


# ---------------------------------------------------------------------------
# compute_derived
# ---------------------------------------------------------------------------

class TestComputeDerived:
    def test_var99_div_var95(self, base_feats):
        derived = compute_derived(base_feats)
        expected = 0.025 / 0.016
        assert abs(derived["var99_div_var95"] - expected) < 1e-6

    def test_vol20_to_volann(self, base_feats):
        derived = compute_derived(base_feats)
        expected = 0.015 / 0.18
        assert abs(derived["vol20_to_volann"] - expected) < 1e-6

    def test_downside_div_vol(self, v2_feats):
        derived = compute_derived(v2_feats)
        expected = v2_feats["downside_dev"] / v2_feats["vol_ann"]
        assert abs(derived["downside_div_vol"] - expected) < 1e-6

    def test_worst_5d_vs_var99(self, v2_feats):
        derived = compute_derived(v2_feats)
        expected = abs(v2_feats["worst_5d_ret"]) / v2_feats["var99"]
        assert abs(derived["worst_5d_vs_var99"] - expected) < 1e-6

    def test_nan_propagation(self):
        feats: Dict[str, Any] = {"vol_ann": None, "var95": None}
        derived = compute_derived(feats)
        assert not math.isfinite(derived["vol_to_var95_scaled"])

    def test_recovery_per_dd_dur_with_zero_duration(self):
        feats: Dict[str, Any] = {
            "dd_duration": 0.0, "recovery_days": 10.0
        }
        derived = compute_derived(feats)
        assert not math.isfinite(derived["recovery_per_dd_dur"])


# ---------------------------------------------------------------------------
# feature_utils functions
# ---------------------------------------------------------------------------

class TestFeatureUtils:
    def test_compute_downside_dev(self, sample_returns):
        from feature_utils import compute_downside_dev
        v = compute_downside_dev(sample_returns)
        assert math.isfinite(v) and v > 0

    def test_compute_vol_of_vol(self, sample_returns):
        from feature_utils import compute_vol_of_vol
        v = compute_vol_of_vol(sample_returns)
        assert math.isfinite(v) and v > 0

    def test_compute_worst_rolling_return(self, sample_returns):
        from feature_utils import compute_worst_rolling_return
        v = compute_worst_rolling_return(sample_returns, 5)
        assert math.isfinite(v) and v <= 0  # worst return is negative

    def test_compute_autocorr(self, sample_returns):
        from feature_utils import compute_autocorr
        v = compute_autocorr(sample_returns)
        assert math.isfinite(v) and -1.0 <= v <= 1.0

    def test_compute_dd_duration_recovery(self, sample_prices):
        from feature_utils import compute_dd_duration_recovery
        dur, rec = compute_dd_duration_recovery(sample_prices)
        assert dur >= 0
        assert rec >= 0

    def test_compute_all_v2_features_keys(self, sample_returns, sample_prices):
        from feature_utils import compute_all_v2_features
        result = compute_all_v2_features(sample_returns, sample_prices, base_var99=0.02)
        expected_keys = {
            "downside_dev", "semivariance", "vol_of_vol",
            "worst_5d_ret", "worst_20d_ret", "autocorr_1", "vol_ewma_ann",
            "stress_var99", "stress_multiplier", "stress_cumret",
            "dd_duration", "recovery_days",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_too_few_returns_returns_nan(self):
        from feature_utils import compute_downside_dev
        v = compute_downside_dev(np.array([0.01, -0.02]))
        assert not math.isfinite(v)
