"""
tests/unit/test_preflight_v3.py
================================
Tests for preflight_v3_api.py invariants and supporting helpers.

Covers:
  1. Threshold monotonicity (t_lo < t_hi)
  2. Feature contract (count + order)
  3. No NaN after frozen imputation
  4. Proba range [0, 1]
  5. Label coherence with thresholds
  6. Recovery sentinel not NaN before imputation
  7. recovery_defined binary constraint
  8. Backtest MaxDD suppressed for cross-sectional dataset
  9. Preflight passes on real models/v3/ artefacts (integration, skipped if absent)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.ml.validation.preflight_v3_api import (
    InvariantError,
    _check,
    apply_frozen_imputation,
    build_X,
    check_feature_contract,
    check_label_coherence,
    check_no_nan_after_imputation,
    check_proba_range,
    check_recovery_defined_binary,
    check_sentinel_not_nan,
    check_threshold_monotonicity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEAT_NAMES_50 = [
    "autocorr_1", "autocorr_5", "bb_distance", "dd_duration", "dd_duration_per_n",
    "dd_to_var99", "downside_dev", "downside_div_vol", "es95", "es95_var95",
    "es99", "es99_es95", "es99_var99", "hill_tail_index", "jump_indicator",
    "kurtosis_excess", "log_n_used", "macd_hist", "max_dd", "max_drawdown",
    "missing_pct", "n_used", "rsi", "rsi_centered", "semivariance",
    "skew", "sma_slope_20", "sma_slope_60", "stress_multiplier", "stress_var99",
    "tail_obs_99", "tuw_pct", "var95", "var99", "var99_var95",
    "vol120_vol_ann", "vol20_vol60", "vol20_vol_ann", "vol60_vol_ann", "vol_120d",
    "vol_20d", "vol_60d", "vol_ann", "vol_ewma_ann", "vol_of_vol",
    "vol_to_var95", "worst_10d_ret", "worst_20d_ret", "worst_5d_ret", "worst_5d_vs_var99",
]


def _make_meta(feat_names: list[str] | None = None) -> dict[str, Any]:
    names = feat_names if feat_names is not None else _FEAT_NAMES_50
    return {
        "schema_version": "3.1",
        "n_features": len(names),
        "feature_cols": list(names),
        "medians": {f: 0.5 for f in names},
    }


def _make_record(extra_feats: dict | None = None) -> dict:
    """Minimal v3 JSONL record with all 50 active features populated."""
    feats: dict[str, Any] = {
        "ticker":       "TEST",
        "asset_type":   "equity",
        "market":       "US",
        "market_proxy": "SPY",
        # Sentinel features
        "dd_duration":       -1.0,
        "recovery_defined":   0.0,
    }
    # Fill all 50 active features with harmless floats
    for f in _FEAT_NAMES_50:
        if f not in feats:
            feats[f] = 0.1
    if extra_feats:
        feats.update(extra_feats)
    return {
        "version": "v3",
        "label": "ok",
        "target_non_ok": 0,
        "window_end_date": "2020-01-01",
        "features": feats,
    }


# ===========================================================================
# 1. Threshold monotonicity
# ===========================================================================

class TestThresholdMonotonicity:

    def test_valid_thresholds_pass(self):
        check_threshold_monotonicity({"t_lo": 0.4, "t_hi": 0.6})

    def test_t_lo_equals_t_hi_fails(self):
        with pytest.raises(InvariantError, match="monotonicity"):
            check_threshold_monotonicity({"t_lo": 0.5, "t_hi": 0.5})

    def test_t_lo_greater_than_t_hi_fails(self):
        with pytest.raises(InvariantError, match="monotonicity"):
            check_threshold_monotonicity({"t_lo": 0.7, "t_hi": 0.4})

    def test_t_lo_zero_fails(self):
        with pytest.raises(InvariantError, match="monotonicity"):
            check_threshold_monotonicity({"t_lo": 0.0, "t_hi": 0.6})

    def test_t_hi_one_fails(self):
        with pytest.raises(InvariantError, match="monotonicity"):
            check_threshold_monotonicity({"t_lo": 0.4, "t_hi": 1.0})

    def test_production_thresholds_pass(self):
        """Confirm current production values (t_lo=0.4863, t_hi=0.6654) are valid."""
        check_threshold_monotonicity({"t_lo": 0.4863, "t_hi": 0.6654})


# ===========================================================================
# 2. Feature contract (count + order)
# ===========================================================================

class TestFeatureContract:

    def test_matching_contract_passes(self):
        meta = _make_meta(_FEAT_NAMES_50)
        check_feature_contract(_FEAT_NAMES_50, meta)

    def test_wrong_count_fails(self):
        meta = _make_meta(_FEAT_NAMES_50[:49])
        with pytest.raises(InvariantError, match="count mismatch"):
            check_feature_contract(_FEAT_NAMES_50, meta)

    def test_wrong_order_fails(self):
        reordered = list(reversed(_FEAT_NAMES_50))
        meta = _make_meta(_FEAT_NAMES_50)
        with pytest.raises(InvariantError, match="order mismatch"):
            check_feature_contract(reordered, meta)

    def test_n_features_mismatch_fails(self):
        meta = _make_meta(_FEAT_NAMES_50)
        meta["n_features"] = 99  # wrong
        with pytest.raises(InvariantError, match="n_features"):
            check_feature_contract(_FEAT_NAMES_50, meta)


# ===========================================================================
# 3. No NaN after frozen imputation
# ===========================================================================

class TestFrozenImputation:

    def test_nan_filled_with_median(self):
        feat_names = ["vol_ann", "var99"]
        medians    = {"vol_ann": 0.25, "var99": 0.04}
        X = np.array([[np.nan, 0.04], [0.30, np.nan]], dtype=float)
        X_out = apply_frozen_imputation(X, feat_names, medians)
        assert np.all(np.isfinite(X_out))
        assert X_out[0, 0] == pytest.approx(0.25)
        assert X_out[1, 1] == pytest.approx(0.04)

    def test_no_nan_passes_invariant(self):
        X = np.ones((5, 3), dtype=float)
        check_no_nan_after_imputation(X, ["a", "b", "c"])

    def test_nan_fails_invariant(self):
        X = np.ones((5, 3), dtype=float)
        X[2, 1] = np.nan
        with pytest.raises(InvariantError, match="I-1"):
            check_no_nan_after_imputation(X, ["a", "b", "c"])

    def test_inf_fails_invariant(self):
        X = np.ones((3, 2), dtype=float)
        X[0, 0] = np.inf
        with pytest.raises(InvariantError, match="I-1"):
            check_no_nan_after_imputation(X, ["x", "y"])

    def test_missing_median_falls_back_to_zero(self):
        feat_names = ["exotic_feat"]
        medians    = {}  # no entry
        X = np.array([[np.nan]], dtype=float)
        X_out = apply_frozen_imputation(X, feat_names, medians)
        assert X_out[0, 0] == pytest.approx(0.0)


# ===========================================================================
# 4. Proba range [0, 1]
# ===========================================================================

class TestProbaRange:

    def test_valid_probas_pass(self):
        check_proba_range(np.array([0.0, 0.3, 0.7, 1.0]))

    def test_negative_proba_fails(self):
        with pytest.raises(InvariantError, match="I-2"):
            check_proba_range(np.array([0.5, -0.01]))

    def test_proba_above_one_fails(self):
        with pytest.raises(InvariantError, match="I-2"):
            check_proba_range(np.array([0.5, 1.001]))


# ===========================================================================
# 5. Label coherence with thresholds
# ===========================================================================

class TestLabelCoherence:

    def test_coherent_labels_pass(self):
        t_lo, t_hi = 0.40, 0.65
        proba  = np.array([0.10, 0.50, 0.80])
        labels = np.array(["ok", "warn", "block"])
        check_label_coherence(proba, labels, t_lo, t_hi)

    def test_incoherent_label_fails(self):
        t_lo, t_hi = 0.40, 0.65
        proba  = np.array([0.10, 0.50, 0.80])
        labels = np.array(["ok", "ok", "block"])   # middle should be "warn"
        with pytest.raises(InvariantError, match="I-3"):
            check_label_coherence(proba, labels, t_lo, t_hi)

    def test_boundary_exactly_at_t_lo(self):
        t_lo, t_hi = 0.40, 0.65
        proba  = np.array([0.40])
        labels = np.array(["warn"])   # >= t_lo → warn
        check_label_coherence(proba, labels, t_lo, t_hi)

    def test_boundary_exactly_at_t_hi(self):
        t_lo, t_hi = 0.40, 0.65
        proba  = np.array([0.65])
        labels = np.array(["block"])   # >= t_hi → block
        check_label_coherence(proba, labels, t_lo, t_hi)


# ===========================================================================
# 6. Sentinel features — not NaN before imputation
# ===========================================================================

class TestSentinelNotNaN:

    def test_sentinel_minus_one_ok(self):
        feat_names = ["dd_duration"]
        X = np.array([[-1.0], [-1.0], [30.0]])
        check_sentinel_not_nan(X, feat_names)

    def test_sentinel_nan_fails(self):
        feat_names = ["dd_duration"]
        X = np.array([[np.nan]])
        with pytest.raises(InvariantError, match="I-7"):
            check_sentinel_not_nan(X, feat_names)

    def test_non_sentinel_feature_ignored(self):
        # vol_ann is not a sentinel feature — NaN is allowed here before imputation
        feat_names = ["vol_ann"]
        X = np.array([[np.nan]])
        check_sentinel_not_nan(X, feat_names)  # should not raise


# ===========================================================================
# 7. recovery_defined binary constraint
# ===========================================================================

class TestRecoveryDefinedBinary:

    def test_zero_and_one_pass(self):
        feat_names = ["recovery_defined"]
        X = np.array([[0.0], [1.0], [0.0]])
        check_recovery_defined_binary(X, feat_names)

    def test_nan_fails(self):
        feat_names = ["recovery_defined"]
        X = np.array([[np.nan]])
        with pytest.raises(InvariantError, match="I-8"):
            check_recovery_defined_binary(X, feat_names)

    def test_non_binary_value_fails(self):
        feat_names = ["recovery_defined"]
        X = np.array([[0.5]])
        with pytest.raises(InvariantError, match="I-8"):
            check_recovery_defined_binary(X, feat_names)

    def test_feature_absent_passes(self):
        feat_names = ["vol_ann"]  # recovery_defined not in list
        X = np.array([[0.5]])
        check_recovery_defined_binary(X, feat_names)  # no-op


# ===========================================================================
# 8. Backtest MaxDD suppressed for cross-sectional dataset
# ===========================================================================

class TestBacktestCrossSectional:
    """Verify that compute_metrics suppresses MaxDD when n_years > threshold."""

    @staticmethod
    def _compute(returns: list[float], periods_per_year: float = 12.6) -> dict:
        """Call compute_metrics as the backtest would."""
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "backtest_signal_v3",
            str(_REPO / "scripts" / "ml" / "backtest_signal_v3.py"),
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        arr = np.array(returns, dtype=float)
        return mod.compute_metrics(arr, label="test", periods_per_year=periods_per_year)

    def test_single_asset_has_max_drawdown(self):
        """With few periods (< 30 years), MaxDD is computed."""
        # 12 periods → 12/12.6 ≈ 0.95 years
        rets = [0.01, -0.02, 0.015, -0.005, 0.01, 0.02,
                0.01, -0.01,  0.01,  0.005, 0.02, -0.01]
        m = self._compute(rets, periods_per_year=12.6)
        assert m.get("cross_sectional") is False
        assert m.get("max_drawdown") is not None
        assert isinstance(m["max_drawdown"], float)

    def test_cross_sectional_suppresses_max_drawdown(self):
        """With many periods (> 30 years equivalent), MaxDD is None."""
        # 400 periods at 12.6/yr → 31.7 years > threshold
        rets = [0.01] * 400
        m = self._compute(rets, periods_per_year=12.6)
        assert m.get("cross_sectional") is True
        assert m.get("max_drawdown") is None
        assert m.get("calmar") is None

    def test_cross_sectional_sharpe_still_computed(self):
        """Sharpe/Sortino should be valid even in cross-sectional mode.
        Use a return series with variance so Sharpe != None."""
        rng_local = np.random.default_rng(0)
        rets = (rng_local.normal(0.005, 0.02, 400)).tolist()  # 400 periods, non-zero vol
        m = self._compute(rets, periods_per_year=12.6)
        assert m.get("cross_sectional") is True
        assert m.get("max_drawdown") is None
        assert m.get("sharpe_ann") is not None
        assert math.isfinite(float(m["sharpe_ann"]))


# ===========================================================================
# 9. Integration — real models/v3/ artefacts
# ===========================================================================

_MODELS_DIR = _REPO / "models" / "v3"
_VAL_FILE   = _REPO / "data" / "training" / "v3" / "fold_5" / "val.jsonl"

@pytest.mark.skipif(
    not (_MODELS_DIR / "v3_meta.json").exists(),
    reason="models/v3/ not present — skipping integration preflight",
)
class TestPreflightIntegration:

    def test_preflight_passes(self):
        """End-to-end: preflight must return True (all invariants satisfied)."""
        from scripts.ml.validation.preflight_v3_api import run_preflight
        result = run_preflight(
            models_dir=_MODELS_DIR,
            val_file=_VAL_FILE,
            n_sample=200,
            seed=42,
        )
        assert result is True

    def test_preflight_threshold_monotonicity(self):
        """Production thresholds must satisfy t_lo < t_hi."""
        thresholds = json.loads((_MODELS_DIR / "v3_thresholds.json").read_text())
        check_threshold_monotonicity(thresholds)

    def test_preflight_feature_count(self):
        """Feature list must have exactly 50 entries."""
        import joblib
        feat_names = joblib.load(_MODELS_DIR / "v3_feature_names.joblib")
        assert len(feat_names) == 50

    def test_preflight_medians_cover_all_features(self):
        """Every active feature must have a stored median."""
        import joblib
        meta       = json.loads((_MODELS_DIR / "v3_meta.json").read_text())
        feat_names = joblib.load(_MODELS_DIR / "v3_feature_names.joblib")
        medians    = meta["medians"]
        missing    = [f for f in feat_names if f not in medians]
        assert missing == [], f"Missing medians for: {missing}"
