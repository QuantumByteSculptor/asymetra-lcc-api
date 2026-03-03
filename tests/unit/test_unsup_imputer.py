"""
tests/unit/test_unsup_imputer.py
=================================
Regression test for:
  Bug  : _unsup_vector_numpy pre-converts NaN→0.0 before SimpleImputer.transform(),
         bypassing training-set medians and sending physically-impossible 0.0 vectors
         to IsolationForest / LOF.
  Fix  : NaN values are kept as float("nan") so SimpleImputer applies its medians.

The test loads models/unsup_bundle.joblib when present (local / Render), or builds
a minimal mock imputer when the file is absent (CI / lightweight env).

Offline, fast (<1s), no network required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from features import features_to_row, DEFAULT_CONFIG, vector_columns  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────
BUNDLE_PATH = REPO / "models" / "unsup_bundle.joblib"
BUNDLE_AVAILABLE = BUNDLE_PATH.exists()


def _make_mock_bundle(medians: list[float]) -> Dict[str, Any]:
    """
    Minimal mock of the unsup bundle with a fitted SimpleImputer.
    The imputer has known medians so we can assert they're applied.
    """
    import joblib as _joblib  # only used here, not at module level
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median")
    # Fit on a tiny dataset that produces deterministic medians
    n_cols = len(medians)
    X_train = np.tile(medians, (4, 1)).astype(float)
    imp.fit(X_train)
    assert list(imp.statistics_) == pytest.approx(medians, rel=1e-6)

    bundle = joblib.load(BUNDLE_PATH) if BUNDLE_AVAILABLE else None

    return {
        "config": DEFAULT_CONFIG,
        "columns": DEFAULT_CONFIG.all_columns[:n_cols],
        "feature_columns": DEFAULT_CONFIG.all_columns[:n_cols],
        "imputer": {"object": imp, "statistics": medians},
        "models": {},
        "meta": {},
    }


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bundle():
    """Load the real bundle if present, otherwise skip tests that need it."""
    if not BUNDLE_AVAILABLE:
        pytest.skip("models/unsup_bundle.joblib not present — skipping real-bundle tests")
    import joblib
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="module")
def real_cfg(bundle):
    return bundle.get("config", DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def real_cols(bundle, real_cfg):
    return bundle.get("columns") or vector_columns(real_cfg)


@pytest.fixture(scope="module")
def real_imputer(bundle):
    return (bundle.get("imputer") or {}).get("object")


# ── NUMERIC COLS in bundle (first 17) ─────────────────────────────────────────
NUMERIC_17 = [
    "vol_ann", "var95", "var99", "es95", "es99",
    "tuw_pct", "n_used", "missing_pct", "tail_obs_99",
    "var99_div_var95", "es99_div_es95", "es95_div_var95", "es99_div_var99",
    "vol_to_var95_scaled", "var95_vs_vol_daily",
    "tail_obs_99_ratio", "missing_pct_clamped",
]

# feats with no numeric metrics (what Lovable sends without pre-computed stats)
FEATS_NO_METRICS: Dict[str, Any] = {
    "asset_type": "equity",
    "market": "US",
    "vol_ann": None, "vol_20d": None, "max_drawdown": None,
    "var95": None, "var99": None, "es95": None, "es99": None,
    "n_used": None, "missing_pct": None, "tuw_pct": None,
    "tail_obs_99": None, "rsi": None, "corr_mkt": None,
}

# feats with realistic numeric metrics (oracle-computed)
FEATS_FULL: Dict[str, Any] = {
    "asset_type": "equity",
    "market": "US",
    "vol_ann": 0.25, "vol_20d": 0.22, "vol_60d": 0.24, "vol_120d": 0.23,
    "max_drawdown": 0.18, "max_dd": 0.18,
    "var95": 0.016, "var99": 0.024, "es95": 0.022, "es99": 0.032,
    "n_used": 251, "missing_pct": 0.004, "tuw_pct": 95.0, "tail_obs_99": 3,
    "rsi": 55.0, "corr_mkt": 0.0,
}


# ── helper: build raw vector the BUGGY way (NaN→0.0, no NaN for imputer) ─────
def _build_vector_buggy(feats, cfg, cols):
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
            if not math.isfinite(fv):
                fv = 0.0   # ← BUG
        except Exception:
            fv = 0.0       # ← BUG
        row.append(fv)
    X = np.asarray([row], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # ← BUG
    return X


# helper: build raw vector the FIXED way (keep NaN for imputer)
def _build_vector_fixed(feats, cfg, cols):
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
            if not math.isfinite(fv):
                fv = float("nan")   # ← FIX
        except Exception:
            fv = float("nan")       # ← FIX
        row.append(fv)
    X = np.asarray([row], dtype=float)
    # No nan_to_num here — imputer handles NaN
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS: real bundle
# ═══════════════════════════════════════════════════════════════════════════════
class TestUnsupImputer:
    """Tests using the real unsup_bundle.joblib."""

    def test_bundle_has_imputer(self, bundle):
        """Bundle must contain a fitted SimpleImputer."""
        imputer_block = bundle.get("imputer") or {}
        imputer = imputer_block.get("object")
        assert imputer is not None, "bundle missing imputer.object"
        assert hasattr(imputer, "statistics_"), "imputer not fitted"
        assert len(imputer.statistics_) > 0

    def test_bundle_imputer_has_nonzero_medians(self, bundle, real_cols):
        """Training medians for the 17 numeric cols must be non-zero (real data)."""
        imputer = (bundle.get("imputer") or {}).get("object")
        stats = imputer.statistics_
        for i, c in enumerate(real_cols):
            if c in NUMERIC_17:
                assert abs(stats[i]) > 1e-9, (
                    f"Median for {c!r} is 0.0 — imputer may not have been trained on real data"
                )

    def test_buggy_path_bypasses_imputer(self, real_cfg, real_cols, real_imputer):
        """
        REGRESSION: old code converts NaN→0.0 before imputer.
        After imputer.transform(), values stay at 0.0 (imputer had nothing to fix).
        """
        if real_imputer is None:
            pytest.skip("no imputer in bundle")

        X_raw = _build_vector_buggy(FEATS_NO_METRICS, real_cfg, real_cols)

        # Confirm: all NaN-converted-to-0 → no actual NaN for imputer to work on
        assert not np.isnan(X_raw).any(), "buggy path should have no NaN left"

        X_imp = real_imputer.transform(X_raw)

        # Imputer has nothing to fix → values unchanged from 0.0
        for i, c in enumerate(real_cols):
            if c in NUMERIC_17:
                assert X_imp[0, i] == pytest.approx(0.0, abs=1e-9), (
                    f"[BUGGY] {c}: expected still-0.0 after imputer, got {X_imp[0,i]}"
                )

    def test_fixed_path_applies_imputer_medians(self, bundle, real_cfg, real_cols, real_imputer):
        """
        FIX: keep NaN so imputer.transform() applies training-set medians.
        After transform, numeric cols must equal the bundle's stored medians.
        """
        if real_imputer is None:
            pytest.skip("no imputer in bundle")

        X_raw = _build_vector_fixed(FEATS_NO_METRICS, real_cfg, real_cols)

        # Confirm: NaN present for all 17 numeric cols
        for i, c in enumerate(real_cols):
            if c in NUMERIC_17:
                assert np.isnan(X_raw[0, i]), f"[FIXED] {c} should be NaN before imputer"

        X_imp = real_imputer.transform(X_raw)

        # After imputer: each numeric col == training median (no NaN left)
        expected = real_imputer.statistics_
        for i, c in enumerate(real_cols):
            if c in NUMERIC_17:
                assert not np.isnan(X_imp[0, i]), f"{c} still NaN after imputer"
                assert X_imp[0, i] == pytest.approx(expected[i], rel=1e-6), (
                    f"{c}: expected median={expected[i]:.6f}, got {X_imp[0,i]:.6f}"
                )

    def test_fixed_path_known_medians(self, bundle, real_cfg, real_cols, real_imputer):
        """
        Spot-check key medians against expected values from the real bundle.
        Guards against silent bundle version drift.
        """
        if real_imputer is None:
            pytest.skip("no imputer in bundle")

        stats = dict(zip(real_cols, real_imputer.statistics_))
        X_raw = _build_vector_fixed(FEATS_NO_METRICS, real_cfg, real_cols)
        X_imp = real_imputer.transform(X_raw)
        imp_vals = dict(zip(real_cols, X_imp[0]))

        # vol_ann median should be in [0.10, 0.60] (annualized vol)
        assert 0.10 <= imp_vals.get("vol_ann", 0.0) <= 0.60, (
            f"vol_ann median {imp_vals.get('vol_ann')} out of plausible range"
        )
        # var95 median should be negative (daily loss)
        assert imp_vals.get("var95", 0.0) < 0, "var95 median should be negative"
        # n_used median should be > 100 (many trading days)
        assert imp_vals.get("n_used", 0.0) > 100, "n_used median should be >100"
        # tail_obs_99 median should be > 0
        assert imp_vals.get("tail_obs_99", 0.0) > 0, "tail_obs_99 median should be >0"

    def test_full_feats_unchanged_by_fix(self, real_cfg, real_cols, real_imputer):
        """
        When feats are complete (oracle-computed), the fix must not change the result.
        (Finite values are not NaN → imputer has nothing to fill → output identical.)
        """
        if real_imputer is None:
            pytest.skip("no imputer in bundle")

        X_buggy = _build_vector_buggy(FEATS_FULL, real_cfg, real_cols)
        X_fixed = _build_vector_fixed(FEATS_FULL, real_cfg, real_cols)

        X_buggy_imp = real_imputer.transform(X_buggy)
        X_fixed_imp = real_imputer.transform(X_fixed)

        # With complete feats, buggy == fixed (no NaN to fill either way)
        np.testing.assert_array_almost_equal(
            X_buggy_imp, X_fixed_imp, decimal=10,
            err_msg="Fix must not alter scoring when feats are already complete",
        )

    def test_categorical_columns_unchanged_by_fix(self, real_cfg, real_cols, real_imputer):
        """
        Categorical one-hot columns (0.0 / 1.0) must be preserved exactly by the fix.
        """
        if real_imputer is None:
            pytest.skip("no imputer in bundle")

        X_raw = _build_vector_fixed(FEATS_NO_METRICS, real_cfg, real_cols)
        X_imp = real_imputer.transform(X_raw)

        cat_cols = [c for c in real_cols if c.startswith("asset_type__") or c.startswith("market__")]
        for c in cat_cols:
            idx = real_cols.index(c)
            # categorical values are finite 0/1 → not NaN → imputer leaves them alone
            assert not np.isnan(X_imp[0, idx]), f"categorical {c} should not be NaN after imputer"
            assert X_imp[0, idx] in (0.0, 1.0), f"categorical {c} should be 0 or 1"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS: mock imputer (CI / no bundle)
# ═══════════════════════════════════════════════════════════════════════════════
class TestUnsupImputerMock:
    """
    Same logic with a mock imputer — runs without the real bundle (CI-friendly).
    Uses 5 synthetic columns to keep it minimal.
    """

    MOCK_COLS = ["vol_ann", "var95", "es95", "asset_type__equity", "market__US"]
    MOCK_MEDIANS = [0.2876, -0.0350, -0.0420, 0.0, 0.0]

    @pytest.fixture
    def mock_imputer(self):
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        X_train = np.tile(self.MOCK_MEDIANS, (4, 1)).astype(float)
        imp.fit(X_train)
        return imp

    def _row_dict(self, feats):
        cfg = DEFAULT_CONFIG
        full_row = features_to_row(feats, cfg=cfg)
        return full_row

    def _build_buggy(self, feats):
        rd = self._row_dict(feats)
        row = []
        for c in self.MOCK_COLS:
            v = rd.get(c, None)
            try:
                fv = float(v)
                if not math.isfinite(fv): fv = 0.0
            except Exception:
                fv = 0.0
            row.append(fv)
        X = np.asarray([row], dtype=float)
        return np.nan_to_num(X, nan=0.0)

    def _build_fixed(self, feats):
        rd = self._row_dict(feats)
        row = []
        for c in self.MOCK_COLS:
            v = rd.get(c, None)
            try:
                fv = float(v)
                if not math.isfinite(fv): fv = float("nan")
            except Exception:
                fv = float("nan")
            row.append(fv)
        return np.asarray([row], dtype=float)

    def test_mock_buggy_bypasses_imputer(self, mock_imputer):
        """Buggy path: NaN→0.0 → imputer sees no NaN → medians NOT applied."""
        feats = {"asset_type": "equity", "market": "US",
                 "vol_ann": None, "var95": None, "es95": None}
        X = self._build_buggy(feats)
        assert not np.isnan(X).any()
        X_imp = mock_imputer.transform(X)
        # numeric cols stay at 0.0 (imputer didn't change anything)
        assert X_imp[0, 0] == pytest.approx(0.0)   # vol_ann
        assert X_imp[0, 1] == pytest.approx(0.0)   # var95
        assert X_imp[0, 2] == pytest.approx(0.0)   # es95

    def test_mock_fixed_applies_medians(self, mock_imputer):
        """Fixed path: NaN kept → imputer applies training medians."""
        feats = {"asset_type": "equity", "market": "US",
                 "vol_ann": None, "var95": None, "es95": None}
        X = self._build_fixed(feats)
        # Numeric cols must be NaN before imputer
        assert np.isnan(X[0, 0])  # vol_ann
        assert np.isnan(X[0, 1])  # var95
        assert np.isnan(X[0, 2])  # es95
        X_imp = mock_imputer.transform(X)
        assert X_imp[0, 0] == pytest.approx(0.2876, rel=1e-4)   # vol_ann
        assert X_imp[0, 1] == pytest.approx(-0.0350, rel=1e-4)  # var95
        assert X_imp[0, 2] == pytest.approx(-0.0420, rel=1e-4)  # es95

    def test_mock_categorical_intact(self, mock_imputer):
        """Categorical 0/1 values must not be altered by the fix."""
        feats = {"asset_type": "equity", "market": "US",
                 "vol_ann": None, "var95": None, "es95": None}
        X = self._build_fixed(feats)
        X_imp = mock_imputer.transform(X)
        assert X_imp[0, 3] == pytest.approx(1.0)   # asset_type__equity = 1 for equity
        assert X_imp[0, 4] == pytest.approx(1.0)   # market__US = 1 for US

    def test_mock_full_feats_unchanged(self, mock_imputer):
        """When feats are complete, buggy == fixed (no NaN to fill)."""
        feats = {"asset_type": "equity", "market": "US",
                 "vol_ann": 0.25, "var95": 0.016, "es95": 0.022}
        X_b = self._build_buggy(feats)
        X_f = self._build_fixed(feats)
        X_b_imp = mock_imputer.transform(X_b)
        X_f_imp = mock_imputer.transform(X_f)
        np.testing.assert_array_almost_equal(X_b_imp, X_f_imp, decimal=10)
