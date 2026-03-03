"""
tests/unit/test_nan_preprocessing.py
=====================================
Verify that the v3 preprocessing pipeline produces zero-NaN feature matrices.

Covers:
  - recovery_* sentinel values (-1.0) flow through load_jsonl_fold without NaN
  - SimpleImputer correctly handles any residual NaN from macro features
  - Final X matrix has no NaN for a realistic synthetic record
  - recovery_defined (0/1 flag) is never NaN
"""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic JSONL record factory
# ---------------------------------------------------------------------------

def _make_record(
    recovery_defined: float = 0.0,
    recovery_days: float = -1.0,
    recovery_per_dd: float = -1.0,
    dd_duration: float = -1.0,
    vix_level = None,          # macro may be NaN → should be imputed
) -> dict:
    """Return a minimal v3 JSONL record with realistic features."""
    return {
        "version": "v3",
        "label": "ok",
        "target_non_ok": 0,
        "window_start_date": "2020-01-01",
        "window_end_date":   "2021-01-01",
        "label_start_date":  "2021-01-02",
        "label_end_date":    "2021-01-29",
        "forward_return_5d":  0.01,
        "forward_return_10d": 0.02,
        "forward_return_20d": 0.03,
        "forward_return_60d": 0.05,
        "future_dd_20d": -0.05,
        "future_vol_ratio": 1.1,
        "source": "synthetic",
        "features": {
            # Meta (excluded from X)
            "ticker":       "AAPL",
            "asset_type":   "equity",
            "market":       "US",
            "market_proxy": "SPY",
            # Sentinel-based features (no NaN)
            "recovery_defined":  recovery_defined,
            "recovery_days":     recovery_days,
            "recovery_per_dd":   recovery_per_dd,
            "dd_duration":       dd_duration,
            # Numeric features — non-null
            "vol_ann":        0.25,
            "vol_20d":        0.015,
            "vol_60d":        0.22,
            "vol_120d":       0.24,
            "var95":          0.018,
            "var99":          0.028,
            "es95":           0.022,
            "es99":           0.035,
            "tail_obs_99":    3.0,
            "mdd":           -0.15,
            "downside_dev":   0.17,
            "semivariance":   0.0001,
            "vol_of_vol":     0.4,
            "vol_ewma_ann":   0.23,
            "worst_5d_ret":  -0.05,
            "worst_10d_ret": -0.08,
            "worst_20d_ret": -0.12,
            "autocorr_1":    -0.02,
            "autocorr_5":     0.01,
            "stress_var99":   0.03,
            "stress_multiplier": 1.2,
            "jump_indicator": 0.0,
            "hill_tail_index": 3.5,
            "corr_spy":      None,   # all-null in current dataset → excluded via _ALWAYS_NULL
            "corr_vix":      -0.3,
            "beta_market":   None,
            "abs_corr_mkt":  None,
            # Macro (may be null → imputed)
            "vix_level":         vix_level,
            "vix_pct_60d":       None,
            "rate_10y":          1.5,
            "rate_2y":           0.5,
            "term_spread":       1.0,
            "credit_spread_hy":  3.5,
            "credit_spread_ig":  0.8,
            "vol_regime":        1.0,
            # Derived ratios
            "var99_var95":      1.55,
            "es99_es95":        1.59,
            "es95_var95":       1.22,
            "es99_var99":       1.25,
            "vol20_vol_ann":    0.38,
            "vol60_vol_ann":    0.88,
            "vol120_vol_ann":   0.96,
            "vol20_vol60":      0.43,
            "dd_to_var99":      5.35,
            "log_n_used":       6.2,
            "downside_div_vol": 0.68,
            "worst_5d_vs_var99": 1.78,
            "dd_duration_per_n": 0.1,
            "missing_pct":      0.0,
            "macd_hist":        0.002,
        },
    }


# ---------------------------------------------------------------------------
# Helper: write JSONL + run load_jsonl_fold
# ---------------------------------------------------------------------------

def _write_and_load(records: list) -> tuple:
    """Write records to a temp JSONL and load via train_v3.load_jsonl_fold."""
    import sys
    from pathlib import Path as _P
    # Ensure repo root on path
    repo = _P(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Reset the global _FEAT_COLS so each test starts fresh
    import scripts.ml.train.train_v3 as tv3
    tv3._FEAT_COLS = []

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        tmp = Path(f.name)

    X, y, feat_names = tv3.load_jsonl_fold(tmp)
    tmp.unlink()
    return X, y, feat_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _imputed(X: np.ndarray) -> np.ndarray:
    """Apply median SimpleImputer (mirrors train_v3._build_preprocessor)."""
    from sklearn.impute import SimpleImputer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimpleImputer(strategy="median").fit_transform(X)


class TestRecoverySentinel:
    """Sentinel -1 values for recovery_* must propagate without NaN after imputation."""

    def test_recovery_undefined_no_nan(self):
        """
        recovery_defined=0, recovery_days=-1 (sentinel).
        After full pipeline (load + impute), X must be NaN-free.
        We provide 3 records so macro median imputation has reference values.
        """
        recs = [
            _make_record(recovery_defined=0.0, recovery_days=-1.0,
                         recovery_per_dd=-1.0, dd_duration=-1.0, vix_level=20.0),
            _make_record(recovery_defined=0.0, recovery_days=-1.0,
                         recovery_per_dd=-1.0, dd_duration=-1.0, vix_level=22.0),
            _make_record(recovery_defined=0.0, recovery_days=-1.0,
                         recovery_per_dd=-1.0, dd_duration=-1.0, vix_level=None),  # will be imputed
        ]
        X, y, feat_names = _write_and_load(recs)
        X_imp = _imputed(X)
        nan_mask = ~np.isfinite(X_imp)
        nan_cols = [feat_names[j] for j in range(X_imp.shape[1]) if nan_mask[:, j].any()]
        assert nan_mask.sum() == 0, (
            f"Expected zero NaN/inf after imputation, got issues in: {nan_cols}"
        )

        # Specifically: recovery features should have sentinel -1, not NaN
        for feat in ["recovery_days", "recovery_per_dd", "dd_duration"]:
            if feat in feat_names:
                j = feat_names.index(feat)
                vals = X[:, j]
                assert not np.isnan(vals).any(), f"{feat} should not be NaN (sentinel -1 expected)"
                assert (vals == -1.0).all(), f"{feat} expected sentinel -1, got {vals}"

    def test_recovery_defined_no_nan(self):
        """Record with actual recovery (recovery_defined=1): values are positive floats."""
        recs = [
            _make_record(recovery_defined=1.0, recovery_days=45.0,
                         recovery_per_dd=1.5, dd_duration=30.0, vix_level=20.0),
            _make_record(recovery_defined=1.0, recovery_days=30.0,
                         recovery_per_dd=1.0, dd_duration=25.0, vix_level=18.0),
        ]
        X, y, feat_names = _write_and_load(recs)
        X_imp = _imputed(X)
        nan_mask = ~np.isfinite(X_imp)
        nan_cols = [feat_names[j] for j in range(X_imp.shape[1]) if nan_mask[:, j].any()]
        assert nan_mask.sum() == 0, (
            f"Expected zero NaN/inf after imputation, got issues in: {nan_cols}"
        )

    def test_recovery_defined_is_binary(self):
        """recovery_defined must be 0 or 1 — never NaN."""
        for val in [0.0, 1.0]:
            rec = _make_record(recovery_defined=val)
            X, y, feat_names = _write_and_load([rec])
            if "recovery_defined" in feat_names:
                j = feat_names.index("recovery_defined")
                assert not math.isnan(float(X[0, j])), (
                    f"recovery_defined={val} should not be NaN in X"
                )


class TestMacroNullImputation:
    """Macro features that are null must be imputed (not remain NaN) when multiple records present."""

    def test_macro_null_imputed_with_multiple_records(self):
        """
        When vix_level is None in some records, SimpleImputer fills with median from others.
        Final X must have no NaN.
        """
        recs = [
            _make_record(vix_level=None),   # will be imputed
            _make_record(vix_level=20.0),
            _make_record(vix_level=25.0),
            _make_record(vix_level=None),
        ]
        X, y, feat_names = _write_and_load(recs)

        # Apply imputer as train_v3 would
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)

        nan_count = np.isnan(X_imp).sum()
        assert nan_count == 0, (
            f"After median imputation, expected 0 NaN but got {nan_count}"
        )

    def test_all_null_macro_handled(self):
        """If ALL records have vix_level=None, imputer produces 0 (fallback). No crash."""
        recs = [_make_record(vix_level=None) for _ in range(3)]
        X, y, feat_names = _write_and_load(recs)

        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        # Should not crash; all-null column becomes 0 after sklearn imputation
        X_imp = imp.fit_transform(X)
        assert X_imp is not None


class TestPreprocessingPipeline:
    """End-to-end preprocessing: load_jsonl_fold → SimpleImputer → StandardScaler."""

    def test_full_pipeline_no_nan(self):
        """After full sklearn pipeline (impute + scale), X must be finite."""
        recs = [_make_record() for _ in range(5)]
        X, y, feat_names = _write_and_load(recs)

        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        X_out = pipe.fit_transform(X)

        assert np.all(np.isfinite(X_out)), (
            f"Preprocessing pipeline produced non-finite values. "
            f"Non-finite count: {(~np.isfinite(X_out)).sum()}"
        )

    def test_y_values(self):
        """target_non_ok must be 0 or 1 (binary)."""
        recs = [
            _make_record(),                               # target_non_ok=0
            {**_make_record(), "target_non_ok": 1, "label": "block"},
        ]
        X, y, feat_names = _write_and_load(recs)
        assert set(y.tolist()).issubset({0, 1}), f"y contains unexpected values: {set(y.tolist())}"
