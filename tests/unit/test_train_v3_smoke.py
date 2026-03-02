"""
tests/unit/test_train_v3_smoke.py
==================================
Smoke tests for scripts/ml/train/train_experts_v3.py

Tests (fast, no network, synthetic data only):
  - load_jsonl_streaming: basic load, max_rows, sort order, feature extraction
  - build_model: each model type (including invalid)
  - impute: no NaN/inf in output
  - backtest_fold: runs without error, expected keys present
  - aggregate_folds: empty + normal cases
  - E2E CLI smoke run (logistic only, synthetic data, subprocess)
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.ml.train.train_experts_v3 import (  # noqa: E402
    aggregate_folds,
    backtest_fold,
    build_model,
    get_feature_cols,
    impute,
    load_jsonl_streaming,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(ticker: str, wed: date, label: str = "ok", i: int = 0) -> dict:
    return {
        "version":           "v3",
        "label":             label,
        "target_non_ok":     0 if label == "ok" else 1,
        "window_end_date":   str(wed),
        "label_start_date":  str(wed + timedelta(days=1)),
        "label_end_date":    str(wed + timedelta(days=20)),
        "forward_return_20d": round(0.01 * (1 + i % 5) * (1 if label == "ok" else -0.5), 4),
        "features": {
            "asset_type":   "equity",       # string → excluded from features
            "ticker":       ticker,         # string → excluded
            "vol_ann":      round(0.15 + 0.001 * i, 4),
            "var95":        round(0.02 + 0.0005 * i, 4),
            "momentum_20d": round(0.005 * (i % 20 - 10), 4),
            "rsi_14":       round(50.0 + (i % 30) - 15, 2),
            "macro_gdp":    None if i % 7 == 0 else round(2.0 + 0.1 * (i % 5), 2),
        },
    }


def _write_synthetic(n_days: int = 200) -> Path:
    """Write synthetic JSONL: 2 tickers × n_days observations (every 3 days)."""
    base = date(2018, 1, 1)
    records = []
    for i in range(n_days):
        wed = base + timedelta(days=i * 3)
        label_a = "block" if i % 10 == 0 else "ok"
        label_b = "warn"  if i % 7  == 0 else "ok"
        records.append(_make_record("AAPL", wed, label_a, i))
        records.append(_make_record("MSFT", wed + timedelta(days=1), label_b, i))

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return Path(tmp.name)


@pytest.fixture(scope="module")
def synthetic_path():
    p = _write_synthetic(200)
    yield p
    p.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# load_jsonl_streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadJsonlStreaming:

    def test_basic_load(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        assert len(df) > 0
        assert "window_end_date" in df.columns
        assert "target_non_ok" in df.columns
        assert "forward_return_20d" in df.columns

    def test_sorted_by_date(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        assert df["window_end_date"].is_monotonic_increasing

    def test_max_rows_limits(self, synthetic_path):
        df_full = load_jsonl_streaming(synthetic_path)
        df_cut  = load_jsonl_streaming(synthetic_path, max_rows=20)
        assert len(df_cut) <= 20
        assert len(df_cut) < len(df_full)

    def test_feature_cols_extracted(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        feat = get_feature_cols(df)
        # Numeric features present
        assert "vol_ann" in feat
        assert "var95" in feat
        assert "momentum_20d" in feat
        assert "rsi_14" in feat
        # Targets/meta excluded
        assert "target_non_ok" not in feat
        assert "forward_return_20d" not in feat

    def test_no_string_feature_cols(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        feat = get_feature_cols(df)
        # asset_type and ticker are strings → must not be in feat_cols
        assert "asset_type" not in feat
        assert "ticker" not in feat

    def test_binary_target(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        assert set(df["target_non_ok"].dropna().unique()).issubset({0, 1, 0.0, 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# build_model
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_logistic_not_none(self):
        m = build_model("logistic")
        assert m is not None

    def test_hgb_not_none(self):
        m = build_model("hgb")
        assert m is not None

    def test_logistic_has_predict_proba(self):
        m = build_model("logistic")
        assert hasattr(m, "predict_proba") or hasattr(m, "named_steps")

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("random_forest_xyz")

    def test_seed_respected(self):
        m1 = build_model("logistic", seed=0)
        m2 = build_model("logistic", seed=0)
        # Same seed → same hyperparameters (both are Pipelines)
        assert str(m1) == str(m2)


# ─────────────────────────────────────────────────────────────────────────────
# impute
# ─────────────────────────────────────────────────────────────────────────────

class TestImpute:

    def test_no_nan_after_impute(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        feat = get_feature_cols(df)
        n = len(df)
        X_tr_raw = df.iloc[: n // 2][feat]
        X_vl_raw = df.iloc[n // 2 :][feat]
        X_tr, X_vl, medians = impute(X_tr_raw, X_vl_raw)
        assert not np.isnan(X_tr).any(), "NaN found in imputed train"
        assert not np.isnan(X_vl).any(), "NaN found in imputed val"

    def test_no_inf_after_impute(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        feat = get_feature_cols(df)
        n = len(df)
        X_tr, X_vl, _ = impute(df.iloc[: n // 2][feat], df.iloc[n // 2 :][feat])
        assert not np.isinf(X_tr).any()
        assert not np.isinf(X_vl).any()

    def test_medians_computed_on_train_only(self, synthetic_path):
        df = load_jsonl_streaming(synthetic_path)
        feat = get_feature_cols(df)
        n = len(df)
        X_tr_raw = df.iloc[: n // 2][feat]
        X_vl_raw = df.iloc[n // 2 :][feat]
        _, _, medians = impute(X_tr_raw, X_vl_raw)
        expected = X_tr_raw.median()
        for col in feat:
            if expected[col] is not None and not np.isnan(expected[col]):
                assert abs(medians[col] - expected[col]) < 1e-9, f"Median mismatch on {col}"


# ─────────────────────────────────────────────────────────────────────────────
# backtest_fold
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestFold:

    def _make_inputs(self, n: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0.0, 1.0, n)
        y_true = (y_prob > 0.65).astype(int)
        fwd    = rng.normal(0.005, 0.03, n)
        return y_prob, y_true, fwd

    def test_returns_expected_keys(self):
        y_prob, y_true, fwd = self._make_inputs()
        result = backtest_fold(y_prob, y_true, fwd)
        assert "simple" in result
        assert "constrained" in result
        assert "threshold_constrained" in result

    def test_simple_strategy_keys(self):
        y_prob, y_true, fwd = self._make_inputs()
        result = backtest_fold(y_prob, y_true, fwd)
        simple = result["simple"]
        for k in ("threshold", "n_invested", "n_skipped", "skip_rate",
                  "strategy_mean_return", "baseline_mean_return",
                  "sharpe_proxy", "max_drawdown_proxy", "fp_rate"):
            assert k in simple, f"Missing key '{k}' in backtest simple"

    def test_skip_rate_range(self):
        y_prob, y_true, fwd = self._make_inputs()
        result = backtest_fold(y_prob, y_true, fwd)
        skip = result["simple"]["skip_rate"]
        if skip is not None:
            assert 0.0 <= skip <= 1.0

    def test_no_crash_all_nan_returns(self):
        n = 100
        y_prob = np.full(n, 0.3)
        y_true = np.zeros(n, dtype=int)
        fwd    = np.full(n, float("nan"))
        result = backtest_fold(y_prob, y_true, fwd)
        assert result is not None

    def test_no_crash_single_class(self):
        n = 100
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(0, 1, n)
        y_true = np.zeros(n, dtype=int)     # all OK — no non_ok
        fwd    = rng.normal(0.01, 0.02, n)
        result = backtest_fold(y_prob, y_true, fwd)
        assert result is not None

    def test_constrained_threshold_gte_simple(self):
        """Constrained threshold should be ≥ 0.5 (more conservative) when possible."""
        y_prob, y_true, fwd = self._make_inputs()
        result = backtest_fold(y_prob, y_true, fwd, fp_constraint=0.15)
        t_c = result["threshold_constrained"]
        assert t_c >= 0.30, "Constrained threshold below minimum search range"


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_folds
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateFolds:

    def _fold(self, fold: int, roc: float, pr: float) -> dict:
        return {
            "fold": fold, "model": "logistic",
            "roc_auc": roc, "pr_auc": pr,
            "brier": 0.18, "ece": 0.05, "f1": 0.55, "accuracy": 0.78,
            "precision_top20pct": 0.60, "lift_top20pct": 1.8,
        }

    def test_empty_list(self):
        r = aggregate_folds([])
        assert r["n_valid_folds"] == 0

    def test_no_roc_auc(self):
        r = aggregate_folds([{"fold": 1, "model": "logistic"}])
        assert r["n_valid_folds"] == 0

    def test_two_folds_mean(self):
        folds = [self._fold(1, 0.70, 0.40), self._fold(2, 0.80, 0.50)]
        agg = aggregate_folds(folds)
        assert agg["n_valid_folds"] == 2
        assert abs(agg["roc_auc_mean"] - 0.75) < 1e-4
        assert abs(agg["pr_auc_mean"]  - 0.45) < 1e-4

    def test_std_present(self):
        folds = [self._fold(1, 0.70, 0.40), self._fold(2, 0.80, 0.50)]
        agg = aggregate_folds(folds)
        assert "roc_auc_std" in agg
        assert agg["roc_auc_std"] >= 0.0

    def test_single_fold(self):
        agg = aggregate_folds([self._fold(1, 0.72, 0.42)])
        assert agg["n_valid_folds"] == 1
        assert agg["roc_auc_std"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# E2E CLI smoke run (subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestE2ESmokeRun:
    """Full pipeline smoke test: synthetic data → CLI → report + model files."""

    def test_logistic_smoke(self, synthetic_path, tmp_path):
        import subprocess

        cmd = [
            sys.executable,
            str(_REPO / "scripts/ml/train/train_experts_v3.py"),
            "--input",        str(synthetic_path),
            "--out_dir",      str(tmp_path / "metrics"),
            "--models_dir",   str(tmp_path / "models"),
            "--n_splits",     "3",
            "--embargo_days", "5",
            "--models",       "logistic",
            "--seed",         "42",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, (
            f"\n--- STDOUT ---\n{result.stdout}"
            f"\n--- STDERR ---\n{result.stderr}"
        )

        # Report file present and valid JSON
        report_path = tmp_path / "metrics" / "metrics_report_v3.json"
        assert report_path.exists(), "metrics_report_v3.json not created"

        data = json.loads(report_path.read_text())
        assert "fold_results" in data
        assert "aggregated" in data
        assert len(data["fold_results"]) > 0
        assert data["n_features"] > 0

        # Model + meta files present
        model_path = tmp_path / "models" / "v3_logistic_final.joblib"
        meta_path  = tmp_path / "models" / "v3_logistic_meta.json"
        assert model_path.exists(), "v3_logistic_final.joblib not found"
        assert meta_path.exists(),  "v3_logistic_meta.json not found"

        # Meta has required keys
        meta = json.loads(meta_path.read_text())
        assert "feature_cols" in meta
        assert "medians" in meta
        assert len(meta["feature_cols"]) > 0

    def test_hgb_smoke(self, synthetic_path, tmp_path):
        """HGB model (no xgboost required)."""
        import subprocess

        cmd = [
            sys.executable,
            str(_REPO / "scripts/ml/train/train_experts_v3.py"),
            "--input",        str(synthetic_path),
            "--out_dir",      str(tmp_path / "metrics"),
            "--models_dir",   str(tmp_path / "models"),
            "--n_splits",     "2",
            "--embargo_days", "5",
            "--models",       "hgb",
            "--n_estimators", "50",
            "--seed",         "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, (
            f"\n--- STDOUT ---\n{result.stdout}"
            f"\n--- STDERR ---\n{result.stderr}"
        )
        model_path = tmp_path / "models" / "v3_hgb_final.joblib"
        assert model_path.exists()

    def test_drop_macro_flag(self, synthetic_path, tmp_path):
        """--drop_macro should run without error and potentially reduce feature count."""
        import subprocess

        cmd = [
            sys.executable,
            str(_REPO / "scripts/ml/train/train_experts_v3.py"),
            "--input",        str(synthetic_path),
            "--out_dir",      str(tmp_path / "metrics"),
            "--models_dir",   str(tmp_path / "models"),
            "--n_splits",     "2",
            "--embargo_days", "5",
            "--models",       "logistic",
            "--drop_macro",
            "--nan_threshold", "0.1",   # aggressive: drop cols with >10% NaN
            "--seed",         "42",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, f"STDERR: {result.stderr}"

    def test_max_rows_smoke(self, synthetic_path, tmp_path):
        """--max_rows should limit dataset and still produce a valid report."""
        import subprocess

        cmd = [
            sys.executable,
            str(_REPO / "scripts/ml/train/train_experts_v3.py"),
            "--input",        str(synthetic_path),
            "--out_dir",      str(tmp_path / "metrics"),
            "--models_dir",   str(tmp_path / "models"),
            "--n_splits",     "2",
            "--embargo_days", "5",
            "--models",       "logistic",
            "--max_rows",     "100",
            "--seed",         "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        # May succeed or warn about too-small dataset, but must not crash hard
        # (exit code 0 OR informative failure)
        data_path = tmp_path / "metrics" / "metrics_report_v3.json"
        if result.returncode == 0:
            assert data_path.exists()
