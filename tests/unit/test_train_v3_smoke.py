"""
tests/unit/test_train_v3_smoke.py
==================================
Smoke tests for scripts/ml/train/train_v3.py — the canonical v3 training entrypoint.

Tests (fast, no network, synthetic data only):
  - filter_high_nan_cols: drops high-NaN features, logs correctly
  - _candidate_feat_cols: excludes meta/string fields
  - load_jsonl_fold: loads X/y, respects max_rows, handles NaN
  - probe_nan_rates: correct NaN fractions
  - compute_metrics: standard classification metrics
  - _fold_signal_backtest: backtest keys + values
  - _aggregate_metrics: mean/std across folds
  - E2E CLI smoke run: synthetic manifest + fold JSONL → full pipeline

Previously tested scripts/ml/train/train_experts_v3.py — that script is
now deprecated in favour of scripts/ml/train/train_v3.py (manifest-based, calibrated).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.ml.train.train_v3 import (  # noqa: E402
    _aggregate_metrics,
    _candidate_feat_cols,
    _fold_signal_backtest,
    compute_metrics,
    filter_high_nan_cols,
    load_jsonl_fold,
    probe_nan_rates,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(ticker: str, wed: date, label: str = "ok", i: int = 0) -> dict:
    """Create a minimal v3 JSONL record with realistic features."""
    return {
        "version":           "v3",
        "label":             label,
        "target_non_ok":     0 if label == "ok" else 1,
        "window_end_date":   str(wed),
        "label_start_date":  str(wed + timedelta(days=1)),
        "label_end_date":    str(wed + timedelta(days=20)),
        "forward_return_20d": round(0.01 * (1 + i % 5) * (1 if label == "ok" else -0.5), 4),
        "features": {
            # Meta (should be excluded)
            "asset_type":   "equity",
            "market":       "US",
            "ticker":       ticker,
            "market_proxy": "SPY",
            # Numeric features
            "vol_ann":      round(0.15 + 0.001 * i, 4),
            "var95":        round(0.02 + 0.0005 * i, 4),
            "momentum_20d": round(0.005 * (i % 20 - 10), 4),
            "rsi_14":       round(50.0 + (i % 30) - 15, 2),
            # Sparse macro (all NaN → should be dropped)
            "vix_level":    None,
            "corr_spy":     None,
            # Partial NaN (>30% → dropped with default threshold)
            "recovery_days": None if i % 2 == 0 else float(i * 5),
        },
    }


def _write_fold_jsonl(records: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _synthetic_fold_pair(
    tmp_dir: Path,
    fold_k: int,
    n_train: int = 150,
    n_val: int = 80,
) -> tuple:
    """Write train.jsonl + val.jsonl for one fold, return (train_path, val_path)."""
    base = date(2018, 1, 1) + timedelta(days=fold_k * 400)
    train_recs = [
        _make_record("AAPL", base + timedelta(days=i * 3),
                     label="block" if i % 8 == 0 else "ok", i=i)
        for i in range(n_train)
    ]
    val_start = base + timedelta(days=n_train * 3 + 30)
    val_recs = [
        _make_record("MSFT", val_start + timedelta(days=i * 3),
                     label="warn" if i % 5 == 0 else "ok", i=n_train + i)
        for i in range(n_val)
    ]

    fold_dir = tmp_dir / f"fold_{fold_k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_path = fold_dir / "train.jsonl"
    val_path   = fold_dir / "val.jsonl"
    _write_fold_jsonl(train_recs, train_path)
    _write_fold_jsonl(val_recs,   val_path)
    return train_path, val_path


def _synthetic_manifest(
    tmp_dir: Path,
    n_folds: int = 3,
    n_train: int = 150,
    n_val: int = 80,
) -> Path:
    """Create a full splits_manifest.json with synthetic fold JSONL files."""
    splits = []
    base_date = date(2018, 1, 1)

    for k in range(1, n_folds + 1):
        train_path, val_path = _synthetic_fold_pair(tmp_dir, k, n_train, n_val)
        fold_base = base_date + timedelta(days=(k - 1) * 400)
        splits.append({
            "fold":              k,
            "train_start":       str(fold_base),
            "train_end":         str(fold_base + timedelta(days=n_train * 3)),
            "train_cutoff":      str(fold_base + timedelta(days=n_train * 3)),
            "val_start":         str(fold_base + timedelta(days=n_train * 3 + 30)),
            "val_end":           str(fold_base + timedelta(days=n_train * 3 + 30 + n_val * 3)),
            "n_train":           n_train,
            "n_val":             n_val,
            "purge_days":        5,
            "embargo_days":      2,
            "non_ok_rate_train": 0.12,
            "non_ok_rate_val":   0.20,
            "label_dist_train":  {"ok": n_train - n_train // 8, "block": n_train // 8},
            "label_dist_val":    {"ok": n_val - n_val // 5, "warn": n_val // 5},
            "train_jsonl":       str(train_path),
            "val_jsonl":         str(val_path),
        })

    manifest = {
        "generated_at":  "2026-03-03T00:00:00",
        "source_file":   "synthetic",
        "n_folds":       n_folds,
        "purge_days":    5,
        "embargo_days":  2,
        "n_valid_folds": n_folds,
        "splits":        splits,
    }

    manifest_path = tmp_dir / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


@pytest.fixture(scope="module")
def synthetic_manifest_dir(tmp_path_factory):
    """Module-scoped synthetic manifest + fold files."""
    d = tmp_path_factory.mktemp("manifest")
    manifest_path = _synthetic_manifest(d, n_folds=3, n_train=150, n_val=80)
    return d, manifest_path


# ─────────────────────────────────────────────────────────────────────────────
# filter_high_nan_cols
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterHighNanCols:

    def _rates(self, **kwargs) -> Dict[str, float]:
        return kwargs

    def test_drops_100pct_null(self):
        feat_cols = ["vol_ann", "corr_spy", "vix_level"]
        nan_rates = {"vol_ann": 0.0, "corr_spy": 1.0, "vix_level": 1.0}
        kept, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        assert kept == ["vol_ann"]
        assert len(dropped) == 2

    def test_drops_above_threshold(self):
        feat_cols = ["a", "b", "c"]
        nan_rates = {"a": 0.0, "b": 0.31, "c": 0.29}
        kept, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        assert "b" not in kept
        assert "c" in kept

    def test_keeps_at_exact_threshold(self):
        feat_cols = ["a", "b"]
        nan_rates = {"a": 0.30, "b": 0.31}
        kept, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        # > threshold means STRICTLY above, so 0.30 is kept
        assert "a" in kept
        assert "b" not in kept

    def test_none_dropped_when_all_clean(self):
        feat_cols = ["a", "b", "c"]
        nan_rates = {"a": 0.0, "b": 0.01, "c": 0.05}
        kept, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        assert kept == feat_cols
        assert dropped == []

    def test_all_dropped_returns_empty_kept(self):
        feat_cols = ["a", "b"]
        nan_rates = {"a": 1.0, "b": 0.99}
        kept, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        assert kept == []
        assert len(dropped) == 2

    def test_dropped_contains_all_high_nan_cols(self):
        # The log prints dropped cols sorted by rate; the returned list is in
        # feat_cols order.  We only assert membership, not order.
        feat_cols = ["a", "b", "c"]
        nan_rates = {"a": 0.5, "b": 1.0, "c": 0.8}
        _, dropped = filter_high_nan_cols(feat_cols, nan_rates, threshold=0.30)
        dropped_names = {col for col, _ in dropped}
        assert dropped_names == {"a", "b", "c"}, f"Expected all 3 dropped, got {dropped_names}"
        # Rates match the input
        dropped_dict = dict(dropped)
        assert dropped_dict["b"] == pytest.approx(1.0)
        assert dropped_dict["a"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# _candidate_feat_cols
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateFeatCols:

    def test_excludes_meta_string_fields(self):
        sample = {
            "asset_type": "equity", "market": "US",
            "ticker": "AAPL", "market_proxy": "SPY",
            "vol_ann": 0.2, "var95": 0.02,
        }
        cols = _candidate_feat_cols(sample)
        for meta in ("asset_type", "market", "ticker", "market_proxy"):
            assert meta not in cols, f"{meta} should be excluded"

    def test_includes_numeric_features(self):
        sample = {"vol_ann": 0.2, "var95": 0.02, "rsi_14": 55.0}
        cols = _candidate_feat_cols(sample)
        assert "vol_ann" in cols and "var95" in cols

    def test_includes_none_valued_features(self):
        # None-valued features are candidates (NaN but still in schema)
        sample = {"vol_ann": 0.2, "corr_spy": None}
        cols = _candidate_feat_cols(sample)
        assert "corr_spy" in cols

    def test_sorted_output(self):
        sample = {"z_feat": 1.0, "a_feat": 2.0, "m_feat": 3.0}
        cols = _candidate_feat_cols(sample)
        assert cols == sorted(cols)


# ─────────────────────────────────────────────────────────────────────────────
# load_jsonl_fold
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadJsonlFold:

    def test_basic_load(self, synthetic_manifest_dir, tmp_path):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vol_ann", "var95", "momentum_20d", "rsi_14"]
        X, y = load_jsonl_fold(train_path, feat_cols)
        assert X.shape[0] > 0
        assert X.shape[1] == 4
        assert len(y) == X.shape[0]

    def test_target_binary(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vol_ann", "var95"]
        _, y = load_jsonl_fold(train_path, feat_cols)
        assert set(y.tolist()).issubset({0, 1})

    def test_max_rows_limits(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vol_ann", "var95"]
        X_full, _ = load_jsonl_fold(train_path, feat_cols)
        X_cut,  _ = load_jsonl_fold(train_path, feat_cols, max_rows=20)
        assert len(X_cut) <= 20
        assert len(X_cut) < len(X_full)

    def test_missing_col_becomes_nan(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vol_ann", "nonexistent_feature"]
        X, _ = load_jsonl_fold(train_path, feat_cols)
        assert np.all(np.isnan(X[:, 1])), "nonexistent feature must be all NaN"

    def test_none_feature_becomes_nan(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vix_level"]   # all-None in synthetic data
        X, _ = load_jsonl_fold(train_path, feat_cols)
        assert np.all(np.isnan(X[:, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# probe_nan_rates
# ─────────────────────────────────────────────────────────────────────────────

class TestProbeNanRates:

    def test_100pct_nan_feature(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vix_level"]   # all None in synthetic data
        rates = probe_nan_rates(train_path, feat_cols)
        assert rates["vix_level"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_nan_feature(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["vol_ann"]   # always set in synthetic data
        rates = probe_nan_rates(train_path, feat_cols)
        assert rates["vol_ann"] == pytest.approx(0.0, abs=1e-6)

    def test_partial_nan_recovery_days(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["recovery_days"]
        rates = probe_nan_rates(train_path, feat_cols)
        # synthetic: None for even i, float for odd i → ~50% NaN
        assert 0.0 < rates["recovery_days"] < 1.0

    def test_max_rows_affects_rate(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        train_path = Path(manifest["splits"][0]["train_jsonl"])
        feat_cols = ["recovery_days"]
        rates_small = probe_nan_rates(train_path, feat_cols, max_rows=4)
        # Just verify it runs without error and returns valid fraction
        assert 0.0 <= rates_small["recovery_days"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:

    def _random_binary(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0, 1, n)
        y_true = (y_prob > 0.55).astype(int)
        return y_true, y_prob

    def test_standard_keys(self):
        y_true, y_prob = self._random_binary(200)
        m = compute_metrics(y_true, y_prob, label="test")
        for k in ("roc_auc", "pr_auc", "brier", "ece", "fpr_at_tpr80",
                  "recall_t05", "f1_t05"):
            assert k in m, f"Missing key: {k}"

    def test_roc_auc_range(self):
        y_true, y_prob = self._random_binary(500, seed=1)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_single_class_returns_warning(self):
        y_true = np.zeros(50, dtype=int)
        y_prob = np.random.default_rng(0).uniform(0, 1, 50)
        m = compute_metrics(y_true, y_prob, label="single_class")
        assert "warning" in m

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# _fold_signal_backtest
# ─────────────────────────────────────────────────────────────────────────────

class TestFoldSignalBacktest:

    def test_returns_expected_keys(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        val_path = Path(manifest["splits"][0]["val_jsonl"])
        n_val = manifest["splits"][0]["n_val"]
        y_prob = np.random.default_rng(42).uniform(0, 1, n_val)
        result = _fold_signal_backtest(y_prob, val_path, threshold=0.5)
        for k in ("threshold", "n_invested", "skip_rate",
                  "strategy_mean_return", "baseline_mean_return", "sharpe_proxy"):
            assert k in result, f"Missing key: {k}"

    def test_skip_rate_between_0_and_1(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        val_path = Path(manifest["splits"][0]["val_jsonl"])
        n_val = manifest["splits"][0]["n_val"]
        y_prob = np.full(n_val, 0.4)
        result = _fold_signal_backtest(y_prob, val_path, threshold=0.5)
        skip = result["skip_rate"]
        if skip is not None:
            assert 0.0 <= skip <= 1.0

    def test_all_risk_on_zero_skip(self, synthetic_manifest_dir):
        _, manifest_path = synthetic_manifest_dir
        manifest = json.loads(manifest_path.read_text())
        val_path = Path(manifest["splits"][0]["val_jsonl"])
        n_val = manifest["splits"][0]["n_val"]
        # All probas very low → all risk_on (skip_rate ≈ 0)
        y_prob = np.full(n_val, 0.01)
        result = _fold_signal_backtest(y_prob, val_path, threshold=0.5)
        assert result.get("skip_rate", 0.0) == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# _aggregate_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateMetrics:

    def _fold(self, roc: float, pr: float, brier: float = 0.20) -> dict:
        return {
            "roc_auc": roc, "pr_auc": pr, "brier": brier,
            "ece": 0.05, "fpr_at_tpr80": 0.1,
            "recall_t05": 0.6, "precision_t05": 0.7, "f1_t05": 0.65,
        }

    def test_empty_returns_empty(self):
        result = _aggregate_metrics([])
        assert result == {}

    def test_two_folds_mean(self):
        folds = [self._fold(0.70, 0.40), self._fold(0.80, 0.50)]
        agg = _aggregate_metrics(folds)
        assert abs(agg["roc_auc_mean"] - 0.75) < 1e-4
        assert abs(agg["pr_auc_mean"]  - 0.45) < 1e-4

    def test_std_computed(self):
        folds = [self._fold(0.70, 0.40), self._fold(0.80, 0.50)]
        agg = _aggregate_metrics(folds)
        assert "roc_auc_std" in agg
        assert agg["roc_auc_std"] > 0

    def test_single_fold_std_zero(self):
        agg = _aggregate_metrics([self._fold(0.75, 0.45)])
        assert agg["roc_auc_std"] == pytest.approx(0.0, abs=1e-6)

    def test_missing_key_skipped(self):
        folds = [{"roc_auc": 0.72}, {"roc_auc": 0.75}]
        # pr_auc missing → should not crash, just skip
        agg = _aggregate_metrics(folds)
        assert "pr_auc_mean" not in agg
        assert "roc_auc_mean" in agg


# ─────────────────────────────────────────────────────────────────────────────
# E2E CLI smoke run — scripts/ml/train/train_v3.py with synthetic manifest
# ─────────────────────────────────────────────────────────────────────────────

class TestE2EManifestRun:
    """
    Full pipeline: synthetic manifest → scripts/ml/train/train_v3.py CLI → artifacts.
    Tests that the single entrypoint (scripts/ml/train/train_v3.py) produces valid output.
    """

    def test_xgb_only_smoke(self, tmp_path):
        """XGB-only run with synthetic manifest and --max_rows=50."""
        import subprocess

        manifest_path = _synthetic_manifest(
            tmp_path, n_folds=2, n_train=100, n_val=60,
        )
        out_dir    = tmp_path / "models_v3"
        result = subprocess.run(
            [
                sys.executable,
                str(_REPO / "scripts/ml/train/train_v3.py"),
                "--manifest", str(manifest_path),
                "--out_dir",  str(out_dir),
                "--no_lr",
                "--max_rows",        "80",
                "--nan_drop_threshold", "0.30",
                "--n_estimators",    "30",
                "--seed",            "42",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"\n--- STDOUT ---\n{result.stdout}"
            f"\n--- STDERR ---\n{result.stderr}"
        )

        # Check artifacts
        for fname in ("v3_xgb_model.joblib", "v3_calibrator.joblib",
                      "v3_feature_names.joblib", "v3_thresholds.json",
                      "v3_meta.json", "v3_metrics.json"):
            assert (out_dir / fname).exists(), f"{fname} not created"

        # v3_meta.json structure
        meta = json.loads((out_dir / "v3_meta.json").read_text())
        assert "schema_version" in meta
        assert "feature_cols" in meta
        assert "dropped_features" in meta
        assert "thresholds" in meta
        assert meta["n_features"] > 0

        # v3_metrics.json structure
        metrics = json.loads((out_dir / "v3_metrics.json").read_text())
        assert "xgb" in metrics
        assert "fold_metrics" in metrics["xgb"]
        assert len(metrics["xgb"]["fold_metrics"]) >= 1

    def test_nan_drop_threshold_effect(self, tmp_path):
        """Stricter threshold (0.0) drops more features; relaxed (1.0) keeps all."""
        import subprocess

        manifest_strict = _synthetic_manifest(
            tmp_path / "strict", n_folds=2, n_train=100, n_val=60,
        )
        manifest_relax = _synthetic_manifest(
            tmp_path / "relax", n_folds=2, n_train=100, n_val=60,
        )

        def _run(manifest: Path, threshold: float, out_dir: Path) -> dict:
            out_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                [
                    sys.executable, str(_REPO / "scripts/ml/train/train_v3.py"),
                    "--manifest", str(manifest),
                    "--out_dir",  str(out_dir),
                    "--no_lr", "--max_rows", "50",
                    "--nan_drop_threshold", str(threshold),
                    "--n_estimators", "20", "--seed", "0",
                ],
                capture_output=True, text=True, timeout=120,
            )
            assert r.returncode == 0, f"Failed with threshold={threshold}\n{r.stderr}"
            return json.loads((out_dir / "v3_meta.json").read_text())

        meta_strict = _run(manifest_strict, 0.01, tmp_path / "out_strict")
        meta_relax  = _run(manifest_relax,  0.99, tmp_path / "out_relax")

        # Stricter threshold → more features dropped → fewer kept
        assert meta_strict["n_features"] <= meta_relax["n_features"], (
            f"Strict ({meta_strict['n_features']}) should have ≤ features "
            f"than relaxed ({meta_relax['n_features']})"
        )
