"""
tests/unit/test_reporting_v3.py
================================
Phase 4 — Unit tests for the v3 reporting pipeline.

Tests:
  - plot_ml_v3: all 8 figure types generated from synthetic data
  - plot_financial_v3: all 7 figure types generated from synthetic data
  - generate_v3_report: PDF generated with correct structure
  - No network required
  - No real model files required
  - Fast: synthetic data only

Run:
    pytest tests/unit/test_reporting_v3.py -v
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

# ── Repo root on path ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_folds_data(
    n_pos: int = 200,
    n_neg: int = 400,
    n_folds: int = 3,
    seed: int = 42,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate synthetic (y_true, y_prob_xgb, y_prob_lr) for n_folds."""
    rng = np.random.default_rng(seed)
    result = {}
    for fk in range(2, 2 + n_folds):
        y = np.array([1] * n_pos + [0] * n_neg)
        # Calibrated-ish probabilities: better than random, not perfect
        probs_pos = rng.beta(3, 2, n_pos)    # biased toward 1
        probs_neg = rng.beta(2, 4, n_neg)    # biased toward 0
        y_xgb = np.concatenate([probs_pos, probs_neg])
        y_lr  = np.concatenate([
            rng.beta(2.5, 2.5, n_pos),
            rng.beta(2, 3.5, n_neg),
        ])
        result[fk] = (y, y_xgb, y_lr)
    return result


def _make_signal_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic signal DataFrame with dates, returns, labels, probabilities."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-05", periods=n, freq="B")
    labels = rng.choice(["ok", "warn", "block"], n, p=[0.55, 0.25, 0.20])
    returns = rng.normal(0.002, 0.02, n)
    proba = rng.beta(2, 3, n)
    asset_types = rng.choice(["equity", "etf", "crypto"], n, p=[0.7, 0.2, 0.1])
    df = pd.DataFrame({
        "date": dates,
        "forward_return_20d": returns,
        "label": labels,
        "proba_non_ok": proba,
        "asset_type": asset_types,
    })
    # Apply signal exposure
    def _exp(p):
        if p >= 0.65: return 0.0
        if p >= 0.50: return 0.5
        return 1.0
    df["exposure"] = df["proba_non_ok"].apply(_exp)
    df["signal_ret"] = df["forward_return_20d"] * df["exposure"]
    return df


def _make_backtest_json() -> Dict:
    """Minimal backtest JSON matching backtest_v3.json schema."""
    return {
        "generated_at": "2026-03-03T00:00:00",
        "horizon_days": 20,
        "n_records": 1000,
        "label_distribution": {"ok": 550, "warn": 250, "block": 200},
        "signal": {
            "label": "signal", "n_periods": 1000, "cagr": 0.36,
            "sharpe_ann": 0.91, "sortino_ann": 1.2, "max_drawdown": -0.18,
            "calmar": 2.0, "hit_rate": 0.54, "avg_win": 0.025,
            "avg_loss": -0.020, "profit_factor": 1.4, "avg_exposure": 0.72,
            "n_ok": 550, "n_warn": 250, "n_block": 200,
            "vol_ann": 0.38, "vol_period": 0.107,
        },
        "always_ok": {
            "label": "always_ok", "n_periods": 1000, "cagr": 0.08,
            "sharpe_ann": 0.31, "sortino_ann": 0.4, "max_drawdown": -0.35,
            "calmar": 0.23, "hit_rate": 0.51, "avg_win": 0.020,
            "avg_loss": -0.019, "profit_factor": 1.05, "avg_exposure": 1.0,
            "vol_ann": 0.63, "vol_period": 0.176,
        },
        "always_block": {
            "label": "always_block", "n_periods": 1000, "cagr": 0.0,
            "sharpe_ann": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0,
            "vol_ann": 0.0, "vol_period": 0.0,
        },
        "by_asset_type": {
            "equity":    {"n_periods": 700, "cagr": 0.41, "sharpe_ann": 0.91, "max_drawdown": -0.40},
            "etf":       {"n_periods": 200, "cagr": 0.21, "sharpe_ann": 1.25, "max_drawdown": -0.37},
            "crypto":    {"n_periods": 100, "cagr": 0.69, "sharpe_ann": 1.33, "max_drawdown": -0.11},
        },
    }


def _make_train_json(feat_cols: list) -> Dict:
    """Minimal train_v3_report.json."""
    fold_metrics = [
        {
            "label": f"xgb_fold{k}", "n": 500, "n_pos": 200, "n_neg": 300,
            "pos_rate": 0.40, "roc_auc": 0.72 + k * 0.01, "pr_auc": 0.65,
            "brier": 0.21, "ece": 0.03, "fpr_at_tpr80": 0.28,
            "recall_t05": 0.61, "precision_t05": 0.58, "f1_t05": 0.59,
            "tp": 120, "fp": 85, "fn": 80, "tn": 215,
            "backtest": {"threshold": 0.5, "n_invested": 350, "n_skipped": 150,
                         "skip_rate": 0.30, "strategy_mean_return": 0.003,
                         "baseline_mean_return": 0.001, "sharpe_proxy": 0.85},
        }
        for k in range(4)
    ]
    return {
        "generated_at": "2026-03-03T00:00:00",
        "n_folds": 4,
        "n_features": len(feat_cols),
        "n_dropped": 3,
        "dropped_features": ["corr_spy", "beta_market", "vix_level"],
        "lr": {
            "fold_metrics": fold_metrics,
            "aggregate": {
                "roc_auc_mean": 0.74, "roc_auc_std": 0.04,
                "pr_auc_mean": 0.67, "pr_auc_std": 0.03,
                "brier_mean": 0.21, "ece_mean": 0.04,
                "fpr_at_tpr80_mean": 0.27, "f1_t05_mean": 0.60,
            },
        },
        "xgb": {
            "fold_metrics": fold_metrics,
            "aggregate": {
                "roc_auc_mean": 0.742, "roc_auc_std": 0.047,
                "pr_auc_mean": 0.68, "pr_auc_std": 0.04,
                "brier_mean": 0.22, "ece_mean": 0.035,
                "fpr_at_tpr80_mean": 0.28, "f1_t05_mean": 0.61,
            },
            "final_calibrated": {
                "label": "xgb_calibrated_final",
                "roc_auc": 0.782, "pr_auc": 0.75, "brier": 0.189,
                "ece": 0.028, "fpr_at_tpr80": 0.22,
                "recall_t05": 0.65, "precision_t05": 0.62, "f1_t05": 0.63,
                "tp": 130, "fp": 80, "fn": 70, "tn": 220,
            },
        },
        "feature_importance_top20": {f: round(0.05 * (20 - i) / 20, 4) for i, f in enumerate(feat_cols[:20])},
        "thresholds": {
            "t_lo": 0.4863, "t_hi": 0.6654,
            "target_fpr_lo": 0.10, "target_fpr_hi": 0.25,
            "fitted_on": "last_fold_val", "model": "xgb_calibrated",
        },
    }


def _make_dataset_json() -> Dict:
    return {
        "valid": True,
        "total_samples": 54824,
        "samples_per_label": {"ok": 29927, "warn": 12940, "block": 11957},
        "label_distribution_pct": {"ok": 54.6, "warn": 23.6, "block": 21.8},
        "samples_per_asset_type": {
            "equity": 41644, "etf": 10620, "crypto": 560,
            "fx": 1040, "commodity": 640, "rate": 320,
        },
    }


def _make_drift_json() -> Dict:
    return {
        "generated_at": "2026-03-03T00:00:00Z",
        "reference_size": 7635,
        "current_size": 47161,
        "n_features_analyzed": 51,
        "global_drift": {
            "global_psi_mean": 0.039,
            "global_psi_max": 0.12,
            "n_features_drift": 0,
            "n_features_moderate": 1,
            "n_features_stable": 50,
            "n_features_total": 51,
            "drift_level": "low",
            "top5_drifting": ["vix_level", "corr_vix", "vol_of_vol", "kurtosis_excess", "skew"],
        },
        "label_drift": {
            "ref_non_ok_rate": 0.41,
            "cur_non_ok_rate": 0.46,
            "non_ok_shift": 0.046,
            "label_drift_significant": False,
        },
    }


def _make_manifest_json(tmp: Path) -> Dict:
    return {
        "n_folds": 4,
        "splits": [
            {
                "fold": k,
                "train_start": "2010-01-01", "train_end": f"201{k}-12-31",
                "val_start": f"201{k+1}-01-01", "val_end": f"201{k+1}-12-31",
                "n_train": 200 * k, "n_val": 500,
                "train_jsonl": str(tmp / f"fold_{k}/train.jsonl"),
                "val_jsonl":   str(tmp / f"fold_{k}/val.jsonl"),
                "purge_days": 20,
            }
            for k in range(2, 6)
        ],
        "params": {"purge_days": 20, "embargo_days": 5},
    }


# ── Phase 1: ML visualization tests ───────────────────────────────────────────

class TestPlotML:
    """Tests for scripts/ml/reporting/plot_ml_v3.py"""

    def test_roc_curves_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_roc_curves
        folds = _make_folds_data()
        paths = plot_roc_curves(folds, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"
        assert paths[0].stat().st_size > 5_000

    def test_pr_curves_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_pr_curves
        folds = _make_folds_data()
        paths = plot_pr_curves(folds, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_calibration_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_calibration
        folds = _make_folds_data()
        paths = plot_calibration(folds, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_prob_distributions_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_prob_distributions
        folds = _make_folds_data()
        paths = plot_prob_distributions(folds, tmp_path, thresholds={"t_lo": 0.49, "t_hi": 0.66})
        assert len(paths) == 1
        assert paths[0].exists()

    def test_lift_curve_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_lift_curve
        folds = _make_folds_data()
        paths = plot_lift_curve(folds, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_confusion_matrix_generated(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_confusion_matrix
        folds = _make_folds_data()
        paths = plot_confusion_matrix(folds, tmp_path, thresholds={"t_lo": 0.49})
        assert len(paths) == 1
        assert paths[0].exists()

    def test_empty_folds_no_crash(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_roc_curves, plot_pr_curves
        empty = {}
        assert plot_roc_curves(empty, tmp_path) == []
        assert plot_pr_curves(empty, tmp_path) == []

    def test_metrics_per_fold_from_json(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_metrics_summary
        feat_cols = [f"feat_{i}" for i in range(10)]
        train_data = _make_train_json(feat_cols)
        metrics_path = tmp_path / "train_report.json"
        metrics_path.write_text(json.dumps(train_data), encoding="utf-8")
        paths = plot_metrics_summary(metrics_path, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_missing_metrics_no_crash(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_metrics_summary
        paths = plot_metrics_summary(tmp_path / "nonexistent.json", tmp_path)
        assert paths == []

    def test_output_dir_created(self, tmp_path):
        from scripts.ml.reporting.plot_ml_v3 import plot_roc_curves
        nested = tmp_path / "deep" / "nested" / "plots"
        folds = _make_folds_data(n_folds=1)
        plot_roc_curves(folds, nested)
        assert nested.exists()

    def test_png_is_valid_image(self, tmp_path):
        """Check PNG header bytes."""
        from scripts.ml.reporting.plot_ml_v3 import plot_roc_curves
        folds = _make_folds_data(n_folds=1)
        paths = plot_roc_curves(folds, tmp_path)
        assert paths
        data = paths[0].read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG file"


# ── Phase 2: Financial visualization tests ────────────────────────────────────

class TestPlotFinancial:
    """Tests for scripts/ml/reporting/plot_financial_v3.py"""

    def test_cumulative_returns_no_df(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_cumulative_returns
        bt = _make_backtest_json()
        paths = plot_cumulative_returns(None, bt, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_cumulative_returns_with_df(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_cumulative_returns
        df = _make_signal_df()
        bt = _make_backtest_json()
        paths = plot_cumulative_returns(df, bt, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_drawdown_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_drawdown
        df = _make_signal_df()
        bt = _make_backtest_json()
        paths = plot_drawdown(df, bt, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_return_distributions_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_return_distributions
        df = _make_signal_df()
        paths = plot_return_distributions(df, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_return_distributions_no_df(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_return_distributions
        paths = plot_return_distributions(None, tmp_path)
        assert paths == []

    def test_skip_rate_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_skip_rate
        df = _make_signal_df(n=2000)
        bt = _make_backtest_json()
        paths = plot_skip_rate(df, bt, t_lo=0.50, out_dir=tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_rolling_sharpe_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_rolling_sharpe
        df = _make_signal_df(n=2000)
        bt = _make_backtest_json()
        paths = plot_rolling_sharpe(df, bt, tmp_path, window_days=126)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_performance_by_asset_type_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_performance_by_asset_type
        bt = _make_backtest_json()
        paths = plot_performance_by_asset_type(bt, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_metrics_card_generated(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_metrics_card
        bt = _make_backtest_json()
        paths = plot_metrics_card(bt, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_empty_by_asset_type_no_crash(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_performance_by_asset_type
        bt = {"by_asset_type": {}}
        paths = plot_performance_by_asset_type(bt, tmp_path)
        assert paths == []

    def test_rolling_sharpe_no_df(self, tmp_path):
        from scripts.ml.reporting.plot_financial_v3 import plot_rolling_sharpe
        bt = _make_backtest_json()
        paths = plot_rolling_sharpe(None, bt, tmp_path)
        assert paths == []


# ── Phase 3: PDF report tests ─────────────────────────────────────────────────

class TestGeneratePDF:
    """Tests for scripts/ml/reporting/generate_v3_report.py"""

    def _setup_tmp(self, tmp_path: Path):
        """Create all required JSON files and dummy PNG stubs."""
        metrics_dir = tmp_path / "metrics"
        plots_dir   = tmp_path / "plots"
        fin_dir     = tmp_path / "fin_plots"
        models_dir  = tmp_path / "models"
        metrics_dir.mkdir()
        plots_dir.mkdir()
        fin_dir.mkdir()
        models_dir.mkdir()

        feat_cols = [f"feat_{i}" for i in range(15)]

        # Write JSON files
        (metrics_dir / "train_v3_report.json").write_text(
            json.dumps(_make_train_json(feat_cols)), encoding="utf-8")
        (metrics_dir / "backtest_v3.json").write_text(
            json.dumps(_make_backtest_json()), encoding="utf-8")
        (metrics_dir / "v3_dataset_report.json").write_text(
            json.dumps(_make_dataset_json()), encoding="utf-8")
        (metrics_dir / "qa_v3_report.json").write_text(
            json.dumps({"verdict": {"ok": True}, "n_rows": 54824}), encoding="utf-8")
        (metrics_dir / "drift_v3_report.json").write_text(
            json.dumps(_make_drift_json()), encoding="utf-8")

        # Write manifest
        manifest = _make_manifest_json(tmp_path)
        manifest_path = tmp_path / "splits_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Create minimal 1x1 transparent PNG stubs using matplotlib
        import matplotlib.pyplot as plt
        for plot_name in [
            "roc_curves", "pr_curves", "calibration", "prob_distributions",
            "lift_curve", "confusion_matrices", "feature_importance", "metrics_per_fold",
        ]:
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.plot([0, 1], [0, 1])
            ax.set_title(plot_name)
            fig.savefig(plots_dir / f"{plot_name}.png", dpi=50)
            plt.close(fig)

        for plot_name in [
            "cumulative_returns", "drawdown", "return_distributions",
            "skip_rate_rolling", "rolling_sharpe", "performance_by_asset_type",
            "backtest_metrics_card",
        ]:
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.plot([0, 1], [1, 0])
            ax.set_title(plot_name)
            fig.savefig(fin_dir / f"{plot_name}.png", dpi=50)
            plt.close(fig)

        return metrics_dir, plots_dir, fin_dir, manifest_path, models_dir

    def test_pdf_generated(self, tmp_path):
        from scripts.ml.reporting.generate_v3_report import build_report
        metrics_dir, plots_dir, fin_dir, manifest_path, models_dir = self._setup_tmp(tmp_path)
        out_path = tmp_path / "report.pdf"
        result = build_report(
            metrics_dir=metrics_dir,
            plots_dir=plots_dir,
            fin_dir=fin_dir,
            manifest_path=manifest_path,
            model_dir=models_dir,
            out_path=out_path,
        )
        assert result.exists()
        assert result.suffix == ".pdf"
        assert result.stat().st_size > 10_000, "PDF is suspiciously small"

    def test_pdf_has_correct_magic_bytes(self, tmp_path):
        from scripts.ml.reporting.generate_v3_report import build_report
        metrics_dir, plots_dir, fin_dir, manifest_path, models_dir = self._setup_tmp(tmp_path)
        out_path = tmp_path / "report.pdf"
        build_report(
            metrics_dir=metrics_dir, plots_dir=plots_dir, fin_dir=fin_dir,
            manifest_path=manifest_path, model_dir=models_dir, out_path=out_path,
        )
        # PDF files start with %PDF-
        header = out_path.read_bytes()[:5]
        assert header == b"%PDF-", f"Invalid PDF header: {header!r}"

    def test_pdf_missing_plots_no_crash(self, tmp_path):
        """PDF should generate gracefully even if plot PNGs are missing."""
        from scripts.ml.reporting.generate_v3_report import build_report
        metrics_dir, _, _, manifest_path, models_dir = self._setup_tmp(tmp_path)
        # empty plot dirs — no PNGs
        empty_plots = tmp_path / "empty_plots"
        empty_plots.mkdir()
        out_path = tmp_path / "report_no_plots.pdf"
        build_report(
            metrics_dir=metrics_dir, plots_dir=empty_plots, fin_dir=empty_plots,
            manifest_path=manifest_path, model_dir=models_dir, out_path=out_path,
        )
        assert out_path.exists()

    def test_pdf_missing_json_no_crash(self, tmp_path):
        """PDF should generate with empty section when JSON is missing."""
        from scripts.ml.reporting.generate_v3_report import build_report
        # Only create minimal JSONs
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "train_v3_report.json").write_text("{}", encoding="utf-8")
        (metrics_dir / "backtest_v3.json").write_text(
            json.dumps({"signal": {}, "always_ok": {}, "by_asset_type": {}}),
            encoding="utf-8")
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        out_path = tmp_path / "report_minimal.pdf"
        build_report(
            metrics_dir=metrics_dir, plots_dir=plots_dir, fin_dir=plots_dir,
            manifest_path=tmp_path / "nonexistent_manifest.json",
            model_dir=tmp_path / "models",
            out_path=out_path,
        )
        assert out_path.exists()

    def test_output_dir_created(self, tmp_path):
        """PDF generator should create output directory if missing."""
        from scripts.ml.reporting.generate_v3_report import build_report
        metrics_dir, plots_dir, fin_dir, manifest_path, models_dir = self._setup_tmp(tmp_path)
        nested_out = tmp_path / "deep" / "nested" / "report.pdf"
        build_report(
            metrics_dir=metrics_dir, plots_dir=plots_dir, fin_dir=fin_dir,
            manifest_path=manifest_path, model_dir=models_dir, out_path=nested_out,
        )
        assert nested_out.exists()

    def test_sanitize_function(self):
        """_s() must produce latin-1 safe strings."""
        from scripts.ml.reporting.generate_v3_report import _s
        # These chars should be replaced
        text = "Signal -- Sharpe 0.91 vs BM 0.31"
        result = _s(text)
        result.encode("latin-1")  # should not raise


# ── Phase 4: Integration — generate_all wrappers ──────────────────────────────

class TestGenerateAll:
    """Integration tests for generate_all() functions with minimal real data."""

    def test_ml_generate_all_no_model(self, tmp_path):
        """generate_all should return empty dict gracefully without models."""
        from scripts.ml.reporting.plot_ml_v3 import generate_all
        # Create empty manifest
        manifest = {"splits": []}
        mp = tmp_path / "splits_manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        metrics_p = tmp_path / "train.json"
        metrics_p.write_text("{}", encoding="utf-8")
        result = generate_all(
            manifest_path=mp,
            model_dir=tmp_path / "no_models",
            metrics_path=metrics_p,
            out_dir=tmp_path / "plots",
        )
        # No model → no fold predictions, but feature importance might still be skipped
        assert isinstance(result, dict)

    def test_financial_generate_all_no_model(self, tmp_path):
        """generate_all should return backtest-only charts without model."""
        from scripts.ml.reporting.plot_financial_v3 import generate_all
        bt_path = tmp_path / "backtest.json"
        bt_path.write_text(json.dumps(_make_backtest_json()), encoding="utf-8")
        result = generate_all(
            backtest_path=bt_path,
            manifest_path=tmp_path / "no_manifest.json",
            model_dir=tmp_path / "no_models",
            out_dir=tmp_path / "fin_plots",
        )
        # At minimum, by_asset_type and metrics_card should be generated
        assert "backtest_metrics_card" in result
        assert "performance_by_asset_type" in result

# ── Phase 5: Robustness module (plot_robustness_v3) ───────────────────────────

class TestPlotRobustness:
    """Unit tests for plot_robustness_v3 — all functions with synthetic data."""

    def test_recent_fold_table_generated(self, tmp_path):
        """plot_recent_fold_table produces a PNG from fold_metrics JSON."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_recent_fold_table
        train = _make_train_json(["vol_ann", "mdd", "var95"])
        paths = plot_recent_fold_table(train, tmp_path)
        assert len(paths) == 1
        p = paths[0]
        assert p.exists()
        assert p.stat().st_size > 1000
        assert p.suffix == ".png"

    def test_recent_fold_table_empty_metrics(self, tmp_path):
        """plot_recent_fold_table returns [] when no fold_metrics."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_recent_fold_table
        paths = plot_recent_fold_table({}, tmp_path)
        assert paths == []

    def test_auc_bootstrap_hist_with_data(self, tmp_path):
        """Bootstrap AUC histogram generated with synthetic fold data."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_auc_bootstrap_hist
        folds_data = _make_folds_data(n_pos=150, n_neg=350, n_folds=2)
        paths = plot_auc_bootstrap_hist(folds_data, tmp_path, n_boot=50)
        assert len(paths) == 1
        p = paths[0]
        assert p.exists()
        assert p.stat().st_size > 5000
        # JSON CI file should also be written
        ci_json = tmp_path / "bootstrap_auc_ci.json"
        assert ci_json.exists()
        data = json.loads(ci_json.read_text())
        assert "roc_auc" in data
        assert "pr_auc" in data
        assert data["roc_auc"]["ci_lo_95"] <= data["roc_auc"]["ci_hi_95"]

    def test_auc_bootstrap_hist_no_data(self, tmp_path):
        """Bootstrap AUC produces stub PNG when no folds_data."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_auc_bootstrap_hist
        paths = plot_auc_bootstrap_hist({}, tmp_path, n_boot=50)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_bootstrap_metric_function(self):
        """_bootstrap_metric returns correct shape and CI direction."""
        from scripts.ml.reporting.plot_robustness_v3 import _bootstrap_metric, _roc_auc_fn
        rng = np.random.default_rng(42)
        y = np.array([1] * 100 + [0] * 200)
        p = np.concatenate([rng.beta(3, 2, 100), rng.beta(2, 4, 200)])
        mean, lo, hi, samples = _bootstrap_metric(_roc_auc_fn, y, p, n_boot=100, seed=0)
        assert 0 < lo <= mean <= hi <= 1.0
        assert len(samples) == 100

    def test_sharpe_significance_compute(self):
        """compute_sharpe_significance returns valid stats dict."""
        from scripts.ml.reporting.plot_robustness_v3 import compute_sharpe_significance
        rng = np.random.default_rng(42)
        sig_rets = rng.normal(0.003, 0.02, 500)
        bm_rets  = rng.normal(0.001, 0.025, 500)
        stats = compute_sharpe_significance(sig_rets, bm_rets, n_boot=100)
        assert "mean_diff" in stats
        assert "p_value" in stats
        assert 0.0 <= stats["p_value"] <= 1.0
        assert stats["ci_lo"] <= stats["ci_hi"]

    def test_sharpe_bootstrap_hist_with_df(self, tmp_path):
        """Sharpe bootstrap histogram generated from signal DataFrame."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_sharpe_bootstrap_hist
        df = _make_signal_df(n=500)
        paths = plot_sharpe_bootstrap_hist(df, t_lo=0.5, t_hi=0.65, out_dir=tmp_path, n_boot=50)
        assert len(paths) == 1
        p = paths[0]
        assert p.exists()
        assert p.stat().st_size > 5000
        # Significance JSON written
        sig_json = tmp_path / "bootstrap_sharpe_significance.json"
        assert sig_json.exists()
        sig = json.loads(sig_json.read_text())
        assert "p_value" in sig
        assert "mean_diff" in sig

    def test_sharpe_bootstrap_hist_no_df(self, tmp_path):
        """Sharpe bootstrap produces stub when df=None."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_sharpe_bootstrap_hist
        paths = plot_sharpe_bootstrap_hist(None, t_lo=0.5, t_hi=0.65, out_dir=tmp_path, n_boot=50)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_confusion_metrics_per_fold_from_json(self, tmp_path):
        """Confusion metrics plot generated from fold_metrics JSON (no raw data)."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_confusion_metrics_per_fold
        train = _make_train_json(["vol_ann", "mdd"])
        paths = plot_confusion_metrics_per_fold(train, {}, tmp_path)
        assert len(paths) == 1
        p = paths[0]
        assert p.exists()
        assert p.stat().st_size > 5000

    def test_confusion_metrics_per_fold_from_raw(self, tmp_path):
        """Confusion metrics plot from raw folds_data predictions."""
        from scripts.ml.reporting.plot_robustness_v3 import plot_confusion_metrics_per_fold
        folds_data = _make_folds_data(n_pos=150, n_neg=350, n_folds=3)
        paths = plot_confusion_metrics_per_fold({}, folds_data, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_generate_all_robustness(self, tmp_path):
        """generate_all_robustness runs end-to-end with synthetic data."""
        from scripts.ml.reporting.plot_robustness_v3 import generate_all_robustness
        train = _make_train_json(["vol_ann", "mdd", "var95"])
        folds_data = _make_folds_data(n_pos=200, n_neg=500, n_folds=3)
        df = _make_signal_df(n=600)
        result = generate_all_robustness(
            folds_data=folds_data,
            metrics_data=train,
            df_signal=df,
            t_lo=0.5,
            t_hi=0.65,
            out_dir=tmp_path,
            n_boot=50,
        )
        # All 4 objectives should produce outputs
        assert "recent_fold_table"         in result
        assert "auc_bootstrap_hist"        in result
        assert "sharpe_bootstrap_hist"     in result
        assert "confusion_metrics_per_fold" in result
        for name, path in result.items():
            assert path.exists(), f"{name} PNG missing: {path}"

    def test_all_pngs_valid_header(self, tmp_path):
        """All robustness PNGs start with PNG magic bytes."""
        from scripts.ml.reporting.plot_robustness_v3 import generate_all_robustness
        train = _make_train_json(["vol_ann", "mdd", "var95"])
        folds_data = _make_folds_data(n_pos=200, n_neg=500, n_folds=3)
        df = _make_signal_df(n=600)
        result = generate_all_robustness(
            folds_data=folds_data, metrics_data=train,
            df_signal=df, t_lo=0.5, t_hi=0.65,
            out_dir=tmp_path, n_boot=50,
        )
        PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
        for name, path in result.items():
            with open(path, "rb") as f:
                header = f.read(8)
            assert header == PNG_MAGIC, f"{name}: not a valid PNG"
