"""
tests/unit/test_qa_v3.py
==========================
Tests unitaires pour scripts/ml/data/qa_dataset_v3.py

Vérifie:
  - Détection correcte des NaN
  - Détection des incohérences temporelles
  - Détection des doublons (ticker + window_end_date)
  - Verdict correct (ok / issues)
  - Rapport JSON bien formé

Rapide, sans réseau.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pytest

# Repo root
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.ml.data.qa_dataset_v3 import run_qa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    ticker: str,
    wed: str,
    label: str = "ok",
    inject_nan: bool = False,
    window_start: str = None,
    label_start: str = None,
    label_end: str = None,
) -> dict:
    """Create a synthetic v3 record."""
    wed_date = date.fromisoformat(wed)
    wsd = window_start or str(wed_date - timedelta(days=252))
    lsd = label_start  or str(wed_date + timedelta(days=1))
    led = label_end    or str(wed_date + timedelta(days=20))

    feat_val = None if inject_nan else 0.25

    return {
        "version":           "v3",
        "label":             label,
        "target_non_ok":     0 if label == "ok" else 1,
        "window_start_date": wsd,
        "window_end_date":   wed,
        "label_start_date":  lsd,
        "label_end_date":    led,
        "label_end_date_60d": str(wed_date + timedelta(days=60)),
        "forward_return_5d":  0.01,
        "forward_return_10d": 0.015,
        "forward_return_20d": 0.02,
        "forward_return_60d": 0.04,
        "features": {
            "asset_type": "equity",
            "market":     "US",
            "ticker":     ticker,
            "vol_ann":    feat_val,
            "var95":      feat_val,
            "max_dd":     feat_val,
            "rsi":        feat_val,
        },
    }


def _write_and_run_qa(records: List[dict]) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_in = Path(f.name)

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False
    ) as f:
        tmp_out = Path(f.name)

    try:
        report = run_qa(tmp_in)
        return report
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests — basic structure
# ---------------------------------------------------------------------------

class TestQABasicStructure:

    def test_report_has_required_keys(self):
        records = [_make_record("AAPL", "2022-01-15")]
        report = _write_and_run_qa(records)
        required = [
            "n_rows", "n_tickers", "by_asset_type", "by_market",
            "label_distribution", "temporal_checks", "duplicates",
            "nan_top30_features", "forward_return_stats", "verdict",
        ]
        for k in required:
            assert k in report, f"Missing key: {k}"

    def test_n_rows_correct(self):
        records = [_make_record("AAPL", f"2022-0{m}-15") for m in range(1, 6)]
        report = _write_and_run_qa(records)
        assert report["n_rows"] == 5

    def test_n_tickers_correct(self):
        records = [
            _make_record("AAPL", "2022-01-15"),
            _make_record("MSFT", "2022-01-15"),
            _make_record("AAPL", "2022-02-15"),
        ]
        report = _write_and_run_qa(records)
        assert report["n_tickers"] == 2

    def test_label_distribution(self):
        records = [
            _make_record("A", "2022-01-15", label="ok"),
            _make_record("A", "2022-02-15", label="warn"),
            _make_record("A", "2022-03-15", label="block"),
        ]
        report = _write_and_run_qa(records)
        dist = report["label_distribution"]
        assert dist.get("ok") == 1
        assert dist.get("warn") == 1
        assert dist.get("block") == 1


# ---------------------------------------------------------------------------
# Tests — NaN detection
# ---------------------------------------------------------------------------

class TestQANanDetection:

    def test_nan_detected(self):
        """Records with None feature values should appear in nan_top30."""
        records = [
            _make_record("AAPL", f"2022-0{m}-15", inject_nan=True)
            for m in range(1, 6)
        ]
        report = _write_and_run_qa(records)
        nan_map = report["nan_top30_features"]
        # vol_ann, var95, max_dd, rsi should all be 100% NaN
        assert "vol_ann" in nan_map
        assert nan_map["vol_ann"] == 100.0

    def test_no_nan_clean_dataset(self):
        """Clean dataset should have 0% NaN for non-macro features."""
        records = [
            _make_record("AAPL", f"2022-0{m}-15", inject_nan=False)
            for m in range(1, 6)
        ]
        report = _write_and_run_qa(records)
        non_macro_high_nan = report["non_macro_high_nan"]
        # vol_ann, var95, max_dd, rsi are NOT macro features — should be 0% NaN
        assert "vol_ann" not in non_macro_high_nan
        assert "var95"   not in non_macro_high_nan

    def test_partial_nan(self):
        """3/5 records with NaN → 60% NaN rate."""
        records = [
            _make_record("AAPL", f"2022-0{m}-15", inject_nan=(m <= 3))
            for m in range(1, 6)
        ]
        report = _write_and_run_qa(records)
        nan_map = report["nan_top30_features"]
        assert "vol_ann" in nan_map
        assert nan_map["vol_ann"] == 60.0


# ---------------------------------------------------------------------------
# Tests — temporal coherence
# ---------------------------------------------------------------------------

class TestQATemporalChecks:

    def test_temporal_clean(self):
        """Well-formed records should have 0 temporal violations."""
        records = [_make_record("AAPL", "2022-01-15")]
        report = _write_and_run_qa(records)
        tc = report["temporal_checks"]
        assert tc["n_violation"] == 0
        assert tc["leakage_free"] is True

    def test_window_end_equals_label_start_is_violation(self):
        """window_end_date == label_start_date should be caught (need window_end < label_start)."""
        rec = _make_record(
            "AAPL", "2022-01-15",
            label_start="2022-01-15",   # same as window_end → violation
            label_end="2022-02-15",
        )
        report = _write_and_run_qa([rec])
        tc = report["temporal_checks"]
        assert tc["n_violation"] >= 1
        assert tc["leakage_free"] is False

    def test_label_start_after_label_end_is_violation(self):
        """label_start > label_end should be caught."""
        rec = _make_record(
            "AAPL", "2022-01-15",
            label_start="2022-03-01",  # after label_end
            label_end="2022-02-01",    # before label_start → inverted
        )
        report = _write_and_run_qa([rec])
        tc = report["temporal_checks"]
        assert tc["n_violation"] >= 1

    def test_window_start_after_window_end_is_violation(self):
        """window_start > window_end should be caught."""
        rec = _make_record(
            "AAPL", "2022-01-15",
            window_start="2022-06-01",   # after window_end → violation
        )
        report = _write_and_run_qa([rec])
        tc = report["temporal_checks"]
        assert tc["n_violation"] >= 1


# ---------------------------------------------------------------------------
# Tests — duplicate detection
# ---------------------------------------------------------------------------

class TestQADuplicates:

    def test_no_duplicates(self):
        records = [
            _make_record("AAPL", "2022-01-15"),
            _make_record("AAPL", "2022-02-15"),
            _make_record("MSFT", "2022-01-15"),
        ]
        report = _write_and_run_qa(records)
        assert report["duplicates"]["n_duplicates"] == 0

    def test_duplicate_detected(self):
        """Same ticker + window_end_date twice → 1 duplicate."""
        records = [
            _make_record("AAPL", "2022-01-15"),
            _make_record("AAPL", "2022-01-15"),  # exact duplicate
        ]
        report = _write_and_run_qa(records)
        assert report["duplicates"]["n_duplicates"] >= 1

    def test_multiple_duplicates(self):
        records = [
            _make_record("AAPL", "2022-01-15"),
            _make_record("AAPL", "2022-01-15"),
            _make_record("MSFT", "2022-01-15"),
            _make_record("MSFT", "2022-01-15"),
        ]
        report = _write_and_run_qa(records)
        assert report["duplicates"]["n_duplicates"] >= 2


# ---------------------------------------------------------------------------
# Tests — verdict
# ---------------------------------------------------------------------------

class TestQAVerdict:

    def test_clean_dataset_verdict_ok(self):
        records = [_make_record("AAPL", f"2022-0{m}-15") for m in range(1, 6)]
        report = _write_and_run_qa(records)
        assert report["verdict"]["ok"] is True
        assert len(report["verdict"]["issues"]) == 0

    def test_temporal_violation_sets_verdict_issue(self):
        rec = _make_record(
            "AAPL", "2022-01-15",
            label_start="2022-01-15",   # violation
        )
        report = _write_and_run_qa([rec])
        assert report["verdict"]["ok"] is False
        assert len(report["verdict"]["issues"]) > 0

    def test_duplicate_sets_verdict_issue(self):
        records = [
            _make_record("AAPL", "2022-01-15"),
            _make_record("AAPL", "2022-01-15"),
        ]
        report = _write_and_run_qa(records)
        assert report["verdict"]["ok"] is False


# ---------------------------------------------------------------------------
# Tests — forward_return stats
# ---------------------------------------------------------------------------

class TestQAForwardReturnStats:

    def test_forward_return_stats_present(self):
        records = [_make_record("AAPL", f"2022-0{m}-15") for m in range(1, 6)]
        report = _write_and_run_qa(records)
        frs = report["forward_return_stats"]
        for h in ["forward_return_5d", "forward_return_10d",
                  "forward_return_20d", "forward_return_60d"]:
            assert h in frs
            assert frs[h]["n"] == 5

    def test_forward_return_stats_correct_values(self):
        """All forward returns are 0.02 → mean should be 0.02."""
        records = [_make_record("AAPL", f"2022-0{m}-15") for m in range(1, 6)]
        report = _write_and_run_qa(records)
        frs = report["forward_return_stats"]["forward_return_20d"]
        assert abs(frs["mean"] - 0.02) < 1e-4
        assert abs(frs["min"]  - 0.02) < 1e-4
        assert abs(frs["max"]  - 0.02) < 1e-4
