"""
tests/unit/test_split_v3.py
============================
Tests unitaires pour scripts/ml/data/split_v3_time.py

Vérifie:
  - Absence de leakage (aucun index commun entre train et val)
  - Purge effectif (pas de records dans la fenêtre purgée)
  - Propriété expanding-window (train_k ⊇ train_{k-1})
  - Pas de crash sur un petit dataset synthétique

Rapide, sans réseau, sans dépendance externe.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

# Repo root
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.ml.data.split_v3_time import (
    load_jsonl,
    generate_splits,
    verify_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(ticker: str, wed: date, label: str = "ok") -> dict:
    return {
        "version":           "v3",
        "label":             label,
        "target_non_ok":     0 if label == "ok" else 1,
        "window_start_date": str(wed - timedelta(days=252)),
        "window_end_date":   str(wed),
        "label_start_date":  str(wed + timedelta(days=1)),
        "label_end_date":    str(wed + timedelta(days=20)),
        "forward_return_20d": 0.02,
        "features": {
            "asset_type": "equity",
            "market":     "US",
            "ticker":     ticker,
        },
    }


def _write_jsonl(records: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_synthetic_dataset(n_days: int = 800, step: int = 10) -> list:
    """
    Synthetic dataset: 2 tickers × windows every `step` days over n_days.
    Gives ~160 records.
    """
    records = []
    base = date(2015, 1, 1)
    for d in range(0, n_days, step):
        wed = base + timedelta(days=d)
        records.append(_make_record("AAPL", wed, label="ok"))
        records.append(_make_record("MSFT", wed, label="block" if d % 50 == 0 else "ok"))
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitV3:

    def setup_method(self):
        """Create a synthetic JSONL in a temp file for each test."""
        self.records = _make_synthetic_dataset(n_days=800, step=10)
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        _write_jsonl(self.records, Path(self._tmp.name))
        self._tmp.close()
        self._path = Path(self._tmp.name)

    def teardown_method(self):
        self._path.unlink(missing_ok=True)

    def _load_and_split(self, n_folds=3, purge_days=20, embargo_days=5):
        records, raw_lines = load_jsonl(self._path)
        splits = generate_splits(
            records,
            n_folds=n_folds,
            purge_days=purge_days,
            embargo_days=embargo_days,
            min_train_samples=5,
        )
        return records, splits

    def test_no_index_overlap(self):
        """No index should appear in both train and val within any fold."""
        records, splits = self._load_and_split()
        assert len(splits) > 0, "Expected at least 1 valid fold"

        for fold in splits:
            t_set = set(fold["train_indices"])
            v_set = set(fold["val_indices"])
            overlap = t_set & v_set
            assert len(overlap) == 0, (
                f"Fold {fold['fold']}: {len(overlap)} indices in both train and val"
            )

    def test_purge_enforced(self):
        """No val record's window_end_date should be <= train_cutoff."""
        import pandas as pd
        records, splits = self._load_and_split(purge_days=20)
        for fold in splits:
            cutoff = pd.Timestamp(fold["train_cutoff"])
            for idx in fold["val_indices"]:
                wed = pd.Timestamp(records[idx]["window_end_date"])
                assert wed > cutoff, (
                    f"Fold {fold['fold']}: val record idx={idx} date={wed.date()} "
                    f"<= train_cutoff={cutoff.date()} — purge violated"
                )

    def test_expanding_window(self):
        """Each fold's train set must be a superset of the previous fold's train set."""
        records, splits = self._load_and_split(n_folds=3)
        if len(splits) < 2:
            pytest.skip("Need at least 2 folds for expanding-window test")

        for i in range(1, len(splits)):
            prev = set(splits[i - 1]["train_indices"])
            curr = set(splits[i]["train_indices"])
            assert prev.issubset(curr), (
                f"Fold {splits[i]['fold']} train NOT superset of fold {splits[i-1]['fold']} "
                "(expanding-window property violated)"
            )

    def test_verify_splits_passes(self):
        """verify_splits() should not raise on a clean split."""
        records, splits = self._load_and_split()
        # Should not raise
        verify_splits(splits, records)

    def test_val_strictly_after_train(self):
        """All val dates must be strictly after all train dates (max train < min val)."""
        import pandas as pd
        records, splits = self._load_and_split()
        for fold in splits:
            if not fold["train_indices"] or not fold["val_indices"]:
                continue
            train_dates = [pd.Timestamp(records[i]["window_end_date"]) for i in fold["train_indices"]]
            val_dates   = [pd.Timestamp(records[i]["window_end_date"]) for i in fold["val_indices"]]
            # Train dates should not overlap with val dates
            max_train = max(train_dates)
            min_val   = min(val_dates)
            assert max_train < min_val, (
                f"Fold {fold['fold']}: max train date {max_train.date()} >= min val date {min_val.date()}"
            )

    def test_n_folds_respected(self):
        """Output should have ≤ n_folds folds (some may be skipped if too few train samples)."""
        records, splits = self._load_and_split(n_folds=4)
        assert len(splits) <= 4

    def test_no_crash_small_dataset(self):
        """Should not crash on a minimal 2-ticker dataset (gracefully skip under-sized folds)."""
        tiny = [_make_record("A", date(2020, 1, 1) + timedelta(days=d * 30)) for d in range(10)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_jsonl(tiny, Path(f.name))
            tmp_path = Path(f.name)
        try:
            records, _ = load_jsonl(tmp_path)
            # May return 0 folds if dataset is too small; should not raise
            splits = generate_splits(records, n_folds=3, purge_days=20,
                                     embargo_days=5, min_train_samples=5)
            # splits may be empty or partial — that's OK
            assert isinstance(splits, list)
        finally:
            tmp_path.unlink(missing_ok=True)
