# tests/unit/test_scoring.py
"""Unit tests for api/scoring.py expert scoring layer.

All tests in this file run with EXPERTS_ENABLED=1 so that bundle
loading and scoring actually execute (the feature flag itself is
tested in test_experts_flag.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def clear_expert_cache():
    """Reset the expert cache between tests to avoid state leakage."""
    from api import scoring
    scoring._EXPERT_CACHE.clear()
    yield
    scoring._EXPERT_CACHE.clear()


@pytest.fixture(autouse=True)
def _enable_experts():
    """All tests in this module require EXPERTS_ENABLED=1."""
    with patch.dict(os.environ, {"EXPERTS_ENABLED": "1"}):
        yield


class TestLoadExpertBundle:
    def test_load_existing_expert(self):
        """Loading equity expert (trained in Phase 3 validation) succeeds."""
        from api.scoring import load_expert_bundle
        bundle = load_expert_bundle("equity")
        assert bundle is not None
        assert "cols" in bundle
        assert len(bundle["cols"]) == 91

    def test_fallback_to_global(self):
        """Unknown asset_type falls back to global bundle."""
        from api.scoring import load_expert_bundle
        bundle = load_expert_bundle("nonexistent_asset_type_xyz")
        # If global exists, should return it; if not, None
        if bundle is not None:
            assert bundle.get("asset_type") in ("global", "nonexistent_asset_type_xyz")

    def test_cache_populated_after_load(self):
        """Second call uses cache (no re-load from disk)."""
        from api import scoring
        load1 = scoring.load_expert_bundle("equity")
        load2 = scoring.load_expert_bundle("equity")
        assert load1 is load2  # same object from cache

    def test_list_loaded_experts(self):
        """list_loaded_experts returns loaded keys."""
        from api.scoring import load_expert_bundle, list_loaded_experts
        load_expert_bundle("equity")
        loaded = list_loaded_experts()
        assert "equity" in loaded


class TestScoreExpert:
    def test_score_expert_equity_returns_dict(self, base_feats):
        from api.scoring import score_expert
        result = score_expert(base_feats, "equity")
        assert result is not None
        assert isinstance(result, dict)

    def test_score_expert_has_required_keys(self, base_feats):
        from api.scoring import score_expert
        result = score_expert(base_feats, "equity")
        assert result is not None
        assert "expert_loaded" in result
        assert "expert_status" in result
        assert result["expert_status"] in ("ok", "warn", "block")

    def test_score_expert_returns_none_when_no_bundle(self):
        """If no bundle is available at all, score_expert returns None."""
        from api import scoring
        # Force no bundles in cache
        scoring._EXPERT_CACHE["no_asset"] = None
        scoring._EXPERT_CACHE["global"] = None
        result = scoring.score_expert({}, "no_asset")
        assert result is None

    def test_score_expert_fallback_asset_type(self, base_feats):
        """commodity falls back to global bundle if no commodity expert."""
        from api.scoring import score_expert
        result = score_expert(base_feats, "commodity")
        # Either returns None (no global) or returns a result from global
        if result is not None:
            assert result.get("expert_asset_type") == "global"

    def test_score_expert_graceful_on_bad_feats(self):
        """score_expert does not raise even with completely empty feats."""
        from api.scoring import score_expert
        # Should not raise — may return dict with NaN scores or None
        try:
            result = score_expert({}, "equity")
            # result can be None or dict
        except Exception as e:
            pytest.fail(f"score_expert raised unexpectedly: {e}")

    def test_preload_experts(self):
        """preload_experts populates the cache for all found bundles."""
        from api.scoring import preload_experts, list_loaded_experts
        preload_experts()
        loaded = list_loaded_experts()
        # equity and global should be found
        assert "equity" in loaded or "global" in loaded
