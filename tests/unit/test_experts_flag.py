# tests/unit/test_experts_flag.py
"""Tests for EXPERTS_ENABLED feature flag behaviour."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _clear_expert_cache():
    """Reset expert cache between tests."""
    from api import scoring
    scoring._EXPERT_CACHE.clear()
    yield
    scoring._EXPERT_CACHE.clear()


# -----------------------------------------------------------------------
# EXPERTS_ENABLED=0 (default) — nothing should load or score
# -----------------------------------------------------------------------
class TestExpertsDisabled:
    """When EXPERTS_ENABLED is unset or '0', experts must be fully inert."""

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_is_experts_enabled_returns_false(self):
        from api.scoring import is_experts_enabled
        assert is_experts_enabled() is False

    @patch.dict(os.environ, {}, clear=False)
    def test_default_is_disabled(self):
        """If EXPERTS_ENABLED is not set at all, default is disabled."""
        os.environ.pop("EXPERTS_ENABLED", None)
        from api.scoring import is_experts_enabled
        assert is_experts_enabled() is False

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_score_expert_returns_none(self, base_feats):
        from api.scoring import score_expert
        result = score_expert(base_feats, "equity")
        assert result is None

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_load_expert_bundle_returns_none(self):
        from api.scoring import load_expert_bundle
        assert load_expert_bundle("equity") is None

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_preload_does_nothing(self):
        from api.scoring import preload_experts, list_loaded_experts
        preload_experts()
        assert list_loaded_experts() == []

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_no_bundles_loaded_in_cache(self, base_feats):
        from api import scoring
        scoring.score_expert(base_feats, "equity")
        assert len(scoring._EXPERT_CACHE) == 0

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_health_shows_disabled(self):
        from api.scoring import experts_health
        h = experts_health()
        assert h["enabled"] is False
        assert h["n_bundles_loaded"] == 0

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_health_still_lists_bundles_on_disk(self):
        """Even when disabled, health reports bundles available on disk."""
        from api.scoring import experts_health
        h = experts_health()
        # The bundles exist in models/experts/
        assert h["n_bundles_on_disk"] >= 0  # might be 0 in CI
        assert isinstance(h["bundles_on_disk"], list)


# -----------------------------------------------------------------------
# EXPERTS_ENABLED=1 — everything should work
# -----------------------------------------------------------------------
class TestExpertsEnabled:
    """When EXPERTS_ENABLED=1, experts load and score normally."""

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_is_experts_enabled_returns_true(self):
        from api.scoring import is_experts_enabled
        assert is_experts_enabled() is True

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_score_expert_returns_dict(self, base_feats):
        from api.scoring import score_expert
        result = score_expert(base_feats, "equity")
        assert result is not None
        assert isinstance(result, dict)
        assert "expert_status" in result

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_preload_populates_cache(self):
        from api.scoring import preload_experts, list_loaded_experts
        preload_experts()
        loaded = list_loaded_experts()
        assert len(loaded) > 0

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_health_shows_enabled(self):
        from api.scoring import experts_health, preload_experts
        preload_experts()
        h = experts_health()
        assert h["enabled"] is True
        assert h["n_bundles_loaded"] > 0
        # Check per-bundle detail
        for at, detail in h["bundles"].items():
            assert "path" in detail
            assert "exists" in detail


# -----------------------------------------------------------------------
# EXPERTS_ENABLED=1 but bundles missing — graceful fallback
# -----------------------------------------------------------------------
class TestExpertsEnabledNoBundles:
    """When enabled but bundles are missing, no crash, clean fallback."""

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_score_expert_returns_none_missing_dir(self, base_feats):
        from api import scoring
        old_dir = scoring._EXPERTS_DIR
        scoring._EXPERTS_DIR = "/tmp/nonexistent_experts_dir_xyz"
        try:
            result = scoring.score_expert(base_feats, "equity")
            assert result is None
        finally:
            scoring._EXPERTS_DIR = old_dir

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_preload_no_crash_missing_dir(self):
        from api import scoring
        old_dir = scoring._EXPERTS_DIR
        scoring._EXPERTS_DIR = "/tmp/nonexistent_experts_dir_xyz"
        try:
            scoring.preload_experts()  # should not raise
            assert scoring.list_loaded_experts() == []
        finally:
            scoring._EXPERTS_DIR = old_dir

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_health_missing_dir(self):
        from api import scoring
        old_dir = scoring._EXPERTS_DIR
        scoring._EXPERTS_DIR = "/tmp/nonexistent_experts_dir_xyz"
        try:
            h = scoring.experts_health()
            assert h["enabled"] is True
            assert h["n_bundles_on_disk"] == 0
            assert h["bundles_loaded"] == []
        finally:
            scoring._EXPERTS_DIR = old_dir


# -----------------------------------------------------------------------
# API integration: /health and /score with flag
# -----------------------------------------------------------------------
class TestAPIWithFlag:
    """Integration tests via TestClient with feature flag."""

    def test_health_experts_disabled(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as c:
            r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "experts" in data
        assert data["experts"]["enabled"] is False

    def test_score_experts_disabled_returns_null_decision(self):
        from fastapi.testclient import TestClient
        from api.main import app
        payload = {
            "ticker": "AAPL",
            "asset_type": "equity",
            "market": "US",
            "vol_ann": 0.18,
            "var95": 0.016,
            "var99": 0.025,
            "es95": 0.022,
            "es99": 0.031,
            "max_dd": -0.12,
            "n_used": 252,
            "missing_pct": 0.0,
        }
        with TestClient(app) as c:
            r = c.post("/score", json=payload, headers={"x-api-key": "test"})
        assert r.status_code == 200
        data = r.json()
        assert data["expert_decision"] is None
        assert data["expert_loaded"] is False
