# tests/integration/test_api.py
"""
Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient or falls back to direct function calls if httpx unavailable.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Check if TestClient is available (requires httpx)
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    _HAS_TEST_CLIENT = True
except Exception:
    _HAS_TEST_CLIENT = False

pytestmark = pytest.mark.skipif(
    not _HAS_TEST_CLIENT,
    reason="fastapi.testclient requires httpx — install it with: pip install httpx",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def _sample_closes(n: int = 260, seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    px = [100.0]
    for _ in range(n - 1):
        px.append(px[-1] * (1 + rng.gauss(0, 0.01)))
    return px


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_ok_true(self, client):
        data = client.get("/health").json()
        assert data.get("ok") is True

    def test_health_has_experts_fields(self, client):
        data = client.get("/health").json()
        assert "experts_loaded" in data
        assert "experts_available" in data
        assert isinstance(data["experts_loaded"], list)

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_200(self, client, base_feats):
        r = client.post("/score", json=base_feats)
        assert r.status_code == 200

    def test_score_has_status_field(self, client, base_feats):
        data = client.post("/score", json=base_feats).json()
        assert "status" in data
        assert data["status"] in ("OK", "WARN", "BLOCK", "SKIP", None)

    def test_score_has_expert_decision(self, client, base_feats):
        data = client.post("/score", json=base_feats).json()
        # expert_decision must be present (may be None if no bundle)
        assert "expert_decision" in data
        assert "expert_loaded" in data

    def test_score_expert_decision_structure(self, client, v2_feats):
        data = client.post("/score", json=v2_feats).json()
        ed = data.get("expert_decision")
        if ed is not None:
            assert "expert_status" in ed
            assert ed["expert_status"] in ("ok", "warn", "block")

    def test_score_etf_expert(self, client):
        feats = {
            "asset_type": "etf", "market": "US",
            "vol_ann": 0.12, "var95": 0.010, "var99": 0.016,
            "es95": 0.013, "es99": 0.020, "max_dd": -0.08,
            "n_used": 252, "missing_pct": 0.0,
        }
        data = client.post("/score", json=feats).json()
        assert "status" in data


# ---------------------------------------------------------------------------
# /score_oracle (with closes, no external network)
# ---------------------------------------------------------------------------

class TestScoreOracle:
    def test_score_oracle_with_closes_200(self, client):
        closes = _sample_closes(260)
        payload = {
            "lovable": {
                "asset_type": "equity",
                "market": "US",
                "ticker": "TEST",
                "vol_ann": 0.18, "var95": 0.016, "var99": 0.025,
                "es95": 0.022, "es99": 0.031, "max_dd": -0.12,
                "n_used": 252, "missing_pct": 0.0,
            },
            "closes": closes,
            "lookback_days": 252,
            "force_oracle": True,
        }
        r = client.post("/score_oracle", json=payload)
        assert r.status_code == 200

    def test_score_oracle_has_oracle_used(self, client):
        closes = _sample_closes(260)
        payload = {
            "lovable": {
                "asset_type": "equity", "market": "US",
                "vol_ann": 0.18, "var95": 0.016, "var99": 0.025,
                "es95": 0.022, "es99": 0.031, "max_dd": -0.12,
                "n_used": 252, "missing_pct": 0.0,
            },
            "closes": closes,
            "lookback_days": 252,
            "force_oracle": True,
        }
        data = client.post("/score_oracle", json=payload).json()
        assert "oracle_used" in data

    def test_score_oracle_has_expert_decision(self, client):
        closes = _sample_closes(260)
        payload = {
            "lovable": {
                "asset_type": "equity", "market": "US",
                "vol_ann": 0.18, "var95": 0.016, "var99": 0.025,
                "es95": 0.022, "es99": 0.031, "max_dd": -0.12,
                "n_used": 252, "missing_pct": 0.0,
            },
            "closes": closes,
            "lookback_days": 252,
            "force_oracle": True,
        }
        data = client.post("/score_oracle", json=payload).json()
        assert "expert_decision" in data
        assert "expert_loaded" in data

    def test_score_oracle_missing_bundle_doesnt_crash(self, client):
        """API must not crash even if expert bundles are absent."""
        closes = _sample_closes(260)
        payload = {
            "lovable": {
                "asset_type": "crypto", "market": "US",
                "vol_ann": 0.80, "var95": 0.06, "var99": 0.09,
                "es95": 0.08, "es99": 0.11, "max_dd": -0.50,
                "n_used": 252, "missing_pct": 0.0,
            },
            "closes": closes,
            "lookback_days": 252,
            "force_oracle": False,
        }
        r = client.post("/score_oracle", json=payload)
        assert r.status_code == 200  # must not raise 500
