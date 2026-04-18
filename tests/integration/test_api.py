# tests/integration/test_api.py
"""
Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient or falls back to direct function calls if httpx unavailable.

Tests run with EXPERTS_ENABLED=0 (default) unless explicitly patched,
matching production default behaviour.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

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

    def test_health_has_experts_block(self, client):
        data = client.get("/health").json()
        assert "experts" in data
        experts = data["experts"]
        assert "enabled" in experts
        assert "bundles_on_disk" in experts
        assert isinstance(experts["bundles_on_disk"], list)

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ---------------------------------------------------------------------------
# /score (experts disabled by default)
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
        # expert_decision must be present (None when disabled)
        assert "expert_decision" in data
        assert "expert_loaded" in data

    def test_score_experts_disabled_returns_null(self, client, base_feats):
        """With EXPERTS_ENABLED=0 patched, expert_decision is null."""
        with patch("api.scoring.is_experts_enabled", return_value=False):
            data = client.post("/score", json=base_feats).json()
        assert data["expert_decision"] is None
        assert data["expert_loaded"] is False

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


# ---------------------------------------------------------------------------
# /metrics — model_3m block
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_model_3m(self, client):
        data = client.get("/metrics").json()
        assert "model_3m" in data
        m = data["model_3m"]
        assert m["model_version"] == "3m_v1"
        assert m["feature_count"] == 30
        assert isinstance(m["backtest_cagr"], float)
        assert isinstance(m["backtest_sharpe"], float)

    def test_metrics_has_call_counts(self, client):
        data = client.get("/metrics").json()
        assert "calls_total" in data
        assert isinstance(data["calls_total"], int)


# ---------------------------------------------------------------------------
# /score_3m
# ---------------------------------------------------------------------------

def _make_stock(ticker: str = "AAPL") -> dict:
    """Minimal valid stock feature dict for /score_3m."""
    return {
        "ticker":         ticker,
        "ret_1m":         0.02,
        "ret_3m":         0.05,
        "ret_6m":         0.10,
        "ret_12m":        0.18,
        "mom_12_1":       0.16,
        "ret_12m_vs_spy": 0.05,
        "vol_ann":        0.22,
        "vol_ratio":      1.1,
        "dd_from_hi52":   -0.08,
        "above_200ma":    1.0,
        "trend_strength": 0.6,
        "gross_margin":   0.40,
        "op_margin":      0.20,
        "net_margin":     0.18,
        "roe":            0.15,
        "debt_to_equity": 0.5,
        "rd_intensity":   0.05,
        "fcf_margin":     0.15,
        "revenue_growth": 0.12,
        "ni_growth":      0.10,
        "pe_ratio":       22.0,
        "pb_ratio":       4.0,
        "earnings_yield": 0.045,
        "ev_to_revenue":  5.0,
        "accruals_ratio": -0.02,
        "asset_growth":   0.08,
        "current_ratio":  2.1,
        "ret_1m_lag":     0.01,
        "skew_6m":        -0.2,
        "sector_id":      10.0,
    }


class TestScore3m:
    def test_score_3m_returns_200(self, client):
        payload = {"stocks": [_make_stock("AAPL"), _make_stock("MSFT")]}
        r = client.post("/score_3m", json=payload)
        assert r.status_code == 200

    def test_score_3m_has_model_version(self, client):
        payload = {"stocks": [_make_stock()]}
        data = client.post("/score_3m", json=payload).json()
        assert data.get("model_version") == "3m_v1"

    def test_score_3m_has_scores(self, client):
        payload = {"stocks": [_make_stock("AAPL"), _make_stock("MSFT")]}
        data = client.post("/score_3m", json=payload).json()
        assert "scores" in data
        assert len(data["scores"]) == 2
        for s in data["scores"]:
            assert "prob_beat_spy_3m" in s
            assert "top_pick" in s
            assert 0.0 <= s["prob_beat_spy_3m"] <= 1.0

    def test_score_3m_top_pick_count(self, client):
        stocks = [_make_stock(f"T{i:03d}") for i in range(20)]
        payload = {"stocks": stocks, "top_pct": 0.10}
        data = client.post("/score_3m", json=payload).json()
        n_top = sum(1 for s in data["scores"] if s["top_pick"])
        assert n_top == data["n_top_picks"]
        assert n_top >= 1

    def test_score_3m_has_summary(self, client):
        payload = {"stocks": [_make_stock() for _ in range(5)]}
        data = client.post("/score_3m", json=payload).json()
        assert "summary" in data
        s = data["summary"]
        for key in ["prob_mean", "prob_median", "prob_p90", "prob_min", "prob_max"]:
            assert key in s

    def test_score_3m_with_missing_features(self, client):
        """Model should impute missing features — no crash."""
        payload = {"stocks": [{"ticker": "AAPL", "ret_1m": 0.02, "vol_ann": 0.22}]}
        r = client.post("/score_3m", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["scores"][0]["prob_beat_spy_3m"] >= 0.0

    def test_score_3m_n_stocks_field(self, client):
        stocks = [_make_stock(f"X{i}") for i in range(10)]
        data = client.post("/score_3m", json={"stocks": stocks}).json()
        assert data["n_stocks"] == 10

    def test_score_3m_empty_payload_rejected(self, client):
        r = client.post("/score_3m", json={"stocks": []})
        assert r.status_code == 422  # validation error
