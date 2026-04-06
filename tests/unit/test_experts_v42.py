# tests/unit/test_experts_v42.py
"""
Tests for v4.2 expert bundles: crypto (new), fx/commodity fallback,
97-feature vector, and v4.2 market-context feature pass-through.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_feats(asset_type: str = "equity", market: str = "US", **extra):
    return {
        "asset_type": asset_type,
        "market": market,
        "vol_ann": 0.22,
        "vol_20d": 0.015,
        "vol_60d": 0.18,
        "vol_120d": 0.19,
        "var95": 0.014,
        "var99": 0.022,
        "es95": 0.018,
        "es99": 0.028,
        "max_dd": -0.18,
        "tuw_pct": 42.0,
        "n_used": 252,
        "missing_pct": 0.0,
        "tail_obs_99": 7,
        "rsi": 48.0,
        "skew": -0.5,
        "kurtosis_excess": 1.2,
        "dd_duration": 30,
        "recovery_days": 20,
        "corr_mkt": 0.7,
        # v4.2 fields
        "corr_spy": 0.72,
        "beta_market": 1.1,
        "vix_level": 19.5,
        **extra,
    }


# ---------------------------------------------------------------------------
# Feature vector — 97 columns
# ---------------------------------------------------------------------------

class TestFeatureVector97:
    def test_vector_length(self):
        from features import DEFAULT_CONFIG, vector_columns
        assert len(vector_columns(DEFAULT_CONFIG)) == 97

    def test_v42_features_present(self):
        from features import DEFAULT_CONFIG, vector_columns
        cols = vector_columns(DEFAULT_CONFIG)
        for f in ("corr_spy", "beta_market", "vix_level",
                  "corr_spy_sq", "beta_abs", "vix_vol_interaction"):
            assert f in cols, f"{f} missing from feature vector"

    def test_corr_spy_computed_correctly(self):
        from features import features_to_row, DEFAULT_CONFIG
        feats = _base_feats(corr_spy=0.8, vix_level=20.0, vol_ann=0.25)
        row = features_to_row(feats, DEFAULT_CONFIG)
        assert abs(row["corr_spy_sq"] - 0.64) < 1e-9
        assert abs(row["vix_vol_interaction"] - 5.0) < 1e-9

    def test_corr_mkt_fallback(self):
        """corr_spy_sq falls back to corr_mkt when corr_spy absent."""
        from features import features_to_row, DEFAULT_CONFIG
        feats = _base_feats()
        feats.pop("corr_spy", None)
        feats["corr_mkt"] = 0.6
        row = features_to_row(feats, DEFAULT_CONFIG)
        assert abs(row["corr_spy_sq"] - 0.36) < 1e-9

    def test_nan_v42_fields_when_missing(self):
        from features import features_to_row, DEFAULT_CONFIG
        feats = {"asset_type": "equity", "market": "US"}
        row = features_to_row(feats, DEFAULT_CONFIG)
        assert not math.isfinite(row["corr_spy"])
        assert not math.isfinite(row["beta_market"])
        assert not math.isfinite(row["vix_level"])
        assert not math.isfinite(row["corr_spy_sq"])
        assert not math.isfinite(row["beta_abs"])
        assert not math.isfinite(row["vix_vol_interaction"])


# ---------------------------------------------------------------------------
# Crypto expert
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / "models" / "experts" / "crypto_bundle.joblib").exists(),
    reason="crypto_bundle.joblib not on disk",
)
class TestCryptoExpert:
    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_crypto_bundle_loads(self):
        from api.scoring import load_expert_bundle
        b = load_expert_bundle("crypto")
        assert b is not None
        assert b.get("asset_type") == "crypto"

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_crypto_bundle_has_97_features(self):
        from api.scoring import load_expert_bundle
        b = load_expert_bundle("crypto")
        assert len(b["cols"]) == 97

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_crypto_score_returns_dict(self):
        from api.scoring import score_expert
        feats = _base_feats("crypto", "GLOBAL", vol_ann=0.8, var95=0.05, var99=0.09)
        result = score_expert(feats, "crypto")
        assert result is not None
        assert result["expert_asset_type"] == "crypto"
        assert "expert_status" in result
        assert result["expert_status"] in ("ok", "warn", "block")
        assert 0.0 <= result["expert_prob_non_ok"] <= 1.0

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_crypto_high_vol_warns_or_blocks(self):
        """Very high-vol crypto should at least warn."""
        from api.scoring import score_expert
        feats = _base_feats("crypto", "GLOBAL",
                            vol_ann=2.5, var95=0.15, var99=0.25,
                            max_dd=-0.75, tuw_pct=90.0)
        result = score_expert(feats, "crypto")
        assert result is not None
        # With vol=250%, max_dd=-75%, expert_prob should be high
        assert result["expert_prob_non_ok"] > 0.3, (
            f"Expected high prob for extreme crypto params, got {result['expert_prob_non_ok']}"
        )


# ---------------------------------------------------------------------------
# FX / Commodity fallback to global
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / "models" / "experts" / "global_bundle.joblib").exists(),
    reason="global_bundle.joblib not on disk",
)
class TestFxCommodityFallback:
    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_fx_falls_back_to_global(self):
        from api.scoring import load_expert_bundle
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        b = load_expert_bundle("fx")
        assert b is not None
        assert b.get("asset_type") == "global"

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_commodity_falls_back_to_global(self):
        from api.scoring import load_expert_bundle
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        b = load_expert_bundle("commodity")
        assert b is not None
        assert b.get("asset_type") == "global"

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_fx_score_uses_global_expert(self):
        from api.scoring import score_expert
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        feats = _base_feats("fx", "GLOBAL", vol_ann=0.08, var95=0.005, var99=0.009)
        result = score_expert(feats, "fx")
        assert result is not None
        # Falls back to global
        assert result["expert_asset_type"] == "global"
        assert result["expert_status"] in ("ok", "warn", "block")

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "1"})
    def test_unknown_asset_type_falls_back(self):
        from api.scoring import score_expert
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        feats = _base_feats("rate", "US")
        result = score_expert(feats, "rate")
        assert result is not None
        assert result["expert_asset_type"] == "global"


# ---------------------------------------------------------------------------
# Experts disabled — no load, clean response
# ---------------------------------------------------------------------------

class TestExpertsDisabledV42:
    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_crypto_returns_none_when_disabled(self):
        from api.scoring import score_expert
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        result = score_expert(_base_feats("crypto", "GLOBAL"), "crypto")
        assert result is None

    @patch.dict(os.environ, {"EXPERTS_ENABLED": "0"})
    def test_no_cache_populated_when_disabled(self):
        from api import scoring
        scoring._EXPERT_CACHE.clear()
        scoring.load_expert_bundle("equity")
        assert len(scoring._EXPERT_CACHE) == 0
