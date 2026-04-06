# tests/unit/test_experts_eval.py
"""
Tests minimalistes pour les expert bundles v2 et le script d'évaluation.

Vérifie :
  - chargement des bundles (loader)
  - prédictions sur un mini-subset sans crash
  - structure des sorties du script d'évaluation
  - EXPERTS_ENABLED=0 par défaut (aucun changement en prod)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def expert_bundles():
    """Load all expert bundles found in models/experts/."""
    import joblib
    experts_dir = REPO_ROOT / "models" / "experts"
    bundles = {}
    for p in sorted(experts_dir.glob("*_bundle.joblib")):
        name = p.stem.replace("_bundle", "")
        b = joblib.load(p)
        bundles[name] = b
    return bundles


@pytest.fixture(scope="module")
def holdout_v2_sample() -> List[Dict[str, Any]]:
    """Load up to 200 records from holdout_v2.jsonl (or fewer if not present)."""
    path = REPO_ROOT / "data" / "training" / "holdout_v2.jsonl"
    if not path.exists():
        pytest.skip("holdout_v2.jsonl not found — run scripts/build_v2_split.py first")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(42)
    rng.shuffle(lines)
    return [json.loads(l) for l in lines[:200]]


# ─────────────────────────────────────────────────────────────────────────────
# Loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertBundleLoader:
    def test_at_least_one_bundle_loaded(self, expert_bundles):
        assert len(expert_bundles) >= 1, "No expert bundles found"

    def test_global_bundle_present(self, expert_bundles):
        assert "global" in expert_bundles, "global_bundle.joblib must exist as fallback"

    def test_bundle_structure(self, expert_bundles):
        required_keys = {"cols", "unsup", "sup_bin", "meta", "feature_version"}
        for name, b in expert_bundles.items():
            missing = required_keys - set(b.keys())
            assert not missing, f"Bundle '{name}' missing keys: {missing}"

    def test_bundle_feature_version_v2(self, expert_bundles):
        for name, b in expert_bundles.items():
            assert b.get("feature_version") == "v2", (
                f"Bundle '{name}' feature_version={b.get('feature_version')!r} expected 'v2'"
            )

    def test_bundle_cols_count(self, expert_bundles):
        for name, b in expert_bundles.items():
            assert len(b["cols"]) == 97, (
                f"Bundle '{name}' has {len(b['cols'])} cols, expected 97 (v4.2)"
            )

    def test_sup_bin_has_model_and_thresholds(self, expert_bundles):
        for name, b in expert_bundles.items():
            sup = b.get("sup_bin", {})
            assert sup.get("model") is not None, f"Bundle '{name}' sup_bin missing model"
            thr = sup.get("thresholds", {})
            assert "t_lo" in thr and "t_hi" in thr, (
                f"Bundle '{name}' thresholds missing t_lo/t_hi"
            )

    def test_thresholds_sane(self, expert_bundles):
        for name, b in expert_bundles.items():
            thr = b["sup_bin"]["thresholds"]
            t_lo = thr["t_lo"]
            t_hi = thr["t_hi"]
            assert 0.0 < t_lo < 1.0, f"Bundle '{name}' t_lo={t_lo} out of range"
            assert 0.0 < t_hi < 1.0, f"Bundle '{name}' t_hi={t_hi} out of range"
            assert t_lo <= t_hi, f"Bundle '{name}' t_lo > t_hi"

    def test_unsup_has_iforest_lof(self, expert_bundles):
        for name, b in expert_bundles.items():
            unsup = b.get("unsup", {})
            assert unsup.get("iforest") is not None, f"Bundle '{name}' missing iforest"
            assert unsup.get("lof") is not None, f"Bundle '{name}' missing lof"

    def test_meta_n_train_positive(self, expert_bundles):
        for name, b in expert_bundles.items():
            n = b.get("meta", {}).get("n_train", 0)
            assert n > 0, f"Bundle '{name}' meta.n_train={n}"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction tests (mini-subset, no crash)
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertPredictions:
    def test_predict_returns_valid_status(self, expert_bundles, holdout_v2_sample):
        from scripts.eval_experts_v2 import predict_expert_bundle
        valid_statuses = {"ok", "warn", "block"}
        for rec in holdout_v2_sample[:50]:
            status, prob, bundle_used = predict_expert_bundle(rec, expert_bundles)
            assert status in valid_statuses, f"Invalid status: {status!r}"
            assert 0.0 <= prob <= 1.0, f"Prob out of range: {prob}"

    def test_predict_prob_is_float(self, expert_bundles, holdout_v2_sample):
        from scripts.eval_experts_v2 import predict_expert_bundle
        for rec in holdout_v2_sample[:30]:
            _, prob, _ = predict_expert_bundle(rec, expert_bundles)
            assert isinstance(prob, float)

    def test_predict_graceful_on_empty_feats(self, expert_bundles):
        from scripts.eval_experts_v2 import predict_expert_bundle
        rec = {"label": "ok", "features": {"asset_type": "equity", "market": "US"}}
        status, prob, _ = predict_expert_bundle(rec, expert_bundles)
        assert status in {"ok", "warn", "block"}

    def test_predict_fallback_to_global(self, expert_bundles, holdout_v2_sample):
        from scripts.eval_experts_v2 import predict_expert_bundle
        rec = dict(holdout_v2_sample[0])
        rec["features"] = dict(rec.get("features", {}))
        rec["features"]["asset_type"] = "nonexistent_type_xyz"
        status, prob, bundle_used = predict_expert_bundle(rec, expert_bundles)
        assert status in {"ok", "warn", "block"}

    def test_roc_auc_above_chance(self, expert_bundles, holdout_v2_sample):
        """ROC-AUC should be well above 0.5 on a 200-sample subset."""
        from scripts.eval_experts_v2 import predict_expert_bundle
        from sklearn.metrics import roc_auc_score
        y_true, y_prob = [], []
        for rec in holdout_v2_sample:
            _, prob, _ = predict_expert_bundle(rec, expert_bundles)
            true_lbl = rec.get("label", "ok")
            y_true.append(0 if true_lbl == "ok" else 1)
            y_prob.append(prob)
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        if len(np.unique(y_true)) < 2:
            pytest.skip("Subset has only one class, skipping ROC-AUC check")
        auc = roc_auc_score(y_true, y_prob)
        assert auc > 0.55, f"ROC-AUC={auc:.3f} suspiciously low on holdout_v2 subset"

    def test_fp_rate_ok_within_constraint(self, expert_bundles, holdout_v2_sample):
        """FP-rate on OK samples should be ≤ 0.25 (relaxed for 200-sample subset)."""
        from scripts.eval_experts_v2 import predict_expert_bundle, fp_rate_ok
        y_true_bin, y_pred_bin = [], []
        for rec in holdout_v2_sample:
            status, _, _ = predict_expert_bundle(rec, expert_bundles)
            true_lbl = rec.get("label", "ok")
            y_true_bin.append(0 if true_lbl == "ok" else 1)
            y_pred_bin.append(0 if status == "ok" else 1)
        fp = fp_rate_ok(np.array(y_true_bin), np.array(y_pred_bin))
        assert fp <= 0.25, f"FP-rate={fp:.3f} exceeds 0.25 on subset"


# ─────────────────────────────────────────────────────────────────────────────
# Eval script output tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalScriptOutput:
    @pytest.fixture(scope="class")
    def report(self):
        path = REPO_ROOT / "data" / "metrics" / "experts_v2_report.json"
        if not path.exists():
            pytest.skip("experts_v2_report.json not found — run scripts/eval_experts_v2.py first")
        return json.loads(path.read_text())

    def test_report_has_required_keys(self, report):
        required = {"evaluation_date", "holdout", "expert_v2", "bin_sigmoid",
                    "comparison", "verdict"}
        assert required.issubset(set(report.keys()))

    def test_expert_v2_global_metrics_present(self, report):
        g = report["expert_v2"]["global"]
        for key in ["accuracy", "balanced_accuracy", "macro_f1",
                    "fp_rate_ok", "recall_non_ok", "roc_auc", "pr_auc", "ece"]:
            assert key in g, f"Missing metric: {key}"

    def test_roc_auc_above_minimum(self, report):
        auc = report["expert_v2"]["global"]["roc_auc"]
        assert auc >= 0.65, f"ROC-AUC={auc:.3f} below minimum 0.65"

    def test_fp_rate_ok_within_constraint(self, report):
        fp = report["expert_v2"]["global"]["fp_rate_ok"]
        assert fp <= 0.15, f"FP-rate={fp:.3f} exceeds constraint 0.15"

    def test_verdict_is_string(self, report):
        assert isinstance(report["verdict"], str)
        assert len(report["verdict"]) > 0

    def test_confusion_matrix_shape(self, report):
        cm = report["expert_v2"]["global"]["confusion_matrix"]
        assert len(cm) == 3 and all(len(row) == 3 for row in cm)


# ─────────────────────────────────────────────────────────────────────────────
# Production safety: EXPERTS_ENABLED=0 by default
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionSafety:
    def test_experts_enabled_default_zero(self):
        """EXPERTS_ENABLED env var must default to 0 (no prod impact)."""
        val = os.getenv("EXPERTS_ENABLED", "0")
        assert val == "0", (
            f"EXPERTS_ENABLED={val!r} — must be 0 to avoid prod impact. "
            "Set explicitly to 1 only when ready to deploy."
        )

    def test_api_health_endpoint_stable(self):
        """Health endpoint must not crash regardless of expert bundle state."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx not installed")
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data.get("ok") is True

    def test_score_endpoint_stable(self, base_feats):
        """Score endpoint must still work (bin_sigmoid path unchanged)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx not installed")
        from api.main import app
        with TestClient(app) as client:
            r = client.post("/score", json=base_feats)
            assert r.status_code == 200
