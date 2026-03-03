"""
scripts/ml/validation/preflight_v3_api.py
==========================================
Preflight check for v3 model production integration.

Loads the full v3 model stack (XGBoost + isotonic calibrator), samples
N records from the validation set, reconstructs the feature matrix,
applies frozen-median imputation, runs inference, and verifies all
invariants defined in docs/V3_API_CONTRACT.md.

Exits with code 0 if all invariants pass, code 1 if any fail.

Usage:
  python scripts/ml/validation/preflight_v3_api.py
  python scripts/ml/validation/preflight_v3_api.py --n_sample 500
  python scripts/ml/validation/preflight_v3_api.py \\
      --models_dir models/v3 \\
      --val_file data/training/v3/fold_5/val.jsonl \\
      --n_sample 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

log = logging.getLogger("preflight_v3")


# ---------------------------------------------------------------------------
# Invariant registry
# ---------------------------------------------------------------------------

class InvariantError(Exception):
    """Raised when a preflight invariant fails."""


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise InvariantError(msg)


# ---------------------------------------------------------------------------
# Load model artefacts
# ---------------------------------------------------------------------------

def load_model_stack(models_dir: Path) -> dict[str, Any]:
    """Load all v3 model artefacts from models_dir. Returns artefact dict."""
    required = [
        "v3_xgb_model.joblib",
        "v3_calibrator.joblib",
        "v3_feature_names.joblib",
        "v3_meta.json",
        "v3_thresholds.json",
    ]
    for fname in required:
        p = models_dir / fname
        _check(p.exists(), f"Missing model artefact: {p}")

    xgb_model   = joblib.load(models_dir / "v3_xgb_model.joblib")
    calibrator  = joblib.load(models_dir / "v3_calibrator.joblib")
    feat_names  = joblib.load(models_dir / "v3_feature_names.joblib")
    meta        = json.loads((models_dir / "v3_meta.json").read_text())
    thresholds  = json.loads((models_dir / "v3_thresholds.json").read_text())

    return {
        "xgb_model":  xgb_model,
        "calibrator": calibrator,
        "feat_names": feat_names,
        "meta":       meta,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Build feature matrix from JSONL records
# ---------------------------------------------------------------------------

_SENTINEL_FEATS = {"recovery_days", "recovery_per_dd", "dd_duration", "recovery_defined"}
_META_COLS      = {"asset_type", "market", "ticker", "market_proxy"}


def build_X(records: list[dict], feat_names: list[str]) -> np.ndarray:
    """
    Reconstruct float64 X matrix from JSONL records.
    Sentinel features (-1.0) are passed through as-is.
    Missing macro features produce NaN (to be imputed later).
    """
    n = len(records)
    m = len(feat_names)
    X = np.full((n, m), np.nan, dtype=np.float64)

    for i, rec in enumerate(records):
        feats = rec.get("features", {})
        for j, col in enumerate(feat_names):
            val = feats.get(col)
            if val is None:
                # Remain NaN — will be imputed
                continue
            X[i, j] = float(val)

    return X


def apply_frozen_imputation(
    X: np.ndarray, feat_names: list[str], medians: dict[str, float]
) -> np.ndarray:
    """
    Replace NaN with training-set medians (frozen from v3_meta.json).
    Sentinel features (-1.0) are never NaN at this point, so they are
    untouched. Falls back to 0.0 if a feature has no stored median.
    """
    X_out = X.copy()
    for j, feat in enumerate(feat_names):
        col = X_out[:, j]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            fill = medians.get(feat, 0.0)
            X_out[nan_mask, j] = fill
    return X_out


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def check_threshold_monotonicity(thresholds: dict[str, Any]) -> None:
    """I-4: t_lo < t_hi."""
    t_lo = thresholds["t_lo"]
    t_hi = thresholds["t_hi"]
    _check(
        isinstance(t_lo, (int, float)) and isinstance(t_hi, (int, float)),
        f"Thresholds must be numeric, got t_lo={type(t_lo)}, t_hi={type(t_hi)}",
    )
    _check(0.0 < t_lo < t_hi < 1.0, (
        f"Threshold monotonicity violated: t_lo={t_lo}, t_hi={t_hi}. "
        f"Expected 0 < t_lo < t_hi < 1."
    ))


def check_feature_contract(feat_names: list[str], meta: dict[str, Any]) -> None:
    """I-5, I-6: feature count and order match meta."""
    expected = meta["feature_cols"]
    _check(
        len(feat_names) == len(expected),
        f"Feature count mismatch: joblib has {len(feat_names)}, meta has {len(expected)}",
    )
    _check(
        list(feat_names) == list(expected),
        "Feature order mismatch between v3_feature_names.joblib and v3_meta.json",
    )
    _check(
        len(feat_names) == meta["n_features"],
        f"n_features in meta ({meta['n_features']}) != actual list length ({len(feat_names)})",
    )


def check_no_nan_after_imputation(X_imp: np.ndarray, feat_names: list[str]) -> None:
    """I-1: X has no NaN or inf after imputation."""
    bad = ~np.isfinite(X_imp)
    n_bad = int(bad.sum())
    if n_bad > 0:
        bad_cols = [feat_names[j] for j in range(X_imp.shape[1]) if bad[:, j].any()]
        raise InvariantError(
            f"I-1 FAIL: {n_bad} non-finite values after imputation in columns: {bad_cols}"
        )


def check_proba_range(proba: np.ndarray) -> None:
    """I-2: proba_non_ok in [0, 1] for all records."""
    out_of_range = (proba < 0.0) | (proba > 1.0)
    n_bad = int(out_of_range.sum())
    _check(n_bad == 0, (
        f"I-2 FAIL: {n_bad} probabilities outside [0, 1]. "
        f"Range: [{float(proba.min()):.6f}, {float(proba.max()):.6f}]"
    ))


def check_label_coherence(
    proba: np.ndarray, labels: np.ndarray, t_lo: float, t_hi: float
) -> None:
    """I-3: Each label is consistent with the threshold rule."""
    expected = np.where(proba >= t_hi, "block",
               np.where(proba >= t_lo, "warn", "ok"))
    mismatches = labels != expected
    n_bad = int(mismatches.sum())
    if n_bad > 0:
        example_p = float(proba[mismatches][0])
        example_got = labels[mismatches][0]
        example_exp = expected[mismatches][0]
        raise InvariantError(
            f"I-3 FAIL: {n_bad} label/threshold mismatches. "
            f"Example: proba={example_p:.4f} "
            f"→ label={example_got!r}, expected={example_exp!r}"
        )


def check_sentinel_not_nan(X_raw: np.ndarray, feat_names: list[str]) -> None:
    """
    I-7: Sentinel features must be -1.0 or a valid float — never NaN.
    (Before imputation, so we check the raw matrix.)
    Only applies to records where the sentinel is actually populated.
    """
    for j, feat in enumerate(feat_names):
        if feat not in _SENTINEL_FEATS:
            continue
        col = X_raw[:, j]
        # Sentinel value should be -1.0 (undefined) or positive (defined).
        # NaN means the sentinel was not stored correctly.
        nan_mask = np.isnan(col)
        if nan_mask.any():
            raise InvariantError(
                f"I-7 FAIL: Sentinel feature '{feat}' has {int(nan_mask.sum())} NaN values. "
                f"Expected sentinel -1.0 or a valid float."
            )


def check_recovery_defined_binary(X_raw: np.ndarray, feat_names: list[str]) -> None:
    """I-8: recovery_defined must be 0.0 or 1.0 — never NaN, never other values."""
    if "recovery_defined" not in feat_names:
        return
    j = feat_names.index("recovery_defined")
    col = X_raw[:, j]
    nan_mask = np.isnan(col)
    _check(not nan_mask.any(), (
        f"I-8 FAIL: recovery_defined has {int(nan_mask.sum())} NaN values"
    ))
    bad = ~np.isin(col[~nan_mask], [0.0, 1.0])
    _check(not bad.any(), (
        f"I-8 FAIL: recovery_defined has non-binary values: "
        f"{np.unique(col[~nan_mask][bad]).tolist()}"
    ))


# ---------------------------------------------------------------------------
# Main preflight routine
# ---------------------------------------------------------------------------

def run_preflight(
    models_dir: Path,
    val_file: Path,
    n_sample: int,
    seed: int = 42,
) -> bool:
    """
    Run all preflight checks. Returns True if all pass, raises InvariantError if any fail.
    """
    log.info("Loading model stack from %s", models_dir)
    stack = load_model_stack(models_dir)

    xgb_model  = stack["xgb_model"]
    calibrator  = stack["calibrator"]
    feat_names  = stack["feat_names"]
    meta        = stack["meta"]
    thresholds  = stack["thresholds"]
    medians     = meta["medians"]
    t_lo        = thresholds["t_lo"]
    t_hi        = thresholds["t_hi"]

    log.info("schema_version=%s  n_features=%d  t_lo=%.4f  t_hi=%.4f",
             meta.get("schema_version"), len(feat_names), t_lo, t_hi)

    # ── Structural invariants (no data needed) ──────────────────────────────
    log.info("[I-4] Checking threshold monotonicity …")
    check_threshold_monotonicity(thresholds)
    log.info("  ✅ t_lo=%.4f < t_hi=%.4f", t_lo, t_hi)

    log.info("[I-5/I-6] Checking feature contract …")
    check_feature_contract(feat_names, meta)
    log.info("  ✅ %d features, order matches meta", len(feat_names))

    # ── Load sample records ─────────────────────────────────────────────────
    log.info("Sampling %d records from %s …", n_sample, val_file)
    _check(val_file.exists(), f"Val file not found: {val_file}")

    all_lines = val_file.read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    sampled_lines = rng.sample(all_lines, min(n_sample, len(all_lines)))
    records = [json.loads(ln) for ln in sampled_lines]
    log.info("  Loaded %d records", len(records))

    # ── Build raw feature matrix ────────────────────────────────────────────
    X_raw = build_X(records, feat_names)
    log.info("  X_raw shape: %s  NaN count: %d", X_raw.shape, int(np.isnan(X_raw).sum()))

    # ── Sentinel / binary checks (before imputation) ────────────────────────
    log.info("[I-7] Checking sentinel features are not NaN …")
    check_sentinel_not_nan(X_raw, feat_names)
    log.info("  ✅ No sentinel NaN")

    log.info("[I-8] Checking recovery_defined is binary …")
    check_recovery_defined_binary(X_raw, feat_names)
    log.info("  ✅ recovery_defined OK")

    # ── Apply frozen imputation ─────────────────────────────────────────────
    X_imp = apply_frozen_imputation(X_raw, feat_names, medians)

    log.info("[I-1] Checking no NaN/inf after imputation …")
    check_no_nan_after_imputation(X_imp, feat_names)
    log.info("  ✅ X is fully finite after imputation")

    # ── Inference ───────────────────────────────────────────────────────────
    log.info("Running inference (%d records) …", len(records))
    proba_raw    = xgb_model.predict_proba(X_imp)[:, 1]
    proba_cal    = np.array(calibrator.predict(proba_raw), dtype=np.float64)
    labels       = np.where(proba_cal >= t_hi, "block",
                   np.where(proba_cal >= t_lo, "warn", "ok"))

    log.info("[I-2] Checking proba ∈ [0, 1] …")
    check_proba_range(proba_cal)
    log.info("  ✅ proba range: [%.4f, %.4f]", float(proba_cal.min()), float(proba_cal.max()))

    log.info("[I-3] Checking label coherence with thresholds …")
    check_label_coherence(proba_cal, labels, t_lo, t_hi)
    log.info("  ✅ All labels consistent with thresholds")

    # ── Summary ─────────────────────────────────────────────────────────────
    ok_count    = int((labels == "ok").sum())
    warn_count  = int((labels == "warn").sum())
    block_count = int((labels == "block").sum())
    log.info(
        "Label distribution: ok=%d (%.1f%%)  warn=%d (%.1f%%)  block=%d (%.1f%%)",
        ok_count,    100 * ok_count    / len(records),
        warn_count,  100 * warn_count  / len(records),
        block_count, 100 * block_count / len(records),
    )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(
        description="Preflight check for v3 model production integration"
    )
    ap.add_argument(
        "--models_dir",
        default=str(repo_root / "models" / "v3"),
        help="Directory containing v3 model artefacts (default: models/v3)",
    )
    ap.add_argument(
        "--val_file",
        default=str(repo_root / "data" / "training" / "v3" / "fold_5" / "val.jsonl"),
        help="JSONL validation file to sample from (default: fold_5/val.jsonl)",
    )
    ap.add_argument(
        "--n_sample",
        type=int,
        default=200,
        help="Number of records to sample for inference checks (default: 200)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    val_file   = Path(args.val_file)

    failures: list[str] = []
    try:
        run_preflight(models_dir, val_file, args.n_sample, args.seed)
    except InvariantError as exc:
        failures.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Unexpected error: {exc}")

    if failures:
        log.error("")
        log.error("=" * 70)
        log.error("PREFLIGHT FAILED — %d invariant(s) violated:", len(failures))
        for i, f in enumerate(failures, 1):
            log.error("  [%d] %s", i, f)
        log.error("=" * 70)
        sys.exit(1)

    log.info("")
    log.info("=" * 70)
    log.info("✅ PREFLIGHT PASSED — all invariants satisfied")
    log.info("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
