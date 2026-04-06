# ml/train_experts.py
"""
Per-asset-type expert bundle training.

Each expert bundle contains:
  - unsup: IF + LOF trained on label_v2=="ok" samples
  - sup_bin: XGB binary (sigmoid-calibrated), trained on all labels
  - calibrated thresholds under FP constraint alpha

Usage:
    python ml/train_experts.py \
        --data data/training/train_v2_all.jsonl \
        --out_dir models/experts/ \
        --min_samples 200 \
        --warn_q 0.95 \
        --block_q 0.99 \
        --alpha 0.15 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError("xgboost is required. pip install xgboost") from e

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL, return list of {label, features} dicts."""
    records: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        feats = obj.get("features", obj)
        label = obj.get("label") or feats.get("label") or feats.get("label_v2")
        records.append({"label": label, "features": feats})
    return records


def records_to_matrix(
    records: List[Dict[str, Any]],
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert records to (X, y) arrays. y: 0=ok, 1=non_ok."""
    X_rows: List[List[float]] = []
    y_list: List[int] = []
    for rec in records:
        feats = rec["features"]
        row = features_to_row(feats, cfg=DEFAULT_CONFIG)
        X_rows.append([float(row.get(c, 0.0) or 0.0) for c in cols])
        label = rec["label"]
        y_list.append(0 if label == "ok" else 1)
    return np.asarray(X_rows, dtype=float), np.asarray(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def find_t_lo_fp_constrained(
    p: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """
    Find the threshold t_lo that maximizes recall(non_ok)
    subject to FP_rate(ok) <= alpha.
    """
    n_ok = int((y == 0).sum())
    if n_ok == 0:
        return None

    thresholds = np.unique(np.clip(p, 0.0, 1.0))
    best: Optional[Dict[str, Any]] = None

    for t in sorted(thresholds):
        pred = (p >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        try:
            cm = confusion_matrix(y, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except ValueError:
            continue
        fp_rate = fp / (n_ok + 1e-12)
        if fp_rate <= alpha:
            recall = tp / (tp + fn + 1e-12)
            if best is None or recall > best["recall"]:
                best = {
                    "t": float(t),
                    "fp": int(fp), "fn": int(fn),
                    "tp": int(tp), "tn": int(tn),
                    "recall": float(recall),
                    "fp_rate_ok": float(fp_rate),
                }

    return best


def compute_t_hi(p: np.ndarray, y: np.ndarray, q: float = 0.995) -> float:
    """Upper threshold: quantile of probabilities among OK samples."""
    p_ok = p[y == 0]
    if len(p_ok) == 0:
        return float(np.quantile(p, q))
    return float(np.quantile(p_ok, q))


# ---------------------------------------------------------------------------
# Unsup training
# ---------------------------------------------------------------------------

def train_unsup(
    X_ok: np.ndarray,
    seed: int,
    warn_q: float,
    block_q: float,
    if_estimators: int = 300,
    if_contamination: float = 0.01,
    lof_neighbors: int = 35,
    w_if: float = 0.5,
    w_lof: float = 0.5,
) -> Dict[str, Any]:
    """Train IF + LOF on clean (ok) samples, return unsup sub-bundle."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_ok)

    iforest = IsolationForest(
        n_estimators=if_estimators,
        contamination=if_contamination,
        random_state=seed,
        n_jobs=-1,
    )
    lof = LocalOutlierFactor(
        n_neighbors=min(lof_neighbors, len(X_imp) - 1),
        novelty=True,
        metric="minkowski",
    )
    iforest.fit(X_imp)
    lof.fit(X_imp)

    raw_if = iforest.score_samples(X_imp).astype(float)
    raw_lof = lof.score_samples(X_imp).astype(float)

    mu_if = float(np.mean(raw_if))
    sg_if = float(np.std(raw_if) + 1e-12)
    mu_lof = float(np.mean(raw_lof))
    sg_lof = float(np.std(raw_lof) + 1e-12)

    z_if = (raw_if - mu_if) / sg_if
    z_lof = (raw_lof - mu_lof) / sg_lof
    s_ens = w_if * z_if + w_lof * z_lof

    thr = {
        "warn": float(np.quantile(s_ens, warn_q)),
        "block": float(np.quantile(s_ens, block_q)),
    }

    return {
        "iforest": iforest,
        "lof": lof,
        "imputer": imputer,
        "imputer_stats": imputer.statistics_,
        "score_norm": {
            "if": {"mu": mu_if, "sigma": sg_if},
            "lof": {"mu": mu_lof, "sigma": sg_lof},
        },
        "weights": {"if": w_if, "lof": w_lof},
        "thresholds": thr,
    }


# ---------------------------------------------------------------------------
# Sup binary training
# ---------------------------------------------------------------------------

def train_sup_bin(
    X: np.ndarray,
    y: np.ndarray,
    cols: List[str],
    seed: int,
    alpha: float,
    test_size: float = 0.25,
    calib_cv: int = 3,
) -> Dict[str, Any]:
    """Train calibrated XGB binary classifier, return sup_bin sub-bundle."""
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    n_total = len(y)
    spw = float(n0 / max(n1, 1))

    # Scale complexity to dataset size to avoid overfitting on small classes
    if n_total < 500:
        n_estimators, max_depth, lr, reg_lambda, min_cw = 300, 3, 0.05, 2.0, 3.0
    elif n_total < 2000:
        n_estimators, max_depth, lr, reg_lambda, min_cw = 500, 4, 0.05, 1.5, 2.0
    else:
        n_estimators, max_depth, lr, reg_lambda, min_cw = 700, 5, 0.04, 1.0, 1.0

    base = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=reg_lambda,
        reg_alpha=0.1,
        min_child_weight=min_cw,
        gamma=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=spw,
    )

    # Need at least 2 classes and enough samples for stratified split
    if n1 < 5 or n0 < 5:
        raise ValueError(f"Insufficient class balance: n0={n0}, n1={n1}")

    actual_test = test_size if len(y) * test_size >= 10 else 0.2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=actual_test, random_state=seed, stratify=y
    )

    model = CalibratedClassifierCV(base, method="sigmoid", cv=calib_cv)
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]

    # Calibrate thresholds
    best_tlo = find_t_lo_fp_constrained(p_te, y_te, alpha=alpha)
    t_lo = best_tlo["t"] if best_tlo else 0.5
    t_hi = compute_t_hi(p_te, y_te)

    # Compute medians for NaN imputation at inference time
    medians: Dict[str, float] = {}
    for i, c in enumerate(cols):
        col_vals = X[:, i]
        finite = col_vals[np.isfinite(col_vals)]
        medians[c] = float(np.median(finite)) if len(finite) > 0 else 0.0

    thresholds = {"t_lo": t_lo, "t_hi": t_hi, "alpha": alpha}
    if best_tlo:
        thresholds.update({
            "recall_non_ok_at_t_lo": best_tlo["recall"],
            "fp_rate_ok_at_t_lo": best_tlo["fp_rate_ok"],
        })

    return {
        "model": model,
        "medians": medians,
        "feature_columns": cols,
        "thresholds": thresholds,
        "train_dist": {"n_ok": n0, "n_non_ok": n1},
    }


# ---------------------------------------------------------------------------
# Main: train one expert bundle
# ---------------------------------------------------------------------------

def train_expert_bundle(
    asset_type: str,
    records: List[Dict[str, Any]],
    cols: List[str],
    min_samples: int,
    warn_q: float,
    block_q: float,
    alpha: float,
    seed: int,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Train one expert bundle for a given asset_type.
    Returns None if not enough samples.
    """
    ok_recs = [r for r in records if r["label"] == "ok"]
    n_ok = len(ok_recs)
    n_all = len(records)

    if verbose:
        label_dist = Counter(r["label"] for r in records)
        print(f"  [{asset_type}] n_total={n_all}, n_ok={n_ok}, dist={dict(label_dist)}")

    # For data-scarce asset classes (fx, crypto, commodity) allow
    # training with as few as 80 total records if at least 20 are ok-labeled.
    effective_min = max(80, min_samples) if n_all < min_samples else min_samples
    min_ok = max(20, effective_min // 5)
    if n_all < effective_min or n_ok < min_ok:
        if verbose:
            print(f"  [{asset_type}] SKIP — insufficient samples "
                  f"(n_all={n_all}<{effective_min} or n_ok={n_ok}<{min_ok})")
        return None

    # Build matrices
    X_ok, _ = records_to_matrix(ok_recs, cols)
    X_all, y_all = records_to_matrix(records, cols)

    # Unsup
    try:
        unsup = train_unsup(X_ok, seed=seed, warn_q=warn_q, block_q=block_q)
    except Exception as e:
        print(f"  [{asset_type}] unsup training FAILED: {e}")
        return None

    # Sup binary
    try:
        sup_bin = train_sup_bin(X_all, y_all, cols=cols, seed=seed, alpha=alpha)
    except Exception as e:
        print(f"  [{asset_type}] sup_bin training FAILED: {e}")
        return None

    bundle = {
        "asset_type": asset_type,
        "feature_version": "v2",
        "cols": cols,
        "unsup": unsup,
        "sup_bin": sup_bin,
        "meta": {
            "version": "experts_v1",
            "asset_type": asset_type,
            "n_train": n_all,
            "n_unsup": n_ok,
            "n_features": len(cols),
            "warn_q": warn_q,
            "block_q": block_q,
            "seed": seed,
        },
    }

    if verbose:
        thr = bundle["sup_bin"]["thresholds"]
        uthr = bundle["unsup"]["thresholds"]
        print(
            f"  [{asset_type}] OK — "
            f"unsup warn>={uthr['warn']:.3f} block>={uthr['block']:.3f} | "
            f"sup t_lo={thr['t_lo']:.3f} t_hi={thr['t_hi']:.3f}"
        )

    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-asset expert bundles (v1)")
    ap.add_argument("--data", required=True, help="JSONL training data path")
    ap.add_argument("--out_dir", default="models/experts/", help="Output directory for expert bundles")
    ap.add_argument("--min_samples", type=int, default=200, help="Min records per asset_type to train (else fallback global)")
    ap.add_argument("--warn_q", type=float, default=0.95, help="Warn quantile for unsup thresholds")
    ap.add_argument("--block_q", type=float, default=0.99, help="Block quantile for unsup thresholds")
    ap.add_argument("--alpha", type=float, default=0.15, help="Max FP rate on OK samples for sup threshold calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--if_estimators", type=int, default=300)
    ap.add_argument("--lof_neighbors", type=int, default=35)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = vector_columns(DEFAULT_CONFIG)
    print(f"Feature columns: {len(cols)} (v2)")

    # Load all records
    print(f"Loading data from: {args.data}")
    all_records = load_records(args.data)
    print(f"Total records loaded: {len(all_records)}")

    if not all_records:
        raise SystemExit("No records loaded. Check --data path.")

    # Group by asset_type
    by_asset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for rec in all_records:
        at = str(rec["features"].get("asset_type") or "").strip().lower()
        if not at:
            skipped += 1
            continue
        by_asset[at].append(rec)

    print(f"Asset types found: {sorted(by_asset.keys())} ({skipped} records skipped, no asset_type)")

    report: Dict[str, Any] = {}

    # Train global bundle first (fallback)
    print("\n--- Training global bundle (all asset types) ---")
    global_bundle = train_expert_bundle(
        asset_type="global",
        records=all_records,
        cols=cols,
        min_samples=max(200, args.min_samples),
        warn_q=args.warn_q,
        block_q=args.block_q,
        alpha=args.alpha,
        seed=args.seed,
    )

    if global_bundle is None:
        print("WARNING: Not enough data for global bundle. Try lowering --min_samples.")
    else:
        out_path = out_dir / "global_bundle.joblib"
        joblib.dump(global_bundle, out_path)
        print(f"  Saved: {out_path}")
        report["global"] = {"n_train": len(all_records), "status": "ok"}

    # Train per-asset bundles
    print("\n--- Training per-asset bundles ---")
    for asset_type in sorted(by_asset.keys()):
        recs = by_asset[asset_type]
        print(f"\nAsset type: {asset_type} ({len(recs)} records)")

        bundle = train_expert_bundle(
            asset_type=asset_type,
            records=recs,
            cols=cols,
            min_samples=args.min_samples,
            warn_q=args.warn_q,
            block_q=args.block_q,
            alpha=args.alpha,
            seed=args.seed,
        )

        if bundle is None:
            report[asset_type] = {"n_train": len(recs), "status": "skipped (< min_samples)"}
            if global_bundle is not None:
                print(f"  [{asset_type}] → will fallback to global bundle at inference")
            continue

        out_path = out_dir / f"{asset_type}_bundle.joblib"
        joblib.dump(bundle, out_path)
        print(f"  [{asset_type}] Saved: {out_path}")
        report[asset_type] = {
            "n_train": len(recs),
            "n_ok": len([r for r in recs if r["label"] == "ok"]),
            "status": "ok",
            "sup_thresholds": bundle["sup_bin"]["thresholds"],
            "unsup_thresholds": bundle["unsup"]["thresholds"],
        }

    # Summary
    print("\n===== TRAINING SUMMARY =====")
    for at, info in sorted(report.items()):
        status = info.get("status", "?")
        n = info.get("n_train", 0)
        print(f"  {at:20s} n={n:6d}  {status}")

    print(f"\nExpert bundles saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
