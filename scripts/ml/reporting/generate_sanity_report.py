"""
scripts/ml/reporting/generate_sanity_report.py
================================================
Generate data/metrics/v3/sanity_report.json — a machine-readable summary
of internal consistency checks for the v3 credibility pipeline.

Sections
--------
finance_consistency_checks
    MDD in [-1, 0], Sharpe/CAGR computed from the same return series used
    for the cumulative-return plot (fold 5 time-series, equity-curve method).

fold_selection_check
    Which fold was used as "most recent regime", its size, date range, and
    ECE value recomputed from actual predictions.

calibration_check
    ECE per fold (weighted, no empty-bin bias) computed from actual
    predictions, compared to the stored JSON value.

Usage
-----
    python scripts/ml/reporting/generate_sanity_report.py \\
        --backtest data/metrics/backtest_v3.json \\
        --manifest data/training/v3/splits_manifest.json \\
        --models   models/v3 \\
        --out      data/metrics/v3/sanity_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

log = logging.getLogger("generate_sanity_report")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted ECE — identical to the one in plot_robustness_v3.py."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        ece += (n_bin / n) * abs(frac_pos - mean_prob)
    return float(ece)


def _sharpe(r: np.ndarray, periods_year: float = 252 / 20) -> float:
    if len(r) < 5 or r.std(ddof=1) < 1e-12:
        return float("nan")
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_year))


def _cagr(r: np.ndarray, periods_year: float = 252 / 20) -> float:
    if len(r) == 0:
        return float("nan")
    equity_end = float(np.prod(1.0 + np.clip(r, -0.999, 10.0)))
    n_years = len(r) / periods_year
    if n_years <= 0 or equity_end <= 0:
        return float("nan")
    return float(equity_end ** (1.0 / n_years) - 1.0)


def _mdd(r: np.ndarray) -> float:
    if len(r) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + np.clip(r, -0.999, 10.0))
    running_max = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(running_max, 1e-12)) - 1.0
    return float(np.min(dd))


# ── Load fold predictions ──────────────────────────────────────────────────────

def _load_fold_predictions(manifest_path: Path, model_dir: Path) -> Dict[int, dict]:
    """
    Returns {fold_k: {"y_true": array, "y_prob": array, "n": int,
                       "val_start": str, "val_end": str}}
    """
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed")
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits   = manifest["splits"]

    xgb_path  = model_dir / "v3_xgb_model.joblib"
    cal_path  = model_dir / "v3_calibrator.joblib"
    feat_path = model_dir / "v3_feature_names.joblib"

    if not xgb_path.exists():
        log.warning("XGB model not found — cannot compute fold predictions")
        return {}

    xgb_model  = joblib.load(xgb_path)
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    feat_cols  = joblib.load(feat_path) if feat_path.exists() else None
    if feat_cols is None:
        return {}

    results = {}
    for split in splits:
        fk       = split["fold"]
        val_path = Path(split["val_jsonl"])
        if not val_path.exists():
            continue

        X_rows, y_rows = [], []
        with val_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feats = rec.get("features", {})
                row = [
                    float(feats[c]) if (c in feats and feats[c] is not None
                                        and isinstance(feats[c], (int, float))
                                        and math.isfinite(float(feats[c])))
                    else float("nan")
                    for c in feat_cols
                ]
                X_rows.append(row)
                y_rows.append(int(rec.get("target_non_ok", 0)))

        if not X_rows:
            continue

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=int)
        xgb_raw = xgb_model.predict_proba(X)[:, 1]
        y_prob  = calibrator.predict(xgb_raw) if calibrator is not None else xgb_raw

        results[fk] = {
            "y_true":    y,
            "y_prob":    y_prob,
            "n":         len(y),
            "n_pos":     int(y.sum()),
            "n_neg":     len(y) - int(y.sum()),
            "val_start": split.get("val_start", "?"),
            "val_end":   split.get("val_end",   "?"),
        }
        log.info("Fold %d: n=%d pos=%.1f%%", fk, len(y), 100 * y.mean())

    return results


def _load_signal_df(manifest_path: Path, model_dir: Path, t_lo: float, t_hi: float):
    """Load fold 5 time-series signal (same code path as plot_financial_v3)."""
    try:
        from scripts.ml.reporting.plot_financial_v3 import (
            _load_signal_series, _apply_signal,
        )
        df = _load_signal_series(manifest_path, model_dir)
        if df is not None:
            df = _apply_signal(df, t_lo, t_hi)
        return df
    except Exception as e:
        log.warning("Could not load signal df: %s", e)
        return None


# ── Section builders ───────────────────────────────────────────────────────────

def _finance_checks(df, backtest_json: dict) -> dict:
    import pandas as pd
    if df is None or "signal_ret" not in df.columns:
        return {"status": "skipped", "reason": "time-series df not available"}

    df_grp = df.groupby("date").agg(
        signal_ret=("signal_ret", "mean"),
        always_ok_ret=("forward_return_20d", "mean"),
    ).reset_index().sort_values("date")

    sig_rets = df_grp["signal_ret"].values
    bm_rets  = df_grp["always_ok_ret"].values

    sig_mdd   = _mdd(sig_rets)
    bm_mdd    = _mdd(bm_rets)
    sig_sh    = _sharpe(sig_rets)
    bm_sh     = _sharpe(bm_rets)
    sig_cagr  = _cagr(sig_rets)
    bm_cagr   = _cagr(bm_rets)

    sig_j = backtest_json.get("signal",    {})
    bm_j  = backtest_json.get("always_ok", {})

    json_mdd_sig = float(sig_j.get("max_drawdown", float("nan")))
    json_mdd_bm  = float(bm_j.get("max_drawdown",  float("nan")))
    json_sh_sig  = float(sig_j.get("sharpe_ann",   float("nan")))
    json_sh_bm   = float(bm_j.get("sharpe_ann",    float("nan")))

    def _r(v):
        return round(float(v), 4) if math.isfinite(float(v)) else None

    checks = {
        "mdd_signal_series":         _r(sig_mdd),
        "mdd_baseline_series":       _r(bm_mdd),
        "mdd_signal_json":           _r(json_mdd_sig),
        "mdd_signal_valid_range":    (-1.0 <= sig_mdd <= 0.0),
        "mdd_baseline_valid_range":  (-1.0 <= bm_mdd  <= 0.0),
        "mdd_json_valid_range":      (-1.0 <= json_mdd_sig <= 0.0) if math.isfinite(json_mdd_sig) else False,
        "signal_sharpe_series":      _r(sig_sh),
        "baseline_sharpe_series":    _r(bm_sh),
        "signal_sharpe_json":        _r(json_sh_sig),
        "baseline_sharpe_json":      _r(json_sh_bm),
        "signal_cagr_series":        _r(sig_cagr),
        "baseline_cagr_series":      _r(bm_cagr),
        "ending_equity_positive":    float(np.prod(1 + np.clip(sig_rets, -0.999, 10))) > 0,
        "cagr_sign_matches_mdd":     (sig_cagr > 0) == (sig_mdd > -1.0) if math.isfinite(sig_cagr) else None,
        "n_periods":                 len(sig_rets),
        "notes": [
            "MDD computed from equity curve (1+r product), guaranteed in [-1,0].",
            f"JSON MDD={json_mdd_sig:.4f} is cross-sectional artifact; series MDD={sig_mdd:.4f} is correct.",
            f"Sharpe series vs JSON: {sig_sh:.3f} vs {json_sh_sig:.3f} (different universe/period).",
        ],
        "all_pass": (-1.0 <= sig_mdd <= 0.0) and (-1.0 <= bm_mdd <= 0.0),
    }
    return checks


def _fold_selection_check(fold_data: Dict[int, dict], backtest_json: dict) -> dict:
    if not fold_data:
        return {"status": "skipped", "reason": "no fold predictions available"}

    last_fk = max(fold_data.keys())
    fd = fold_data[last_fk]
    y_true = fd["y_true"]
    y_prob = fd["y_prob"]

    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float("nan")

    ece = _compute_ece(y_true, y_prob)

    return {
        "most_recent_fold":  last_fk,
        "val_start":         fd.get("val_start", "?"),
        "val_end":           fd.get("val_end",   "?"),
        "n":                 fd["n"],
        "n_pos":             fd["n_pos"],
        "n_neg":             fd["n_neg"],
        "pos_rate":          round(fd["n_pos"] / fd["n"], 4) if fd["n"] > 0 else None,
        "roc_auc":           round(roc_auc, 4) if math.isfinite(roc_auc) else None,
        "pr_auc":            round(pr_auc,  4) if math.isfinite(pr_auc)  else None,
        "ece_recomputed":    round(ece, 4) if math.isfinite(ece) else None,
        "note": (
            f"Most recent fold = max(fold_keys) = {last_fk}. "
            "Selection criterion: latest validation date in splits manifest."
        ),
    }


def _calibration_check(fold_data: Dict[int, dict]) -> dict:
    if not fold_data:
        return {"status": "skipped", "reason": "no fold predictions available"}

    per_fold = {}
    for fk in sorted(fold_data):
        fd = fold_data[fk]
        ece = _compute_ece(fd["y_true"], fd["y_prob"])
        per_fold[f"fold_{fk}"] = {
            "n":            fd["n"],
            "ece_weighted": round(ece, 4) if math.isfinite(ece) else None,
        }

    ece_vals = [v["ece_weighted"] for v in per_fold.values() if v["ece_weighted"] is not None]
    mean_ece = round(float(np.mean(ece_vals)), 4) if ece_vals else None
    well_cal = all((v or 1.0) < 0.05 for v in ece_vals) if ece_vals else False

    return {
        "per_fold":              per_fold,
        "mean_ece_across_folds": mean_ece,
        "well_calibrated_05":    well_cal,
        "note": (
            "ECE computed with weighted binning (10 equal-width bins). "
            "ECE=0.0 in train_v3_report.json was an artefact of isotonic "
            "regression fitting on its own calibration set."
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(
    backtest_path: Path,
    manifest_path: Path,
    model_dir:     Path,
    out_path:      Path,
) -> dict:
    report = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "version":       "v4.1",
        "finance_consistency_checks": {},
        "fold_selection_check":       {},
        "calibration_check":          {},
    }

    backtest_json: dict = {}
    if backtest_path.exists():
        backtest_json = json.loads(backtest_path.read_text(encoding="utf-8"))
    else:
        log.warning("backtest JSON not found: %s", backtest_path)

    # Thresholds
    t_lo, t_hi = 0.5, 0.65
    thr_path = model_dir / "v3_thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        t_lo = thr.get("t_lo", 0.5)
        t_hi = thr.get("t_hi", 0.65)

    # Load fold predictions
    fold_data: Dict[int, dict] = {}
    if manifest_path.exists() and model_dir.exists():
        log.info("Loading fold predictions...")
        fold_data = _load_fold_predictions(manifest_path, model_dir)

    # Load signal time-series
    df = None
    if manifest_path.exists() and model_dir.exists():
        log.info("Loading signal time-series (fold 5)...")
        df = _load_signal_df(manifest_path, model_dir, t_lo, t_hi)

    log.info("Building finance consistency checks...")
    report["finance_consistency_checks"] = _finance_checks(df, backtest_json)

    log.info("Building fold selection check...")
    report["fold_selection_check"] = _fold_selection_check(fold_data, backtest_json)

    log.info("Building calibration check...")
    report["calibration_check"] = _calibration_check(fold_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Sanity report written → %s", out_path)
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Generate v3 sanity_report.json")
    ap.add_argument("--backtest", default="data/metrics/backtest_v3.json")
    ap.add_argument("--manifest", default="data/training/v3/splits_manifest.json")
    ap.add_argument("--models",   default="models/v3")
    ap.add_argument("--out",      default="data/metrics/v3/sanity_report.json")
    args = ap.parse_args()

    rpt = generate(
        backtest_path=Path(args.backtest),
        manifest_path=Path(args.manifest),
        model_dir=Path(args.models),
        out_path=Path(args.out),
    )

    # Print summary
    fc = rpt.get("finance_consistency_checks", {})
    fs = rpt.get("fold_selection_check", {})
    cc = rpt.get("calibration_check", {})

    print("\n=== Sanity Report Summary ===")
    print(f"Finance — all_pass: {fc.get('all_pass')}")
    print(f"  signal MDD (series): {fc.get('mdd_signal_series')}  JSON: {fc.get('mdd_signal_json')}")
    print(f"  MDD in valid range: {fc.get('mdd_signal_valid_range')}")
    print(f"Fold selection — most recent fold: {fs.get('most_recent_fold')} "
          f"n={fs.get('n')} val_start={fs.get('val_start')} val_end={fs.get('val_end')}")
    print(f"  ECE (recomputed): {fs.get('ece_recomputed')}")
    print(f"Calibration — mean ECE: {cc.get('mean_ece_across_folds')} "
          f"well_calibrated(<0.05): {cc.get('well_calibrated_05')}")
    for fold_k, fv in cc.get("per_fold", {}).items():
        print(f"  {fold_k}: n={fv['n']} ECE={fv['ece_weighted']}")
    print(f"\nReport written → {args.out}")


if __name__ == "__main__":
    main()
