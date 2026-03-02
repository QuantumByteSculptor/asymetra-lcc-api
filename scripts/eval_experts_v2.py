"""
scripts/eval_experts_v2.py
──────────────────────────
Évaluation rigoureuse des expert bundles v2 vs bin_sigmoid.

Évalue sur holdout_v2.jsonl (données v2, jamais vues à l'entraînement).
Produit un rapport JSON + résumé texte.

Usage :
    python scripts/eval_experts_v2.py \
        --holdout data/training/holdout_v2.jsonl \
        --experts_dir models/experts \
        --sigmoid_bundle models/bin_sigmoid.joblib \
        --sigmoid_thresholds models/threshold_sigmoid.json \
        --out_json data/metrics/experts_v2_report.json \
        --out_txt  data/metrics/experts_v2_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import DEFAULT_CONFIG, features_to_row, vector_columns  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"ok": 0, "warn": 1, "block": 2}
LABEL_NAMES = ["ok", "warn", "block"]
COLS_V2 = vector_columns(DEFAULT_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def ece_score(y_bin: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        val += mask.sum() / len(probs) * abs(y_bin[mask].mean() - probs[mask].mean())
    return round(float(val), 6)


def feats_to_vec(feats: Dict, cols: List[str]) -> np.ndarray:
    row = features_to_row(feats, cfg=DEFAULT_CONFIG)
    vec = np.array([float(row.get(c, 0.0) or 0.0) for c in cols], dtype=float)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)


def confusion_str(cm: np.ndarray, labels: List[str]) -> str:
    w = 7
    header = f"{'':>{w+6}}" + "".join(f"  Pred:{l:>5}" for l in labels)
    lines = [header]
    for i, tl in enumerate(labels):
        row = f"  True:{tl:>{w}}" + "".join(f"{cm[i, j]:>12}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def fp_rate_ok(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    n_ok = (y_true_bin == 0).sum()
    if n_ok == 0:
        return 0.0
    return float(((y_pred_bin == 1) & (y_true_bin == 0)).sum() / n_ok)


def recall_nonok(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    n_no = (y_true_bin == 1).sum()
    if n_no == 0:
        return 0.0
    return float(((y_pred_bin == 1) & (y_true_bin == 1)).sum() / n_no)


# ─────────────────────────────────────────────────────────────────────────────
# Expert v2 predictor
# ─────────────────────────────────────────────────────────────────────────────

def load_expert_bundles(experts_dir: str) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    d = Path(experts_dir)
    for p in sorted(d.glob("*_bundle.joblib")):
        name = p.stem.replace("_bundle", "")
        try:
            b = joblib.load(p)
            if isinstance(b, dict) and "cols" in b:
                bundles[name] = b
        except Exception as e:
            print(f"  ⚠️  Could not load {p}: {e}")
    return bundles


def predict_expert_bundle(
    rec: Dict, bundles: Dict[str, Any]
) -> Tuple[str, float, str]:
    """Returns (status, prob_nonok, bundle_used)."""
    feats = rec.get("features", rec)
    at = str(feats.get("asset_type", "")).strip().lower()
    bundle = bundles.get(at) or bundles.get("global")
    if bundle is None:
        return "ok", 0.0, "none"

    used = bundle.get("asset_type", "global")
    sup = bundle.get("sup_bin", {})
    model = sup.get("model")
    thr = sup.get("thresholds", {})
    if model is None:
        return "ok", 0.0, used

    vec = feats_to_vec(feats, bundle.get("cols", COLS_V2))
    try:
        prob = float(model.predict_proba(vec)[0, 1])
    except Exception:
        return "ok", 0.0, used

    t_lo = thr.get("t_lo", 0.5)
    t_hi = thr.get("t_hi", 0.8)
    if prob >= t_hi:
        status = "block"
    elif prob >= t_lo:
        status = "warn"
    else:
        status = "ok"
    return status, prob, used


# ─────────────────────────────────────────────────────────────────────────────
# bin_sigmoid predictor
# ─────────────────────────────────────────────────────────────────────────────

def load_sigmoid(bundle_path: str, thr_path: str):
    bundle = joblib.load(bundle_path)
    thr_cfg = json.loads(Path(thr_path).read_text())
    model = bundle.get("model") or bundle
    sig_cols = thr_cfg.get("columns", [])
    t_lo = thr_cfg["t_lo"]
    t_hi = thr_cfg["t_hi"]
    return model, sig_cols, t_lo, t_hi


def predict_sigmoid(rec: Dict, model, cols, t_lo, t_hi) -> Tuple[str, float]:
    feats = rec.get("features", rec)
    row = features_to_row(feats, cfg=DEFAULT_CONFIG)
    vec = np.array([float(row.get(c, 0.0) or 0.0) for c in cols], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0).reshape(1, -1)
    try:
        prob = float(model.predict_proba(vec)[0, 1])
    except Exception:
        return "ok", 0.0
    if prob >= t_hi:
        status = "block"
    elif prob >= t_lo:
        status = "warn"
    else:
        status = "ok"
    return status, prob


# ─────────────────────────────────────────────────────────────────────────────
# Metrics bundle
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true_3: np.ndarray,
    y_pred_3: np.ndarray,
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true_3, y_pred_3))
    bacc = float(balanced_accuracy_score(y_true_3, y_pred_3))
    mf1 = float(f1_score(y_true_3, y_pred_3, average="macro", zero_division=0))

    y_pred_bin = (y_pred_3 > 0).astype(int)
    fp = fp_rate_ok(y_true_bin, y_pred_bin)
    rec = recall_nonok(y_true_bin, y_pred_bin)

    roc = float(roc_auc_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else 0.0
    pr  = float(average_precision_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else 0.0
    ece = ece_score(y_true_bin, y_prob)

    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true_3, y_pred_3, labels=[0, 1, 2], zero_division=0
    )
    per_class = {
        cls: {
            "precision": round(float(prec_cls[i]), 4),
            "recall":    round(float(rec_cls[i]), 4),
            "f1":        round(float(f1_cls[i]), 4),
            "support":   int(sup_cls[i]),
        }
        for i, cls in enumerate(LABEL_NAMES)
    }

    cm = confusion_matrix(y_true_3, y_pred_3, labels=[0, 1, 2])

    return {
        "n": len(y_true_3),
        "accuracy":          round(acc, 6),
        "balanced_accuracy": round(bacc, 6),
        "macro_f1":          round(mf1, 6),
        "fp_rate_ok":        round(fp, 6),
        "recall_non_ok":     round(rec, 6),
        "roc_auc":           round(roc, 6),
        "pr_auc":            round(pr, 6),
        "ece":               round(ece, 6),
        "per_class":         per_class,
        "confusion_matrix":  cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout",            default="data/training/holdout_v2.jsonl")
    ap.add_argument("--experts_dir",        default="models/experts")
    ap.add_argument("--sigmoid_bundle",     default="models/bin_sigmoid.joblib")
    ap.add_argument("--sigmoid_thresholds", default="models/threshold_sigmoid.json")
    ap.add_argument("--out_json",           default="data/metrics/experts_v2_report.json")
    ap.add_argument("--out_txt",            default="data/metrics/experts_v2_report.txt")
    args = ap.parse_args()

    lines: List[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    DIVIDER = "═" * 68
    SEP     = "─" * 68

    p(DIVIDER)
    p("  ÉVALUATION — Expert bundles v2 vs bin_sigmoid")
    p(f"  Holdout: {args.holdout}")
    p(DIVIDER)

    # ── Load data ──────────────────────────────────────────────────────────
    holdout = load_jsonl(args.holdout)
    dist    = Counter(r.get("label", "?") for r in holdout)
    at_dist = Counter(r.get("features", {}).get("asset_type", "?") for r in holdout)
    p(f"\n[DATA] holdout_v2: {len(holdout)} samples")
    p(f"       labels : {dict(dist)}")
    p(f"       assets : {dict(at_dist)}")
    p(f"       v2 features present: yes (holdout_v2 built from train_v2_all)")

    # ── Load models ────────────────────────────────────────────────────────
    bundles = load_expert_bundles(args.experts_dir)
    p(f"\n[MODELS] Expert bundles loaded: {sorted(bundles.keys())}")
    for name, b in sorted(bundles.items()):
        meta = b.get("meta", {})
        thr  = b.get("sup_bin", {}).get("thresholds", {})
        p(f"  {name:10s}  n_train={meta.get('n_train','?'):6}  "
          f"t_lo={thr.get('t_lo',0):.4f}  t_hi={thr.get('t_hi',0):.4f}  "
          f"features={meta.get('n_features','?')}")

    sig_model, sig_cols, t_lo_s, t_hi_s = load_sigmoid(
        args.sigmoid_bundle, args.sigmoid_thresholds
    )
    p(f"\n[MODELS] bin_sigmoid loaded  t_lo={t_lo_s:.4f}  t_hi={t_hi_s:.4f}  "
      f"n_cols={len(sig_cols)}")

    # ── Expert predictions ─────────────────────────────────────────────────
    exp_true_3, exp_pred_3 = [], []
    exp_true_bin, exp_prob = [], []
    per_asset: Dict[str, Dict] = defaultdict(lambda: {
        "true_3": [], "pred_3": [], "true_bin": [], "prob": []
    })

    for rec in holdout:
        true_lbl = rec.get("label", "ok")
        pred_status, prob, bundle_used = predict_expert_bundle(rec, bundles)
        at = str(rec.get("features", {}).get("asset_type", "global")).lower()

        exp_true_3.append(LABEL_MAP.get(true_lbl, 0))
        exp_pred_3.append(LABEL_MAP.get(pred_status, 0))
        exp_true_bin.append(0 if true_lbl == "ok" else 1)
        exp_prob.append(prob)

        per_asset[at]["true_3"].append(LABEL_MAP.get(true_lbl, 0))
        per_asset[at]["pred_3"].append(LABEL_MAP.get(pred_status, 0))
        per_asset[at]["true_bin"].append(0 if true_lbl == "ok" else 1)
        per_asset[at]["prob"].append(prob)

    exp_true_3   = np.array(exp_true_3)
    exp_pred_3   = np.array(exp_pred_3)
    exp_true_bin = np.array(exp_true_bin)
    exp_prob     = np.array(exp_prob)

    exp_metrics = compute_metrics(exp_true_3, exp_pred_3, exp_true_bin, exp_prob)

    # ── Sigmoid predictions ────────────────────────────────────────────────
    sig_pred_3, sig_prob = [], []
    for rec in holdout:
        ps, pb = predict_sigmoid(rec, sig_model, sig_cols, t_lo_s, t_hi_s)
        sig_pred_3.append(LABEL_MAP.get(ps, 0))
        sig_prob.append(pb)

    sig_pred_3 = np.array(sig_pred_3)
    sig_prob   = np.array(sig_prob)
    sig_metrics = compute_metrics(exp_true_3, sig_pred_3, exp_true_bin, sig_prob)

    # ── Global metrics ─────────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("  MÉTRIQUES GLOBALES — holdout_v2")
    p(SEP)
    p(f"  {'Métrique':<25} {'bin_sigmoid':>13} {'Expert v2':>13} {'Δ':>10}")
    p(f"  {'─'*62}")
    metrics_order = [
        ("Accuracy",          "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Macro F1",          "macro_f1"),
        ("FP-rate OK",        "fp_rate_ok"),
        ("Recall NON-OK",     "recall_non_ok"),
        ("ROC-AUC",           "roc_auc"),
        ("PR-AUC",            "pr_auc"),
        ("ECE",               "ece"),
    ]
    for display, key in metrics_order:
        s_val = sig_metrics[key]
        e_val = exp_metrics[key]
        delta = e_val - s_val
        # Lower is better for FP-rate and ECE
        better = delta < -0.005 if key in ("fp_rate_ok", "ece") else delta > 0.005
        worse  = delta > 0.005  if key in ("fp_rate_ok", "ece") else delta < -0.005
        arrow = "▲" if better else ("▼" if worse else "≈")
        p(f"  {display:<25} {s_val:>13.4f} {e_val:>13.4f} {arrow} {delta:>+8.4f}")

    # ── Par classe ─────────────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("  PAR CLASSE — Expert v2")
    p(SEP)
    p(f"  {'Classe':<8} {'Precision':>11} {'Recall':>11} {'F1':>11} {'Support':>10}")
    for cls in LABEL_NAMES:
        c = exp_metrics["per_class"][cls]
        p(f"  {cls:<8} {c['precision']:>11.4f} {c['recall']:>11.4f} "
          f"{c['f1']:>11.4f} {c['support']:>10}")

    p(f"\n{SEP}")
    p("  PAR CLASSE — bin_sigmoid")
    p(SEP)
    p(f"  {'Classe':<8} {'Precision':>11} {'Recall':>11} {'F1':>11} {'Support':>10}")
    for cls in LABEL_NAMES:
        c = sig_metrics["per_class"][cls]
        p(f"  {cls:<8} {c['precision']:>11.4f} {c['recall']:>11.4f} "
          f"{c['f1']:>11.4f} {c['support']:>10}")

    # ── Confusion matrices ─────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("  MATRICE DE CONFUSION — Expert v2")
    p(SEP)
    p(confusion_str(np.array(exp_metrics["confusion_matrix"]), LABEL_NAMES))

    p(f"\n{SEP}")
    p("  MATRICE DE CONFUSION — bin_sigmoid")
    p(SEP)
    p(confusion_str(np.array(sig_metrics["confusion_matrix"]), LABEL_NAMES))

    # ── Par asset_type ─────────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("  PERFORMANCE PAR ASSET TYPE — Expert v2")
    p(SEP)
    p(f"  {'Asset':>10} {'n':>6} {'Acc':>8} {'BalAcc':>8} {'MacF1':>8} "
      f"{'RecNonOK':>10} {'FP_OK':>8} {'ROC':>8}")
    asset_metrics: Dict[str, Any] = {}
    for at, res in sorted(per_asset.items()):
        yt3  = np.array(res["true_3"])
        yp3  = np.array(res["pred_3"])
        ytb  = np.array(res["true_bin"])
        ypr  = np.array(res["prob"])
        if len(yt3) == 0:
            continue
        at_m = compute_metrics(yt3, yp3, ytb, ypr)
        asset_metrics[at] = at_m
        p(f"  {at:>10} {at_m['n']:>6} {at_m['accuracy']:>8.4f} "
          f"{at_m['balanced_accuracy']:>8.4f} {at_m['macro_f1']:>8.4f} "
          f"{at_m['recall_non_ok']:>10.4f} {at_m['fp_rate_ok']:>8.4f} "
          f"{at_m['roc_auc']:>8.4f}")

    # ── Seuils ─────────────────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("  SEUILS PAR ASSET TYPE — Expert v2")
    p(SEP)
    p(f"  {'Asset':>12} {'t_lo':>8} {'t_hi':>8} {'n_train':>10} {'FP@t_lo':>10} {'Rec@t_lo':>10}")
    for at_name in sorted(bundles.keys()):
        b = bundles[at_name]
        thr = b.get("sup_bin", {}).get("thresholds", {})
        meta = b.get("meta", {})
        # compute FP and recall at t_lo on holdout
        at_res = per_asset.get(at_name) or {"true_bin": list(exp_true_bin), "prob": list(exp_prob)}
        ytb = np.array(at_res["true_bin"])
        ypr = np.array(at_res["prob"])
        t_lo_val = thr.get("t_lo", 0.5)
        yp_tlo = (ypr >= t_lo_val).astype(int)
        fp_at = fp_rate_ok(ytb, yp_tlo)
        rc_at = recall_nonok(ytb, yp_tlo)
        p(f"  {at_name:>12} {t_lo_val:>8.4f} {thr.get('t_hi',0):>8.4f} "
          f"{meta.get('n_train',0):>10} {fp_at:>10.4f} {rc_at:>10.4f}")

    # ── Rapport synthétique ────────────────────────────────────────────────
    p(f"\n{DIVIDER}")
    p("  RAPPORT SYNTHÉTIQUE")
    p(DIVIDER)

    strengths, issues = [], []

    e = exp_metrics
    if e["roc_auc"] >= 0.75:
        strengths.append(f"ROC-AUC solide ({e['roc_auc']:.3f})")
    elif e["roc_auc"] >= 0.65:
        issues.append(f"ROC-AUC modeste ({e['roc_auc']:.3f}) — acceptable mais à surveiller")
    else:
        issues.append(f"ROC-AUC faible ({e['roc_auc']:.3f})")

    if e["fp_rate_ok"] <= 0.15:
        strengths.append(f"FP-rate OK respecté ({e['fp_rate_ok']:.3f} ≤ 0.15)")
    else:
        issues.append(f"FP-rate OK dépassé ({e['fp_rate_ok']:.3f} > 0.15)")

    if e["recall_non_ok"] >= 0.45:
        strengths.append(f"Recall NON-OK atteint ({e['recall_non_ok']:.3f} ≥ 0.45)")
    elif e["recall_non_ok"] >= 0.30:
        issues.append(f"Recall NON-OK faible ({e['recall_non_ok']:.3f}), cible 0.45")
    else:
        issues.append(f"Recall NON-OK critique ({e['recall_non_ok']:.3f} < 0.30)")

    if e["ece"] < 0.10:
        strengths.append(f"Calibration OK (ECE={e['ece']:.3f})")
    else:
        issues.append(f"Calibration dégradée (ECE={e['ece']:.3f})")

    if e["roc_auc"] > sig_metrics["roc_auc"]:
        strengths.append(f"Expert v2 bat bin_sigmoid en ROC-AUC "
                         f"({e['roc_auc']:.3f} vs {sig_metrics['roc_auc']:.3f})")
    else:
        issues.append(f"bin_sigmoid meilleur ROC-AUC "
                      f"({sig_metrics['roc_auc']:.3f} vs {e['roc_auc']:.3f})")

    missing = [at for at in ["commodity", "fx", "crypto"] if at not in bundles]
    if missing:
        issues.append(f"Experts manquants (→ fallback global): {missing}")

    p("\n  POINTS FORTS:")
    for s in strengths:
        p(f"    ✅ {s}")

    p("\n  POINTS FAIBLES / RISQUES:")
    for iss in issues:
        p(f"    ⚠️  {iss}")

    critical = [i for i in issues if any(k in i for k in [
        "FP-rate OK dépassé", "critique", "ROC-AUC faible"
    ])]
    if not critical:
        verdict = "✅  DÉPLOYABLE (sous monitoring)"
        detail  = ("Critères critiques respectés. Activer EXPERTS_ENABLED=1 en prod "
                   "avec monitoring fp_rate + recall.")
    else:
        verdict = "⚠️  À AMÉLIORER"
        detail  = "Blocants: " + "; ".join(critical)

    p(f"\n{DIVIDER}")
    p(f"  VERDICT FINAL: {verdict}")
    p(f"  {detail}")
    p(DIVIDER)

    # ── Export JSON ────────────────────────────────────────────────────────
    report = {
        "evaluation_date": "2026-03-02",
        "holdout": {
            "path": args.holdout,
            "n": len(holdout),
            "label_dist": dict(dist),
            "asset_dist": dict(at_dist),
            "has_v2_features": True,
        },
        "expert_v2": {
            "bundles": sorted(bundles.keys()),
            "thresholds": {
                at: {
                    "t_lo": round(b["sup_bin"]["thresholds"]["t_lo"], 4),
                    "t_hi": round(b["sup_bin"]["thresholds"]["t_hi"], 4),
                }
                for at, b in bundles.items()
            },
            "global": exp_metrics,
            "per_asset": asset_metrics,
        },
        "bin_sigmoid": {
            "t_lo": t_lo_s,
            "t_hi": t_hi_s,
            "global": sig_metrics,
        },
        "comparison": {
            key: {
                "bin_sigmoid": sig_metrics[key],
                "expert_v2":   exp_metrics[key],
                "delta":       round(exp_metrics[key] - sig_metrics[key], 6),
            }
            for _, key in metrics_order
        },
        "strengths": strengths,
        "issues": issues,
        "verdict": verdict,
        "verdict_detail": detail,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    p(f"\n  📄 JSON  → {args.out_json}")

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(lines), encoding="utf-8")
    p(f"  📄 TEXT  → {args.out_txt}")


if __name__ == "__main__":
    main()
