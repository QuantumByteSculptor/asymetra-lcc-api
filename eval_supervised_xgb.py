# eval_supervised_xgb.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import itertools

import joblib
import numpy as np
import pandas as pd


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        feats = r.get("features") or {}
        row = dict(feats)
        row["_true_label"] = r.get("label")
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Unsupervised z-scores
# -----------------------------
def load_unsup_bundle(path: str) -> Dict[str, Any]:
    b = joblib.load(path)
    models = b.get("models") or {}
    cols = b.get("columns") or []
    score_norm = b.get("score_norm") or {}

    if not cols or "iforest" not in models or "lof" not in models:
        raise ValueError("Invalid unsup bundle: need models.iforest/lof + columns + score_norm.")
    if "if" not in score_norm or "lof" not in score_norm:
        raise ValueError("Invalid score_norm keys (expected 'if' and 'lof').")
    return b


def add_unsup_zscores_to_df(df: pd.DataFrame, unsup: Dict[str, Any]) -> int:
    cols: List[str] = list(unsup["columns"])
    iforest = unsup["models"]["iforest"]
    lof = unsup["models"]["lof"]
    score_norm = unsup["score_norm"]

    mu_if = float(score_norm["if"]["mu"])
    sd_if = float(score_norm["if"]["sigma"] or 1e-12)
    mu_lof = float(score_norm["lof"]["mu"])
    sd_lof = float(score_norm["lof"]["sigma"] or 1e-12)

    X = np.full((len(df), len(cols)), np.nan, dtype=float)

    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            continue

        # aliases
        if c == "max_dd" and "max_drawdown" in df.columns:
            X[:, j] = pd.to_numeric(df["max_drawdown"], errors="coerce").to_numpy(dtype=float)
            continue
        if c == "max_drawdown" and "max_dd" in df.columns:
            X[:, j] = pd.to_numeric(df["max_dd"], errors="coerce").to_numpy(dtype=float)
            continue

    # Safe impute (no All-NaN warnings)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(np.isnan(col)):
            fillv = 0.0
        else:
            fillv = float(np.nanmedian(col))
            if not np.isfinite(fillv):
                fillv = 0.0
        col[np.isnan(col)] = fillv
        X[:, j] = col

    raw_if = np.asarray(iforest.score_samples(X), dtype=float)
    raw_lof = np.asarray(lof.score_samples(X), dtype=float)

    z_if = (raw_if - mu_if) / (sd_if + 1e-12)
    z_lof = (raw_lof - mu_lof) / (sd_lof + 1e-12)
    z_gap = z_if - z_lof

    df["raw_if"] = raw_if
    df["raw_lof"] = raw_lof
    df["z_if"] = z_if
    df["z_lof"] = z_lof
    df["z_gap_if_lof"] = z_gap

    return len(df)


# -----------------------------
# Training prep apply
# -----------------------------
def apply_training_prep(df: pd.DataFrame, prep: Dict[str, Any]) -> pd.DataFrame:
    feature_columns: List[str] = list(prep.get("feature_columns") or [])
    numeric_cols: List[str] = list(prep.get("numeric_cols") or [])
    cat_cols: List[str] = list(prep.get("cat_cols") or prep.get("categorical_cols") or [])
    medians: Dict[str, float] = dict(prep.get("medians") or {})
    onehot_cols: List[str] = list(prep.get("onehot_cols") or [])

    # numeric
    X_num = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        if c in df.columns:
            X_num[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            if c == "max_dd" and "max_drawdown" in df.columns:
                X_num[c] = pd.to_numeric(df["max_drawdown"], errors="coerce")
            elif c == "max_drawdown" and "max_dd" in df.columns:
                X_num[c] = pd.to_numeric(df["max_dd"], errors="coerce")
            else:
                X_num[c] = np.nan

    # impute numeric (safe)
    for c in numeric_cols:
        if c in medians and medians[c] is not None:
            fillv = float(medians[c])
        else:
            col = X_num[c].to_numpy(dtype=float)
            if np.all(np.isnan(col)):
                fillv = 0.0
            else:
                fillv = float(np.nanmedian(col))
                if not np.isfinite(fillv):
                    fillv = 0.0
        X_num[c] = X_num[c].fillna(fillv)

    # categorical -> one-hot
    X_cat = pd.DataFrame(index=df.index)
    if cat_cols:
        base = df.copy()
        for c in cat_cols:
            if c not in base.columns:
                base[c] = None

        d1 = pd.get_dummies(base[cat_cols], prefix=cat_cols, prefix_sep="_", dummy_na=False)
        d2 = pd.get_dummies(base[cat_cols], prefix=cat_cols, prefix_sep="=", dummy_na=False)
        X_cat = pd.concat([d1, d2], axis=1)

    if onehot_cols:
        for c in onehot_cols:
            if c not in X_cat.columns:
                X_cat[c] = 0
        X_cat = X_cat[onehot_cols]

    X_all = pd.concat([X_num, X_cat], axis=1)

    if feature_columns:
        for c in feature_columns:
            if c not in X_all.columns:
                X_all[c] = 0
        X_all = X_all[feature_columns]

    return X_all


# -----------------------------
# Helpers
# -----------------------------
def _norm(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def extract_sup_model(bundle: Dict[str, Any]):
    for k in ["model", "clf", "xgb", "estimator"]:
        m = bundle.get(k)
        if hasattr(m, "predict_proba"):
            return m

    models = bundle.get("models")
    if isinstance(models, dict):
        for k in ["xgb", "clf", "model", "gbt", "classifier"]:
            m = models.get(k)
            if hasattr(m, "predict_proba"):
                return m
        for m in models.values():
            if hasattr(m, "predict_proba"):
                return m

    raise ValueError("Could not find a model with predict_proba in sup_bundle.")


def best_label_permutation(proba: np.ndarray, y_true: List[str]) -> List[str]:
    """
    Si on ne sait pas à quoi correspondent les colonnes du predict_proba,
    on teste les 6 permutations possibles et on garde celle qui maximise l'accuracy.
    """
    want = ["ok", "warn", "block"]
    yt = [_norm(x) for x in y_true]
    known = set(want)

    idx_valid = [i for i, t in enumerate(yt) if t in known]
    if not idx_valid:
        return want  # fallback

    P = proba[idx_valid, :]
    T = [yt[i] for i in idx_valid]

    best_acc = -1.0
    best_perm = want

    for perm in itertools.permutations(want, 3):
        pred = [perm[int(np.argmax(row))] for row in P]
        acc = sum(p == t for p, t in zip(pred, T)) / max(1, len(T))
        if acc > best_acc:
            best_acc = acc
            best_perm = list(perm)

    print(f"✅ inferred proba column mapping = {best_perm} (holdout-acc={best_acc:.4f})")
    return best_perm


def predict_proba_3(bundle: Dict[str, Any], X: pd.DataFrame, y_true: List[str]) -> Tuple[np.ndarray, List[str]]:
    model = extract_sup_model(bundle)
    proba = np.asarray(model.predict_proba(X), dtype=float)

    # Try to use model.classes_ if readable
    labels = None
    if hasattr(model, "classes_"):
        cls = list(model.classes_)
        cls_norm = [_norm(c) for c in cls]
        want = ["ok", "warn", "block"]

        # if it matches, use it
        if all(w in cls_norm for w in want):
            idx = [cls_norm.index(w) for w in want]
            out = proba[:, idx]
            # renorm just in case
            s = out.sum(axis=1, keepdims=True)
            s = np.where(s <= 1e-12, 1.0, s)
            out = out / s
            return out, want

    # Otherwise infer mapping via permutations on holdout
    labels = best_label_permutation(proba, y_true)
    return proba, labels


def confusion_matrix_3(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    labels = ["ok", "warn", "block"]
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        t = _norm(t)
        p = _norm(p)
        if t in labels and p in labels:
            m[labels.index(t), labels.index(p)] += 1
    return m


def clean_ticker(v: Any) -> str:
    try:
        if isinstance(v, pd.Series):
            return "None" if len(v) == 0 else str(v.iloc[0])
        return "None" if v is None else str(v)
    except Exception:
        return "None"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--sup", required=True)
    ap.add_argument("--unsup_bundle", default=None)
    ap.add_argument("--show_top", type=int, default=30)
    args = ap.parse_args()

    sup_bundle = joblib.load(args.sup)
    prep = sup_bundle.get("prep") or {}
    if not prep:
        raise ValueError("sup_bundle missing 'prep' dict.")

    records = read_jsonl(args.holdout)
    df = flatten_records(records)

    if args.unsup_bundle:
        unsup = load_unsup_bundle(args.unsup_bundle)
        added = add_unsup_zscores_to_df(df, unsup)
        print(f"✅ added unsup z-scores to holdout records: {added}/{len(df)}")

    X = apply_training_prep(df, prep)

    true = [_norm(x) for x in df["_true_label"].tolist()]
    proba, labels = predict_proba_3(sup_bundle, X, true)

    pred_idx = np.argmax(proba, axis=1)
    pred = [labels[i] for i in pred_idx]

    known = {"ok", "warn", "block"}
    mask = [t in known for t in true]
    acc = float(np.mean([pred[i] == true[i] for i in range(len(true)) if mask[i]])) if any(mask) else float("nan")
    cm = confusion_matrix_3(true, pred)

    print("\n--- EVAL ---")
    print(f"holdout: {args.holdout}")
    print(f"n={len(df)}  accuracy={acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred)  [OK, WARN, BLOCK]:")
    print(cm)

    # Show top by P(BLOCK) *after mapping*
    if "block" in labels:
        p_block = proba[:, labels.index("block")]
    else:
        p_block = np.zeros(len(df), dtype=float)

    topn = min(args.show_top, len(df))
    order = np.argsort(-p_block)[:topn]

    print(f"\nTop {topn} by P(BLOCK):")
    for i in order:
        asset = df.iloc[i].get("asset_type")
        mkt = df.iloc[i].get("market")
        tic = clean_ticker(df.iloc[i].get("ticker"))
        print(f"- p_block={p_block[i]:.4f} pred={pred[i]} true={true[i]} asset={asset} mkt={mkt} ticker={tic}")


if __name__ == "__main__":
    main()






