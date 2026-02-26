# calibrate_thresholds.py
import json
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict
from features import features_to_row, row_to_vector  # module local

def predict_probas(bundle_path: str, jsonl_path: str):
    b = joblib.load(bundle_path)
    model = b["model"]
    cols = b["cols"]
    p_list = []
    y_list = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            feats = r.get("features", {})
            row = features_to_row(feats, cfg=b.get("cfg"))
            vec = row_to_vector(row, cfg=b.get("cfg"))
            p = model.predict_proba(vec.reshape(1, -1))[:,1][0]
            p_list.append(p)
            y_list.append(0 if r.get("label") == "ok" else 1)
    return np.array(p_list), np.array(y_list)

def find_t_lo_under_fp_constraint(p, y, alpha=0.25):
    # on veut FP/OK <= alpha et maxi recall possible
    ok_mask = (y == 0)
    p_ok = p[ok_mask]
    unique_thresholds = np.unique(np.clip(p, 0.0, 1.0))
    unique_thresholds = np.sort(unique_thresholds)
    best = None
    for t in unique_thresholds:
        pred_non_ok = (p >= t).astype(int)
        cm = confusion_matrix(y, pred_non_ok)
        tn, fp, fn, tp = cm.ravel()
        fp_rate_ok = fp / (y==0).sum()
        if fp_rate_ok <= alpha:
            recall_non_ok = tp / (tp + fn + 1e-12)
            if best is None or recall_non_ok > best["recall"]:
                best = {"t": float(t), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                        "tn": int(tn), "recall": float(recall_non_ok), "fp_rate_ok": float(fp_rate_ok)}
    return best

def compute_t_hi_by_quantile(p, y, q=0.995):
    # t_hi = quantile des probabilités observées parmi les OK (pour limiter BLOCK sur OK)
    p_ok = p[y==0]
    return float(np.quantile(p_ok, q))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--q_hi", type=float, default=0.995)
    parser.add_argument("--out_json", default="models/threshold_auto.json")
    args = parser.parse_args()

    p, y = predict_probas(args.bundle, args.holdout)
    best = find_t_lo_under_fp_constraint(p, y, alpha=args.alpha)
    t_hi = compute_t_hi_by_quantile(p, y, q=args.q_hi)

    out = {
        "t_lo": best["t"] if best else None,
        "t_hi": t_hi,
        "alpha": args.alpha,
        "q_hi": args.q_hi,
        "metrics_at_t_lo": best,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("written:", args.out_json)
    print(json.dumps(out, indent=2))


