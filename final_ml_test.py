# final_ml_test.py
import json
from pathlib import Path
import joblib
import numpy as np

from features import DEFAULT_CONFIG, features_to_row, vector_columns

BUNDLE = "models/unsup_bundle.joblib"
NORMAL = "lcc_runs.jsonl"
BROKEN = "lcc_broken_tests.jsonl"

def load_jsonl(path):
    out = []
    for l in Path(path).read_text().splitlines():
        if l.strip():
            obj = json.loads(l)
            out.append(obj.get("features", obj))
    return out

def anomaly_scores(bundle, X):
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)

    ifm = bundle["models"]["iforest"]
    lof = bundle["models"]["lof"]
    w_if = bundle["ensemble_weights"]["if"]
    w_lof = bundle["ensemble_weights"]["lof"]

    def score(pipe, kind):
        imp = pipe.named_steps["imputer"]
        sca = pipe.named_steps["scaler"]
        m = pipe.named_steps["iforest"] if kind=="if" else pipe.named_steps["lof"]
        return -m.score_samples(sca.transform(imp.transform(X)))

    s_if = score(ifm, "if")
    s_lof = score(lof, "lof")
    return w_if*z(s_if) + w_lof*z(s_lof)

def classify(scores, warn, block):
    return np.where(scores>=block,"BLOCK",np.where(scores>=warn,"WARN","OK"))

def run(name, feats, bundle, use_per_asset=True):
    cols = vector_columns(DEFAULT_CONFIG)
    rows = [features_to_row(f, DEFAULT_CONFIG) for f in feats]
    X = np.array([[r.get(c,np.nan) for c in cols] for r in rows], float)

    scores = anomaly_scores(bundle, X)
    g = bundle["thresholds_global"]
    pa = bundle.get("thresholds_per_asset_type", {})

    out = []
    for i,f in enumerate(feats):
        at = (f.get("asset_type") or "").lower()
        warn, block = g["warn"], g["block"]
        if use_per_asset and at in pa:
            warn, block = pa[at]["warn"], pa[at]["block"]
        out.append((scores[i], classify(np.array([scores[i]]), warn, block)[0]))
    print(f"\n{name}")
    print("-"*40)
    for i,(s,st) in enumerate(out,1):
        print(f"[{i:02d}] {st:5s} score={s:.4f}")
    print("Summary:", {k:[x[1] for x in out].count(k) for k in ["OK","WARN","BLOCK"]})

def main():
    bundle = joblib.load(BUNDLE)
    normal = load_jsonl(NORMAL)
    broken = load_jsonl(BROKEN)

    run("NORMAL CASES (expected mostly OK)", normal[:20], bundle)
    run("BROKEN CASES (expected OK here; blocked by deterministic)", broken, bundle)

if __name__ == "__main__":
    main()

