# generate_lcc_jsonl.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# -----------------------------
# CONFIG (change here)
# -----------------------------
OUT_DIR = Path(".")
N_CLEAN = 5000
N_BORDERLINE = 1000
N_SEMI_BROKEN = 700         # semi-broken goes INTO training
N_BROKEN = 500
HOLDOUT_CLEAN = 800         # clean holdout for false-positive rate

SEED = 42

# If you want more/less diversity:
ASSET_TYPES = ["equity", "etf", "fx", "rates", "crypto", "commodity"]
MARKETS = ["US", "EU", "UK", "JP", "EM"]

# Commodity subtypes (used only when asset_type == "commodity")
COMMODITY_SUBTYPES = ["energy_oil", "energy_gas", "metals_gold", "metals_copper", "agri_wheat", "agri_corn"]

REGIMES = ["low_vol", "normal", "stress", "crash", "rebound"]

# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rand_log_uniform(lo: float, hi: float) -> float:
    """Log-uniform random number between lo and hi (lo>0, hi>0)."""
    u = random.random()
    return lo * (hi / lo) ** u

def rand_trunc_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    """Quick trunc normal via rejection (fine for this use)."""
    for _ in range(50):
        x = random.gauss(mu, sigma)
        if lo <= x <= hi:
            return x
    return clamp(mu, lo, hi)

def iso_ts(i: int) -> str:
    # spread timestamps over the last ~90 days
    now = datetime.now(timezone.utc)
    dt = now - timedelta(days=random.randint(0, 90), minutes=random.randint(0, 24 * 60))
    return dt.isoformat().replace("+00:00", "Z")

@dataclass
class RegimeParams:
    vol_mult: float
    dd_mult: float
    corr_mult: float

REGIME_PARAMS: Dict[str, RegimeParams] = {
    "low_vol":  RegimeParams(vol_mult=0.6, dd_mult=0.7, corr_mult=0.8),
    "normal":   RegimeParams(vol_mult=1.0, dd_mult=1.0, corr_mult=1.0),
    "stress":   RegimeParams(vol_mult=1.6, dd_mult=1.8, corr_mult=1.2),
    "crash":    RegimeParams(vol_mult=2.2, dd_mult=2.6, corr_mult=1.3),
    "rebound":  RegimeParams(vol_mult=1.4, dd_mult=1.3, corr_mult=1.1),
}

# Typical annualized vol ranges by asset class (very rough but realistic enough)
VOL_RANGES = {
    "equity":    (0.12, 0.45),
    "etf":       (0.08, 0.35),
    "fx":        (0.05, 0.18),
    "rates":     (0.03, 0.15),
    "crypto":    (0.35, 1.20),
    "commodity": (0.15, 0.60),
}

# Typical drawdown ranges (negative numbers)
DD_RANGES = {
    "equity":    (-0.55, -0.05),
    "etf":       (-0.45, -0.04),
    "fx":        (-0.20, -0.01),
    "rates":     (-0.25, -0.01),
    "crypto":    (-0.85, -0.10),
    "commodity": (-0.60, -0.06),
}

# Correlation typical (absolute correlation with a broad market proxy)
CORR_RANGES = {
    "equity":    (0.15, 0.85),
    "etf":       (0.20, 0.90),
    "fx":        (0.00, 0.35),
    "rates":     (0.00, 0.45),
    "crypto":    (0.05, 0.60),
    "commodity": (0.00, 0.55),
}

def sample_asset() -> Dict[str, Any]:
    asset_type = random.choice(ASSET_TYPES)
    market = random.choice(MARKETS)
    regime = random.choice(REGIMES)
    subtype = None
    if asset_type == "commodity":
        subtype = random.choice(COMMODITY_SUBTYPES)
    return {"asset_type": asset_type, "market": market, "regime": regime, "commodity_subtype": subtype}

def build_base_features(meta: Dict[str, Any]) -> Dict[str, Any]:
    asset_type = meta["asset_type"]
    regime = meta["regime"]
    rp = REGIME_PARAMS[regime]

    vol_lo, vol_hi = VOL_RANGES[asset_type]
    dd_lo, dd_hi = DD_RANGES[asset_type]
    corr_lo, corr_hi = CORR_RANGES[asset_type]

    # Base annualized volatility (vol_ann)
    vol_ann = rand_trunc_normal(mu=(vol_lo + vol_hi) / 2, sigma=(vol_hi - vol_lo) / 6, lo=vol_lo, hi=vol_hi)
    vol_ann *= rp.vol_mult
    vol_ann = clamp(vol_ann, 0.01, 2.5)

    # 20d realized vol approximation from annual (reverse annualization)
    # NOTE: daily sigma ≈ vol_ann / sqrt(252)
    vol_20d = vol_ann / math.sqrt(252) * math.sqrt(20)
    # Some noise
    vol_20d *= rand_trunc_normal(1.0, 0.08, 0.85, 1.20)

    # Maximum drawdown (negative)
    dd = rand_trunc_normal(mu=(dd_lo + dd_hi) / 2, sigma=(abs(dd_lo - dd_hi)) / 6, lo=dd_lo, hi=dd_hi)
    dd *= rp.dd_mult
    dd = clamp(dd, -0.95, -0.001)

    # Correlation with market proxy (0..1)
    corr = rand_trunc_normal(mu=(corr_lo + corr_hi) / 2, sigma=(corr_hi - corr_lo) / 6, lo=corr_lo, hi=corr_hi)
    corr *= rp.corr_mult
    corr = clamp(corr, 0.0, 0.99)

    # Data coverage / quality
    n_used = random.randint(180, 900)  # number of daily points used
    missing_pct = rand_trunc_normal(0.003, 0.01, 0.0, 0.08)  # fraction missing
    tuw_pct = rand_trunc_normal(88.0, 8.0, 50.0, 100.0)  # some "time usable windows" %
    tail_obs_99 = max(8, int(n_used * 0.01 * rand_trunc_normal(1.0, 0.20, 0.6, 1.6)))

    # Risk (VaR/ES) magnitudes: negative returns at confidence
    # Rule of thumb: VaR95 around ~1.65*sigma_daily, VaR99 ~2.33*sigma_daily
    sigma_daily = vol_ann / math.sqrt(252)
    # Add fat-tail-ish multiplier
    fat = rand_trunc_normal(1.15, 0.10, 1.0, 1.5)

    var95 = -abs(1.65 * sigma_daily * fat) * rand_trunc_normal(1.0, 0.08, 0.85, 1.25)
    var99 = -abs(2.33 * sigma_daily * fat) * rand_trunc_normal(1.0, 0.08, 0.85, 1.25)

    # ES typically more extreme than VaR at same alpha
    es95 = var95 * rand_trunc_normal(1.20, 0.07, 1.05, 1.45)
    es99 = var99 * rand_trunc_normal(1.15, 0.07, 1.03, 1.35)

    # Ensure proper ordering (more negative at 99)
    var99 = min(var99, var95 - abs(var95) * 0.05)
    es99 = min(es99, es95 - abs(es95) * 0.03)

    # Simple RSI proxy (0..100) - not critical but helps realism
    rsi = rand_trunc_normal(52.0, 14.0, 5.0, 95.0)

    feat: Dict[str, Any] = {
        "timestamp": iso_ts(random.randint(0, 1000000)),
        "status_rules": "OK",
        "asset_type": asset_type,
        "market": meta["market"],
        "regime": meta["regime"],
        "commodity_subtype": meta["commodity_subtype"],

        # Core numbers (keep consistent units: decimals not percents)
        "vol_ann": float(vol_ann),     # 0.18 means 18% annualized
        "vol_20d": float(vol_20d),     # 20d vol in decimal terms (not annualized)
        "max_drawdown": float(dd),     # negative decimal
        "corr_mkt": float(corr),       # 0..1

        "var95": float(var95),
        "var99": float(var99),
        "es95": float(es95),
        "es99": float(es99),

        # Data quality
        "n_used": int(n_used),
        "missing_pct": float(missing_pct),
        "tuw_pct": float(tuw_pct),
        "tail_obs_99": int(tail_obs_99),

        "rsi": float(rsi),
    }
    return feat

# -----------------------------
# Borderline variants (still "plausible", near thresholds)
# -----------------------------
def make_borderline(f: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(f)

    # Push into suspicious but not obviously wrong territory
    # low tail observations, higher missing, borderline ordering margins
    g["missing_pct"] = float(clamp(g["missing_pct"] * rand_trunc_normal(3.0, 0.6, 1.5, 6.0), 0.0, 0.25))
    g["tuw_pct"] = float(clamp(g["tuw_pct"] * rand_trunc_normal(0.85, 0.08, 0.55, 0.98), 20.0, 100.0))

    # Reduce n_used sometimes
    if random.random() < 0.35:
        g["n_used"] = int(clamp(int(g["n_used"] * rand_trunc_normal(0.45, 0.15, 0.18, 0.75)), 40, 400))

    # tail obs too small relative to sample
    g["tail_obs_99"] = int(clamp(int(g["tail_obs_99"] * rand_trunc_normal(0.55, 0.15, 0.15, 0.85)), 3, 30))

    # Make VaR99 ~ VaR95 (too close) -> suspicious
    if random.random() < 0.60:
        v95 = g["var95"]
        g["var99"] = float(v95 - abs(v95) * rand_trunc_normal(0.01, 0.01, 0.0, 0.04))

    # Make ES not much larger than VaR -> suspicious
    if random.random() < 0.60:
        g["es95"] = float(g["var95"] * rand_trunc_normal(1.03, 0.02, 1.00, 1.08))
    if random.random() < 0.60:
        g["es99"] = float(g["var99"] * rand_trunc_normal(1.02, 0.02, 1.00, 1.07))

    # Vol borderline for asset
    at = g["asset_type"]
    if at in ("equity", "etf") and random.random() < 0.45:
        g["vol_ann"] = float(clamp(g["vol_ann"] * rand_trunc_normal(0.55, 0.08, 0.35, 0.75), 0.01, 2.5))
    if at == "crypto" and random.random() < 0.35:
        g["vol_ann"] = float(clamp(g["vol_ann"] * rand_trunc_normal(0.55, 0.10, 0.30, 0.75), 0.01, 2.5))

    # Re-derive vol_20d from vol_ann to keep consistent
    g["vol_20d"] = float(g["vol_ann"] / math.sqrt(252) * math.sqrt(20) * rand_trunc_normal(1.0, 0.10, 0.8, 1.25))

    return g

# -----------------------------
# Broken realistic bugs
# -----------------------------
BROKEN_MODES = [
    "units_percent_instead_decimal",
    "annualization_missing",
    "annualization_double",
    "var_es_order_inversion",
    "sign_errors",
    "n_used_too_low",
    "tuw_out_of_range",
    "missing_pct_out_of_range",
    "drawdown_positive",
]

def make_broken(f: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(f)
    mode = random.choice(BROKEN_MODES)
    g["broken_mode"] = mode
    g["status_rules"] = "BROKEN_SIM"  # just a marker

    if mode == "units_percent_instead_decimal":
        # convert decimals to percents (0.18 -> 18)
        for k in ["vol_ann", "vol_20d", "var95", "var99", "es95", "es99", "max_drawdown"]:
            if k in g and isinstance(g[k], (int, float)):
                g[k] = float(g[k] * 100.0)

    elif mode == "annualization_missing":
        # annual vol mistakenly stored as daily
        # 0.18 becomes ~0.011
        g["vol_ann"] = float(g["vol_ann"] / math.sqrt(252))
        g["vol_20d"] = float(g["vol_ann"] * math.sqrt(20))  # compounding wrong chain too

    elif mode == "annualization_double":
        # annual vol mistakenly annualized again
        g["vol_ann"] = float(g["vol_ann"] * math.sqrt(252))
        g["vol_20d"] = float(g["vol_20d"] * math.sqrt(252))

    elif mode == "var_es_order_inversion":
        # VaR99 less extreme than VaR95 (violates ordering)
        g["var99"] = float(g["var95"] * rand_trunc_normal(0.85, 0.05, 0.70, 0.95))
        # ES99 less extreme too
        g["es99"] = float(g["es95"] * rand_trunc_normal(0.90, 0.05, 0.75, 0.98))

    elif mode == "sign_errors":
        # risk metrics positive (wrong sign)
        g["var95"] = abs(g["var95"])
        g["var99"] = abs(g["var99"])
        g["es95"] = abs(g["es95"])
        g["es99"] = abs(g["es99"])

    elif mode == "n_used_too_low":
        g["n_used"] = random.randint(10, 45)
        g["tail_obs_99"] = random.randint(0, 5)

    elif mode == "tuw_out_of_range":
        g["tuw_pct"] = random.choice([-15.0, 135.0, 250.0])

    elif mode == "missing_pct_out_of_range":
        g["missing_pct"] = random.choice([-0.2, 1.2, 3.5])

    elif mode == "drawdown_positive":
        g["max_drawdown"] = abs(g["max_drawdown"])  # should be negative

    return g

# -----------------------------
# Semi-broken realistic modes (soft anomalies)
# -----------------------------
SEMI_BROKEN_MODES = [
    "partial_units_percent_mix",
    "vol_window_inconsistent",
    "var_es_too_close",
    "tail_obs_inconsistent",
    "coverage_contradiction",
    "es_not_extreme_enough",
]

def make_semi_broken(f: Dict[str, Any]) -> Dict[str, Any]:
    """
    Semi-broken cases: realistic bugs that are not always caught by hard rules,
    and should be learnable by ML (soft detector).
    """
    g = dict(f)
    mode = random.choice(SEMI_BROKEN_MODES)
    g["semi_broken_mode"] = mode
    g["status_rules"] = "SEMI_BROKEN_SIM"

    if mode == "partial_units_percent_mix":
        # Only some metrics mistakenly in percent (not all)
        # e.g., VaR/ES in percent but vols remain decimal, or vice-versa.
        if random.random() < 0.5:
            for k in ["var95", "var99", "es95", "es99"]:
                if k in g and isinstance(g[k], (int, float)):
                    g[k] = float(g[k] * 100.0)
        else:
            for k in ["vol_ann", "vol_20d"]:
                if k in g and isinstance(g[k], (int, float)):
                    g[k] = float(g[k] * 100.0)

    elif mode == "vol_window_inconsistent":
        # vol_ann is plausible, but vol_20d is wrong due to bad scaling/window confusion
        expected = float(g["vol_ann"] / math.sqrt(252) * math.sqrt(20))
        mult = rand_trunc_normal(1.0, 0.35, 0.45, 1.85)
        g["vol_20d"] = float(expected * mult)

    elif mode == "var_es_too_close":
        v95 = float(g["var95"])
        g["var99"] = float(v95 - abs(v95) * rand_trunc_normal(0.005, 0.006, 0.0005, 0.02))
        g["es95"] = float(g["var95"] * rand_trunc_normal(1.01, 0.01, 1.00, 1.05))
        g["es99"] = float(g["var99"] * rand_trunc_normal(1.01, 0.01, 1.00, 1.05))

    elif mode == "tail_obs_inconsistent":
        n = int(g.get("n_used", 300))
        g["tail_obs_99"] = max(1, int(rand_trunc_normal(mu=3.0, sigma=2.0, lo=1.0, hi=10.0)))
        if random.random() < 0.35:
            g["n_used"] = int(clamp(int(n * rand_trunc_normal(0.75, 0.10, 0.55, 0.92)), 80, 900))

    elif mode == "coverage_contradiction":
        if random.random() < 0.5:
            g["missing_pct"] = float(clamp(g["missing_pct"] * rand_trunc_normal(0.35, 0.15, 0.0, 0.6), 0.0, 0.06))
            g["tuw_pct"] = float(clamp(g["tuw_pct"] * rand_trunc_normal(0.45, 0.10, 0.20, 0.65), 10.0, 100.0))
        else:
            g["missing_pct"] = float(clamp(g["missing_pct"] * rand_trunc_normal(4.0, 0.8, 1.8, 7.0), 0.02, 0.35))
            g["tuw_pct"] = float(clamp(g["tuw_pct"] * rand_trunc_normal(1.08, 0.08, 0.95, 1.20), 60.0, 100.0))

    elif mode == "es_not_extreme_enough":
        g["es95"] = float(g["var95"] * rand_trunc_normal(1.02, 0.015, 1.00, 1.06))
        g["es99"] = float(g["var99"] * rand_trunc_normal(1.02, 0.015, 1.00, 1.06))
        g["es99"] = min(float(g["es99"]), float(g["es95"]) - abs(float(g["es95"])) * 0.001)

    return g

# -----------------------------
# Writer
# -----------------------------
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    random.seed(SEED)

    clean: List[Dict[str, Any]] = []
    borderline: List[Dict[str, Any]] = []
    semi_broken: List[Dict[str, Any]] = []
    broken: List[Dict[str, Any]] = []

    # CLEAN
    for i in range(N_CLEAN):
        meta = sample_asset()
        f = build_base_features(meta)
        clean.append(f)

    # BORDERLINE: start from clean then perturb
    for i in range(N_BORDERLINE):
        base = random.choice(clean)
        borderline.append(make_borderline(base))

    # SEMI-BROKEN: realistic near-bugs to INCLUDE in training
    for i in range(N_SEMI_BROKEN):
        base = random.choice(clean)
        semi_broken.append(make_semi_broken(base))

    # BROKEN: start from clean then inject realistic bugs
    for i in range(N_BROKEN):
        base = random.choice(clean)
        broken.append(make_broken(base))

    # Shuffle a bit
    random.shuffle(clean)
    random.shuffle(borderline)
    random.shuffle(semi_broken)
    random.shuffle(broken)

    # CLEAN HOLDOUT: measure false positives (should stay low)
    clean_holdout = clean[:HOLDOUT_CLEAN]

    # Raw datasets
    write_jsonl(OUT_DIR / "lcc_runs.jsonl", clean)
    write_jsonl(OUT_DIR / "lcc_borderline.jsonl", borderline)
    write_jsonl(OUT_DIR / "lcc_semi_broken.jsonl", semi_broken)
    write_jsonl(OUT_DIR / "lcc_broken_tests.jsonl", broken)

    # Ready-to-use splits
    # Train: clean + borderline + semi-broken (soft anomalies)
    train_rows = clean + borderline + semi_broken
    random.shuffle(train_rows)
    write_jsonl(OUT_DIR / "lcc_train.jsonl", train_rows)

    # Eval: true broken realistic bugs (should be mostly WARN/BLOCK)
    write_jsonl(OUT_DIR / "lcc_eval.jsonl", broken)

    # Holdout: clean-only subset (track false positives)
    write_jsonl(OUT_DIR / "lcc_clean_holdout.jsonl", clean_holdout)

    print("✅ Generated:")
    print(f" - {OUT_DIR/'lcc_runs.jsonl'}              ({len(clean)} rows)  [clean base]")
    print(f" - {OUT_DIR/'lcc_borderline.jsonl'}        ({len(borderline)} rows)  [borderline]")
    print(f" - {OUT_DIR/'lcc_semi_broken.jsonl'}       ({len(semi_broken)} rows)  [semi-broken]")
    print(f" - {OUT_DIR/'lcc_broken_tests.jsonl'}      ({len(broken)} rows)  [broken]")
    print("")
    print("✅ Splits ready:")
    print(f" - {OUT_DIR/'lcc_train.jsonl'}             ({len(train_rows)} rows)  [TRAIN]")
    print(f" - {OUT_DIR/'lcc_eval.jsonl'}              ({len(broken)} rows)  [EVAL]")
    print(f" - {OUT_DIR/'lcc_clean_holdout.jsonl'}     ({len(clean_holdout)} rows)  [CLEAN HOLDOUT]")
    print("")
    print("Tip: train on lcc_train.jsonl, evaluate on lcc_eval.jsonl, and track false positives on lcc_clean_holdout.jsonl.")

if __name__ == "__main__":
    main()



