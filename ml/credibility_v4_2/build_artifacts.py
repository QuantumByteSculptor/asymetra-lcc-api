"""
ml/credibility_v4_2/build_artifacts.py
─────────────────────────────────────────
Génère dataset_profile.json, dataset_hash.txt, run_provenance.json.

Usage:
    python ml/credibility_v4_2/build_artifacts.py \
        --run_id v42_... \
        --out_dir artifacts/credibility_v4_2/v42_...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Git helpers
# ═══════════════════════════════════════════════════════════════════════════════

def git_commit_hash(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_commit_hash_short(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_branch(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(repo_root: Optional[Path] = None) -> bool:
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Hash
# ═══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    """Stable SHA-256 of a file, reading in binary."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset profiling
# ═══════════════════════════════════════════════════════════════════════════════

def profile_dataset(records: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    """
    Compute dataset_profile.json content.
    """
    n_total = len(records)
    label_counter: Counter = Counter()
    asset_counter: Counter = Counter()
    market_counter: Counter = Counter()
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    n_missing_date = 0
    n_missing_corr_spy = 0
    n_missing_beta = 0
    n_missing_vix = 0

    # Per asset-type breakdown
    asset_label: Dict[str, Counter] = defaultdict(Counter)
    per_period: Counter = Counter()  # year bucket

    for rec in records:
        feats = rec.get("features", rec)
        label = rec.get("label", feats.get("label_v2", "unknown"))
        asset_type = str(feats.get("asset_type", "unknown"))
        market = str(feats.get("market", "unknown"))
        wed = feats.get("window_end_date")

        label_counter[label] += 1
        asset_counter[asset_type] += 1
        market_counter[market] += 1
        asset_label[asset_type][label] += 1

        if wed:
            if date_min is None or wed < date_min:
                date_min = wed
            if date_max is None or wed > date_max:
                date_max = wed
            year = wed[:4]
            per_period[year] += 1
        else:
            n_missing_date += 1

        if feats.get("corr_spy") is None:
            n_missing_corr_spy += 1
        if feats.get("beta_market") is None:
            n_missing_beta += 1
        if feats.get("vix_level") is None:
            n_missing_vix += 1

    # Pos/neg (binary: block+warn = "positive", ok = "negative")
    n_pos = label_counter.get("block", 0) + label_counter.get("warn", 0)
    n_neg = label_counter.get("ok", 0)

    return {
        "run_id": run_id,
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_rate": round(n_pos / n_total, 4) if n_total else None,
        "date_min": date_min,
        "date_max": date_max,
        "n_missing_window_end_date": n_missing_date,
        "n_missing_corr_spy": n_missing_corr_spy,
        "n_missing_beta_market": n_missing_beta,
        "n_missing_vix_level": n_missing_vix,
        "label_distribution": dict(label_counter),
        "asset_type_distribution": dict(asset_counter),
        "market_distribution": dict(market_counter.most_common(20)),
        "asset_type_label_breakdown": {
            at: dict(cnt) for at, cnt in sorted(asset_label.items())
        },
        "records_per_year": dict(sorted(per_period.items())),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Build Credibility v4.2 artifacts (profile, hash, provenance)")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=20)
    ap.add_argument("--purge_days", type=int, default=20)
    ap.add_argument("--embargo_days", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dataset_path = out_dir / "dataset_raw.jsonl"
    profile_path = out_dir / "dataset_profile.json"
    hash_path = out_dir / "dataset_hash.txt"
    prov_path = out_dir / "run_provenance.json"

    if not dataset_path.exists():
        sys.exit(f"dataset_raw.jsonl not found: {dataset_path}")

    print(f"=== build_artifacts.py — run_id={args.run_id} ===")

    # ── Repo root = 2 levels up from this script ────────────────────────────
    repo_root = Path(__file__).resolve().parents[2]

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("  Loading dataset_raw.jsonl …")
    records: List[Dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records")

    # ── run_provenance.json ──────────────────────────────────────────────────
    commit_full = git_commit_hash(repo_root)
    commit_short = git_commit_hash_short(repo_root)
    branch = git_branch(repo_root)
    dirty = git_dirty(repo_root)

    provenance = {
        "run_id": args.run_id,
        "schema_version": "v4.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit_full,
            "commit_short": commit_short,
            "branch": branch,
            "dirty": dirty,
        },
        "parameters": {
            "start": args.start,
            "end": args.end,
            "lookback_days": args.lookback_days,
            "horizon_days": args.horizon_days,
            "step_days": args.step_days,
            "purge_days": args.purge_days,
            "embargo_days": args.embargo_days,
        },
        "pipeline_scripts": [
            "ml/credibility_v4_2/build_dataset.py",
            "ml/credibility_v4_2/build_folds.py",
            "ml/credibility_v4_2/build_artifacts.py",
        ],
        "note": (
            "Credibility v4.2 — single run_id, end-to-end reproducible. "
            "corr_spy + beta_market + vix_level computed from real price data. "
            "Expanding-window 5-fold CV with purge+embargo anti-leakage."
        ),
    }
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ {prov_path}")

    # ── dataset_hash.txt ─────────────────────────────────────────────────────
    digest = sha256_file(dataset_path)
    hash_content = (
        f"run_id: {args.run_id}\n"
        f"file: dataset_raw.jsonl\n"
        f"sha256: {digest}\n"
        f"n_records: {len(records)}\n"
        f"generated_at: {datetime.now(timezone.utc).isoformat()}\n"
    )
    hash_path.write_text(hash_content, encoding="utf-8")
    print(f"  ✓ {hash_path}  (sha256={digest[:16]}…)")

    # ── dataset_profile.json ─────────────────────────────────────────────────
    print("  Computing dataset profile …")
    profile = profile_dataset(records, args.run_id)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ {profile_path}")

    # Quick summary
    print(f"\n  n_total       : {profile['n_total']:,}")
    print(f"  n_positive    : {profile['n_positive']:,}  ({100*profile['pos_rate']:.1f}%)")
    print(f"  date_min/max  : {profile['date_min']} → {profile['date_max']}")
    print(f"  label dist    : {profile['label_distribution']}")
    print(f"  asset_types   : {profile['asset_type_distribution']}")

    if profile["n_missing_corr_spy"] > 0:
        pct = 100 * profile["n_missing_corr_spy"] / max(profile["n_total"], 1)
        print(f"  ⚠  corr_spy missing in {profile['n_missing_corr_spy']} records ({pct:.1f}%)")
    if profile["n_missing_beta_market"] > 0:
        pct = 100 * profile["n_missing_beta_market"] / max(profile["n_total"], 1)
        print(f"  ⚠  beta_market missing in {profile['n_missing_beta_market']} records ({pct:.1f}%)")
    if profile["n_missing_vix_level"] > 0:
        pct = 100 * profile["n_missing_vix_level"] / max(profile["n_total"], 1)
        print(f"  ⚠  vix_level missing in {profile['n_missing_vix_level']} records ({pct:.1f}%)")

    print("\n=== build_artifacts DONE ===")


if __name__ == "__main__":
    main()
