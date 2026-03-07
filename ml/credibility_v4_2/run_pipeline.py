"""
ml/credibility_v4_2/run_pipeline.py
─────────────────────────────────────
Orchestrateur du pipeline Credibility v4.2.

1. Génère un run_id unique (date + commit_short + uuid)
2. Crée artifacts/credibility_v4_2/<run_id>/
3. Lance build_dataset.py  → dataset_raw.jsonl
4. Lance build_folds.py    → splits.json, fold_boundaries.csv
5. Lance build_artifacts.py → dataset_profile.json, dataset_hash.txt, run_provenance.json
6. Lance verify_artifacts.py → FAIL si incohérence

Usage:
    python ml/credibility_v4_2/run_pipeline.py [OPTIONS]

Options importantes:
    --start         2010-01-01
    --end           2025-12-31
    --step_days     20           (fenêtres glissantes, pas en jours de bourse)
    --max_per_ticker 200
    --dry_run       Si set, affiche le plan sans télécharger de données.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ── Repo root ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_ARTIFACTS_BASE = _REPO_ROOT / "artifacts" / "credibility_v4_2"


def _git_short(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _make_run_id(repo_root: Path) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    commit = _git_short(repo_root)
    uid = uuid.uuid4().hex[:8]
    return f"v42_{today}_{commit}_{uid}"


def _run(cmd: list[str], cwd: Path) -> int:
    """Run a subprocess, streaming stdout/stderr. Returns exit code."""
    print(f"\n{'─'*70}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full Credibility v4.2 pipeline")
    ap.add_argument("--run_id", default=None,
                    help="Override run_id (default: auto-generated)")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--lookback_days", type=int, default=252)
    ap.add_argument("--horizon_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=20,
                    help="Step between rolling windows in trading days.")
    ap.add_argument("--max_per_ticker", type=int, default=200)
    ap.add_argument("--purge_days", type=int, default=20)
    ap.add_argument("--embargo_days", type=int, default=5)
    ap.add_argument("--sleep_ticker", type=float, default=0.3)
    ap.add_argument("--max_tries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print plan without running anything.")
    args = ap.parse_args()

    # ── run_id ────────────────────────────────────────────────────────────────
    run_id = args.run_id or _make_run_id(_REPO_ROOT)
    out_dir = _ARTIFACTS_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        CREDIBILITY v4.2 — PIPELINE ORCHESTRATOR             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  run_id  : {run_id}")
    print(f"  out_dir : {out_dir}")
    print(f"  period  : {args.start} → {args.end}")
    print(f"  params  : lookback={args.lookback_days}d  horizon={args.horizon_days}d  step={args.step_days}d")
    print(f"  folds   : 5 expanding-window  purge={args.purge_days}bd  embargo={args.embargo_days}bd")

    if args.dry_run:
        print("\n  [DRY RUN] No scripts launched.")
        print(f"  Would write to: {out_dir}/")
        print("  Steps:")
        print("    1. build_dataset.py")
        print("    2. build_folds.py")
        print("    3. build_artifacts.py")
        print("    4. verify_artifacts.py")
        return

    python = sys.executable

    # ── Step 1: Build dataset ─────────────────────────────────────────────────
    rc = _run([
        python, str(_SCRIPT_DIR / "build_dataset.py"),
        "--run_id", run_id,
        "--out_dir", str(out_dir),
        "--start", args.start,
        "--end", args.end,
        "--universe", args.universe,
        "--lookback_days", str(args.lookback_days),
        "--horizon_days", str(args.horizon_days),
        "--step_days", str(args.step_days),
        "--max_per_ticker", str(args.max_per_ticker),
        "--sleep_ticker", str(args.sleep_ticker),
        "--max_tries", str(args.max_tries),
    ], cwd=_REPO_ROOT)

    if rc != 0:
        sys.exit(f"\n✗ build_dataset.py FAILED (exit={rc}). Pipeline aborted.")

    # Check output exists and is non-empty
    dataset_path = out_dir / "dataset_raw.jsonl"
    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        sys.exit("\n✗ dataset_raw.jsonl missing or empty after build_dataset.py. Aborting.")

    # ── Step 2: Build folds ───────────────────────────────────────────────────
    rc = _run([
        python, str(_SCRIPT_DIR / "build_folds.py"),
        "--run_id", run_id,
        "--out_dir", str(out_dir),
        "--purge_days", str(args.purge_days),
        "--embargo_days", str(args.embargo_days),
    ], cwd=_REPO_ROOT)

    if rc != 0:
        sys.exit(f"\n✗ build_folds.py FAILED (exit={rc}). Pipeline aborted.")

    # ── Step 3: Build artifacts ───────────────────────────────────────────────
    rc = _run([
        python, str(_SCRIPT_DIR / "build_artifacts.py"),
        "--run_id", run_id,
        "--out_dir", str(out_dir),
        "--start", args.start,
        "--end", args.end,
        "--lookback_days", str(args.lookback_days),
        "--horizon_days", str(args.horizon_days),
        "--step_days", str(args.step_days),
        "--purge_days", str(args.purge_days),
        "--embargo_days", str(args.embargo_days),
    ], cwd=_REPO_ROOT)

    if rc != 0:
        sys.exit(f"\n✗ build_artifacts.py FAILED (exit={rc}). Pipeline aborted.")

    # ── Step 4: Verify artifacts ──────────────────────────────────────────────
    rc = _run([
        python, str(_REPO_ROOT / "verify_artifacts.py"),
        "--out_dir", str(out_dir),
        "--run_id", run_id,
    ], cwd=_REPO_ROOT)

    if rc != 0:
        sys.exit(f"\n✗ verify_artifacts.py FAILED (exit={rc}). Artifacts are INVALID.")

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       ✓ CREDIBILITY v4.2 PIPELINE COMPLETE                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  run_id  : {run_id}")
    print(f"  out_dir : {out_dir}/")
    print()
    print("  Artifacts produced:")
    for fname in [
        "dataset_raw.jsonl",
        "splits.json",
        "fold_boundaries.csv",
        "dataset_profile.json",
        "dataset_hash.txt",
        "run_provenance.json",
    ]:
        p = out_dir / fname
        size = f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING"
        print(f"    {fname:<30} {size}")

    print()
    print("  → Hand off to Agent 2 (model training) and Agent 3 (report generation)")
    print(f"  → Pass run_id={run_id} and out_dir={out_dir}")


if __name__ == "__main__":
    main()
