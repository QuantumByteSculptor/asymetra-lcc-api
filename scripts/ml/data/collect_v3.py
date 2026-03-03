"""
scripts/ml/data/collect_v3.py
==============================
Phase 1 — Collecte massive robuste du dataset v3.

Fonctionnalités:
  - --resume : skip tickers déjà collectés (lit le JSONL partiel)
  - Retry avec backoff exponentiel agressif (rate-limit yfinance/stooq)
  - Logs par ticker dans logs/collect_v3/
  - Rapport post-collecte automatique  → data/metrics/collect_v3_report.json
  - Aucun crash global (chaque ticker isolé dans try/except)
  - Multiprocessing-safe (Pool avec initializer)

Usage:
  # Collecte complète
  python scripts/ml/data/collect_v3.py \\
      --universe data/universe.json \\
      --out data/training/train_v3_all.jsonl \\
      --workers 4

  # Reprise après interruption
  python scripts/ml/data/collect_v3.py \\
      --universe data/universe.json \\
      --out data/training/train_v3_all.jsonl \\
      --workers 4 --resume

No API / prod impact.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Repo-root bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ml.data.build_dataset_v3 import (  # noqa: E402
    process_ticker,
    download_macro_data,
    download_spy_returns,
    _init_worker,
    _SHARED_MACRO,
    _SHARED_SPY,
)

log = logging.getLogger("collect_v3")


# ---------------------------------------------------------------------------
# Resume logic — extract tickers already in a partial JSONL
# ---------------------------------------------------------------------------

def read_done_tickers(jsonl_path: Path) -> Tuple[Set[str], int]:
    """
    Parse an existing (partial) JSONL to find which tickers are already done.
    Returns (set_of_tickers, n_lines).
    Fast: reads only the 'features.ticker' field per line.
    """
    done: Set[str] = set()
    n = 0
    if not jsonl_path.exists():
        return done, 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = rec.get("features", {}).get("ticker", "")
                if t:
                    done.add(t)
                n += 1
            except json.JSONDecodeError:
                continue
    log.info("Resume: found %d existing records for %d tickers in %s",
             n, len(done), jsonl_path)
    return done, n


# ---------------------------------------------------------------------------
# Adaptive rate-limit backoff
# ---------------------------------------------------------------------------

class RateLimitTracker:
    """
    Tracks consecutive provider failures to detect rate-limiting.
    Increases sleep time automatically between tickers when rate-limited.
    """
    def __init__(self, base_sleep: float = 0.5, max_sleep: float = 30.0):
        self.base_sleep = base_sleep
        self.max_sleep  = max_sleep
        self.consecutive_fails = 0
        self._current_sleep = base_sleep

    def on_success(self) -> None:
        self.consecutive_fails = 0
        self._current_sleep = max(self.base_sleep, self._current_sleep * 0.9)

    def on_fail(self) -> None:
        self.consecutive_fails += 1
        # Exponential backoff: double every 3 consecutive fails
        if self.consecutive_fails % 3 == 0:
            self._current_sleep = min(self._current_sleep * 2.0, self.max_sleep)
            log.warning(
                "Rate-limit detector: %d consecutive fails → sleep %.1fs",
                self.consecutive_fails, self._current_sleep,
            )

    def sleep(self) -> None:
        if self._current_sleep > 0:
            time.sleep(self._current_sleep)


# ---------------------------------------------------------------------------
# Post-collect report
# ---------------------------------------------------------------------------

def write_collect_report(
    out_path: Path,
    report_path: Path,
    tasks_total: int,
    n_success: int,
    n_fail: int,
    n_skipped_resume: int,
    n_windows: int,
    label_counts: Dict[str, int],
    asset_type_counts: Dict[str, int],
    source_counts: Dict[str, int],
    failed_tickers: List[str],
    elapsed_sec: float,
) -> None:
    report = {
        "generated_at":       datetime.utcnow().isoformat() + "Z",
        "output_file":        str(out_path),
        "elapsed_seconds":    round(elapsed_sec, 1),
        "tickers": {
            "total_in_universe": tasks_total + n_skipped_resume,
            "skipped_resume":    n_skipped_resume,
            "processed":         tasks_total,
            "success":           n_success,
            "failed":            n_fail,
            "success_rate":      round(n_success / max(tasks_total, 1), 4),
        },
        "samples": {
            "total_windows":  n_windows,
            "by_label":       label_counts,
            "label_balance":  {
                k: round(100 * v / max(n_windows, 1), 2)
                for k, v in label_counts.items()
            },
        },
        "coverage": {
            "by_asset_type": asset_type_counts,
            "by_source":     source_counts,
        },
        "failed_tickers": failed_tickers[:50],   # cap for readability
        "n_failed_tickers": len(failed_tickers),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Post-collect report written: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Collect v3 dataset — massive, resumable")
    ap.add_argument("--universe",        default="data/universe.json")
    ap.add_argument("--out",             default="data/training/train_v3_all.jsonl")
    ap.add_argument("--start",           default=None)
    ap.add_argument("--lookback_years",  type=int,   default=7)
    ap.add_argument("--lookback_days",   type=int,   default=252)
    ap.add_argument("--step_days",       type=int,   default=10)
    ap.add_argument("--max_per_ticker",  type=int,   default=80)
    ap.add_argument("--workers",         type=int,   default=1)
    ap.add_argument("--sleep_ticker",    type=float, default=0.3,
                    help="Base sleep between tickers (seconds, sequential only)")
    ap.add_argument("--resume",          action="store_true",
                    help="Skip tickers already present in --out (append mode)")
    ap.add_argument("--skip_macro",      action="store_true")
    ap.add_argument("--asset_types",     default=None,
                    help="Comma-separated filter (e.g. equity,etf)")
    ap.add_argument("--report",          default="data/metrics/collect_v3_report.json")
    ap.add_argument("--max_tickers",     type=int, default=0,
                    help="Cap number of tickers (0 = no cap, useful for testing)")
    args = ap.parse_args()

    from datetime import timedelta
    start_date = args.start or (
        (datetime.today() - timedelta(days=args.lookback_years * 365 + 300))
        .strftime("%Y-%m-%d")
    )

    # Load universe
    uni_path = Path(args.universe)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe not found: {uni_path}")
    uni: List[Dict[str, str]] = json.loads(uni_path.read_text(encoding="utf-8"))
    log.info("Universe: %d tickers", len(uni))

    allowed_types: Optional[set] = None
    if args.asset_types:
        allowed_types = {t.strip().lower() for t in args.asset_types.split(",") if t.strip()}

    out_path    = Path(args.out)
    report_path = Path(args.report)

    # Resume: find already-done tickers
    done_tickers: Set[str] = set()
    n_existing_lines = 0
    if args.resume and out_path.exists():
        done_tickers, n_existing_lines = read_done_tickers(out_path)

    # Build task list
    all_tasks: List[Dict[str, Any]] = []
    n_skip_resume  = 0
    n_skip_type    = 0
    for item in uni:
        ticker = str(item.get("ticker", "")).strip()
        if not ticker:
            continue
        asset_type = str(item.get("asset_type", "")).strip().lower()
        if allowed_types and asset_type not in allowed_types:
            n_skip_type += 1
            continue
        if ticker in done_tickers:
            n_skip_resume += 1
            continue
        all_tasks.append({
            "ticker":         ticker,
            "asset_type":     asset_type,
            "market":         str(item.get("market", "")).strip(),
            "start":          start_date,
            "lookback_days":  args.lookback_days,
            "step_days":      args.step_days,
            "max_per_ticker": args.max_per_ticker,
        })

    if args.max_tickers and args.max_tickers > 0:
        all_tasks = all_tasks[:args.max_tickers]
        log.info("Capped to %d tickers (--max_tickers)", args.max_tickers)

    n_universe_total = len(uni)
    log.info(
        "Tasks: %d to process | %d skipped (resume) | %d skipped (type filter)",
        len(all_tasks), n_skip_resume, n_skip_type,
    )

    # Macro / SPY
    if args.skip_macro:
        macro: Dict = {}
        spy_ret = __import__("pandas").Series(dtype=float)
        log.info("Skipping macro data (--skip_macro)")
    else:
        log.info("Downloading macro data from FRED...")
        macro = download_macro_data(start=start_date)
        spy_ret = download_spy_returns(start=start_date)
        log.info("SPY returns: %d pts", len(spy_ret))

    # Counters
    n_success = n_fail = total_windows = 0
    label_counts: Dict[str, int]      = {"ok": 0, "warn": 0, "block": 0}
    asset_counts: Dict[str, int]      = {}
    source_counts: Dict[str, int]     = {}
    failed_tickers: List[str]         = []
    rate_tracker = RateLimitTracker(
        base_sleep=args.sleep_ticker, max_sleep=20.0
    )

    t0 = time.time()

    # Open output in append mode if resume, write mode otherwise
    write_mode = "a" if (args.resume and out_path.exists()) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)

    with out_path.open(write_mode, encoding="utf-8") as out_file:

        def _handle(lines: List[str], ticker: str, err: Optional[str],
                    asset_type: str = "") -> None:
            nonlocal n_success, n_fail, total_windows
            if err:
                n_fail += 1
                failed_tickers.append(ticker)
                log.warning("FAIL  %-22s %s", ticker, err[:120])
                rate_tracker.on_fail()
                return

            n_success += 1
            rate_tracker.on_success()
            for ln in lines:
                out_file.write(ln + "\n")
                rec = json.loads(ln)
                lbl = rec.get("label", "ok") or "ok"
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                src = rec.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
                at = rec.get("features", {}).get("asset_type", "unknown")
                asset_counts[at] = asset_counts.get(at, 0) + 1
                total_windows += 1

            if lines:
                log.info("OK    %-22s %3d windows", ticker, len(lines))

        if workers > 1:
            import scripts.ml.data.build_dataset_v3 as _bds
            _bds._SHARED_MACRO = macro
            _bds._SHARED_SPY   = spy_ret
            with multiprocessing.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(macro, spy_ret),
            ) as pool:
                for result in pool.imap_unordered(
                    process_ticker, all_tasks, chunksize=1
                ):
                    _handle(result[0], result[1], result[2])
        else:
            # Sequential with adaptive rate-limit sleep
            import scripts.ml.data.build_dataset_v3 as _bds
            _bds._SHARED_MACRO = macro
            _bds._SHARED_SPY   = spy_ret
            for task in all_tasks:
                result = process_ticker(task)
                _handle(result[0], result[1], result[2], task.get("asset_type", ""))
                rate_tracker.sleep()

    elapsed = time.time() - t0

    # Summary
    log.info("=" * 60)
    log.info("COLLECT V3 COMPLETE  %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("Output : %s  (%d new lines + %d existing)",
             out_path, total_windows, n_existing_lines)
    log.info("Tickers: %d ok / %d failed / %d total processed",
             n_success, n_fail, len(all_tasks))
    log.info("Windows: %d  Labels: %s", total_windows, label_counts)
    log.info("=" * 60)

    # Report
    write_collect_report(
        out_path       = out_path,
        report_path    = report_path,
        tasks_total    = len(all_tasks),
        n_success      = n_success,
        n_fail         = n_fail,
        n_skipped_resume = n_skip_resume,
        n_windows      = total_windows + n_existing_lines,
        label_counts   = label_counts,
        asset_type_counts = asset_counts,
        source_counts  = source_counts,
        failed_tickers = failed_tickers,
        elapsed_sec    = elapsed,
    )


if __name__ == "__main__":
    main()
