"""
scripts/ml/reporting/export_v3_credibility_assets.py
======================================================
Collect, verify and package v3 credibility assets into build/credibility/v3/.

Usage:
    python scripts/ml/reporting/export_v3_credibility_assets.py [--dry-run]

Outputs:
    build/credibility/v3/
        stat/          # ML/statistical plots (8 PNGs)
        finance/       # Financial plots (7 PNGs)
        report.pdf     # Scientific report
        manifest.json  # Asset registry for frontend consumption

Exit codes:
    0  all assets present and copied
    1  one or more source files missing (non-dry-run aborts)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Repo root (4 levels up from this file)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Asset registry — explicit captions specified by the mission
# ---------------------------------------------------------------------------

class Asset(NamedTuple):
    key: str
    src_rel: str          # relative to _REPO
    dst_sub: str          # subfolder under build/credibility/v3/
    section: str          # "stat" | "finance"
    title: str
    caption: str


ASSETS: list[Asset] = [
    # ---- Statistical Validation (ML metrics) ----
    Asset(
        key="roc_curves",
        src_rel="data/metrics/v3/plots/roc_curves.png",
        dst_sub="stat",
        section="stat",
        title="ROC Curves (per fold)",
        caption=(
            "Receiver Operating Characteristic curves for each cross-validation fold. "
            "Mean AUC >= 0.70 indicates the model discriminates warn/block events "
            "well above chance across all time periods tested."
        ),
    ),
    Asset(
        key="pr_curves",
        src_rel="data/metrics/v3/plots/pr_curves.png",
        dst_sub="stat",
        section="stat",
        title="Precision-Recall Curves (per fold)",
        caption=(
            "Precision-Recall tradeoff for the positive class (non-ok events). "
            "High average precision at the thresholds used in production confirms "
            "the model flags real risk without excessive false alarms."
        ),
    ),
    Asset(
        key="calibration",
        src_rel="data/metrics/v3/plots/calibration.png",
        dst_sub="stat",
        section="stat",
        title="Probability Calibration",
        caption=(
            "Reliability diagram comparing predicted probabilities to observed event rates. "
            "The isotonic-regression calibration layer keeps the calibration curve close "
            "to the diagonal, meaning a score of 0.7 corresponds to ~70% actual risk."
        ),
    ),
    Asset(
        key="prob_distributions",
        src_rel="data/metrics/v3/plots/prob_distributions.png",
        dst_sub="stat",
        section="stat",
        title="Score Distributions (ok vs non-ok)",
        caption=(
            "Histogram of model scores split by true label. Clear separation between "
            "ok (low score) and non-ok (high score) distributions validates the "
            "discriminative power of the pipeline."
        ),
    ),
    Asset(
        key="lift_curve",
        src_rel="data/metrics/v3/plots/lift_curve.png",
        dst_sub="stat",
        section="stat",
        title="Lift Curve",
        caption=(
            "Cumulative lift over random baseline. Targeting the top-scored decile "
            "captures non-ok events at 3-4x the rate of random selection, "
            "justifying the decision-gate architecture."
        ),
    ),
    Asset(
        key="confusion_matrices",
        src_rel="data/metrics/v3/plots/confusion_matrices.png",
        dst_sub="stat",
        section="stat",
        title="Confusion Matrices (per fold)",
        caption=(
            "Counts of true/false positives and negatives at the production threshold. "
            "Low false-negative rate is the primary optimization target: missing a "
            "real risk event is more costly than issuing an excess warning."
        ),
    ),
    Asset(
        key="feature_importance",
        src_rel="data/metrics/v3/plots/feature_importance.png",
        dst_sub="stat",
        section="stat",
        title="Feature Importance (XGBoost gain)",
        caption=(
            "Top features ranked by mean gain across all trees. Volatility regime, "
            "drawdown duration and tail-risk metrics dominate, confirming the model "
            "learns from economically meaningful signals."
        ),
    ),
    Asset(
        key="metrics_per_fold",
        src_rel="data/metrics/v3/plots/metrics_per_fold.png",
        dst_sub="stat",
        section="stat",
        title="Metrics Stability Across Folds",
        caption=(
            "AUC-ROC, Average Precision and F1 shown fold-by-fold. "
            "Low variance across the 5 expanding windows confirms the model "
            "generalises to unseen time periods without significant degradation."
        ),
    ),
    # ---- Practical Value (financial / backtest) ----
    Asset(
        key="cumulative_returns",
        src_rel="data/metrics/v3/financial_plots/cumulative_returns.png",
        dst_sub="finance",
        section="finance",
        title="Cumulative Returns — Signal vs Always-OK",
        caption=(
            "Net cumulative return of the signal-filtered portfolio (skip warn/block) "
            "versus the always-invested baseline. The signal strategy avoids the "
            "largest drawdown episodes while preserving upside participation."
        ),
    ),
    Asset(
        key="drawdown",
        src_rel="data/metrics/v3/financial_plots/drawdown.png",
        dst_sub="finance",
        section="finance",
        title="Drawdown Profile",
        caption=(
            "Maximum drawdown over time for both strategies. The filtered portfolio "
            "consistently experiences smaller and shorter drawdown episodes, "
            "translating directly into lower client stress and smoother equity curves."
        ),
    ),
    Asset(
        key="return_distributions",
        src_rel="data/metrics/v3/financial_plots/return_distributions.png",
        dst_sub="finance",
        section="finance",
        title="Return Distributions",
        caption=(
            "Distribution of 20-day forward returns conditioned on model signal. "
            "Skipped periods (warn/block) show systematically more negative left-tail "
            "outcomes, validating that the model identifies genuinely risky windows."
        ),
    ),
    Asset(
        key="skip_rate_rolling",
        src_rel="data/metrics/v3/financial_plots/skip_rate_rolling.png",
        dst_sub="finance",
        section="finance",
        title="Rolling Skip Rate",
        caption=(
            "Fraction of tickers flagged warn or block on a 60-day rolling basis. "
            "Skip-rate spikes align with known market stress periods (2020 COVID, "
            "2022 rate shock), confirming macro-regime sensitivity."
        ),
    ),
    Asset(
        key="rolling_sharpe",
        src_rel="data/metrics/v3/financial_plots/rolling_sharpe.png",
        dst_sub="finance",
        section="finance",
        title="Rolling Sharpe Ratio (90-day)",
        caption=(
            "90-day rolling Sharpe ratio of the signal portfolio versus the baseline. "
            "The signal strategy maintains a higher and more stable Sharpe ratio "
            "across market regimes, including periods of elevated volatility."
        ),
    ),
    Asset(
        key="performance_by_asset_type",
        src_rel="data/metrics/v3/financial_plots/performance_by_asset_type.png",
        dst_sub="finance",
        section="finance",
        title="Performance by Asset Type",
        caption=(
            "Sharpe ratio and return improvement broken down by asset category "
            "(equity, ETF, crypto, etc.). Consistent positive lift across all classes "
            "confirms the model is not overfitting to a single asset universe."
        ),
    ),
    Asset(
        key="backtest_metrics_card",
        src_rel="data/metrics/v3/financial_plots/backtest_metrics_card.png",
        dst_sub="finance",
        section="finance",
        title="Backtest Summary Card",
        caption=(
            "Aggregated backtest statistics: Sharpe ratio, Calmar ratio, max drawdown, "
            "skip rate and annualised excess return versus always-invested baseline. "
            "All metrics computed on out-of-sample fold data only (no look-ahead bias)."
        ),
    ),
]

PDF_ASSET = {
    "key": "scientific_report",
    "src_rel": "data/metrics/v3/V3_Scientific_Report.pdf",
    "dst_name": "report.pdf",
    "title": "V3 Scientific Validation Report",
    "description": (
        "Complete scientific validation document including dataset statistics, "
        "cross-validation methodology, calibration analysis, backtest results, "
        "feature drift analysis and production-readiness verdict."
    ),
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _check_sources(dry_run: bool) -> list[str]:
    """Return list of missing source files (relative paths)."""
    missing = []
    for a in ASSETS:
        src = _REPO / a.src_rel
        if not src.exists():
            missing.append(a.src_rel)
    pdf_src = _REPO / PDF_ASSET["src_rel"]
    if not pdf_src.exists():
        missing.append(PDF_ASSET["src_rel"])
    return missing


def _copy_assets(out_dir: Path, dry_run: bool) -> dict:
    """
    Copy all assets to out_dir and return the manifest dict.
    out_dir will be created if needed.
    """
    manifest: dict = {
        "version": "v3",
        "generated_by": "export_v3_credibility_assets.py",
        "sections": {
            "stat": {
                "label": "Validation statistique",
                "label_en": "Statistical Validation",
                "assets": [],
            },
            "finance": {
                "label": "Valeur pratique",
                "label_en": "Practical Value",
                "assets": [],
            },
        },
        "report": None,
    }

    for a in ASSETS:
        src = _REPO / a.src_rel
        dst_dir = out_dir / a.dst_sub
        dst = dst_dir / src.name

        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size // 1024
        else:
            size_kb = src.stat().st_size // 1024 if src.exists() else -1

        entry = {
            "key": a.key,
            "file": f"{a.dst_sub}/{src.name}",
            "title": a.title,
            "caption": a.caption,
            "size_kb": size_kb,
        }
        manifest["sections"][a.section]["assets"].append(entry)

    # PDF
    pdf_src = _REPO / PDF_ASSET["src_rel"]
    pdf_dst = out_dir / PDF_ASSET["dst_name"]
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_src, pdf_dst)
        pdf_size_kb = pdf_dst.stat().st_size // 1024
    else:
        pdf_size_kb = pdf_src.stat().st_size // 1024 if pdf_src.exists() else -1

    manifest["report"] = {
        "key": PDF_ASSET["key"],
        "file": PDF_ASSET["dst_name"],
        "title": PDF_ASSET["title"],
        "description": PDF_ASSET["description"],
        "size_kb": pdf_size_kb,
    }

    return manifest


def _write_manifest(manifest: dict, out_dir: Path, dry_run: bool) -> None:
    manifest_path = out_dir / "manifest.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK]  manifest written → {manifest_path.relative_to(_REPO)}")
    else:
        print("[DRY] Would write manifest.json:")
        print(json.dumps(manifest, indent=2, ensure_ascii=False)[:800] + " ...")


def export(dry_run: bool = False) -> int:
    """
    Main entry point. Returns 0 on success, 1 on missing files.
    """
    out_dir = _REPO / "build" / "credibility" / "v3"
    print(f"{'[DRY RUN] ' if dry_run else ''}Exporting credibility assets → {out_dir}")

    # 1. Check sources
    missing = _check_sources(dry_run)
    if missing:
        print(f"\n[ERROR] {len(missing)} source file(s) missing:")
        for m in missing:
            print(f"  - {m}")
        if not dry_run:
            return 1
        print("  (continuing in dry-run mode despite missing files)")

    # 2. Copy assets
    manifest = _copy_assets(out_dir, dry_run)

    # 3. Write manifest
    _write_manifest(manifest, out_dir, dry_run)

    # 4. Summary
    stat_count = len(manifest["sections"]["stat"]["assets"])
    fin_count  = len(manifest["sections"]["finance"]["assets"])
    print(f"\n[OK]  {stat_count} statistical plots")
    print(f"[OK]  {fin_count} financial plots")
    print(f"[OK]  PDF report ({manifest['report']['size_kb']} KB)")
    print(f"\nTotal assets: {stat_count + fin_count + 1} files → build/credibility/v3/")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export v3 credibility assets to build/credibility/v3/"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without copying files"
    )
    args = parser.parse_args()
    sys.exit(export(dry_run=args.dry_run))
