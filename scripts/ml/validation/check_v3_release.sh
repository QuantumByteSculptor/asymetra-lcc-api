#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  check_v3_release.sh — V3 Release Gate
#
#  Single entry point: exits 0 (GO) or 1 (NO-GO).
#  Run before any API integration or version tagging.
#
#  Usage:
#    bash scripts/ml/validation/check_v3_release.sh
# ============================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODELS_DIR="$REPO_ROOT/models/v3"

# ── Locate Python ─────────────────────────────────────────────────────────────
# Resolve the actual repo root — handles git worktrees where REPO_ROOT is the
# worktree directory but the .venv lives in the main checkout.
_GIT_COMMON="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || echo "")"
if [[ -n "$_GIT_COMMON" && "$_GIT_COMMON" != ".git" ]]; then
    # Linked worktree: git-common-dir is an absolute path to main-repo/.git
    MAIN_REPO="$(cd "$_GIT_COMMON/.." && pwd)"
else
    MAIN_REPO="$REPO_ROOT"
fi
VENV_PYTHON="$MAIN_REPO/.venv/bin/python3"

if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: python3 not found. Activate venv or install python3." >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  V3 Release Gate"
echo "  repo   : $REPO_ROOT"
echo "  models : $MODELS_DIR"
echo "  python : $PYTHON"
echo "════════════════════════════════════════════════════"
echo ""

FAILED=0

# ── Step 1 — Artifact inventory (7 files) ─────────────────────────────────────
echo "Step 1/4 — Artifact inventory"

ARTIFACTS=(
    "v3_xgb_model.joblib"
    "v3_lr_model.joblib"
    "v3_calibrator.joblib"
    "v3_feature_names.joblib"
    "v3_meta.json"
    "v3_thresholds.json"
    "v3_metrics.json"
)

FOUND=0
for f in "${ARTIFACTS[@]}"; do
    if [[ -f "$MODELS_DIR/$f" ]]; then
        echo "  ✅ $f"
        FOUND=$((FOUND + 1))
    else
        echo "  ❌ MISSING: $f"
        FAILED=1
    fi
done
echo "  artifacts: $FOUND/${#ARTIFACTS[@]}"
echo ""

# ── Step 2 — Smoke load (Python inline) ───────────────────────────────────────
echo "Step 2/4 — Smoke load (Python inline)"

SMOKE_RESULT=0
MODELS_DIR="$MODELS_DIR" "$PYTHON" - <<'PYEOF' || SMOKE_RESULT=$?
import joblib, json, os, pathlib, sys

models_dir = pathlib.Path(os.environ["MODELS_DIR"])

# Load joblib artefacts
print("  Loading v3_xgb_model.joblib ...")
xgb_pipe = joblib.load(models_dir / "v3_xgb_model.joblib")

print("  Loading v3_lr_model.joblib ...")
lr_pipe = joblib.load(models_dir / "v3_lr_model.joblib")

print("  Loading v3_calibrator.joblib ...")
calibrator = joblib.load(models_dir / "v3_calibrator.joblib")

print("  Loading v3_feature_names.joblib ...")
feat_names = joblib.load(models_dir / "v3_feature_names.joblib")
print(f"  feature_names  : {len(feat_names)} features")

# Validate v3_meta.json
print("  Loading v3_meta.json ...")
meta = json.loads((models_dir / "v3_meta.json").read_text())
sv = meta.get("schema_version")
assert sv == "3.1", f"schema_version mismatch: expected '3.1', got {sv!r}"
fc = meta.get("feature_cols", [])
assert len(fc) > 0, "feature_cols is empty in v3_meta.json"
print(f"  schema_version : {sv}")
print(f"  feature_cols   : {len(fc)} features")
print(f"  dropped        : {len(meta.get('dropped_features', []))} features")

# Validate v3_thresholds.json
print("  Loading v3_thresholds.json ...")
thr = json.loads((models_dir / "v3_thresholds.json").read_text())
t_lo, t_hi = thr["t_lo"], thr["t_hi"]
assert 0.0 < t_lo < t_hi < 1.0, \
    f"Threshold violation: expected 0 < t_lo={t_lo} < t_hi={t_hi} < 1"
print(f"  t_lo (warn)    : {t_lo}")
print(f"  t_hi (block)   : {t_hi}")

print("  ✅ 3 models loaded — meta + thresholds valid")
PYEOF

if [[ $SMOKE_RESULT -ne 0 ]]; then
    echo "  ❌ Smoke load FAILED (exit $SMOKE_RESULT)"
    FAILED=1
fi
echo ""

# ── Step 3 — pytest v3 suite (fast, <10s) ─────────────────────────────────────
echo "Step 3/4 — pytest v3 suite"

cd "$REPO_ROOT"
PYTEST_RESULT=0
"$PYTHON" -m pytest \
    tests/unit/test_train_v3_smoke.py \
    tests/unit/test_qa_v3.py \
    tests/unit/test_split_v3.py \
    tests/unit/test_nan_preprocessing.py \
    -q --tb=short || PYTEST_RESULT=$?

if [[ $PYTEST_RESULT -ne 0 ]]; then
    echo "  ❌ pytest FAILED (exit $PYTEST_RESULT)"
    FAILED=1
fi
echo ""

# ── Step 4 — GO / NO-GO banner ────────────────────────────────────────────────
if [[ $FAILED -eq 0 ]]; then
    echo "════════════════════════════════════"
    echo "  ✅  V3 RELEASE — GO"
    echo "  artifacts : $FOUND/${#ARTIFACTS[@]}"
    echo "  models    : 3 loaded"
    echo "  tests     : passed"
    echo "════════════════════════════════════"
    exit 0
else
    echo "════════════════════════════════════"
    echo "  ❌  V3 RELEASE — NO-GO"
    echo "  Investigate failures above."
    echo "════════════════════════════════════"
    exit 1
fi
