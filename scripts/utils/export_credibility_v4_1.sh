#!/usr/bin/env bash
# =============================================================================
# scripts/utils/export_credibility_v4_1.sh
# =============================================================================
# Export v4.1 credibility assets (consistency fixes + validated metrics) to
# ~/Asymetra_Exports/credibility_v4_1/ with a self-contained static viewer.
#
# v4.1 fixes vs v4:
#   - Finance drawdown: equity-curve method, % units (was log-return units)
#   - Finance metrics card: MDD/Sharpe from fold-5 time-series (equity curve)
#   - Most-recent fold: fold 5 (2023-2025, n=12,291) instead of fold 2 (n=50)
#   - ECE: weighted computation (was 0.0000 artefact)
#   - sanity_report.json: machine-readable consistency checks
#
# Usage:
#   bash scripts/utils/export_credibility_v4_1.sh [--rebuild]
#
# Options:
#   --rebuild   Re-run the full pipeline before copying
#
# Output:
#   ~/Asymetra_Exports/credibility_v4_1/
#       stat/             12 ML/statistical PNG figures
#       finance/           7 financial PNG figures
#       report.pdf         Scientific validation report
#       manifest.json      Asset registry (version: "v4.1")
#       sanity_report.json Finance + fold + calibration consistency checks
#       index.html         Static viewer (no server required)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_DIR="${REPO_ROOT}/build/credibility/v4.1"
DEST_DIR="${HOME}/Asymetra_Exports/credibility_v4_1"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    *) echo "[WARN] Unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 1: Optionally rebuild v4.1 assets
# ---------------------------------------------------------------------------
if [[ "$REBUILD" -eq 1 ]]; then
  echo "[1/4] Rebuilding v4.1 credibility assets..."

  echo "  [1a] Finance plots (equity-curve drawdown)..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/plot_financial_v3.py" \
    --backtest data/metrics/backtest_v3.json \
    --manifest data/training/v3/splits_manifest.json \
    --models   models/v3 \
    --out      data/metrics/v3/financial_plots

  echo "  [1b] Robustness plots (fold-5 table, weighted ECE)..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/plot_robustness_v3.py"

  echo "  [1c] PDF report..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/generate_v3_report.py" --no_plots

  echo "  [1d] Sanity report..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/generate_sanity_report.py" \
    --out data/metrics/v3/sanity_report.json

  echo "  [1e] Package as v4.1..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/export_v3_credibility_assets.py" --version v4.1

else
  echo "[1/4] Using existing build/credibility/v4.1/ (pass --rebuild to regenerate)"
fi

# ---------------------------------------------------------------------------
# Step 2: Verify source directory
# ---------------------------------------------------------------------------
if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Source directory not found: $SRC_DIR"
  echo "        Run: bash $0 --rebuild"
  exit 1
fi

if [[ ! -f "${SRC_DIR}/manifest.json" ]]; then
  echo "[ERROR] manifest.json missing from ${SRC_DIR}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Copy to export directory (clean copy)
# ---------------------------------------------------------------------------
echo "[2/4] Copying assets to ${DEST_DIR}..."
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

cp -r "${SRC_DIR}/stat"    "${DEST_DIR}/"
cp -r "${SRC_DIR}/finance" "${DEST_DIR}/"
cp    "${SRC_DIR}/report.pdf"    "${DEST_DIR}/"
cp    "${SRC_DIR}/manifest.json" "${DEST_DIR}/"

# Copy sanity report if present
SANITY_SRC="${REPO_ROOT}/data/metrics/v3/sanity_report.json"
if [[ -f "$SANITY_SRC" ]]; then
  cp "$SANITY_SRC" "${DEST_DIR}/sanity_report.json"
  echo "        sanity_report.json included"
fi

# ---------------------------------------------------------------------------
# Step 4: Generate static index.html
# ---------------------------------------------------------------------------
echo "[3/4] Generating static index.html..."

cat > "${DEST_DIR}/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Asymetra — Fiabilité modèle v4.1 (Consistance)</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #fff;
      color: #111;
      padding: 2rem 1.5rem 4rem;
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      border-bottom: 2px solid #111;
      padding-bottom: 1.5rem;
      margin-bottom: 2.5rem;
    }

    h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
    .subtitle { font-size: 0.95rem; color: #555; }
    .version-badge {
      display: inline-block;
      background: #1e2d46;
      color: #fff;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.2rem 0.6rem;
      letter-spacing: 0.06em;
      margin-left: 0.5rem;
      vertical-align: middle;
    }

    .changelog {
      background: #f0f7ff;
      border-left: 3px solid #3498db;
      padding: 1rem 1.5rem;
      margin-bottom: 2rem;
      font-size: 0.88rem;
    }
    .changelog h2 { font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.5rem; color: #3498db; }
    .changelog ul { padding-left: 1.25rem; }
    .changelog li { margin-bottom: 0.3rem; color: #333; line-height: 1.5; }

    .sanity-box {
      background: #f0fff4;
      border-left: 3px solid #27ae60;
      padding: 1rem 1.5rem;
      margin-bottom: 2rem;
      font-size: 0.88rem;
    }
    .sanity-box h2 { font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
                      letter-spacing: 0.08em; margin-bottom: 0.5rem; color: #27ae60; }
    .sanity-box p { color: #333; line-height: 1.6; }
    .sanity-link { color: #27ae60; font-weight: 600; text-decoration: none; }
    .sanity-link:hover { text-decoration: underline; }

    .methodology {
      background: #f7f7f7;
      border-left: 3px solid #111;
      padding: 1.25rem 1.5rem;
      margin-bottom: 3rem;
    }
    .methodology h2 {
      font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 0.75rem;
    }
    .methodology ul { padding-left: 1.25rem; }
    .methodology li { font-size: 0.88rem; line-height: 1.6; color: #333; margin-bottom: 0.45rem; }

    section.content-section { margin-bottom: 3.5rem; }
    h2.section-title {
      font-size: 1.2rem; font-weight: 700; border-bottom: 1px solid #ddd;
      padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    h2.section-title span { font-weight: 400; color: #666; font-size: 0.95rem; }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1.25rem;
    }

    figure.card {
      cursor: pointer; border: 1px solid #e0e0e0;
      background: #fafafa; margin: 0;
      transition: box-shadow .15s ease, border-color .15s ease;
    }
    figure.card:hover { border-color: #111; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
    figure.card.new-in-v4 { border-color: #3498db; }
    figure.card.new-in-v4::before {
      content: "NEW v4"; display: block; background: #3498db; color: #fff;
      font-size: 0.65rem; font-weight: 700; text-align: center;
      padding: 2px 0; letter-spacing: 0.06em;
    }
    figure.card.fixed-v41 { border-color: #e67e22; }
    figure.card.fixed-v41::before {
      content: "FIXED v4.1"; display: block; background: #e67e22; color: #fff;
      font-size: 0.65rem; font-weight: 700; text-align: center;
      padding: 2px 0; letter-spacing: 0.06em;
    }

    figure.card img {
      width: 100%; height: 180px; object-fit: cover;
      display: block; border-bottom: 1px solid #e0e0e0; background: #f0f0f0;
    }
    figcaption { padding: 0.6rem 0.75rem; font-size: 0.78rem; font-weight: 600; color: #222; line-height: 1.3; }

    .report-section { margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #ddd; }
    .report-link {
      display: inline-block; background: #111; color: #fff;
      padding: 0.65rem 1.25rem; font-size: 0.9rem; font-weight: 600;
      text-decoration: none; margin-bottom: 0.75rem; margin-right: 0.5rem;
    }
    .report-link.secondary { background: #27ae60; }
    .report-link:hover { background: #333; }
    .report-link.secondary:hover { background: #1e8449; }
    .report-desc { font-size: 0.85rem; color: #555; max-width: 700px; line-height: 1.6; margin-top: 0.5rem; }

    #lightbox {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,.85); align-items: center;
      justify-content: center; z-index: 1000; padding: 1.5rem;
    }
    #lightbox.open { display: flex; }
    #lb-inner {
      position: relative; background: #fff;
      max-width: 90vw; max-height: 90vh;
      display: flex; flex-direction: column; overflow: hidden;
    }
    #lb-close {
      position: absolute; top: 0.5rem; right: 0.75rem;
      background: none; border: none; font-size: 1.75rem;
      cursor: pointer; color: #111; z-index: 10; line-height: 1;
    }
    #lb-img { max-width: 100%; max-height: 65vh; object-fit: contain; display: block; }
    #lb-caption {
      padding: 1rem 1.25rem; border-top: 1px solid #e0e0e0;
      max-height: 200px; overflow-y: auto;
    }
    #lb-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.4rem; }
    #lb-text  { font-size: 0.85rem; color: #444; line-height: 1.6; }

    @media (max-width: 640px) {
      h1 { font-size: 1.5rem; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<header>
  <h1>Fiabilité du modèle <span class="version-badge">v4.1</span></h1>
  <p class="subtitle">Validation scientifique — correctifs de cohérence (finance equity-curve, ECE, fold 5)</p>
</header>

<div class="changelog">
  <h2>Correctifs v4.1 vs v4</h2>
  <ul>
    <li><strong>Finance — Drawdown</strong> — calculé depuis la courbe d'équité (1+r produit cumulé), unités en % [−1, 0]. Élimine le biais log-return.</li>
    <li><strong>Finance — Metrics Card</strong> — MDD et Sharpe désormais issus de la série temporelle fold 5, non du JSON cross-sectionnel (MDD JSON était −99.98%).</li>
    <li><strong>Fold le plus récent</strong> — fold 5 (2023-06-30→2025-12-31, n=12 291). Avant : fold 2 (n=50).</li>
    <li><strong>ECE</strong> — calcul pondéré (10 bins, pondération par taille de bin). Élimine l'artefact ECE=0.0000.</li>
    <li><strong>sanity_report.json</strong> — rapport de cohérence machine-readable (finance + fold + calibration).</li>
  </ul>
</div>

<div class="sanity-box">
  <h2>Rapport de cohérence interne</h2>
  <p>
    Toutes les vérifications de cohérence sont documentées dans
    <a class="sanity-link" href="./sanity_report.json" target="_blank">sanity_report.json</a>
    — finance (MDD valide, CAGR signe, Sharpe série vs JSON), sélection du fold
    et calibration ECE par fold.
  </p>
</div>

<div class="methodology">
  <h2>Méthodologie</h2>
  <ul>
    <li>Validation temporelle par expanding-window (5 folds, 2010–2025) avec purge de 20 jours et embargo de 5 jours.</li>
    <li>Calibration isotonique par régression — ECE pondéré &lt; 0.05 (cohérent sur les 5 folds).</li>
    <li>Backtest hors-échantillon strict — drawdown calculé depuis la courbe d'équité, non depuis des log-returns cumulés.</li>
  </ul>
</div>

<div id="app">Chargement...</div>

<div id="lightbox" role="dialog" aria-modal="true">
  <div id="lb-inner">
    <button id="lb-close" aria-label="Fermer">&times;</button>
    <img id="lb-img" src="" alt="" />
    <div id="lb-caption">
      <div id="lb-title"></div>
      <div id="lb-text"></div>
    </div>
  </div>
</div>

<script>
HTMLEOF

echo "const MANIFEST = $(cat "${SRC_DIR}/manifest.json");" >> "${DEST_DIR}/index.html"

cat >> "${DEST_DIR}/index.html" << 'JSEOF'
const BASE = "./";
const NEW_V4_KEYS = new Set([
  "recent_fold_table",
  "auc_bootstrap_hist",
  "sharpe_bootstrap_hist",
  "confusion_metrics_per_fold",
]);
const FIXED_V41_KEYS = new Set([
  "drawdown",
  "backtest_metrics_card",
  "recent_fold_table",
]);

function renderSection(sectionKey, sectionData) {
  const assets = sectionData.assets;
  const cards = assets.map(a => {
    let cssClass = "";
    if (FIXED_V41_KEYS.has(a.key)) cssClass = " fixed-v41";
    else if (NEW_V4_KEYS.has(a.key)) cssClass = " new-in-v4";
    return `
    <figure class="card${cssClass}" data-key="${a.key}" data-file="${a.file}"
            data-title="${a.title.replace(/"/g,'&quot;')}"
            data-caption="${a.caption.replace(/"/g,'&quot;')}"
            tabindex="0" role="button" aria-label="Agrandir: ${a.title.replace(/"/g,'&quot;')}">
      <img src="${BASE}${a.file}" alt="${a.title.replace(/"/g,'&quot;')}" loading="lazy" />
      <figcaption>${a.title}</figcaption>
    </figure>`;
  }).join("");

  return `
    <section class="content-section">
      <h2 class="section-title">${sectionData.label_en} <span>— ${sectionData.label}</span></h2>
      <div class="grid">${cards}</div>
    </section>`;
}

function renderReport(report) {
  if (!report) return "";
  const mb = (report.size_kb / 1024).toFixed(1);
  return `
    <section class="report-section">
      <h2 class="section-title">Rapport complet</h2>
      <a class="report-link" href="${BASE}${report.file}" target="_blank" download>
        Télécharger le rapport scientifique v4.1 (PDF — ${mb} MB)
      </a>
      <a class="report-link secondary" href="${BASE}sanity_report.json" target="_blank">
        Voir sanity_report.json
      </a>
      <p class="report-desc">${report.description}</p>
    </section>`;
}

function init() {
  const app = document.getElementById("app");
  app.innerHTML =
    renderSection("stat",    MANIFEST.sections.stat) +
    renderSection("finance", MANIFEST.sections.finance) +
    renderReport(MANIFEST.report);

  const lb      = document.getElementById("lightbox");
  const lbImg   = document.getElementById("lb-img");
  const lbTitle = document.getElementById("lb-title");
  const lbText  = document.getElementById("lb-text");
  const lbClose = document.getElementById("lb-close");

  function openLb(file, title, caption) {
    lbImg.src = BASE + file;
    lbImg.alt = title;
    lbTitle.textContent = title;
    lbText.textContent  = caption;
    lb.classList.add("open");
  }
  function closeLb() {
    lb.classList.remove("open");
    lbImg.src = "";
  }

  lbClose.addEventListener("click", closeLb);
  lb.addEventListener("click", (e) => { if (e.target === lb) closeLb(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLb(); });

  app.addEventListener("click", (e) => {
    const card = e.target.closest(".card");
    if (!card) return;
    openLb(card.dataset.file, card.dataset.title, card.dataset.caption);
  });
  app.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const card = e.target.closest(".card");
    if (!card) return;
    openLb(card.dataset.file, card.dataset.title, card.dataset.caption);
  });
}

init();
</script>
</body>
</html>
JSEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[4/4] Finalizing..."
echo ""
echo "Export complete → ${DEST_DIR}"
echo ""
ls -lh "${DEST_DIR}"
echo ""
echo "Finder: open '${DEST_DIR}'"
echo "Browser: open '${DEST_DIR}/index.html'"
