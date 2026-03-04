#!/usr/bin/env bash
# =============================================================================
# scripts/utils/export_credibility_assets.sh
# =============================================================================
# Export v3 credibility assets to ~/Asymetra_Exports/credibility_v3/
# for offline sharing and archiving.
#
# Usage:
#   bash scripts/utils/export_credibility_assets.sh [--rebuild]
#
# Options:
#   --rebuild   Re-run export_v3_credibility_assets.py before copying
#
# Output:
#   ~/Asymetra_Exports/credibility_v3/
#       stat/             8 ML/statistical PNG figures
#       finance/          7 financial PNG figures
#       report.pdf        Scientific validation report
#       manifest.json     Asset registry
#       index.html        Static viewer (no server required)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_DIR="${REPO_ROOT}/build/credibility/v3"
DEST_DIR="${HOME}/Asymetra_Exports/credibility_v3"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

# Fall back to system python3 if venv not present
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
# Step 1: Optionally rebuild assets
# ---------------------------------------------------------------------------
if [[ "$REBUILD" -eq 1 ]]; then
  echo "[1/3] Rebuilding credibility assets..."
  "$PYTHON" "${REPO_ROOT}/scripts/ml/reporting/export_v3_credibility_assets.py"
else
  echo "[1/3] Using existing build (pass --rebuild to regenerate)"
fi

# ---------------------------------------------------------------------------
# Step 2: Verify source directory
# ---------------------------------------------------------------------------
if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Source directory not found: $SRC_DIR"
  echo "        Run: python scripts/ml/reporting/export_v3_credibility_assets.py"
  exit 1
fi

# Check manifest
if [[ ! -f "${SRC_DIR}/manifest.json" ]]; then
  echo "[ERROR] manifest.json missing from ${SRC_DIR}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Copy to export directory
# ---------------------------------------------------------------------------
echo "[2/3] Copying assets to ${DEST_DIR}..."
mkdir -p "$DEST_DIR"

cp -r "${SRC_DIR}/stat"    "${DEST_DIR}/"
cp -r "${SRC_DIR}/finance" "${DEST_DIR}/"
cp    "${SRC_DIR}/report.pdf"    "${DEST_DIR}/"
cp    "${SRC_DIR}/manifest.json" "${DEST_DIR}/"

# ---------------------------------------------------------------------------
# Step 4: Generate static index.html
# ---------------------------------------------------------------------------
echo "[3/3] Generating static index.html..."

MANIFEST_JSON=$(cat "${SRC_DIR}/manifest.json")

cat > "${DEST_DIR}/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Asymetra — Fiabilité modèle v3</title>
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

    figure.card img {
      width: 100%; height: 180px; object-fit: cover;
      display: block; border-bottom: 1px solid #e0e0e0; background: #f0f0f0;
    }
    figcaption { padding: 0.6rem 0.75rem; font-size: 0.78rem; font-weight: 600; color: #222; line-height: 1.3; }

    .report-section { margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #ddd; }
    .report-link {
      display: inline-block; background: #111; color: #fff;
      padding: 0.65rem 1.25rem; font-size: 0.9rem; font-weight: 600;
      text-decoration: none; margin-bottom: 0.75rem;
    }
    .report-link:hover { background: #333; }
    .report-desc { font-size: 0.85rem; color: #555; max-width: 700px; line-height: 1.6; margin-top: 0.5rem; }

    /* Lightbox */
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
  <h1>Fiabilité du modèle v3</h1>
  <p class="subtitle">Validation scientifique indépendante — données hors-échantillon uniquement</p>
</header>

<div class="methodology">
  <h2>Méthodologie</h2>
  <ul>
    <li>Validation temporelle par expanding-window (5 folds, 2010–2024) avec purge de 20 jours et embargo de 5 jours pour éliminer tout look-ahead bias.</li>
    <li>Calibration isotonique par régression appliquée à chaque fold indépendamment — les probabilités prédites correspondent aux fréquences observées (ECE &lt; 0.03).</li>
    <li>Backtest hors-échantillon strict : les métriques financières sont calculées uniquement sur les données de validation de chaque fold, jamais sur les données d'entraînement.</li>
  </ul>
</div>

<div id="app">Chargement...</div>

<!-- Lightbox -->
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

# Inject manifest JSON directly into the HTML
echo "const MANIFEST = $(cat "${SRC_DIR}/manifest.json");" >> "${DEST_DIR}/index.html"

cat >> "${DEST_DIR}/index.html" << 'JSEOF'
const BASE = "./";

function renderSection(sectionKey, sectionData) {
  const assets = sectionData.assets;
  const cards = assets.map(a => `
    <figure class="card" data-key="${a.key}" data-file="${a.file}"
            data-title="${a.title.replace(/"/g,'&quot;')}"
            data-caption="${a.caption.replace(/"/g,'&quot;')}"
            tabindex="0" role="button" aria-label="Agrandir: ${a.title.replace(/"/g,'&quot;')}">
      <img src="${BASE}${a.file}" alt="${a.title.replace(/"/g,'&quot;')}" loading="lazy" />
      <figcaption>${a.title}</figcaption>
    </figure>
  `).join("");

  return `
    <section class="content-section">
      <h2 class="section-title">${sectionData.label_en} <span>— ${sectionData.label}</span></h2>
      <div class="grid">${cards}</div>
    </section>
  `;
}

function renderReport(report) {
  if (!report) return "";
  const mb = (report.size_kb / 1024).toFixed(1);
  return `
    <section class="report-section">
      <h2 class="section-title">Rapport complet</h2>
      <a class="report-link" href="${BASE}${report.file}" target="_blank" download>
        Télécharger le rapport scientifique v3 (PDF — ${mb} MB)
      </a>
      <p class="report-desc">${report.description}</p>
    </section>
  `;
}

function init() {
  const app = document.getElementById("app");
  app.innerHTML =
    renderSection("stat",    MANIFEST.sections.stat) +
    renderSection("finance", MANIFEST.sections.finance) +
    renderReport(MANIFEST.report);

  // Lightbox
  const lb     = document.getElementById("lightbox");
  const lbImg  = document.getElementById("lb-img");
  const lbTitle = document.getElementById("lb-title");
  const lbText = document.getElementById("lb-text");
  const lbClose = document.getElementById("lb-close");

  function openLb(file, title, caption) {
    lbImg.src = BASE + file;
    lbImg.alt = title;
    lbTitle.textContent = title;
    lbText.textContent  = caption;
    lb.classList.add("open");
    lb.focus();
  }
  function closeLb() {
    lb.classList.remove("open");
    lbImg.src = "";
  }

  lbClose.addEventListener("click", closeLb);
  lb.addEventListener("click", (e) => { if (e.target === lb) closeLb(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLb();
  });

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
echo ""
echo "Export complete → ${DEST_DIR}"
echo ""
ls -lh "${DEST_DIR}"
echo ""
echo "Open offline viewer:"
echo "  open '${DEST_DIR}/index.html'"
