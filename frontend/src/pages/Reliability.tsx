/**
 * frontend/src/pages/Reliability.tsx
 * ====================================
 * Page /reliability — Asymetra credibility & scientific validation page.
 *
 * Reads build/credibility/v3/manifest.json (served as static asset).
 * Displays images in two sections with lightbox on click.
 *
 * Usage (React Router v6):
 *   <Route path="/reliability" element={<Reliability />} />
 *
 * Add link in App.tsx nav (À propos section):
 *   <NavLink to="/reliability">Fiabilité</NavLink>
 *
 * The manifest.json must be served at /credibility/v3/manifest.json
 * (copy build/credibility/v3/ to your public/ folder or serve statically).
 */

import React, { useEffect, useState, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types matching manifest.json structure
// ---------------------------------------------------------------------------

interface AssetEntry {
  key: string;
  file: string;
  title: string;
  caption: string;
  size_kb: number;
}

interface SectionData {
  label: string;
  label_en: string;
  assets: AssetEntry[];
}

interface ReportEntry {
  key: string;
  file: string;
  title: string;
  description: string;
  size_kb: number;
}

interface Manifest {
  version: string;
  generated_by: string;
  sections: {
    stat: SectionData;
    finance: SectionData;
  };
  report: ReportEntry;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MANIFEST_URL = "/credibility/v3/manifest.json";
const ASSETS_BASE  = "/credibility/v3/";

const METHODOLOGY_BULLETS = [
  "Validation temporelle par expanding-window (5 folds, 2010–2024) avec purge de 20 jours et embargo de 5 jours pour éliminer tout look-ahead bias.",
  "Calibration isotonique par régression appliquée à chaque fold indépendamment — les probabilités prédites correspondent aux fréquences observées (ECE < 0.03).",
  "Backtest hors-échantillon strict : les métriques financières sont calculées uniquement sur les données de validation de chaque fold, jamais sur les données d'entraînement.",
];

// ---------------------------------------------------------------------------
// Lightbox component
// ---------------------------------------------------------------------------

interface LightboxProps {
  asset: AssetEntry | null;
  onClose: () => void;
}

function Lightbox({ asset, onClose }: LightboxProps) {
  // Close on Escape key
  useEffect(() => {
    if (!asset) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [asset, onClose]);

  if (!asset) return null;

  return (
    <div
      className="lightbox-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={asset.title}
    >
      <div
        className="lightbox-content"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          className="lightbox-close"
          onClick={onClose}
          aria-label="Fermer"
        >
          ×
        </button>
        <img
          src={ASSETS_BASE + asset.file}
          alt={asset.title}
          className="lightbox-image"
        />
        <div className="lightbox-caption">
          <h3 className="lightbox-title">{asset.title}</h3>
          <p className="lightbox-text">{asset.caption}</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AssetCard component
// ---------------------------------------------------------------------------

interface AssetCardProps {
  asset: AssetEntry;
  onClick: (asset: AssetEntry) => void;
}

function AssetCard({ asset, onClick }: AssetCardProps) {
  return (
    <figure
      className="asset-card"
      onClick={() => onClick(asset)}
      role="button"
      tabIndex={0}
      aria-label={`Agrandir: ${asset.title}`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onClick(asset);
      }}
    >
      <img
        src={ASSETS_BASE + asset.file}
        alt={asset.title}
        className="asset-thumbnail"
        loading="lazy"
      />
      <figcaption className="asset-figcaption">
        <span className="asset-title">{asset.title}</span>
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------
// Section component
// ---------------------------------------------------------------------------

interface SectionProps {
  data: SectionData;
  onSelect: (asset: AssetEntry) => void;
}

function Section({ data, onSelect }: SectionProps) {
  return (
    <section className="reliability-section">
      <h2 className="section-heading">
        {data.label_en}
        <span className="section-heading-fr"> — {data.label}</span>
      </h2>
      <div className="asset-grid">
        {data.assets.map((a) => (
          <AssetCard key={a.key} asset={a} onClick={onSelect} />
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Reliability() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [selected, setSelected] = useState<AssetEntry | null>(null);

  useEffect(() => {
    fetch(MANIFEST_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: Manifest) => {
        setManifest(data);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleSelect = useCallback((asset: AssetEntry) => {
    setSelected(asset);
  }, []);

  const handleClose = useCallback(() => {
    setSelected(null);
  }, []);

  return (
    <main className="reliability-page">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                           */}
      {/* ---------------------------------------------------------------- */}
      <header className="reliability-header">
        <h1 className="reliability-title">Fiabilité du modèle v3</h1>
        <p className="reliability-subtitle">
          Validation scientifique indépendante — données hors-échantillon uniquement
        </p>
      </header>

      {/* ---------------------------------------------------------------- */}
      {/* Methodology bullets                                              */}
      {/* ---------------------------------------------------------------- */}
      <section className="methodology-section">
        <h2 className="methodology-heading">Méthodologie</h2>
        <ul className="methodology-list">
          {METHODOLOGY_BULLETS.map((b, i) => (
            <li key={i} className="methodology-item">
              {b}
            </li>
          ))}
        </ul>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Asset sections                                                   */}
      {/* ---------------------------------------------------------------- */}
      {loading && (
        <p className="reliability-loading">Chargement des assets...</p>
      )}
      {error && (
        <p className="reliability-error">
          Impossible de charger le manifeste : {error}
        </p>
      )}
      {manifest && (
        <>
          <Section data={manifest.sections.stat}    onSelect={handleSelect} />
          <Section data={manifest.sections.finance} onSelect={handleSelect} />

          {/* Report download */}
          {manifest.report && (
            <section className="report-download-section">
              <h2 className="section-heading">Rapport complet</h2>
              <a
                href={ASSETS_BASE + manifest.report.file}
                target="_blank"
                rel="noopener noreferrer"
                className="report-download-link"
                download
              >
                Télécharger le rapport scientifique v3 (PDF —{" "}
                {Math.round(manifest.report.size_kb / 1024 * 10) / 10} MB)
              </a>
              <p className="report-description">{manifest.report.description}</p>
            </section>
          )}
        </>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Lightbox                                                         */}
      {/* ---------------------------------------------------------------- */}
      <Lightbox asset={selected} onClose={handleClose} />
    </main>
  );
}
