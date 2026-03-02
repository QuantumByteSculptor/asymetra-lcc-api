# Claude Project Memory — Asymetra LCC API
<!-- Mis à jour à chaque étape importante. Relire AVANT toute action. -->

## Objectif global
Pipeline ML de fiabilité pour Asymetra.fr :
- SignalCheckML (IF+LOF unsupervised + XGB binary) détecte les anomalies dans les stats financières
- PriceGuard (Oracle) recompute les stats depuis les closes bruts quand suspect
- Expert bundles v2 (par asset_type) en cours d'intégration — PAS encore en prod

## Infra
- **Backend** : FastAPI sur Render (`asymetra-lcc-api.onrender.com`)
- **Domaine** : `api.asymetra.fr` (via Cloudflare)
- **Docker** : `Dockerfile` COPY models dans l'image, CMD = uvicorn
- **Python** : 3.12 (local), 3.11 (Docker), venv à `.venv/`
- **Branche active** : `feat/ml-experts-v2`

## Endpoints — contrats à ne jamais casser
| Endpoint | Méthode | Rôle |
|----------|---------|------|
| `/health` | GET | status + experts_loaded |
| `/score` | POST | SignalCheckML + expert_decision (additionnel) |
| `/oracle/analyze` | POST | PriceGuard pur |
| `/score_oracle` | POST | Pipeline complet |

## État des bundles (2026-03-02)
| Bundle | Fichier | Etat |
|--------|---------|------|
| Unsup global | `models/unsup_bundle.joblib` | ✅ prod |
| bin_sigmoid | `models/bin_sigmoid.joblib` | ✅ prod (modèle principal) |
| Seuils sigmoid | `models/threshold_sigmoid.json` | ✅ prod (t_lo=0.507, t_hi=0.85) |
| Expert equity v2 | `models/experts/equity_bundle.joblib` | ✅ local validé |
| Expert etf v2 | `models/experts/etf_bundle.joblib` | ✅ local validé |
| Expert global v2 | `models/experts/global_bundle.joblib` | ✅ local validé |

## Features
- **v1** : 76 colonnes (NUMERIC_BASE 26 + DERIVED 15 + one-hot 35)
- **v2** : 91 colonnes (+9 NUMERIC_BASE : downside_dev, semivariance, vol_of_vol,
  worst_5d/20d_ret, autocorr_1, vol_ewma_ann, stress_var99, stress_multiplier)
- `features.py` version = "v2", DEFAULT_CONFIG

## Datasets (2026-03-02)
| Fichier | N | v2 | Notes |
|---------|---|----|-------|
| `train_v2_all.jsonl` | 18 587 | ✅ | collecté via 736 tickers, 5 ans |
| `train_v2_split.jsonl` | 14 902 | ✅ | 80% du précédent (sans holdout) |
| `holdout_v2.jsonl` | 3 685 | ✅ | 20%, zéro ticker overlap |
| `holdout_3930.jsonl` | 3 930 | ❌ | v1 (legacy, ne pas réutiliser pour v2) |
| `train_11790.jsonl` | 11 790 | ❌ | v1 legacy |

## Résultats évaluation experts v2 (holdout_v2, 2026-03-02)
| Métrique | bin_sigmoid | Expert v2 | Δ |
|----------|-------------|-----------|---|
| ROC-AUC | 0.764 | **0.814** | +0.051 ✅ |
| PR-AUC | 0.719 | **0.797** | +0.078 ✅ |
| Recall NON-OK | 0.448 | **0.527** | +0.079 ✅ |
| FP-rate OK | 0.146 | **0.107** | -0.039 ✅ |
| ECE | 0.078 | **0.033** | -0.045 ✅ |
| Macro F1 | 0.346 | **0.453** | +0.107 ✅ |

**Verdict** : ✅ DÉPLOYABLE sous monitoring (tous critères critiques OK)

## Problèmes identifiés et résolus
- ❌ [résolu] Sur holdout v1 (sans features v2), experts v2 prédit 100% "ok" →
  holdout inadapté. Résolu en construisant `holdout_v2.jsonl` correct.
- ⚠️ [en cours] Stooq rate-limit (400/711 tickers échoués lors de la collecte).
  Relancer build_dataset_daily.py demain pour compléter à 50k+ samples.
- ⚠️ [structurel] z-score features (raw_if, raw_lof, z_if, z_lof, z_gap_if_lof)
  toutes NaN dans train_v2 (multi-worker ne les calcule pas). Impact limité (XGB
  et imputer gèrent NaN). À corriger dans un prochain sprint si nécessaire.
- ⚠️ Experts manquants pour commodity, fx, crypto → fallback global.

## Décisions en cours
- **EXPERTS_ENABLED** reste à 0 en prod jusqu'à déploiement explicite
- bin_sigmoid reste modèle principal ; experts en layer additionnel (`expert_decision`)
- Prochaine étape : passer EXPERTS_ENABLED=1 sur Render après validation équipe

## Checklist déploiement Render (QUAND décidé)
1. [ ] Vérifier que `models/experts/` est dans le Dockerfile COPY
2. [ ] Ajouter `EXPERTS_ENABLED=1` dans les env vars Render
3. [ ] Passer `feat/ml-experts-v2` → PR → merge `main`
4. [ ] Vérifier `/health` → `experts_loaded` dans la réponse
5. [ ] Smoke test `expert_decision` présent dans `/score` et `/score_oracle`
6. [ ] Monitorer fp_rate_ok et recall_non_ok en prod

## TODO immédiats
- [ ] Compléter collecte (relancer build_dataset après reset rate-limit Stooq)
- [ ] Entraîner experts commodity/fx/crypto si assez de samples
- [ ] Merger feat/ml-experts-v2 → main quand équipe validée

## TODO long terme
- [ ] CI/CD GitHub Actions pour tests auto sur PR
- [ ] Monitoring prod (fp_rate, recall) dans les logs Supabase
- [ ] Relancer évaluation après collecte complète (~50k samples)

## Scripts clés
| Script | Rôle |
|--------|------|
| `scripts/build_v2_split.py` | Split train/holdout v2 (group-aware) |
| `scripts/eval_experts_v2.py` | Évaluation complète experts vs sigmoid |
| `ml/train_experts.py` | Entraînement bundles per-asset |
| `build_dataset_daily.py` | Collecte données yfinance/Stooq |
| `docs/HOWTO_EVAL_EXPERTS_V2.md` | Guide complet pipeline évaluation |
