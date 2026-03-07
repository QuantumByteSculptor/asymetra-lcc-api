# Credibility v4.2 — Agent Handoff (Agent 1 → Agents 2 & 3)

## État après Agent 1

Agent 1 a produit un pipeline complet et reproductible.
**Un seul `run_id` est la source de vérité.**

---

## 1. Lancer le pipeline (Agent 1 → Dataset + Artefacts)

### Prérequis

```bash
cd "Asymetra code"
# Installer les dépendances manquantes dans le venv
.venv/bin/pip install pandas yfinance requests
```

### Commande principale

```bash
.venv/bin/python ml/credibility_v4_2/run_pipeline.py \
    --start 2010-01-01 \
    --end   2025-12-31 \
    --step_days 20 \
    --max_per_ticker 200 \
    --sleep_ticker 0.3
```

> **Durée estimée** : 736 tickers × ~0.5s moyen (stooq) = ~6–12h.
> Pour un test rapide : ajouter `--max_per_ticker 10 --sleep_ticker 0.1`
> et filtrer l'univers à ~50 tickers.

### Résultat

```
artifacts/credibility_v4_2/<run_id>/
├── dataset_raw.jsonl       # Dataset complet (N ≈ 50k–100k records attendus)
├── splits.json             # 5 folds temporels + bornes + N counts
├── fold_boundaries.csv     # Tableau CSV des bornes
├── dataset_profile.json    # Profil complet (N, pos/neg, par asset_type…)
├── dataset_hash.txt        # SHA-256 stable de dataset_raw.jsonl
└── run_provenance.json     # commit hash, date, paramètres
```

---

## 2. Pour Agent 2 — Entraînement du modèle

**Inputs à utiliser :**

| Fichier | Usage |
|---------|-------|
| `dataset_raw.jsonl` | Source de vérité, ne pas modifier |
| `splits.json` | Bornes des 5 folds (train/val par `window_end_date`) |
| `run_provenance.json` | À insérer dans chaque rapport de métrique |

**Nouvelles features disponibles dans v4.2 :**
- `corr_spy` : corrélation 252j vs SPY (float, peut être NaN si données insuffisantes)
- `beta_market` : beta 252j vs SPY
- `vix_level` : niveau ^VIX au `window_end_date`
- `window_end_date` : date ISO de la fenêtre (clé pour les splits temporels)

**Convention de split :**
```python
# Pour le fold k, charger les records selon window_end_date :
import json, pandas as pd

with open("dataset_raw.jsonl") as f:
    records = [json.loads(l) for l in f if l.strip()]

splits = json.load(open("splits.json"))
fold = splits["folds"][k - 1]  # fold k (1-indexed)

train = [r for r in records
         if fold["train_start"] <= r["features"]["window_end_date"] <= fold["train_end"]
         and r["features"]["window_end_date"] < fold["purge_start"]]  # purge exclus

val = [r for r in records
       if fold["val_start"] <= r["features"]["window_end_date"] <= fold["val_end"]]
```

**Invariant critique :**
- Tout rapport de métrique DOIT inclure `run_id` et `fold_id`.
- Ne jamais mélanger des résultats de `run_id` différents.

---

## 3. Pour Agent 3 — Génération du rapport Credibility v4.2

**Sources autorisées (uniquement) :**

| Fichier | Section du rapport |
|---------|--------------------|
| `dataset_profile.json` | Section Dataset, Table 1 |
| `splits.json` | Section Cross-validation, Table 2, Figure fold timeline |
| `fold_boundaries.csv` | Figure fold boundaries |
| `run_provenance.json` | Section Reproducibility, footer |
| Métriques Agent 2 (JSON) | Section Résultats, Figures AUC/PR |

**Problèmes v4.1 à NE PAS reproduire :**
- ❌ N dans les tableaux ≠ N dans les figures
- ❌ `xgb_fold2` mentionné dans "Most recent fold" alors que c'est fold_5
- ❌ corr_spy / beta_market / vix_level marqués "N/A"
- ❌ AUC moyen ≠ moyenne des AUC par fold

**Checklist v4.2 :**
- [ ] Toutes les métriques proviennent du même `run_id`
- [ ] Fold IDs numériques cohérents (1 à 5)
- [ ] corr_spy, beta_market, vix_level présents dans le profil et dans les figures de feature importance
- [ ] N_train / N_val cohérents entre tableau et texte
- [ ] AUC moyen = mean(AUC_fold1, …, AUC_fold5) — calculé, pas copié

---

## 4. Vérification post-run

```bash
.venv/bin/python verify_artifacts.py \
    --out_dir artifacts/credibility_v4_2/<run_id> \
    --run_id <run_id>
```

Doit afficher `✓ ALL CHECKS PASSED` avec exit code 0.

---

## 5. Paramètres du pipeline v4.2

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `start` | 2010-01-01 | Couverture historique complète |
| `end` | 2025-12-31 | Inclut post-COVID, inflation 2022, IA 2023–2025 |
| `lookback_days` | 252 | 1 an de trading |
| `horizon_days` | 20 | ~1 mois, cohérent avec les règles warn/block |
| `step_days` | 20 | Fenêtres mensuelles (~60–70k records attendus) |
| `purge_days` | 20 | = horizon_days (labels regardent sur 20j forward) |
| `embargo_days` | 5 | Buffer sériel (≈1 semaine de trading) |
| `n_folds` | 5 | Expanding-window temporel |
