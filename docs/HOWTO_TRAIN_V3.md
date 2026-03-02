# HOWTO — Training v3 (expanding-window CV)

## Prérequis

```bash
cd "/Users/paul.nytts/Asymetra code/.claude/worktrees/gifted-einstein"
source .venv/bin/activate   # ou l'env Python du projet
```

Dépendances : `scikit-learn`, `xgboost` (optionnel), `joblib`, `pandas`, `numpy`

---

## Smoke run (dataset partiel, rapide)

```bash
python scripts/ml/train/train_experts_v3.py \
    --input        data/training/train_v3_all.jsonl \
    --out_dir      data/metrics/v3 \
    --models_dir   models/v3 \
    --n_splits     3 \
    --embargo_days 10 \
    --max_rows     5000 \
    --models       logistic \
    --seed         42
```

Durée attendue : **< 30s** sur 5 000 lignes, modèle logistic.

---

## Entraînement complet

```bash
python scripts/ml/train/train_experts_v3.py \
    --input        data/training/train_v3_all.jsonl \
    --out_dir      data/metrics/v3 \
    --models_dir   models/v3 \
    --n_splits     5 \
    --embargo_days 20 \
    --models       xgb,logistic \
    --seed         42
```

Remplacer `xgb` par `hgb` si xgboost n'est pas installé.

### Options utiles

| Option | Défaut | Description |
|--------|--------|-------------|
| `--max_rows N` | (tout) | Limite le JSONL à N lignes (smoke) |
| `--drop_macro` | false | Supprime les features avec >50% NaN |
| `--nan_threshold 0.3` | 0.5 | Seuil NaN pour `--drop_macro` |
| `--n_estimators 400` | 300 | Profondeur des modèles tree |
| `--models lgbm` | — | Ajouter LightGBM si disponible |

---

## Résultats

### Rapport métriques
```
data/metrics/v3/metrics_report_v3.json
```

Structure :
```json
{
  "generated_at":  "2026-03-02T...",
  "n_samples":     12540,
  "n_features":    47,
  "n_splits":      5,
  "embargo_days":  20,
  "fold_results":  [
    {
      "fold": 1, "model": "logistic",
      "roc_auc": 0.71, "pr_auc": 0.43, "brier": 0.19,
      "ece": 0.04, "f1": 0.53, "accuracy": 0.77,
      "precision_top20pct": 0.61, "lift_top20pct": 1.9,
      "backtest": {
        "threshold_constrained": 0.65,
        "simple":      { "threshold": 0.5, "skip_rate": 0.32, "sharpe_proxy": 0.8, "max_drawdown_proxy": -0.12, "fp_rate": 0.09 },
        "constrained": { "threshold": 0.65, "skip_rate": 0.48, "sharpe_proxy": 1.1, "max_drawdown_proxy": -0.08, "fp_rate": 0.05 }
      }
    },
    ...
  ],
  "aggregated": {
    "logistic": { "roc_auc_mean": 0.72, "roc_auc_std": 0.03, ... },
    "xgb":      { "roc_auc_mean": 0.76, "roc_auc_std": 0.02, ... }
  },
  "models_saved": ["models/v3/v3_xgb_final.joblib", ...]
}
```

### Modèles exportés
```
models/v3/
  v3_logistic_final.joblib   — modèle sklearn Pipeline (StandardScaler + LR)
  v3_logistic_meta.json      — feature_cols + medians (fill values pour inférence)
  v3_xgb_final.joblib        — XGBClassifier (si xgboost disponible)
  v3_xgb_meta.json
```

---

## Tests

```bash
# Tous les smoke tests (rapide, ~60s)
python -m pytest tests/unit/test_train_v3_smoke.py -v

# Uniquement les tests unitaires (sans subprocess E2E)
python -m pytest tests/unit/test_train_v3_smoke.py -v -k "not E2E"

# CV split integrity
python -m pytest tests/unit/test_split_v3.py -v
```

---

## Interpréter les résultats

| Métrique | Bonne valeur | Attention |
|----------|-------------|-----------|
| `roc_auc_mean` | ≥ 0.70 | < 0.60 → signal faible |
| `pr_auc_mean` | ≥ 0.30 | Dépend du taux de base non_ok |
| `ece_mean` | ≤ 0.05 | Calibration correcte |
| `backtest.constrained.sharpe_proxy` | > 0.5 | Signal utile |
| `backtest.constrained.fp_rate` | ≤ 0.15 | Contrainte respectée |

---

## Backtest signal intégré

Pour chaque fold de validation, le trainer calcule deux stratégies :
- **`simple`** : investit si `proba_non_ok < 0.5`
- **`constrained`** : threshold optimisé pour que `FP rate ≤ 0.15`
  (FP = investir dans un actif réellement non_ok)

Métriques par stratégie : skip_rate, sharpe_proxy, max_drawdown_proxy, fp_rate, strategy vs baseline mean return.
