# HOWTO — Évaluation des expert bundles v2

## Vue d'ensemble

Le pipeline d'évaluation des experts v2 se déroule en 3 étapes :

```
train_v2_all.jsonl
       │
       ├─ 80% ──► train_v2_split.jsonl ──► train_experts.py ──► models/experts/
       │
       └─ 20% ──► holdout_v2.jsonl ──────► eval_experts_v2.py ──► data/metrics/
```

**Règle d'or** : `holdout_v2.jsonl` ne doit JAMAIS être utilisé pour l'entraînement
ou la calibration des seuils.

---

## Étape 1 — (Re)collecter les données v2

```bash
# Regénérer l'univers si besoin
python collect_universe.py --out data/universe.json

# Collecter (1-2h, rate-limit Stooq se reset à minuit UTC)
python build_dataset_daily.py \
  --universe data/universe.json \
  --out data/training/train_v2_all.jsonl \
  --lookback_years 5 \
  --step_days 10 \
  --workers 4 \
  --max_per_ticker 60 \
  > logs/build_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## Étape 2 — Construire le split train/holdout v2

```bash
python scripts/build_v2_split.py \
  --input   data/training/train_v2_all.jsonl \
  --train   data/training/train_v2_split.jsonl \
  --holdout data/training/holdout_v2.jsonl \
  --holdout_ratio 0.20 \
  --seed 42
```

**Garanties du split :**
- Stratifié par `(asset_type, dominant_label)` au niveau ticker
- Zéro overlap de tickers entre train et holdout
- Ratio effectif ~20%

---

## Étape 3 — Réentraîner les experts

```bash
python ml/train_experts.py \
  --data     data/training/train_v2_split.jsonl \
  --out_dir  models/experts/ \
  --min_samples 200 \
  --warn_q 0.95 \
  --block_q 0.99 \
  --alpha 0.15 \
  --seed 42
```

Bundles produits :
- `models/experts/global_bundle.joblib`
- `models/experts/equity_bundle.joblib`
- `models/experts/etf_bundle.joblib`

---

## Étape 4 — Évaluer

```bash
python scripts/eval_experts_v2.py \
  --holdout             data/training/holdout_v2.jsonl \
  --experts_dir         models/experts \
  --sigmoid_bundle      models/bin_sigmoid.joblib \
  --sigmoid_thresholds  models/threshold_sigmoid.json \
  --out_json            data/metrics/experts_v2_report.json \
  --out_txt             data/metrics/experts_v2_report.txt
```

Rapport produit dans `data/metrics/` :
- `experts_v2_report.json` — métriques complètes (ROC-AUC, PR-AUC, FP-rate, recall, ECE…)
- `experts_v2_report.txt` — résumé lisible terminal

---

## Critères de déployabilité

| Métrique          | Seuil minimal | Critique |
|-------------------|:-------------:|:--------:|
| ROC-AUC NON-OK    | ≥ 0.75        | oui      |
| FP-rate OK        | ≤ 0.15        | oui      |
| Recall NON-OK     | ≥ 0.45        | oui      |
| ECE               | < 0.10        | non      |
| Balanced Accuracy | ≥ 0.40        | non      |

---

## Lancer les tests

```bash
# Suite complète (72 tests)
.venv/bin/python -m pytest tests/ -v

# Uniquement les tests experts v2
.venv/bin/python -m pytest tests/unit/test_experts_eval.py -v
```

---

## Déploiement (à faire APRÈS validation)

1. Vérifier que `EXPERTS_ENABLED=0` en prod (aucun impact actuel)
2. Passer `EXPERTS_ENABLED=1` dans les variables Render une fois les critères atteints
3. Monitorer `fp_rate_ok` et `recall_non_ok` post-déploiement

**Ne jamais pusher les bundles `.joblib` directement sur Render sans validation
préalable avec ce pipeline.**
