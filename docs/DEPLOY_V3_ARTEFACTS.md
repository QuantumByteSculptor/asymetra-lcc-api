# DEPLOY_V3_ARTEFACTS.md — Déploiement des artefacts ML v3.1.0

> Statut : **v3.1.0 — artefacts publiés sur GitHub Releases**
> Pas de modification du code API dans cette étape.

---

## 1. Où récupérer les artefacts

Les 7 fichiers de modèle sont attachés à la **GitHub Release v3.1.0** :

```
https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/tag/v3.1.0
```

### Téléchargement direct (CLI)

```bash
RELEASE_URL="https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/download/v3.1.0"

mkdir -p models/v3 && cd models/v3
for f in v3_xgb_model.joblib v3_lr_model.joblib v3_calibrator.joblib \
          v3_feature_names.joblib v3_meta.json v3_thresholds.json v3_metrics.json; do
  curl -fLO "${RELEASE_URL}/${f}"
done
```

Ou via `gh` CLI :

```bash
gh release download v3.1.0 \
  --repo QuantumByteSculptor/asymetra-lcc-api \
  --dir models/v3
```

---

## 2. Inventaire et checksums (SHA-256)

| Fichier | Taille | SHA-256 | Usage prod |
|---------|--------|---------|------------|
| `v3_xgb_model.joblib` | 176 653 B | `ed4bb126…5073f50e` | ✅ **REQUIS** — pipeline XGB (SimpleImputer + XGBClassifier) |
| `v3_calibrator.joblib` | 1 181 B | `d48e28fb…011ec6e` | ✅ **REQUIS** — IsotonicRegression (step séparé) |
| `v3_meta.json` | 3 503 B | `c759a2ac…7188c4` | ✅ **REQUIS** — feature_cols (64), medians, schema_version=3.1 |
| `v3_thresholds.json` | 196 B | `e7538800…313dd697` | ✅ **REQUIS** — t_lo=0.5203, t_hi=0.6667 |
| `v3_lr_model.joblib` | 3 264 B | `a92e439e…8e27b6e` | ⬜ optionnel — baseline LR, non utilisé en prod |
| `v3_feature_names.joblib` | 480 B | `25d45e4f…bed64cc` | ⬜ optionnel — redondant avec feature_cols dans v3_meta.json |
| `v3_metrics.json` | 9 870 B | `100abc59…8bed7554` | ⬜ optionnel — artefact d'évaluation, pas d'usage runtime |

**4 fichiers requis en prod** : `v3_xgb_model.joblib`, `v3_calibrator.joblib`, `v3_meta.json`, `v3_thresholds.json`

### Vérification checksums

```bash
cd models/v3
shasum -a 256 v3_xgb_model.joblib v3_calibrator.joblib v3_meta.json v3_thresholds.json
# Sortie attendue :
# ed4bb1261996c1fd072872bc646184c97bce768c23ad9440d70c5f8f5073f50e  v3_xgb_model.joblib
# d48e28fb3710cfbb6b9578cce36eeecf49419042e69ecb52e7bd1025c011ec6e  v3_calibrator.joblib
# c759a2ac51dc397b7f2ada9852ac770df12dfd9d0e100a1fbdee9dde0e7188c4  v3_meta.json
# e753880fedbe4751ac6a61411dffdc484c9c646df4f836bea5ddb881313dd697  v3_thresholds.json
```

---

## 3. Procédure de déploiement sur Render

> Les artefacts sont intentionnellement exclus du repo git (`models/v3/` dans `.gitignore`).
> Le déploiement est actuellement **manuel** — un wiring automatique (Render pre-deploy script) est prévu.

### Option 1 — Render Shell (one-shot, recommandée pour test rapide)

Dans le dashboard Render → votre service → **Shell** :

```bash
# Télécharger les 4 artefacts requis depuis la Release v3.1.0
mkdir -p /opt/render/project/src/models/v3
cd /opt/render/project/src/models/v3

RELEASE_URL="https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/download/v3.1.0"
for f in v3_xgb_model.joblib v3_calibrator.joblib v3_meta.json v3_thresholds.json; do
  curl -fLO "${RELEASE_URL}/${f}"
done

# Vérifier
ls -lh .
python3 -c "
import joblib, json
xgb = joblib.load('v3_xgb_model.joblib')
cal = joblib.load('v3_calibrator.joblib')
meta = json.load(open('v3_meta.json'))
assert meta['schema_version'] == '3.1'
print(f'✅ OK — schema=3.1, {len(meta[\"feature_cols\"])} features')
"
```

⚠️ **Limitation** : les fichiers sont éphémères sur Render free tier (effacés à chaque rebuild Docker). Voir Option 2 pour une solution persistante.

### Option 2 — Render pre-deploy command (recommandée pour production)

Ajouter dans `render.yaml` (ou dans le dashboard Render → Build Command) :

```yaml
# render.yaml
services:
  - type: web
    name: asymetra-lcc-api
    env: docker
    buildCommand: ""
    preDeployCommand: |
      mkdir -p models/v3 &&
      RURL="https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/download/v3.1.0" &&
      for f in v3_xgb_model.joblib v3_calibrator.joblib v3_meta.json v3_thresholds.json; do
        curl -fLO "${RURL}/${f}" -o "models/v3/${f}";
      done
```

> Cette option se déclenche automatiquement à chaque rebuild et garantit la présence des artefacts.
> **Elle nécessite une modification de `render.yaml`** — hors scope de ce PR, à faire dans un commit séparé.

### Option 3 — Variable d'environnement `MODELS_V3_URL` (futur)

Wiring API prévu : l'API pourra télécharger les artefacts au démarrage si la variable
`MODELS_V3_URL` est définie. Non implémenté dans v3.1.0.

---

## 4. Snippet de chargement API (copier-coller)

```python
import joblib
import json
import numpy as np
from pathlib import Path

MODELS_V3 = Path("models/v3")

# ── Chargement au démarrage ─────────────────────────────────────
xgb_pipe   = joblib.load(MODELS_V3 / "v3_xgb_model.joblib")
calibrator = joblib.load(MODELS_V3 / "v3_calibrator.joblib")
meta       = json.loads((MODELS_V3 / "v3_meta.json").read_text())
thresholds = json.loads((MODELS_V3 / "v3_thresholds.json").read_text())

# ── Fail-fast si artefact périmé ────────────────────────────────
assert meta["schema_version"] == "3.1", f"schema mismatch: {meta['schema_version']}"

FEATURE_COLS = meta["feature_cols"]    # 64 features ordonnées
T_LO         = thresholds["t_lo"]     # 0.5203 — seuil warn
T_HI         = thresholds["t_hi"]     # 0.6667 — seuil block

# ── Inférence ──────────────────────────────────────────────────
def score_v3(input_features: dict) -> dict:
    X = np.array(
        [input_features.get(col, np.nan) for col in FEATURE_COLS],
        dtype=np.float64,
    ).reshape(1, -1)
    proba_raw = xgb_pipe.predict_proba(X)[:, 1]
    score = float(calibrator.predict(proba_raw)[0])
    label = "block" if score >= T_HI else ("warn" if score >= T_LO else "ok")
    return {"score": score, "label": label, "t_lo": T_LO, "t_hi": T_HI}
```

---

## 5. Rappel modèle v3.1.0

| Métrique | Valeur |
|----------|--------|
| Modèle prod | XGBoost + IsotonicRegression calibrateur |
| ROC-AUC (calibré) | **0.787** |
| PR-AUC | 0.769 |
| Brier | 0.187 |
| ECE | 0.000 |
| Features | 64 (0 dropped — corr_spy/beta_market/corr_vix peuplés) |
| Folds CV | 5 (expanding-window, purge=20j, embargo=5j) |
| Dataset | 54 963 records, 689 tickers, 2010–2025 |
| t_lo (warn) | 0.5203 (FPR ≤ 10%) |
| t_hi (block) | 0.6667 (FPR ≤ 25%) |

---

*Généré le 2026-03-03 — release gate: ✅ GO (66/66 tests, 7/7 artefacts)*
