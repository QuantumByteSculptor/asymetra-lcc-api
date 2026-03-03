# HOWTO — Training v3 (expanding-window CV)

> **DEPRECATED** — This document describes `scripts/ml/train/train_experts_v3.py`,
> which has been superseded by the canonical entrypoint `scripts/ml/train/train_v3.py`.
>
> **Please refer to [HOWTO_V3_PIPELINE.md](HOWTO_V3_PIPELINE.md) Section 4** for the
> up-to-date training instructions (manifest-based, calibrated, NaN-aware).

---

## Migration rapide

```bash
# Ancien (DEPRECATED)
python scripts/ml/train/train_experts_v3.py \
    --input  data/training/train_v3_all.jsonl \
    --models xgb,logistic

# Nouveau (entrypoint officiel)
python scripts/ml/train/train_v3.py \
    --manifest data/training/v3/splits_manifest.json \
    --out_dir  models/v3
```

Voir [HOWTO_V3_PIPELINE.md](HOWTO_V3_PIPELINE.md) pour les détails complets.

---

*Ce document est conservé pour référence historique uniquement.*
