# lcc_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

# ✅ Liste stable de features numériques (doit matcher tes logs)
NUM_FEATURES = [
    "vol_ann",
    "var95",
    "var99",
    "es95",
    "es99",
    "max_dd",
    "tuw_pct",
    "n_used",
    "missing_pct",
    "tail_obs_99",
]

# ✅ Encodage catégoriel minimal (tu peux l'étendre)
CAT_FEATURES = ["asset_type", "market"]

# Valeurs autorisées (sinon "other")
ASSET_TYPES = ["equity", "etf", "crypto", "commodity", "fx", "index"]
MARKETS = ["US", "EU", "UK", "JP", "CN", "IN", "SA", "HK", "OTHER"]


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    v = value if value in vocab else ("OTHER" if "OTHER" in vocab else "other")
    return [1.0 if v == token else 0.0 for token in vocab]


def features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Convertit un dict features (JSON) en vecteur numérique pour le modèle.
    - Remplit les NaN/None par np.nan puis imputation côté pipeline.
    - Garde un ordre stable.
    """
    row: List[float] = []
    for k in NUM_FEATURES:
        x = features.get(k, np.nan)
        try:
            row.append(float(x))
        except Exception:
            row.append(np.nan)

    # one-hot simple
    asset_type = str(features.get("asset_type", "other")).lower()
    market = str(features.get("market", "OTHER")).upper()

    row.extend(_one_hot(asset_type, ASSET_TYPES))
    row.extend(_one_hot(market, MARKETS))

    return np.array(row, dtype=float)


@dataclass
class LccMlDecision:
    status: str  # OK/WARN/BLOCK
    anomaly_score: float
    threshold_warn: float
    threshold_block: float


def score_to_status(score: float, warn: float, block: float) -> str:
    # score plus grand = plus anormal (on va calibrer comme ça)
    if score >= block:
        return "BLOCK"
    if score >= warn:
        return "WARN"
    return "OK"