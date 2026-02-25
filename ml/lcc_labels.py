# app/lcc/lcc_labels.py
from enum import Enum


class LCCLabel(str, Enum):
    """
    High-level label for a price series window.
    """
    OK = "ok"                  # Série saine / exploitable
    SUSPICIOUS = "suspicious"  # Bizarre, à surveiller / borderline
    BROKEN = "broken"          # Data quality vraiment cassée


class LCCSubtype(str, Enum):
    """
    More specific reason code for explainability & analytics.
    """
    GAP = "gap"                           # trous dans la série
    STALE = "stale"                       # prix qui ne bougent pas
    SPLIT_UNADJUSTED = "split_unadjusted"  # split non ajusté (jump énorme)
    SPIKE = "spike"                       # spike de rendement (moins extrême)
    VOLUME_INCOHERENT = "volume_incoherent"
    STRUCTURAL = "structural"             # dates, timezone, index, etc.
    OTHER = "other"


