# ml/lcc_label_utils.py

from typing import Optional, Tuple

from ml.lcc_labels import LCCLabel, LCCSubtype


def map_legacy_label(label: Optional[str]) -> Optional[LCCLabel]:
    """
    Map existing legacy labels used in this repo (ok/warn/block) to the new LCCLabel enum.
    Returns None if unknown.
    """
    if label is None:
        return None

    s = str(label).strip().lower()
    if s == "ok":
        return LCCLabel.OK
    if s in {"warn", "warning"}:
        return LCCLabel.SUSPICIOUS
    if s in {"block", "broken"}:
        return LCCLabel.BROKEN

    return None


def choose_best_label(
    legacy_label: Optional[str],
    heuristic_label: LCCLabel,
    heuristic_subtype: Optional[LCCSubtype],
    heuristic_rule_id: Optional[str],
) -> Tuple[LCCLabel, Optional[LCCSubtype], Optional[str]]:
    """
    Combine legacy human/curated labels (if present) with heuristic output.

    Policy:
    - If legacy label exists and is 'block' -> always BROKEN (strong signal).
    - If legacy label exists and is 'ok' -> keep OK unless heuristic says BROKEN.
    - If legacy label exists and is 'warn' -> keep SUSPICIOUS unless heuristic says BROKEN.
    - If legacy label is missing/unknown -> use heuristic.
    """
    legacy = map_legacy_label(legacy_label)
    if legacy is None:
        return heuristic_label, heuristic_subtype, heuristic_rule_id

    # BROKEN overrides everything
    if legacy == LCCLabel.BROKEN:
        return LCCLabel.BROKEN, LCCSubtype.OTHER, "legacy_block"

    # Heuristic BROKEN should override legacy OK/WARN (data is probably unsafe)
    if heuristic_label == LCCLabel.BROKEN and legacy != LCCLabel.BROKEN:
        return LCCLabel.BROKEN, heuristic_subtype, heuristic_rule_id

    # Otherwise keep the legacy label (more trusted), but keep subtype/rule from heuristic if you want
    if legacy == LCCLabel.OK:
        return LCCLabel.OK, None, "legacy_ok"

    if legacy == LCCLabel.SUSPICIOUS:
        return LCCLabel.SUSPICIOUS, heuristic_subtype, heuristic_rule_id or "legacy_warn"

    return heuristic_label, heuristic_subtype, heuristic_rule_id


