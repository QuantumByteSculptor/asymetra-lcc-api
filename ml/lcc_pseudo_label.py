# ml/lcc_pseudo_label.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from ml.lcc_labels import LCCLabel, LCCSubtype


def lcc_pseudo_label(
    series: pd.Series,
) -> Tuple[LCCLabel, Optional[LCCSubtype], Optional[str]]:
    """
    Generate a weak heuristic label for a price series window.

    Parameters
    ----------
    series : pd.Series
        Price series (Adj Close preferred), indexed by date.

    Returns
    -------
    (label, subtype, rule_id)
    """

    if series is None or len(series) < 30:
        return LCCLabel.BROKEN, LCCSubtype.STRUCTURAL, "too_short"

    series = series.sort_index()
    returns = series.pct_change()

    # ---- Missing ratio
    pct_missing = series.isna().mean()
    if pct_missing > 0.2:
        return LCCLabel.BROKEN, LCCSubtype.GAP, "missing_gt_20pct"

    # ---- Gap length detection
    is_na = series.isna().astype(int)
    gap_groups = (is_na != is_na.shift()).cumsum()
    gap_lengths = is_na.groupby(gap_groups).sum()
    max_gap = gap_lengths.max() if len(gap_lengths) > 0 else 0

    if max_gap >= 10:
        return LCCLabel.BROKEN, LCCSubtype.GAP, "gap_ge_10d"

    # ---- Stale prices
    stale_mask = (series == series.shift()) & series.notna()
    n_stale = stale_mask.sum()
    if n_stale >= 10:
        return LCCLabel.SUSPICIOUS, LCCSubtype.STALE, "stale_ge_10d"

    # ---- Extreme returns (possible split or bad data)
    max_abs_ret = np.nanmax(np.abs(returns.values))

    if np.isnan(max_abs_ret):
        return LCCLabel.BROKEN, LCCSubtype.STRUCTURAL, "nan_returns"

    if max_abs_ret > 1.5:
        return LCCLabel.BROKEN, LCCSubtype.SPLIT_UNADJUSTED, "abs_ret_gt_150pct"

    if max_abs_ret > 0.5:
        return LCCLabel.SUSPICIOUS, LCCSubtype.SPIKE, "abs_ret_gt_50pct"

    return LCCLabel.OK, None, None

