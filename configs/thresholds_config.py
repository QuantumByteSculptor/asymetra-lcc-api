# thresholds_config.py

# Policy-level default thresholds by asset class.
# These are "initial suggestions" that you can override with calibrated thresholds
# from train_unsupervised.py (thresholds_per_asset_type).
#
# Higher threshold => more permissive (fewer flags).
# Lower threshold  => more strict (more WARN/BLOCK).

THRESHOLDS_BY_ASSET_TYPE = {
    # Equities & ETFs: moderate strictness
    "equity": {"warn_q": 0.97, "block_q": 0.995},
    "etf": {"warn_q": 0.97, "block_q": 0.995},

    # Bonds/Rates: should be tighter (low vol expected -> unit mistakes show up fast)
    "bond": {"warn_q": 0.965, "block_q": 0.993},

    # FX: medium strict
    "fx": {"warn_q": 0.97, "block_q": 0.995},

    # Commodities: slightly more tolerant (can spike)
    "commodity": {"warn_q": 0.975, "block_q": 0.996},

    # Crypto: much more tolerant, otherwise everything looks anomalous
    "crypto": {"warn_q": 0.985, "block_q": 0.998},
}

# A global fallback policy if asset_type unknown
GLOBAL_POLICY = {"warn_q": 0.97, "block_q": 0.995}

