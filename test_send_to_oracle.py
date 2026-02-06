# test_send_to_oracle.py
from lovable_client_utils import prepare_and_send_score_oracle
import random, time

# Simule Lovable outputs
lovable = {
    "asset_type": "equity",
    "market": "US",
    "ticker": "AAPL",
    # autres metrics Lovable pourrait envoyer...
}

# génère closes synthétiques (ou prendre les closes réels issus de Yahoo)
cl = [100.0]
for _ in range(260):
    cl.append(cl[-1] * (1 + random.gauss(0, 0.01)))

resp = prepare_and_send_score_oracle(
    lovable=lovable,
    closes=cl[-254:],   # suffit
    dates=None,         # si tu as des dates, envoie-les
    lookback_days=252,
    force_oracle=False,
    api_key=None,       # si nécessaire
)
import json
print(json.dumps(resp, indent=2)[:2000])