# test_send_to_oracle.py
import os, random, json, requests

cl = [100.0]
for _ in range(260):
    cl.append(cl[-1] * (1 + random.gauss(0, 0.01)))

payload = {
    "lovable": {
        "asset_type": "equity",
        "market": "US",
        "ticker": "AAPL",
    },
    "closes": cl[-254:],
    "lookback_days": 252,
    "force_oracle": False,
}

headers = {}
api_key = os.getenv("API_KEY")
if api_key:
    headers["x-api-key"] = api_key

r = requests.post(
    "https://asymetra-lcc-api.onrender.com/score_oracle",
    json=payload,
    headers=headers,
    timeout=60,
)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:2000])
