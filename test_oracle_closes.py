import random, json, requests

cl=[100.0]
for _ in range(260):
    cl.append(cl[-1]*(1+random.gauss(0,0.01)))

payload={
  "asset_type":"equity",
  "market":"US",
  "ticker":"AAPL",
  "closes":cl[-254:],
  "lookback_days":252
}

r=requests.post("https://asymetra-lcc-api.onrender.com/oracle/analyze", json=payload, timeout=60)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:2000])

