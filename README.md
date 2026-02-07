# Asymetra LCC API

Backend API for **Asymetra.fr** — a reliability layer that sanity-checks market analytics and can **recompute key risk stats from raw closes** when upstream data looks suspicious.

This service powers the “analysis / reliability” part of the Asymetra product and is designed to be **fast**, **defensive**, and **production-friendly** (cache, fallbacks, and never-crash shadow models).

---

## What this API does

### 1) SignalCheckML (formerly “LCC ML”)
An unsupervised anomaly detector (Isolation Forest + LOF) that inspects computed features (volatility, drawdowns, tail risk…) and flags suspicious outputs:

- **OK**: looks consistent  
- **WARN**: borderline / needs caution  
- **BLOCK**: strongly inconsistent → likely data/feature issue  

It also reports **coverage** (how many input features were missing), so you can skip scoring or force recomputation when upstream payloads are incomplete.

### 2) PriceGuard (formerly “Oracle”)
A “ground-truth-ish” recomputation engine for core stats using **raw close prices**:

- realized vol (20d / annualized)  
- max drawdown  
- VaR / ES (95 / 99)  
- RSI  
- extra diagnostics: skew, kurtosis, EWMA vol, optional GARCH vol, stress VaR  

PriceGuard supports:
- **rescue mode**: compute directly from `closes[]` sent by the frontend (no external data dependency)
- **recompute mode**: download data (yfinance) with a **Stooq fallback**, plus a SQLite cache to avoid rate limits

### 3) score_oracle pipeline
The main endpoint:
- receives “Lovable computed stats”
- runs SignalCheckML
- checks integrity constraints (VaR/ES ordering, etc.)
- if suspicious or incomplete → runs PriceGuard
- returns `features_final` + decision trace (debug-friendly)

---

## API endpoints

### Health
`GET /health`

Returns service version + cache status.

### Root (avoid noisy 404s)
`GET /`

Returns a tiny JSON payload with pointers to useful routes.

### Pure PriceGuard (recompute features)
`POST /oracle/analyze`

Body:
```json
{
  "asset_type": "equity",
  "market": "US",
  "ticker": "AAPL",
  "lookback_days": 252
}
```

Optional: provide closes[] (and dates[]) to compute without downloads:

```
{
  "asset_type": "equity",
  "market": "US",
  "ticker": "AAPL",
  "closes": [189.2, 190.1, 188.9],
  "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
  "lookback_days": 252
}
```

Pure SignalCheckML

POST /score

Body:

```
{
  "asset_type": "equity",
  "market": "US",
  "ticker": "AAPL",
  "vol_ann": 0.23,
  "vol_20d": 0.18,
  "max_drawdown": -0.17,
  "var95": 0.032,
  "var99": 0.055,
  "es95": 0.041,
  "es99": 0.072,
  "rsi": 52.1,
  "n_used": 252,
  "missing_pct": 0.0
}
```

Full reliability pipeline

POST /score_oracle

Body:

```
{
  "lovable": {
    "asset_type": "equity",
    "market": "US",
    "ticker": "AAPL",
    "vol_ann": 0.23,
    "max_drawdown": -0.17,
    "var99": 0.055
  },
  "closes": [189.2, 190.1, 188.9],
  "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
  "force_oracle": false,
  "lookback_days": 252
}
```

Response includes:
	•	features_final (lovable or recomputed)
	•	oracle_used + oracle_mode (none | rescue | recompute)
	•	unsup_status (OK/WARN/BLOCK/SKIP)
	•	decision_trace + gating_debug

⸻

Reliability rules (high-level)

PriceGuard is triggered when one of these happens:
	•	forced by force_oracle=true
	•	integrity violations (e.g. VaR99 < VaR95, ES < VaR)
	•	missing critical fields (vol_ann, var95/99, es95/99, max_drawdown…)
	•	SignalCheckML says BLOCK
	•	SignalCheckML says WARN and it’s not a “shallow warn” (and not relaxed by the shadow XGB)

⸻

Caching & fallbacks

Cache

PriceGuard stores results in SQLite (ORACLE_CACHE_DB) keyed by:
asset_type|market|ticker|lookback_days|source

Each market has its own TTL (configurable). This reduces yfinance rate-limit pain.

Download strategy
	•	Try yfinance with start/end
	•	fallback to yfinance period=max
	•	fallback to Stooq (robust CSV validation + headers)
	•	final fallback (if provided) → compute from closes[]


⸻

Configuration (environment variables)

Bundles / models
	•	UNSUP_BUNDLE_PATH (default models/unsup_bundle.joblib)
	•	SUP_BUNDLE_PATH (default models/sup_bundle.joblib)
	•	XGB_SHADOW_ENABLED (1 by default)

Security
	•	API_KEY (optional). If set, requests must include header x-api-key.

Oracle / PriceGuard
	•	ORACLE_MAX_TRIES (default 4)
	•	ORACLE_SLEEP_TRY (default 0.8)
	•	ORACLE_CACHE_DB (default data/oracle_cache.sqlite3)

Cache TTL
	•	ORACLE_TTL_US_EU (default 28800 = 8h)
	•	ORACLE_TTL_ASIA_OCE (default 43200 = 12h)
	•	ORACLE_TTL_GLOBAL (default 21600 = 6h)

Gating knobs
	•	ORACLE_WARN_MARGIN (default 0.08)
	•	ORACLE_XGB_OK_PBLOCK_MAX (default 0.20)

UNSUP coverage gates
	•	UNSUP_MAX_MISSING_RATIO (default 0.25)
	•	UNSUP_MAX_MISSING_COUNT (default 0 = ignore)

⸻

Local development

Install

```
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```
Run

```
uvicorn api.main:app --host 0.0.0.0 --port 10000 --reload
```


⸻

Deployment notes (Render)
	•	This repo is designed to run as a Render Web Service with Docker.
	•	Models are usually pulled as a tarball release artifact and extracted into /app/models.
	•	If you move files, ensure the Dockerfile COPY paths stay correct.

Typical pitfalls:
	•	COPY thresholds_config.py ... not found → file moved; update Dockerfile path
	•	yfinance rate-limits → rely on cache + Stooq fallback + rescue mode (closes payload)

⸻

Naming
	•	SignalCheckML: the “sanity check / anomaly detection” layer (unsupervised + coverage)
	•	PriceGuard: the recomputation engine used when results are suspicious or incomplete

⸻

License

Private / internal project (Asymetra).
