FROM python:3.11-slim

# ============================
# System setup
# ============================
WORKDIR /app

# Install system deps needed for:
# - curl (download models)  <- laissé si tu veux garder curl pour debug local
# - tar  (extract models)
# - ca-certificates (HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
  && rm -rf /var/lib/apt/lists/*

# ============================
# Python dependencies
# ============================
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# Application models (copie depuis le repo)
# ============================
# Copier les modèles ML fournis dans le repo (bin_sigmoid.joblib, thresholds, unsup_bundle, ...)
COPY models /app/models

# ============================
# Application code
# ============================
COPY api /app/api
COPY features.py /app/features.py
COPY configs/thresholds_config.py /app/thresholds_config.py

# ============================
# Runtime configuration
# ============================
ENV PORT=8000
ENV UNSUP_BUNDLE_PATH=/app/models/unsup_bundle.joblib
ENV SUP_BUNDLE_PATH=/app/models/sup_bundle.joblib

# ============================
# Start API
# ============================
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]






