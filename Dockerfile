FROM python:3.11-slim

# ============================
# System setup
# ============================
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ============================
# Python dependencies
# ============================
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# Copy application code
# ============================
COPY api /app/api
COPY features.py /app/features.py
COPY feature_utils.py /app/feature_utils.py
COPY configs/thresholds_config.py /app/thresholds_config.py

# 🔥 IMPORTANT: copy models from repo
COPY models /app/models

# ============================
# Runtime configuration
# ============================
ENV PORT=8000
ENV UNSUP_BUNDLE_PATH=/app/models/unsup_bundle.joblib
ENV SUP_BUNDLE_PATH=/app/models/sup_bundle.joblib

# ============================
# Start API
# ============================
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]








