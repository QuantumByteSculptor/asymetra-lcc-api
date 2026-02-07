FROM python:3.11-slim

# ============================
# System setup
# ============================
WORKDIR /app

# Install system deps needed for:
# - curl (download models)
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
# Download ML models (GitHub Release)
# ============================
# This downloads models.tar.gz from GitHub Releases
# and extracts it into /app/models/
RUN mkdir -p /app/models \
  && curl -L \
     -H "Accept: application/octet-stream" \
     https://api.github.com/repos/QuantumByteSculptor/asymetra-lcc-api/releases/assets/351136422 \
     -o /tmp/models.tar.gz \
  && tar -xzf /tmp/models.tar.gz -C /app \
  && rm -f /tmp/models.tar.gz

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





