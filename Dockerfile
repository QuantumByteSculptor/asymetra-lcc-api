FROM python:3.11-slim

# ============================
# System setup
# ============================
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
  && rm -rf /var/lib/apt/lists/*

# ============================
# Python dependencies
# ============================
COPY api/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

# ============================
# Download ML models (GitHub Release Asset)
# ============================
# Optional: set GITHUB_TOKEN in Render env vars to avoid rate limits.
# Example: Render -> Environment -> Add "GITHUB_TOKEN"
ARG GITHUB_TOKEN=""
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

RUN mkdir -p /app/models \
  && echo "Downloading models.tar.gz from GitHub Release asset..." \
  && curl -fSL \
     -H "Accept: application/octet-stream" \
     -H "User-Agent: asymetra-render-build" \
     $(if [ -n "$GITHUB_TOKEN" ]; then echo "-H Authorization: token $GITHUB_TOKEN"; fi) \
     https://api.github.com/repos/QuantumByteSculptor/asymetra-lcc-api/releases/assets/351136422 \
     -o /tmp/models.tar.gz \
  && (tar -tzf /tmp/models.tar.gz >/dev/null) \
  && tar -xzf /tmp/models.tar.gz -C /app \
  && rm -f /tmp/models.tar.gz

# ============================
# Application code
# ============================
COPY api /app/api
COPY features.py /app/features.py
COPY thresholds_config.py /app/thresholds_config.py

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







