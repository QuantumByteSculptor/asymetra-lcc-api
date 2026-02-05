FROM python:3.11-slim

WORKDIR /app

# ---- system deps (curl + tar) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar \
  && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- app code ----
COPY api /app/api
COPY features.py /app/features.py
COPY thresholds_config.py /app/thresholds_config.py

# ---- models: download from GitHub Release ----
# You can override MODELS_TARBALL_URL in Render env vars if needed
ARG MODELS_TARBALL_URL="https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/download/v1.0.0/models.tar.gz"
RUN echo "Downloading models from: ${MODELS_TARBALL_URL}" \
  && curl -L --fail "${MODELS_TARBALL_URL}" -o /tmp/models.tar.gz \
  && mkdir -p /app/models \
  # The tar you built likely contains a top-level "models/" directory.
  # Extract to /app so we end up with /app/models/...
  && tar -xzf /tmp/models.tar.gz -C /app \
  && rm -f /tmp/models.tar.gz \
  # sanity check:
  && ls -lah /app/models \
  && test -f /app/models/unsup_bundle.joblib

ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



