FROM python:3.11-slim

WORKDIR /app

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar \
  && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- download & extract models (GitHub Release asset) ----
ARG MODELS_URL="https://github.com/QuantumByteSculptor/asymetra-lcc-api/releases/download/v1.0.0/models.tar.gz"
# Optional: verify integrity (recommended)
ARG MODELS_SHA256="e82306f3c47bdf87a5729cb53c8e108fe7149abf67be22556318390c122635e7"

RUN mkdir -p /app/models \
  && curl -L --fail --retry 5 --retry-delay 2 -o /tmp/models.tar.gz "${MODELS_URL}" \
  && echo "${MODELS_SHA256}  /tmp/models.tar.gz" | sha256sum -c - \
  && tar -xzf /tmp/models.tar.gz -C /app \
  && rm -f /tmp/models.tar.gz

# Si ton tar contient un dossier "models/" à la racine, tu auras /app/models/*.
# Si jamais il extrait vers /app/models/models/*, on corrige:
RUN if [ -d "/app/models/models" ]; then mv /app/models/models/* /app/models/ && rmdir /app/models/models; fi

# ---- app code ----
COPY api /app/api
COPY features.py /app/features.py
COPY thresholds_config.py /app/thresholds_config.py

# ---- runtime ----
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Render fournit $PORT au runtime → il faut un shell pour l’expansion
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]



