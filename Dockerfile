FROM python:3.11-slim

WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api
COPY models /app/models
COPY features.py /app/features.py
COPY thresholds_config.py /app/thresholds_config.py

ENV PORT=8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

