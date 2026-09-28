# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps:
#   libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender-dev  – OpenCV
#   libgomp1  – PyTorch CPU OpenMP kernels
#   curl      – container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install Python dependencies (cache layer) ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ── Copy application source ───────────────────────────────────────────────────
# The *.pth checkpoint is excluded by .dockerignore; mount it via a Docker
# volume (see docker-compose.yml MODEL_HOST_PATH) or copy it manually:
#   docker cp nih_densenet121.pth ai-service:/app/test_nih/saved_models/
LABEL build-version="2026-09-NIH-v1"
COPY . .

# Ensure the saved_models directory exists even when no volume is mounted
RUN mkdir -p /app/test_nih/saved_models

EXPOSE 8000

# Healthcheck – generous start_period because DenseNet weights take time on CPU
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
