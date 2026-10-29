# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies:
#   libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender-dev – OpenCV & imaging
#   libgomp1                                                  – PyTorch CPU OpenMP kernels
#   curl                                                      – Container healthcheck
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
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# ── Install Python dependencies (cached build layer) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ── Copy application source ───────────────────────────────────────────────────
LABEL maintainer="NIH Multi-Disease AI Team"
LABEL build-version="2026-10-NIH-v2.2.0"
COPY . .

# Ensure required runtime and output directories exist
RUN mkdir -p /app/outputs/onnx \
             /app/outputs/retrained \
             /app/outputs/vlm

EXPOSE 8000

# Healthcheck – allows start-period for PyTorch and pretrained model downloads
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
