# Multi-platform build:
#   Laptop / x86 / generic ARM (default):  docker compose up --build
#       -> python:3.11-slim + matched CPU torch/torchvision wheels
#   NVIDIA Jetson (GPU):  docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
#       -> NVIDIA L4T PyTorch base image that already ships CUDA torch/torchvision (TORCH_MODE=skip)
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# cpu  = install the CPU torch/torchvision pair from download.pytorch.org
# skip = keep the torch/torchvision already present in the base image (Jetson)
ARG TORCH_MODE=cpu

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
    if [ "$TORCH_MODE" = "cpu" ]; then \
        pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip freeze | grep -i -E '^(torch|torchvision)==' > /tmp/torch-constraints.txt && \
    cat /tmp/torch-constraints.txt && \
    pip install --no-cache-dir -r requirements.txt -c /tmp/torch-constraints.txt

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
HEALTHCHECK --interval=10s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "warning"]
