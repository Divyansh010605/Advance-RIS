#!/usr/bin/env bash
# Retrain all three NIH ChestX-ray14 models sequentially
set -e

echo "=========================================================="
echo "Starting Sequential Retraining of all 3 NIH Models"
echo "=========================================================="

# 1. DenseNet-121
echo ""
echo "[1/3] Retraining DenseNet-121..."
python -m src.train --config configs/densenet121.yaml

# 2. ResNet-50
echo ""
echo "[2/3] Retraining ResNet-50..."
python -m src.train --config configs/resnet50.yaml

# 3. Swin Transformer Tiny
echo ""
echo "[3/3] Retraining Swin Transformer Tiny..."
python -m src.train --config configs/swin_tiny.yaml

echo ""
echo "=========================================================="
echo "All retraining jobs completed successfully!"
echo "Checkpoints and metrics saved under outputs/retrained/"
echo "=========================================================="
