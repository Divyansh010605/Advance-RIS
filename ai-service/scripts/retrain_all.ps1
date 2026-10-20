# PowerShell script to retrain all 3 NIH ChestX-ray14 models sequentially
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Sequential Retraining of all 3 NIH Models" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. DenseNet-121
Write-Host "`n[1/3] Retraining DenseNet-121..." -ForegroundColor Yellow
python -m src.train --config configs/densenet121.yaml

# 2. ResNet-50
Write-Host "`n[2/3] Retraining ResNet-50..." -ForegroundColor Yellow
python -m src.train --config configs/resnet50.yaml

# 3. Swin Transformer Tiny
Write-Host "`n[3/3] Retraining Swin Transformer Tiny..." -ForegroundColor Yellow
python -m src.train --config configs/swin_tiny.yaml

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "All retraining jobs completed successfully!" -ForegroundColor Green
Write-Host "Checkpoints and metrics saved under outputs/retrained/" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
