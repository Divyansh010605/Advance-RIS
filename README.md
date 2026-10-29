# Multi-Disease Chest X-Ray Screening with Explainable AI & Automated Radiology Reports

### NIH ChestX-ray14 · TorchXRayVision Ensemble · Test-Time Augmentation · DICOM · VLM Instruction Tuning

[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18+-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This capstone system performs **multi-label thoracic pathology screening** on chest radiographs using an upgraded multi-model architecture benchmarked on the **NIH ChestX-ray14** dataset (112,120 frontal-view radiographs, 14 thoracic findings).

The platform integrates:
1. **Two-Model Soft-Voting Ensemble** combining TorchXRayVision **DenseNet-121** and **ResNet-50** pretrained weights.
2. **Test-Time Augmentation (TTA)** & **Predictive Uncertainty Quantification** (standard deviation, 95% confidence intervals, and high-uncertainty clinician flags).
3. **ONNX Runtime Acceleration Engine** achieving sub-35ms inference latency.
4. **Server-Side DICOM Ingestion** with Window Center/Width windowing and MONOCHROME1 inversion.
5. **ACR-Style PDF Report Generation** with multi-disease findings grids, uncertainty indicators, and embedded Grad-CAM overlays.
6. **Interactive Medical Viewport & Longitudinal Study Comparison** in React with zoom, pan, W/L contrast, prior-exam delta indicators, and `NEW`/`RESOLVED` badges.
7. **Unified Vision-Language Model (VLM) Fine-Tuning Pipeline** for MedGemma-4B, LLaVA-Med-7B, and BiomedGPT using 4-bit QLoRA and hallucination-free multi-hot instruction synthesis.

> [!IMPORTANT]
> **Research & Educational Disclaimer**: This software is developed strictly for research, educational, and capstone evaluation purposes. It is **NOT** a certified medical diagnostic system (FDA/CE marked) and must not replace professional clinical diagnosis by a licensed radiologist.

The inference service loads TorchXRayVision weights `densenet121-res224-all` and `resnet50-res512-all`, downloading them on first initialization and caching them locally. It maps the 14 shared pathology scores into the service's labels; `No_Finding` is derived as the complement because neither XRV model has a trained `No Finding` output. Swin-T remains available for research experiments but is not loaded by the production service.

---

## Target Classes (NIH ChestX-ray14 — 14 Thoracic Findings)

| # | Pathology | # | Pathology | # | Pathology | # | Pathology |
|---|---|---|---|---|---|---|---|
| **1** | Atelectasis | **5** | Mass | **9** | Consolidation | **13** | Pleural Thickening |
| **2** | Cardiomegaly | **6** | Nodule | **10** | Edema | **14** | Hernia |
| **3** | Effusion | **7** | Pneumonia | **11** | Emphysema | | |
| **4** | Infiltration | **8** | Pneumothorax | **12** | Fibrosis | | |

---

## Key Features & Upgrades

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                         CORE CAPABILITIES                              │
  ├────────────────────────────┬───────────────────────────────────────────┤
  │ XRV Ensemble               │ DenseNet-121 (0.45) + ResNet-50 (0.25),   │
  │                            │ normalized over loaded models             │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ Test-Time Augmentation     │ 4 passes (Identity, Flip, Zoom, Contrast) │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ Uncertainty Estimation     │ Predictive variance σ, 95% normal CI,     │
  │                            │ is_uncertain clinician review flags       │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ ONNX Runtime Engine        │ Dynamic batching, CPU/CUDA graph opt      │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ DICOM Ingestion            │ Window/Level windowing, MONOCHROME1 invert│
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ Async Job Queue            │ Submit → Poll → Retrieve with cleanup     │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ ACR PDF Reports            │ 2-column institutional layout (ReportLab) │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ Medical Viewport           │ React pan, zoom (0.3×-5×), W/L, Grad-CAM  │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ Longitudinal Comparison    │ Prior vs. current scan delta, NEW badges  │
  ├────────────────────────────┼───────────────────────────────────────────┤
  │ VLM Fine-Tuning Pipeline   │ MedGemma, LLaVA-Med, BiomedGPT (QLoRA)    │
  └────────────────────────────┴───────────────────────────────────────────┘
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           xray-net (Docker Bridge)                              │
│                                                                                 │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────────┐  │
│   │   frontend   │────▶ │   backend    │────▶ │          ai-service          │  │
│   │  React 18    │      │  Node.js     │      │   FastAPI + PyTorch + ONNX   │  │
│   │  Nginx :80   │      │  Express     │      │   DenseNet + ResNet + Swin   │  │
│   │  Host :3000  │      │  Host :5000  │      │   Host :8000                 │  │
│   └──────────────┘      └──────┬───────┘      └──────────────────────────────┘  │
│                                │                                                │
│                         ┌──────▼───────┐                                        │
│                         │    mongo     │                                        │
│                         │  MongoDB 7   │                                        │
│                         │  Port :27017 │                                        │
│                         └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Docker Compose)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/Divyansh010605/Advance-RIS.git
```

### 2. Build & Launch Containers
```bash
# Build and start all services in the background
docker compose up --build -d

# Check service health
docker compose ps
```

### 3. Access Service Endpoints
- **Web Application**: [http://localhost:3000](http://localhost:3000)
- **Node.js Gateway API**: [http://localhost:5000/health](http://localhost:5000/health)
- **AI Microservice Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Local Development (Without Docker)

To run all components locally on bare metal:

### Terminal 1: AI Microservice (FastAPI)
```bash
cd ai-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Backend Gateway (Node.js / Express)
```bash
cd Backend
npm install
npm run dev
```

### Terminal 3: Frontend (React 18)
```bash
cd frontend
npm install
npm start
```

---

## Model Training & Retraining Commands

All training commands are run from the `ai-service/` directory:
```bash
cd ai-service
```

### 1. Automated Sequential Retraining Pipeline (Recommended)
Retrains DenseNet-121, ResNet-50, and Swin-T sequentially with validation-based checkpointing and automated ONNX export:
```bash
# Fast pipeline dry-run (validates data pipeline and exports in seconds)
python scripts/retrain_all.py --dry-run

# Full retraining run (overnight / unattended)
python scripts/retrain_all.py \
    --csv_path test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/nih/images/ \
    --output_dir outputs/retrained/
```
> Retraining writes checkpoints under `outputs/retrained/`. Production inference uses the TorchXRayVision weights above and does not load the legacy `test_nih/saved_models/nih_densenet121.pth` checkpoint.

### 2. Individual Model Training
```bash
# DenseNet-121
python -m src.train --config configs/densenet121.yaml

# ResNet-50
python -m src.train --config configs/resnet50.yaml

# Swin Transformer
python -m src.train --config configs/swin_tiny.yaml
```

### 3. Standalone ONNX Export
```bash
python scripts/export_onnx.py --model densenet121 \
    --weights outputs/retrained/densenet121/best_model.pth \
    --output outputs/onnx/densenet121.onnx
```

### 4. Metrics & Grad-CAM Localization Evaluation
```bash
# AUROC, F1, Sensitivity, Specificity
python -m src.eval.metrics \
    --model_path outputs/retrained/densenet121/best_model.pth \
    --model_type densenet121 \
    --test_csv test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/images/

# Bounding box pointing game & IoU
python -m src.eval.xai_eval \
    --model_path outputs/retrained/densenet121/best_model.pth \
    --model_type densenet121 \
    --bbox_csv test_nih/BBox_List_2017.csv \
    --image_dir /path/to/images/ \
    --method gradcam++
```

---

## Vision-Language Model (VLM) Fine-Tuning Pipeline

Instruction tune foundation VLMs on verified NIH multi-hot labels using 4-bit QLoRA:

```bash
# 1. MedGemma-4B (QLoRA)
python -m src.vlm.trainer --vlm medgemma \
    --csv_path test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/images/ \
    --load_in_4bit --epochs 3

# 2. LLaVA-Med-7B (QLoRA)
python -m src.vlm.trainer --vlm llava_med \
    --csv_path test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/images/ \
    --load_in_4bit --epochs 3

# 3. BiomedGPT
python -m src.vlm.trainer --vlm biomedgpt \
    --csv_path test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/images/ \
    --epochs 3

# 4. VLM Hallucination & Consistency Evaluation
python -m src.eval.vlm_eval \
    --vlm medgemma \
    --adapter_path outputs/vlm/medgemma_lora/checkpoint-final \
    --csv_path test_nih/Data_Entry_2017_v2020.csv \
    --image_dir /path/to/images/
```

---

## API Reference Summary

### AI Microservice (`FastAPI` — Port 8000)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Standard & Ensemble inference + TTA + Uncertainty + Grad-CAM + Report |
| `POST` | `/api/v1/ensemble/analyze` | Dedicated DenseNet-121 + ResNet-50 ensemble analysis endpoint |
| `POST` | `/api/v1/dicom/upload` | Multipart DICOM upload (`.dcm`), windowing, metadata extraction |
| `POST` | `/api/v1/jobs/submit` | Submit analysis task for background async execution |
| `GET` | `/api/v1/jobs/{job_id}` | Poll background job status and diagnostic results |
| `GET` | `/api/v1/jobs/{job_id}/pdf` | Download ACR PDF report for a completed background job |
| `POST` | `/api/v1/report/pdf` | Direct synchronous ACR-style PDF report generation and download |
| `POST` | `/api/v1/report/validate` | Clinical report consistency and hallucination verification |
| `GET` | `/api/v1/model/info` | Inspect loaded models, ensemble weights, and detection thresholds |
| `GET` | `/health` | Service health status and model load confirmation |
| `POST` | `/predict` | Legacy backward-compatible single-prediction endpoint |

### Backend Service (`Express` — Port 5000)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Multipart X-ray upload (JPEG, PNG, DICOM) |
| `GET` | `/api/history` | List diagnostic history records |
| `PUT` | `/api/history/:id/report` | Clinician review, edit, and approve report |
| `GET` | `/api/history/:id/report/pdf` | Stream diagnostic PDF report |
| `GET` | `/health` | Gateway health check |

---

## Repository Structure

```
Capstone-1/
├── ai-service/
│   ├── src/                          # Modular NIH-14 ML/VLM pipeline
│   │   ├── data/                     # NIHChestXrayDataset & patient splits (zero leakage)
│   │   ├── models/                   # DenseNet, ResNet, Swin-T, Ensemble & ONNX Engine
│   │   ├── train/                    # MultiLabelTrainer, loss functions (BCE/Focal)
│   │   ├── xai/                      # Grad-CAM, Grad-CAM++, Swin spatial transform
│   │   ├── vlm/                      # MedGemma, LLaVA-Med, BiomedGPT fine-tuning & builder
│   │   ├── eval/                     # Macro-AUROC metrics, XAI pointing game, VLM eval
│   │   ├── dicom/                    # Server-side DICOM parser & windowing
│   │   └── jobs/                     # In-memory async job store & lifecycle
│   ├── app.py                        # FastAPI production service entry point
│   ├── configs/                      # base.yaml, densenet121.yaml, resnet50.yaml, swin_tiny.yaml
│   ├── models/                       # densenet_multilabel.py, nih_model.py, tb_model.py
│   ├── test_nih/  # Optional dataset and research checkpoints

│   ├── outputs/                      # Saved retraining runs, ONNX models, eval reports
│   ├── scripts/                      # retrain_all.py, export_onnx.py, retrain_all.sh/ps1
│   ├── report_generation/            # pdf_report.py, generator.py, vlm_report.py
│   ├── schemas/                      # prediction.py (Pydantic request/response schemas)
│   ├── utils/                        # image_processor.py, gradcam.py
│   ├── tests/                        # pytest test suites (Phase 1 & Phase 2)
│   ├── Architecture.md               # Complete v2.2.0 technical architecture specification
│   ├── Decision.md                   # Machine learning decision log & mathematical formulas
│   ├── Dockerfile
│   └── requirements.txt
├── Backend/                          # Node.js / Express API gateway
├── frontend/                         # React 18 single page application
│   ├── src/
│   │   ├── components/               # XrayViewer, LongitudinalComparison, MultiDiseaseResults
│   │   └── pages/                    # ResultsPage, UploadPage, HistoryPage, AnalyticsPage
├── docker-compose.yml
├── DATASET_SETUP.md
└── VLM_SETUP.md
```

---

## Running the Automated Test Suite

```bash
cd ai-service

# Run all test suites
pytest tests/ -v

# Run Phase 1 tests (Ensemble, TTA, Uncertainty, ONNX):
pytest tests/test_phase1_ensemble_onnx.py -v

# Run Phase 2 tests (DICOM Ingestion, Async Job Queue, ACR PDF Generator):
pytest tests/test_phase2_dicom_jobs_pdf.py -v
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
