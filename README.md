# Multi-Disease Chest X-Ray Screening with Explainable AI and Automated Radiology Report Generation

### 14-Class NIH ChestX-ray14 · DenseNet-121 · Per-Disease Grad-CAM · Grounded Radiology Reports

---

## Overview

This capstone system performs **multi-label pathology screening** on chest X-rays using a **DenseNet-121** model trained on the **NIH ChestX-ray14** dataset (112,120 frontal-view X-rays, 14 thoracic findings).  
It produces per-disease Grad-CAM localization heatmaps, generates grounded draft radiology reports, and provides an interactive Human-in-the-Loop workflow for clinical verification.

> **Research & Educational Disclaimer**: This software is developed strictly for research, educational, and capstone evaluation purposes. It is **NOT** a certified medical diagnostic system and must not replace professional clinical diagnosis by a licensed radiologist.

---

## Target Classes (NIH ChestX-ray14 – 14 findings)

| #  | Class              | #  | Class             |
|----|--------------------|----|-------------------|
| 1  | Atelectasis        | 8  | Pneumothorax      |
| 2  | Cardiomegaly       | 9  | Consolidation     |
| 3  | Effusion           | 10 | Edema             |
| 4  | Infiltration       | 11 | Emphysema         |
| 5  | Mass               | 12 | Fibrosis          |
| 6  | Nodule             | 13 | Pleural_Thickening|
| 7  | Pneumonia          | 14 | Hernia            |

---

## Key Features

- **14-Class Multi-Label Classification**: DenseNet-121 trained on NIH ChestX-ray14 detects up to 14 co-occurring thoracic pathologies from a single chest X-ray.
- **Per-Disease Grad-CAM Explainability**: Generates individual Grad-CAM heatmaps and blended overlays for each detected condition.
- **Automated Grounded Radiology Report Generation**: Structured Findings and Impressions via Stage A (template-based grounding) and Stage B (BioMedCLIP zero-shot VLM) with pathology consistency validation.
- **Hallucination & Consistency Checker**: Validates generated text against positive model predictions to flag unsupported medical claims.
- **Human-in-the-Loop Workflow**: AI-generated reports start as `DRAFT` status and can be interactively edited, approved, and exported to PDF by clinicians.
- **DICOM & Standard Image Support**: Native DICOM (`.dcm`), JPEG, PNG, WebP, and BMP support with EXIF metadata stripping.
- **Interactive Radiology Viewer**: Brightness, Contrast, Invert, Zoom, Pan, and per-disease Heatmap Opacity controls.
- **Multi-Heatmap PDF Diagnostic Reports**: Automated PDF formatter for patient details, multi-pathology tables, Findings/Impressions, original X-ray, and Grad-CAM overlays.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   xray-net (Docker Bridge)                      │
│                                                                 │
│  ┌────────────┐     ┌──────────────┐     ┌───────────────────┐  │
│  │  frontend  │───▶ │   backend    │───▶ │    ai-service     │  │
│  │ React 18   │     │ Node.js      │     │ FastAPI + PyTorch │  │
│  │ :3000      │     │ Express      │     │ DenseNet-121      │  │
│  │            │     │ :5000        │     │ NIH 14-class      │  │
│  └────────────┘     └──────┬───────┘     │ :8000             │  │
│                            │             └───────────────────┘  │
│                      ┌─────▼──────┐                             │
│                      │   mongo    │                             │
│                      │ MongoDB 7  │                             │
│                      │ :27017     │                             │
│                      └────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer        | Technology                                                              |
|--------------|-------------------------------------------------------------------------|
| Frontend     | React 18, React Router v6, Daikon (DICOM parser), Axios, Vanilla CSS   |
| Backend      | Node.js, Express, Mongoose (MongoDB), Opossum (circuit breaker), PDFKit |
| AI Service   | Python 3.11, FastAPI, PyTorch 2.2, torchvision 0.17, OpenCV, BioMedCLIP |
| ML Model     | DenseNet-121 trained on NIH ChestX-ray14 (14 classes)                  |
| DevOps       | Docker, Docker Compose, Jenkins CI/CD                                   |

---

## Model Details

| Property          | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Architecture      | DenseNet-121 (torchvision)                                   |
| Dataset           | NIH ChestX-ray14 (`Data_Entry_2017_v2020.csv`, ~112k images) |
| Output            | 14-class sigmoid multi-label                                 |
| Input size        | 224 × 224 RGB                                                |
| Normalisation     | ImageNet mean/std (0.485/0.456/0.406, 0.229/0.224/0.225)    |
| Checkpoint        | `ai-service/test_nih/saved_models/nih_densenet121.pth`       |
| Detection threshold | 0.45 per class (configurable in `configs/thresholds.json`) |

---

## Dataset & Training

For full NIH dataset download instructions, folder layout, and training commands see [`DATASET_SETUP.md`](./DATASET_SETUP.md).

### Quick Training

```bash
cd ai-service/test_nih
python train_nih.py \
    --csv_path Data_Entry_2017_v2020.csv \
    --images_dir images \
    --epochs 20 \
    --batch_size 32 \
    --pretrained
```

The best validation checkpoint is automatically saved to `saved_models/nih_densenet121.pth`.

### Quick Inference (CLI)

```bash
cd ai-service/test_nih
python predict_nih.py \
    --image path/to/xray.png \
    --model_path saved_models/nih_densenet121.pth
```

---

## Quick Start (Docker Compose)

### Prerequisites
1. **NIH checkpoint** must be present at:  
   `ai-service/test_nih/saved_models/nih_densenet121.pth`  
   (trained via `train_nih.py` or downloaded separately — ~85 MB, excluded from git)

2. Docker & Docker Compose installed.

### Launch

```bash
docker compose up --build -d
```

The compose file automatically mounts `./ai-service/test_nih/saved_models` into the container.  
Override the mount path:

```bash
MODEL_HOST_PATH=/path/to/your/models docker compose up -d
```

### Service URLs

| Service            | URL                          |
|--------------------|------------------------------|
| Frontend App       | http://localhost:3000        |
| Backend API        | http://localhost:5000        |
| AI Service Swagger | http://localhost:8000/docs   |

---

## API Documentation Summary

### AI Service (FastAPI — port 8000)

| Method | Endpoint                  | Description                                           |
|--------|---------------------------|-------------------------------------------------------|
| POST   | `/api/v1/analyze`         | 14-class inference + per-disease Grad-CAM + report    |
| POST   | `/api/v1/report/validate` | Pathology consistency / hallucination check           |
| GET    | `/api/v1/model/info`      | Model metadata, classes, thresholds                  |
| GET    | `/health`                 | Health & model status                                 |
| POST   | `/predict`                | Legacy single-score endpoint (backward compat.)       |

### Backend Service (Express — port 5000)

| Method | Endpoint                        | Description                          |
|--------|---------------------------------|--------------------------------------|
| POST   | `/api/upload`                   | Multipart X-ray upload (JPEG/PNG/DICOM)|
| GET    | `/api/history`                  | List diagnostic history records       |
| PUT    | `/api/history/:id/report`       | Edit / approve radiology report       |
| GET    | `/api/history/:id/report/pdf`   | Stream multi-disease PDF report       |

---

## VLM Report Generation

For BioMedCLIP Vision-Language Model download, offline caching, and Stage B configuration, see [`VLM_SETUP.md`](./VLM_SETUP.md).

---

## Testing & CI/CD

- **AI Service Tests**: `cd ai-service && pytest`
- **Backend Tests**: `cd Backend && npm test`
- **Jenkins Pipeline**: Linting → unit tests → Docker builds → deployment verification.

---

## Project Structure

```
Capstone-1/
├── ai-service/
│   ├── src/                          # Modular NIH-14 ML/VLM pipeline
│   │   ├── data/                     # NIHChestXrayDataset & patient splits (zero leakage)
│   │   ├── models/                   # DenseNet-121, ResNet-50, Swin-T & checkpoint loader
│   │   ├── train/                    # MultiLabelTrainer, loss functions (BCE/Focal)
│   │   ├── xai/                      # Grad-CAM, Grad-CAM++, Swin spatial transform
│   │   ├── vlm/                      # MedGemma, LLaVA-Med, BiomedGPT fine-tuning & data builder
│   │   └── eval/                     # Macro-AUROC metrics, XAI pointing game, VLM claim eval
│   ├── app.py                        # FastAPI production service entry point
│   ├── configs/                      # base.yaml, densenet121.yaml, resnet50.yaml, swin_tiny.yaml
│   ├── models/                       # densenet_multilabel.py, nih_model.py, tb_model.py
│   ├── test_nih/
│   │   ├── Data_Entry_2017_v2020.csv # Official NIH metadata
│   │   └── saved_models/
│   │       └── nih_densenet121.pth   # Verified production checkpoint (14 classes)
│   ├── outputs/                      # Saved retraining runs, XAI and VLM evaluation outputs
│   ├── scripts/                      # retrain_all.sh & retrain_all.ps1
│   ├── report_generation/            # generator.py, vlm_report.py, report_grounding.py
│   ├── schemas/                      # prediction.py
│   ├── utils/                        # image_processor.py, gradcam.py
│   ├── tests/                        # pytest test suite
│   ├── Architecture.md               # Technical architecture & neural network specs
│   ├── Decision.md                   # Machine learning design decisions & justifications
│   ├── Dockerfile
│   └── requirements.txt
├── Backend/                          # Node.js / Express API gateway
├── frontend/                         # React 18 single page application
├── docker-compose.yml
├── DATASET_SETUP.md
└── VLM_SETUP.md
```
