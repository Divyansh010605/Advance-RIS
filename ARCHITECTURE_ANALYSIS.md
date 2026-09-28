# Architecture Analysis

## 1. Current Architecture

The system uses a **microservices architecture** with four containers running on a shared Docker bridge network (`xray-net`):

| Service       | Technology                         | Port  |
|---------------|------------------------------------|-------|
| **frontend**  | React 18 SPA (Nginx)               | 3000  |
| **backend**   | Node.js / Express + MongoDB        | 5000  |
| **ai-service**| FastAPI + PyTorch (DenseNet-121)   | 8000  |
| **mongo**     | MongoDB 7.0                        | 27017 |

---

## 2. ML Model

| Property          | Value                                                                  |
|-------------------|------------------------------------------------------------------------|
| Architecture      | DenseNet-121 (torchvision – plain ImageNet backbone, no XRV)          |
| Dataset           | NIH ChestX-ray14 (`Data_Entry_2017_v2020.csv`, 112 120 images)        |
| Classes           | 14 thoracic pathologies (multi-label, sigmoid output)                  |
| Input             | 224 × 224 RGB, ImageNet normalisation                                  |
| Loss              | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)                |
| Optimiser         | AdamW (lr=1e-4, weight_decay=1e-4)                                    |
| Validation split  | Patient-level (10 % hold-out by default)                              |
| Checkpoint        | `ai-service/test_nih/saved_models/nih_densenet121.pth` (~85 MB)       |
| Detection threshold | 0.45 per class (configurable in `configs/thresholds.json`)           |

---

## 3. Service Communication

```
Browser
  │  HTTP
  ▼
frontend (React / Nginx :3000)
  │  REST /api/*
  ▼
backend (Express :5000)
  │  HTTP POST /api/v1/analyze  ◀─── base64 image + mime_type
  ▼
ai-service (FastAPI :8000)
  │  Returns JSON: predictions, Grad-CAM heatmaps, report draft
  ▼
backend
  │  Stores ScanRecord in MongoDB; generates PDF on demand
  ▼
frontend  ◀─── predictions, heatmaps, report, history
```

---

## 4. API Endpoints

### AI Service (FastAPI – port 8000)

| Method | Path                      | Description                                          |
|--------|---------------------------|------------------------------------------------------|
| GET    | `/health`                 | Liveness / model-load status                        |
| GET    | `/api/v1/model/info`      | Model metadata: architecture, classes, thresholds   |
| POST   | `/api/v1/analyze`         | 14-class inference + per-disease Grad-CAM + report  |
| POST   | `/api/v1/report/validate` | Hallucination / consistency check                   |
| POST   | `/predict`                | Legacy single-score endpoint (backward compat.)     |

### Backend (Express – port 5000)

| Method | Path                        | Description                        |
|--------|-----------------------------|------------------------------------|
| POST   | `/api/auth/login`           | JWT login                          |
| POST   | `/api/auth/signup`          | User registration                  |
| POST   | `/api/upload`               | Multipart X-ray upload             |
| GET    | `/api/history`              | List scan records                  |
| PUT    | `/api/history/:id/report`   | Edit / approve report              |
| GET    | `/api/history/:id/report/pdf` | Stream PDF diagnostic report     |
| GET    | `/api/analytics/stats`      | Aggregate statistics               |

---

## 5. Data Flow

1. User uploads a chest X-ray (JPEG / PNG / DICOM) via the React frontend.
2. Backend receives it, strips EXIF metadata, and POSTs the base64-encoded image to `/api/v1/analyze`.
3. **AI Service** (`app.py`):
   a. Preprocesses the image (`utils/image_processor.py`).
   b. Runs 14-class sigmoid inference via `TBModel.predict_multi()`.
   c. Performs per-class Grad-CAM backpropagation (`utils/gradcam.py`).
   d. Generates a grounded draft radiology report (`report_generation/vlm_report.py`).
   e. Returns `MultiDiseaseAnalyzeResponse` (predictions, heatmaps, report).
4. Backend stores a `ScanRecord` document in MongoDB and returns the result to the frontend.
5. Frontend displays the interactive results (pathology table, Grad-CAM gallery, report editor).
6. Clinician can edit / approve the report; a PDF can be streamed on demand.

---

## 6. Key Source Files

| File                                  | Role                                                      |
|---------------------------------------|-----------------------------------------------------------|
| `ai-service/app.py`                   | FastAPI routes, lifespan startup                         |
| `ai-service/models/densenet_multilabel.py` | `NIHDenseNet` (torchvision) + `MultiDiseaseModel`   |
| `ai-service/models/tb_model.py`       | `TBModel` – config loading, predict_multi, Grad-CAM      |
| `ai-service/configs/classes.json`     | 14 NIH class names                                       |
| `ai-service/configs/thresholds.json`  | Per-class detection thresholds                           |
| `ai-service/test_nih/train_nih.py`    | Training script (NIH ChestX-ray14)                       |
| `ai-service/test_nih/predict_nih.py`  | CLI inference script                                     |
| `ai-service/test_nih/nih_dataset.py`  | `NIHDensenedDataset` + `parse_nih_csv()`                 |
| `ai-service/utils/gradcam.py`         | Grad-CAM heatmap generation                              |
| `ai-service/report_generation/vlm_report.py` | BioMedCLIP + template-based report generator    |

---

## 7. Docker Setup

| Component              | Detail                                                                      |
|------------------------|-----------------------------------------------------------------------------|
| Docker network         | `xray-net` (bridge)                                                         |
| MongoDB volume         | `mongo-data:/data/db`                                                        |
| NIH checkpoint volume  | `${MODEL_HOST_PATH:-./ai-service/test_nih/saved_models}:/app/test_nih/saved_models` |
| Model env var          | `MODEL_PATH=/app/test_nih/saved_models/nih_densenet121.pth`                 |
| Config env var         | `CONFIG_DIR=/app/configs`                                                   |

---

## 8. Known Risks & Mitigations

| Risk                                     | Mitigation                                                              |
|------------------------------------------|-------------------------------------------------------------------------|
| Multi-class Grad-CAM increases inference time & RAM | Grad-CAM is only computed for **detected** classes (or top-1 fallback) |
| Report hallucinations                    | Stage A grounding + consistency checker (`/api/v1/report/validate`)    |
| Large checkpoint not in git              | Mounted via Docker volume; `.gitignore` excludes `*.pth`               |
| `No Finding` images affect threshold tuning | Threshold set per-class (0.45); tunable via `thresholds.json`        |
| Legacy MongoDB documents (old schema)    | Backend handles missing fields gracefully with defaults                |
