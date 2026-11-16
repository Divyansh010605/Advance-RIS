# Advance-RIS: Multi-Disease Chest X-Ray AI Screening & Radiology Information System

### NIH ChestX-ray14 · Tri-Model Consensus Ensemble · LayerCAM XAI · VLM Grounded Reporting · DICOM PACS · Full-Stack RIS

[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Node.js](https://img.shields.io/badge/Backend-Node.js%2018+-339933.svg?logo=node.js&logoColor=white)](https://nodejs.org)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB%207.0-47A248.svg?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**Advance-RIS** is an end-to-end, hospital-grade **Radiology Information System (RIS)** and **Computer-Aided Diagnosis (CAD)** platform designed for automated multi-label thoracic disease screening on chest radiographs (CXRs). 

Trained and benchmarked on the **NIH ChestX-ray14** dataset (112,120 frontal radiographs, 14 thoracic pathologies), the platform features a clinical **Tri-Model Consensus Ensemble** (DenseNet-121, ResNet-50, and SwinCheX Vision Transformer), **Multi-Backbone LayerCAM Explainable AI (XAI)**, **Vision-Language Model (VLM)** grounded radiology reporting, and full PACS/DICOM image handling.

> [!IMPORTANT]
> **Clinical & Research Disclaimer**: This platform is designed and evaluated strictly for scientific research, educational demonstrations, and capstone evaluation. It is **NOT** a CE-marked or FDA-cleared medical device and must never replace clinical judgment by a licensed medical practitioner.

---

## 🏗️ System Architecture

The solution is orchestrated across four decoupled, scalable tiers:

```
                                    +-----------------------------------------+
                                    |         React 18 Clinical UI            |
                                    |  (Tailwind CSS, Canvas DICOM / XRay)    |
                                    +-----------------------------------------+
                                                         |  :3000 / :80
                                                         v
                                    +-----------------------------------------+
                                    |         Nginx Reverse Proxy             |
                                    |   /api -> Backend  |  /ai-api -> AI     |
                                    +-----------------------------------------+
                                          |                                |
                       REST (JSON / JWT)  |                                | Inference / Base64
                                          v                                v
    +-----------------------------------------+      +-----------------------------------------+
    |         Node.js / Express RIS           |      |        Python FastAPI Microservice      |
    |  - Auth & RBAC (Radiologist / Admin)    |----->|  - Tri-Model Ensemble (PyTorch)         |
    |  - Audit Trail & HIPAA Compliance       |      |  - Multi-Backbone LayerCAM Attribution  |
    |  - Longitudinal Prior Comparisons       |      |  - VLM BioMedCLIP Report Generation     |
    +-----------------------------------------+      |  - DICOM Loader & ACR PDF Engine        |
                         |                           +-----------------------------------------+
                         v                                                |
    +-----------------------------------------+                           |
    |             MongoDB 7.0                 |              Local / Mounted Models:
    |  - Studies, Patients, & ScanRecords     |              - densenet121_cxr.pth
    |  - Normalized Multi-Disease Predictions |              - resnet50_cxr.pth
    |  - Audit Logs & Diagnostic Reports      |              - SwinCheX Transformer (.pth)
    +-----------------------------------------+
```

---

## 🔬 Core AI & Clinical Capabilities

### 1. Tri-Model Consensus Ensemble
The AI inference engine evaluates chest radiographs across three diverse neural network architectures simultaneously:
* **DenseNet-121 (Densely Connected CNN)**: Extracts fine-grained localized radiologic patterns using dense feature reuse.
* **ResNet-50 (Deep Residual Network)**: Leverages skip-connections for deep anatomical representation.
* **SwinCheX (Shifted Window Vision Transformer)**: Self-attention across shifted local windows captures long-range anatomical context across the entire thoracic field.

#### Clinical Consensus Rule
To protect against false negatives in critical screening workflows, the platform employs a **Majority Voting + Weighted Soft-Vote** consensus:
$$\text{Detected} \iff (\text{Weighted Soft-Vote Probability} \ge \text{Threshold}) \;\lor\; (\text{Count of Agreeing Models} \ge 2)$$
If any **two or more models** independently identify a disease above their operating threshold, the pathology is declared **Detected**, and the study is flagged as **Pathology Detected** (`PATHOLOGY_DETECTED`).

### 2. Multi-Backbone LayerCAM Explainable AI (XAI)
Rather than a generic Grad-CAM, the system generates high-resolution **LayerCAM** heatmaps across all backbones:
* **DenseNet-121**: Extracted from DenseBlock 4 (`norm5`).
* **ResNet-50**: Extracted from Layer 4 residual block (`layer4`).
* **SwinCheX**: Extracted from Stage 4 transformer blocks with spatial token reconstruction ($B \times N \times C \rightarrow B \times C \times s \times s$).
* **Anatomical Super-Gaussian Windowing**: Suppresses artifacts outside the thoracic rib cage and diaphragm:
  $$W_{\text{thoracic}}(x, y) = \exp\left( -0.5 \left[ \left(\frac{x - c_x}{s_x}\right)^4 + \left(\frac{y - c_y}{s_y}\right)^4 \right] \right)$$
* **Noise Floor Filtering**: Drops signals below 20% of peak intensity and applies Gaussian smoothing ($15\times 15, \sigma=0$).

### 3. Predictive Uncertainty Quantification & TTA
* **Test-Time Augmentation (TTA)**: Evaluates multiple perturbed variations per scan (identity, horizontal flip, scale/crop, contrast modulation).
* **Uncertainty Metrics**: Reports predictive standard deviation ($\sigma$), 95% Confidence Intervals ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$), and warns clinicians when uncertainty exceeds acceptable variance thresholds.

### 4. Vision-Language Grounded Radiology Reports
* **BioMedCLIP & Structured Clinical Lexicon**: Produces standardized American College of Radiology (ACR) style **Findings** and **Impressions**.
* **Hallucination Prevention**: Validates that all findings in generated text correspond directly to verified classifier probabilities.
* **PDF Report Generation**: Exports publication-quality clinical reports containing the radiograph, side-by-side LayerCAM heatmaps, model agreement matrix, and radiologist approval sign-off.

---

## 🩺 Supported Pathologies (NIH ChestX-ray14)

| Pathology | Description & Clinical Focus | Cutoff Threshold |
| :--- | :--- | :---: |
| **Atelectasis** | Partial or complete collapse of lung volume | 0.45 |
| **Cardiomegaly** | Enlargement of the transverse cardiac silhouette | 0.50 |
| **Effusion** | Fluid accumulation in the pleural cavity | 0.45 |
| **Infiltration** | Fluid/pus in the alveoli (early infection indicator) | 0.40 |
| **Mass** | Solid localized pulmonary lesion $> 3\text{ cm}$ | 0.50 |
| **Nodule** | Rounded focal opacity $\le 3\text{ cm}$ | 0.45 |
| **Pneumonia** | Active inflammatory exudate within lung parenchyma | 0.40 |
| **Pneumothorax** | Air within the pleural space resulting in lung collapse | 0.50 |
| **Consolidation** | Alveolar air space replaced by exudative fluid | 0.45 |
| **Edema** | Fluid extravasation into pulmonary interstitial tissues | 0.45 |
| **Emphysema** | Permanent enlargement and destruction of air spaces | 0.50 |
| **Fibrosis** | Scar tissue accumulation in pulmonary interstitial septa | 0.50 |
| **Pleural Thickening** | Apical or lateral pleural fibrotic changes | 0.50 |
| **Hernia** | Diaphragmatic displacement of abdominal organs | 0.50 |
| **No Finding** | Clear thoracic fields with no acute pathology | Complementary |

---

## 📁 Repository Structure

```
Advance-RIS/
├── ai-service/                             # Python AI Microservice (FastAPI + PyTorch)
│   ├── configs/                            # Pathology definitions, thresholds & model configs
│   │   ├── classes.json                    # 14 NIH classes + No_Finding
│   │   ├── model_configs.yaml              # Ensemble weights, input resolutions, normalization
│   │   └── thresholds.json                 # Per-disease calibrated classification thresholds
│   ├── data/                               # Bounding box & official ground-truth index metadata
│   ├── dicom/                              # Medical DICOM loader (pydicom + windowing)
│   ├── jobs/                               # Async study job queue & memory store
│   ├── models/                             # Deep learning backbone implementations
│   │   ├── checkpoint.py                   # Checkpoint loader with architecture fallback
│   │   ├── densenet.py                     # TorchXRayVision DenseNet-121 adapter
│   │   ├── ensemble.py                     # Tri-Model Soft-Voting & Consensus Engine
│   │   ├── resnet.py                       # NIH ResNet-50 multi-label backbone
│   │   └── swin.py                         # SwinCheX Vision Transformer backbone
│   ├── outputs/                            # Retrained checkpoints & evaluation artifacts
│   │   ├── retrained/                      # Local model weights directory (ignored from Git)
│   │   └── vlm_eval/                       # VLM grounding benchmark metrics
│   ├── schemas/                            # Pydantic request/response schemas
│   ├── tests/                              # Pytest test suite (LayerCAM, ensemble, VLM)
│   ├── training/                           # Data pipelines, loss functions, & retrain scripts
│   ├── vlm/                                # Vision-Language grounded reporting & PDF export
│   │   ├── generator.py                    # Structured report generator
│   │   ├── grounding.py                    # Clinical validation checker
│   │   ├── pdf_export.py                   # ReportLab ACR-style PDF generator
│   │   └── templates.py                    # Clinical findings & impression templates
│   ├── xai/                                # Explainable AI & image processing
│   │   └── layercam.py                     # Anatomical thoracic LayerCAM implementation
│   ├── app.py                              # FastAPI REST service entrypoint
│   ├── Dockerfile                          # Optimized CPU/CUDA container specification
│   └── requirements.txt                    # Python dependencies
│
├── Backend/                                # Node.js / Express RIS API Service
│   ├── config/                             # Database, logger, and seed administrator setup
│   ├── controllers/                        # Analytics, audit, auth, history & upload controllers
│   ├── middlewares/                        # JWT authentication, role verification & privacy filters
│   ├── models/                             # Mongoose schemas (User, ScanRecord, AuditLog)
│   ├── routes/                             # Express REST route definitions
│   ├── services/                           # AI client service & audit logging
│   ├── app.js                              # Express app configuration
│   ├── server.js                           # HTTP server listener
│   ├── Dockerfile                          # Node.js production image
│   └── package.json                        # Dependencies & scripts
│
├── frontend/                               # React 18 Clinical Web Application
│   ├── public/                             # HTML template & icons
│   ├── src/                                # React source code
│   │   ├── components/                     # UI Components
│   │   │   ├── HeatmapGallery.jsx          # Per-disease LayerCAM heatmap viewer
│   │   │   ├── LongitudinalComparison.jsx  # Prior study comparison with delta badges
│   │   │   ├── ModelComparison.jsx         # Tri-model side-by-side benchmark & LayerCAMs
│   │   │   ├── MultiDiseaseResults.jsx     # Probability bars, consensus badges, & metrics
│   │   │   ├── Navbar.js                   # Application navigation & user role status
│   │   │   ├── ReportEditor.jsx            # Interactive findings/impression editor
│   │   │   └── XrayViewer.jsx              # Canvas radiograph viewport (zoom, pan, invert)
│   │   ├── pages/                          # Application pages (Upload, Results, History, Analytics)
│   │   ├── services/                       # API communication services
│   │   └── App.js                          # Router & global state
│   ├── nginx.conf                          # Production reverse proxy configuration
│   ├── Dockerfile                          # Multi-stage build + Nginx container
│   └── package.json                        # Frontend dependencies
│
├── docker-compose.yml                      # Full-stack multi-container composition
├── .dockerignore                           # Docker build ignore patterns
├── .gitignore                              # Comprehensive Git ignore rules (weights & PHI safe)
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### Option 1: Run with Docker Compose (Recommended)

Docker Compose starts all 4 services (**Frontend**, **Backend**, **AI Service**, **MongoDB**) in isolated networks with automated healthchecks.

#### 1. Clone & Configure Environment
```bash
git clone https://github.com/Divyansh010605/Capstone-1.git
cd Capstone-1

# Copy sample environment configuration (or use defaults)
cp .env.example .env 2>/dev/null || :
```

#### 2. Start the Full Stack
```bash
docker compose up --build -d
```

#### 3. Verify Container Health
```bash
docker compose ps
```

| Service | Container Name | URL / Port | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | `frontend` | [http://localhost:3000](http://localhost:3000) | Clinical Web Portal (Nginx) |
| **Backend** | `backend` | [http://localhost:5000](http://localhost:5000) | RIS Node.js / Express API |
| **AI Service** | `ai-service` | [http://localhost:8000](http://localhost:8000) | FastAPI PyTorch Microservice |
| **MongoDB** | `mongo` | `localhost:27017` | Diagnostic Database |

#### 4. Stop Containers
```bash
docker compose down
```

---

### Option 2: Run Locally (Without Docker)

If you prefer to run services individually without containers:

#### Prerequisites
* **Python**: 3.10 or 3.11
* **Node.js**: 18.x or 20.x
* **MongoDB**: Running locally at `mongodb://localhost:27017`

---

#### 1. Start MongoDB
Ensure MongoDB is running locally:
```bash
# Windows
net start MongoDB

# Linux / macOS
sudo systemctl start mongod
```

---

#### 2. Start the AI Microservice (FastAPI)
```bash
cd ai-service

# Create and activate a Python virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*API documentation will be accessible at [http://localhost:8000/docs](http://localhost:8000/docs).*

---

#### 3. Start the Backend API (Node.js)
Open a new terminal:
```bash
cd Backend

# Install dependencies
npm install

# Configure environment variables
set MONGO_URI=mongodb://localhost:27017/nihxray
set JWT_SECRET=supersecretjwtkey123
set AI_SERVICE_URL=http://localhost:8000
set PORT=5000

# Start server
npm run dev
```
*The Backend API will listen on [http://localhost:5000](http://localhost:5000).*

---

#### 4. Start the Frontend (React)
Open a third terminal:
```bash
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
```
*The frontend will open automatically at [http://localhost:3000](http://localhost:3000).*

---

## 📡 API Reference

### AI Microservice Endpoints (`:8000`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health, loaded model architecture, and GPU status |
| `POST` | `/api/v1/analyze` | Synchronous classification, LayerCAM heatmaps & report |
| `POST` | `/api/v1/dicom/upload` | Ingests `.dcm` DICOM, extracts tags, and converts to viewable JPEG |
| `POST` | `/api/v1/jobs/submit` | Asynchronous inference submission for batched studies |
| `GET` | `/api/v1/jobs/{job_id}` | Polls async inference status and retrieves results |
| `POST` | `/api/v1/report/pdf` | Generates a downloadable, publication-grade ACR PDF report |

#### Sample `/api/v1/analyze` Request
```json
{
  "image": "<base64_encoded_jpeg_or_png>",
  "mime_type": "image/jpeg",
  "use_ensemble": true,
  "use_tta": true,
  "tta_steps": 2
}
```

---

### Backend RIS Endpoints (`:5000`)

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/auth/register` | Register new user (Radiologist, Physician, Admin) | No |
| `POST` | `/api/auth/login` | Authenticate user and issue JWT token | No |
| `POST` | `/api/upload` | Upload chest radiograph / DICOM for screening | Yes |
| `GET` | `/api/history` | Retrieve study history with longitudinal tracking | Yes |
| `GET` | `/api/analytics` | Aggregated disease distribution and screening metrics | Yes |
| `GET` | `/api/audit` | HIPAA activity logs and compliance history | Admin |

---

## 🛡️ Security, Privacy & Model Weight Handling

* **Protected Health Information (PHI)**: All patient metadata from DICOM headers is stripped and anonymized prior to storage and model processing.
* **Large Model Weights**: Neural network weights (`*.pth`, `*.bin`, `*.safetensors`) exceed GitHub's 100 MB file limit. They are stored locally on disk under `ai-service/outputs/retrained/` (mounted into Docker at runtime) and are strictly excluded from git tracking via [.gitignore](.gitignore).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
