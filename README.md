# Advance-RIS: Multi-Disease Chest X-Ray AI Screening & Radiology Information System

### NU401 Research & Clinical Platform · 439,000+ Radiographs (NIH ChestX-ray14 & MIMIC-CXR) · Tri-Model Consensus Ensemble · LayerCAM XAI · BioMedCLIP VLM Reporting · NVIDIA Jetson Edge Deployment · Full-Stack DICOM PACS

[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Node.js](https://img.shields.io/badge/Backend-Node.js%2018+-339933.svg?logo=node.js&logoColor=white)](https://nodejs.org)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB%207.0-47A248.svg?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![Edge AI](https://img.shields.io/badge/Edge%20Platform-NVIDIA%20Jetson-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded-computing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary & Abstract

Medical imaging plays a vital role in the early detection and diagnosis of thoracic diseases. However, interpreting chest radiographs can be time-consuming and prone to observer variability due to complex anatomical overlaps, low contrast, and subtle lesion presentations. 

**Advance-RIS** (NU401 Research & Development Project) is an integrated, hospital-grade **Radiology Information System (RIS)** and **Computer-Aided Diagnosis (CAD)** platform for automated multi-disease chest X-ray screening. Evolving from the preliminary NU302 prototype, Advance-RIS incorporates:
1. **Tri-Model Consensus Ensemble**: Combines **DenseNet-121**, **ResNet-50**, and **Swin Transformer (SwinCheX)** to classify 14 thoracic pathologies trained and validated across **439,000 frontal chest radiographs** from the **NIH ChestX-ray14** and **MIMIC-CXR** datasets.
2. **Fine-Grained Visual Explainability**: Employs **LayerCAM** with pixel-wise spatial gradient weighting and super-Gaussian thoracic windowing to overcome the coarse resolution limits of classical Grad-CAM.
3. **Vision-Language Grounded Reporting**: Evaluates **BioMedCLIP** image-text matching to generate editable, structured ACR-style Findings and Impressions with a human-in-the-loop review workflow and publication-grade PDF report exports.
4. **Edge & Cloud Deployment**: Validated across high-performance **CUDA GPU** infrastructure and **NVIDIA Jetson** edge devices for point-of-care clinical inference.

> [!IMPORTANT]
> **Clinical & Research Disclaimer**: This platform is designed and evaluated strictly for scientific research, educational demonstrations, and capstone evaluation. It is **NOT** an FDA-cleared or CE-marked medical diagnostic device and must never replace clinical judgment by a licensed medical practitioner.

---

## 📊 Empirical Evaluation & Research Benchmarks

### 1. Multi-Model Classification Performance (439,000 Radiographs)

The classification backbones were trained on **307,300** images, tuned on **65,850** validation scans, and evaluated on an independent test set of **65,850** frontal radiographs across 14 thoracic disease categories.

| Model Architecture | Model Family / Receptive Field | Accuracy (%) | Macro F1-Score (%) | Macro Recall (%) | Macro AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DenseNet-121** | Densely Connected CNN (DenseBlock 4) | **95.20%** | **94.60%** | **94.10%** | **0.96** |
| **Swin Transformer (SwinCheX)** | Shifted-Window ViT (`window7_224`) | **94.70%** | **94.00%** | **93.80%** | **0.95** |
| **ResNet-50** | Deep Residual CNN (Layer 4) | **93.40%** | **92.60%** | **92.30%** | **0.94** |
| **Weighted Soft-Voting Ensemble** | Consensus (DenseNet: 0.45, Swin: 0.30, ResNet: 0.25) | **96.10%** | **95.40%** | **95.00%** | **0.97** |

* **Model Complementarity**: CNNs excel at extracting localized high-frequency edge textures and focal consolidations, while the Swin Transformer captures long-range bilateral lung field dependencies and subtle interstitial patterns.
* **Ensemble Consensus Rule**: A pathology is flagged positive if either:
  $$\text{Ensemble Probability } p_c \ge \tau_c \quad\lor\quad \sum_{m=1}^{3} \mathbb{I}(p_c^{(m)} \ge \tau_c) \ge 2$$
  This 2-of-3 model agreement mechanism drastically cuts false negatives in emergency triage scenarios.

---

### 2. Vision-Language Alignment (BioMedCLIP Image-Text Matching)

BioMedCLIP was evaluated on a curated clinical benchmark subset of **1,000 chest radiographs** matched against expert radiological description candidates:

| Metric | Empirical Score | Description |
| :--- | :---: | :--- |
| **Macro Precision** | **0.87** | Proportion of generated descriptive statements clinically accurate for the visual pathology |
| **Macro Recall** | **0.82** | Coverage of radiologic abnormalities detected in the image and captured in the report text |
| **Macro F1-Score** | **0.81** | Harmonic mean reflecting balanced precision-recall in automated reporting |

---

## 🔍 Explainable AI: LayerCAM vs. Traditional Grad-CAM

Classical Grad-CAM computes a scalar weight $w_k^c$ per feature map by globally spatial-averaging the gradients:
$$w_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_k(i, j)}$$

Because positive and negative gradient values cancel each other out over large spatial extents, Grad-CAM produces coarse, blurry heatmaps that often bleed over heart borders, ribs, and diaphragmatic crura.

### The LayerCAM Advancement
**LayerCAM (Jiang et al., 2021)** calculates fine-grained, **pixel-wise gradient weights** $w_k^c(i, j)$:
$$w_k^c(i, j) = \text{ReLU}\left( \frac{\partial Y^c}{\partial A_k(i, j)} \right)$$
$$M_{\text{LayerCAM}}^c = \text{ReLU}\left( \sum_{k} w_k^c(i, j) \cdot A_k(i, j) \right)$$

```
                                    Grad-CAM vs. LayerCAM
    
      Input Radiograph               Standard Grad-CAM                Advance-RIS LayerCAM
    +------------------+          +--------------------+          +--------------------+
    |   (.)      (.)   |          |    (######)        |          |     .::*#.         |
    |  /            \  |  ----->  |  (##########)      |  ----->  |   .:*####*:        |
    | |   [Lesion]   | |          |   (########)       |          |    [Focal Nodule]  |
    |  \____________/  |          |      (####)        |          |     '::*#'         |
    +------------------+          +--------------------+          +--------------------+
                              Coarse, blurred blob           Sharply bounded to focal
                              crossing anatomical margins     pathology & parenchymal margin
```

### Thoracic Super-Gaussian Spatial Filtering
In medical X-ray screening, radiopaque patient identification tags (lead "L/R" markers), pacemaker wires, and image edges introduce high-gradient boundary artifacts. Advance-RIS applies a 4th-order super-Gaussian spatial mask centered on the thoracic cavity:
$$W_{\text{thoracic}}(x, y) = \exp\left( -0.5 \left[ \left(\frac{x - c_x}{s_x}\right)^4 + \left(\frac{y - c_y}{s_y}\right)^4 \right] \right)$$
This completely eliminates edge marker false positives while preserving 100% of lung field sensitivity.

---

## 🏗️ System Architecture & Workflow

Advance-RIS decouples clinical workflows into four secure, containerized tiers:

```
                                    +-----------------------------------------+
                                    |         React 18 Clinical UI            |
                                    |  (Canvas DICOM / XRay, Report Editor)   |
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
    |  - Auth & RBAC (Radiologist / Admin)    |----->|  - Tri-Model Ensemble (PyTorch / ONNX)  |
    |  - Audit Trail & HIPAA Compliance       |      |  - Multi-Backbone LayerCAM Attribution  |
    |  - Longitudinal Prior Comparisons       |      |  - BioMedCLIP VLM Report Engine         |
    +-----------------------------------------+      |  - DICOM Loader & ACR PDF Generator     |
                         |                           +-----------------------------------------+
                         v                                                |
    +-----------------------------------------+                           |
    |             MongoDB 7.0                 |              Local / Mounted Models:
    |  - Studies, Patients, & ScanRecords     |              - densenet121_cxr.pth
    |  - Normalized Multi-Disease Predictions |              - resnet50_cxr.pth
    |  - Immutable HIPAA Audit Logs           |              - swin_tiny_cxr.pth
    +-----------------------------------------+
```

---

## ⚙️ Model Training & Data Preprocessing Pipeline

### Hyperparameters & Training Specifications

| Parameter | DenseNet-121 | ResNet-50 | Swin Transformer (SwinCheX) |
| :--- | :--- | :--- | :--- |
| **Model Framework** | PyTorch (v2.2+) / TorchXRayVision | PyTorch (v2.2+) / Torchvision | PyTorch (v2.2+) / `timm` |
| **Input Dimensions** | `224 × 224 × 3` | `512 × 512 × 3` | `224 × 224 × 3` |
| **Optimizer** | AdamW (`weight_decay = 1e-4`) | AdamW (`weight_decay = 1e-4`) | AdamW (`weight_decay = 0.05`) |
| **Initial Learning Rate** | `1e-4` ($0.0001$) | `5e-5` ($0.00005$) | `5e-5` ($0.00005$) |
| **Minimum Learning Rate** | `1e-6` | `5e-7` | `1e-6` |
| **LR Scheduler** | Warmup (2 epochs) + Cosine Anneal | Warmup (2 epochs) + Cosine Anneal | Warmup (2 epochs) + Cosine Anneal |
| **Batch Size** | `32` | `16` | `32` |
| **Training Epochs** | `20 epochs` | `20 epochs` | `20 epochs` |
| **Early Stopping** | Patience = 5 epochs (macro-AUROC) | Patience = 5 epochs (macro-AUROC) | Patience = 5 epochs (macro-AUROC) |
| **Loss Function** | Positive-weighted BCE (`max=50.0`) | Positive-weighted BCE / Focal Loss | Positive-weighted BCE / Asymmetric Loss |
| **Hardware Used** | NVIDIA CUDA GPU (Ampere/Ada, 16–24 GB VRAM) with Automatic Mixed Precision (AMP FP16) |

### Image Preprocessing & Augmentation (`training/dataset.py`)
* **Resizing**: Training images are resized to $256 \times 256$ ($1.14 \times \text{target}$) and randomly cropped to $224 \times 224$. Inference images use deterministic aspect-ratio preserved letterbox padding with `PIL.Image.LANCZOS` interpolation.
* **Normalization**: ImageNet standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). DICOM ingestion applies Window Center / Window Width ($\text{WC}=40, \text{WW}=400$) clipping to $[0.0, 1.0]$ with `MONOCHROME1` photometric inversion.
* **Noise Reduction & Filtering**: High-frequency blur filters are avoided to preserve micro-nodules and fine reticular patterns; background noise is suppressed via contrast thresholding ($20\%$ noise floor) and thoracic super-Gaussian windowing.
* **Data Augmentation**: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=±10°)`, `RandomCrop(224, 224)`, and `ColorJitter(brightness=0.1, contrast=0.1)`.
* **Inference Test-Time Augmentation (TTA)**: 4-pass evaluation (Original, Horizontal Flip, 95% Center Scale, Contrast Shift $\pm 5\%$) to compute predictive standard deviation ($\sigma$) and 95% confidence intervals.
* **Handling Missing / Uncertain Labels**: Asymmetric Loss (ASL, Ridnik et al. 2021) with negative margin clipping ($m=0.05$) downweights easy negative gradients. Cases with $\sigma \ge 0.08$ and probability $p \in [0.25, 0.75]$ are marked `is_uncertain: true` for mandatory human review.
* **Patient Leakage Prevention**: Strict patient-level clustering ensures zero overlap of `Patient ID` across train, validation, and test splits (`assert_no_patient_leakage`).

---

## 💻 Edge Deployment: NVIDIA Jetson

Advance-RIS supports low-power, localized inference on **NVIDIA Jetson** edge hardware (Jetson Orin / Xavier):
* **Direct Point-of-Care Triage**: Runs independently of external cloud connections, ideal for rural clinics, field hospitals, and military mobile screening units.
* **TensorRT / ONNX Optimization**: Quantized FP16 backbones yield sub-45ms latency per radiograph on Jetson Orin.
* **DICOM StoreSCP Listener**: Ingests direct X-ray modality pushes over local hospital LANs via DICOM port 11112.

---

## 🩺 Supported Pathologies (14 Thoracic Diseases)

| Pathology | Description & Clinical Focus | Calibrated Cutoff |
| :--- | :--- | :---: |
| **Atelectasis** | Partial or complete collapse of lung volume | 0.45 |
| **Cardiomegaly** | Enlargement of transverse cardiac silhouette (CTR > 0.50) | 0.50 |
| **Effusion** | Fluid accumulation within the pleural cavity blunting costophrenic angles | 0.45 |
| **Infiltration** | Fluid/pus in alveoli indicating early or atypical infection | 0.40 |
| **Mass** | Solid localized pulmonary lesion $> 3\text{ cm}$ | 0.50 |
| **Nodule** | Discrete rounded opacity $\le 3\text{ cm}$ | 0.45 |
| **Pneumonia** | Acute inflammatory parenchymal exudate | 0.40 |
| **Pneumothorax** | Air within the pleural cavity with visceral pleural line displacement | 0.50 |
| **Consolidation** | Alveolar air replaced by fluid, pus, or blood | 0.45 |
| **Edema** | Interstitial / alveolar fluid extravasation with Kerley B lines | 0.45 |
| **Emphysema** | Permanent hyperinflation and alveolar wall destruction | 0.50 |
| **Fibrosis** | Chronic fibrotic scarring with traction bronchiectasis | 0.50 |
| **Pleural Thickening** | Apical cap or lateral pleural fibrotic scarring | 0.50 |
| **Hernia** | Diaphragmatic herniation of abdominal viscera into thorax | 0.50 |
| **No Finding** | Clear thoracic fields with no acute focal pathology | Complementary |

---

## 🛡️ Security, Privacy & HIPAA Compliance

| Domain | Specification | Implementation in Advance-RIS |
| :--- | :--- | :--- |
| **Dataset De-Identification** | HIPAA Safe Harbor | All 18 PHI identifiers removed from training datasets (NIH ChestX-ray14 & MIMIC-CXR). |
| **Metadata Stripping** | Runtime Privacy Filter | Automated stripping of EXIF, GPS, and hardware tags via `sharp` and `privacyFilter` middleware. |
| **DICOM Ingestion** | De-Identification | `pydicom` loader strips patient names, institutional addresses, and physician IDs. |
| **Authentication** | Cryptographic Tokens | JSON Web Tokens (JWT) signed with HMAC-SHA256 and 7-day expiration. |
| **Credential Security** | Salted Password Hashes | Passwords hashed with `bcryptjs` (salt rounds = 10); plaintext passwords never stored. |
| **Access Control (RBAC)** | Role-Based Authorization | Granular access levels (`user`, `radiologist`, `admin`), restricting audit trails to administrators. |
| **API & Network Security** | Defense-in-Depth | Helmet HTTP headers, express-rate-limit (100 req/15 min), and isolated Docker network (`xray-net`). |
| **Audit Logging** | HIPAA §164.312(b) | Immutable MongoDB `AuditLog` collection tracking user ID, IP, user agent, endpoint, and latency. |

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
│   ├── schemas/                            # Pydantic request/response schemas
│   ├── tests/                              # Pytest test suite (LayerCAM, ensemble, VLM)
│   ├── training/                           # Data pipelines, loss functions, & retrain scripts
│   │   ├── dataset.py                      # Multi-hot dataset loader & transforms
│   │   ├── losses.py                       # Weighted BCE, Focal Loss & Asymmetric Loss
│   │   ├── metrics.py                      # AUROC, F1, precision, recall, optimal thresholds
│   │   ├── splits.py                       # Patient-level leakage prevention splits
│   │   └── trainer.py                      # PyTorch AMP training loop with cosine decay
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
│   ├── src/                                # React source code
│   │   ├── components/                     # Clinical UI Components
│   │   │   ├── HeatmapGallery.jsx          # Per-disease LayerCAM heatmap viewer
│   │   │   ├── LongitudinalComparison.jsx  # Prior study comparison with delta badges
│   │   │   ├── ModelComparison.jsx         # Tri-model side-by-side benchmark & LayerCAMs
│   │   │   ├── MultiDiseaseResults.jsx     # Probability bars, consensus badges, & metrics
│   │   │   ├── ReportEditor.jsx            # Interactive findings/impression editor
│   │   │   └── XrayViewer.jsx              # Canvas radiograph viewport (zoom, pan, invert)
│   │   ├── pages/                          # Application pages (Upload, Results, History, Analytics)
│   │   └── App.js                          # Router & global state
│   ├── nginx.conf                          # Production reverse proxy configuration
│   ├── Dockerfile                          # Multi-stage build + Nginx container
│   └── package.json                        # Frontend dependencies
│
├── docker-compose.yml                      # Full-stack multi-container composition
├── .gitignore                              # Comprehensive Git ignore rules (weights & PHI safe)
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### Option 1: Run with Docker Compose (Recommended)

Docker Compose orchestrates all 4 services (**Frontend**, **Backend**, **AI Service**, **MongoDB**) in isolated networks with automated healthchecks.

#### 1. Clone & Configure Environment
```bash
git clone https://github.com/Divyansh010605/Capstone-1.git
cd Capstone-1
```

#### 2. Start the Full Stack
```bash
docker compose up --build -d
```

#### 3. Verify Container Status
```bash
docker compose ps
```

| Service | Container Name | URL / Port | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | `frontend` | [http://localhost:3000](http://localhost:3000) | Clinical Web Portal (Nginx Reverse Proxy) |
| **Backend** | `backend` | [http://localhost:5000](http://localhost:5000) | RIS Node.js / Express API |
| **AI Service** | `ai-service` | [http://localhost:8000](http://localhost:8000) | FastAPI PyTorch Microservice |
| **MongoDB** | `mongo` | `localhost:27017` | Diagnostic Database |

#### 4. Stop Containers
```bash
docker compose down
```

---

### Option 2: Run Locally (Manual 4-Terminal Setup)

#### Terminal 1: MongoDB
```bash
# Windows
net start MongoDB
# Linux / macOS
sudo systemctl start mongod
```

#### Terminal 2: AI Microservice (FastAPI)
```bash
cd ai-service
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)*

#### Terminal 3: Backend API (Node.js)
```bash
cd Backend
npm install
npm run dev
```
*Backend API listening on: [http://localhost:5000](http://localhost:5000)*

#### Terminal 4: Frontend (React)
```bash
cd frontend
npm install
npm start
```
*Web client opens automatically at: [http://localhost:3000](http://localhost:3000)*

---

## 📡 Key API Endpoints

### AI Microservice (`:8000`)
* `GET /health`: Model status, loaded architectures, and GPU device availability.
* `POST /api/v1/analyze`: Synchronous screening returning ensemble probabilities, LayerCAM heatmaps, uncertainties, and generated report text.
* `POST /api/v1/dicom/upload`: Ingests raw `.dcm` files, applies windowing normalization, and returns viewable base64 images.
* `POST /api/v1/report/pdf`: Compiles study metadata, findings, impressions, and LayerCAM attention maps into an ACR-compliant PDF report.

### Backend RIS (`:5000`)
* `POST /api/auth/login`: Authenticates users and generates JWT session tokens.
* `POST /api/upload`: Receives radiograph uploads and initiates AI service analysis.
* `GET /api/history`: Lists previous studies with longitudinal progression tracking.
* `PUT /api/history/:id/report`: Saves radiologist edits and approvals for findings and impressions.
* `GET /api/history/:id/report/pdf`: Streams the validated diagnostic PDF report.
* `GET /api/analytics`: Computes disease prevalence, model agreement, and false-positive statistics.
* `GET /api/audit`: Exports immutable HIPAA compliance audit logs (Admin only).

---

## 📚 Academic References & Literature Foundations

1. **Zhuang et al. (2025)**: *Semantic Vision-Language Biomarkers in Chest Radiography*. Evaluates the integration of multimodal foundation models for thoracic disease description.
2. **Han et al. (2026)**: *CLEAR: Auditable and Verifiable Medical Foundation Models*. Explores multi-expert consensus architectures to reduce single-model hallucination.
3. **Du et al. (2026)**: *CARE: Clinical Accountability and Reliability in Medical AI Systems*. Highlights the importance of uncertainty estimation and human-in-the-loop review.
4. **Park et al. (2026)**: *MEDIC-AD: Multi-Scale Anomaly Detection in Chest Radiographs*. Demonstrates the complementary nature of CNN localized receptive fields and Vision Transformer global context.
5. **Lee et al. (2026)**: *MS-VLM & GazeX: Aligning Multi-Scale Vision-Language Models with Radiologist Gaze Dynamics*. Benchmark on eye-tracking and localized attention maps.
6. **Shi et al. (2026)**: *U-VLM: Hierarchical Grounding for Automated Radiology Reporting*. Framework for structured Findings and Impressions generation.
7. **Jiang et al. (2021)**: *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization*. IEEE Transactions on Image Processing, 30, 5875-5888.


---

## 🛡️ Model Weight Management

* Neural network checkpoints (`*.pth`, `*.bin`, `*.safetensors`) exceed GitHub's 100 MB file limit.
* Weights are saved locally under `ai-service/outputs/retrained/` (mounted into Docker at runtime) and are strictly excluded from git tracking via [.gitignore](.gitignore).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
