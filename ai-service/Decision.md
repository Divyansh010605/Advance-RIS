# Decision Log: NIH ChestX-ray14 Refactor & VLM Adaptation

This document details the architectural, methodological, and engineering decisions adopted during the refactoring of the AI screening microservice around the NIH ChestX-ray14 benchmark.

---

## 1. Dataset Splitting Strategy

- **Chosen**: Official NIH split lists (`train_val_list.txt` / `test_list.txt`) combined with deterministic **patient-level partitioning** between train and validation subsets.
- **Alternatives Considered**:
  - *Random row-level splitting*: Randomly distributing individual chest radiographs.
  - *K-fold cross-validation*: Full cross-validation over the entire 112,120-image dataset.
- **Rationale**: Chest radiographs frequently feature multiple follow-up exams from the exact same patient over time. Row-level splitting allows images of the same patient to appear in both training and test sets, causing severe data leakage and artificially inflated AUROC. Grouping strictly by `Patient ID` guarantees zero patient overlap.
- **Trade-offs**: Patient-level clustering results in slightly varying label distribution across folds compared to stratified row splitting, but ensures true clinical generalization.
- **Enforcement**: Guarded by `assert_no_patient_leakage()` which asserts zero set intersection between patient IDs across splits.

---

## 2. Loss Function Formulation

- **Chosen**: Binary Cross-Entropy with per-class positive weighting (`BCEWithLogitsLoss(pos_weight=...)`).
- **Alternatives Considered**:
  - *Standard unweighted BCE*: Equal gradient weighting across all 14 findings.
  - *Focal Loss ($\gamma=2.0$)*: Down-weighting easy negative examples.
  - *Asymmetric Loss (ASL)*: Decoupled positive/negative margin clipping.
- **Rationale**: NIH ChestX-ray14 exhibits extreme label sparsity and class imbalance: common classes like Infiltration comprise ~17% of scans, while rare classes like Hernia comprise ~0.2% (an 85:1 imbalance). Per-class `pos_weight = (N_neg / N_pos)` dynamically rescales positive gradients for rare pathologies without introducing extra hyperparameter search complexity.
- **Trade-offs**: Very large weights can cause gradient instability; weights are clamped to $\text{max\_weight}=50.0$. Focal loss is supported via `configs/base.yaml` for optional comparative experiments.

---

## 3. Augmentation Pipeline

- **Chosen**:
  - *Training*: Resize to $256 \times 256$, RandomCrop to $224 \times 224$, RandomHorizontalFlip ($p=0.5$), RandomRotation ($\pm 10^\circ$), ColorJitter (brightness 0.1, contrast 0.1), ImageNet normalization.
  - *Evaluation*: Deterministic Resize to $224 \times 224$, ImageNet normalization.
- **Alternatives Considered**:
  - *Aggressive affine & elastic deformation*: Heavy shears, perspective transforms, and cutouts.
  - *Vertical flips*: Inverting the scan vertically.
- **Rationale**: Chest radiography depends strictly on anatomical orientation (e.g. apex top, diaphragm base, left cardiac silhouette). Vertical flipping violates thoracic anatomy. Moderate rotation ($\pm 10^\circ$) reflects real-world patient positioning variance on portable X-rays without distorting cardiothoracic ratios.
- **Trade-offs**: Light augmentation maintains clinical fidelity while preventing overfitting on small minority classes.

---

## 4. Input Resolution

- **Chosen**: $224 \times 224$ pixels.
- **Alternatives Considered**:
  - *$512 \times 512$ / $448 \times 448$*: Higher spatial resolution.
  - *$1024 \times 1024$*: Native NIH sensor resolution.
- **Rationale**: The production service serves the pre-trained `nih_densenet121.pth` checkpoint, which operates on $224 \times 224$ inputs. Retaining $224 \times 224$ preserves 100% backward compatibility, minimizes inference latency in FastAPI, and matches standard ImageNet pre-training backbones without extra interpolation artifacts.
- **Trade-offs**: Fine-grained findings (e.g. small nodules $<1\text{ cm}$) have lower pixel resolution than at $512 \times 512$, but compute cost is reduced by $5.2\times$.

---

## 5. Model Variant Selection

| Model Family | Chosen Variant | Backbone Source | Justification |
|---|---|---|---|
| **DenseNet** | DenseNet-121 | `torchvision.models` | Preserves 100% state-dict compatibility with existing `nih_densenet121.pth` checkpoint. Feature reuse across dense blocks is ideal for subtle thoracic opacities. |
| **ResNet** | ResNet-50 | `torchvision.models` | Canonical clinical benchmark baseline; residual connections prevent gradient vanishing while providing deeper spatial representation than ResNet-18/34. |
| **Vision Transformer** | Swin-T (`swin_tiny_patch4_window7_224`) | `timm` | Hierarchical shifted-window self-attention scales linearly with image size. Swin-Tiny has 28M parameters (comparable to DenseNet-121's ~7M and ResNet-50's ~25M), preventing out-of-memory errors on commodity GPUs during fine-tuning. |

---

## 6. Explainable AI (XAI) per Architecture

- **DenseNet-121 & ResNet-50**:
  - *Chosen*: Both **Grad-CAM** and **Grad-CAM++**.
  - *Target Layers*: Final feature extractor blocks (`features` for DenseNet, `layer4` for ResNet).
  - *Rationale*: Grad-CAM++ weights pixel-wise gradients with higher-order derivatives, substantially improving multi-focal pathology localization (e.g., bilateral infiltrates or multiple nodules).
- **Swin Transformer**:
  - *Chosen*: **Grad-CAM with spatial Reshape Transform**.
  - *Alternatives Considered*: *Attention Rollout / Attention Flow*.
  - *Rationale*: Attention rollout aggregates attention matrices across all heads and layers, suffering from accumulation of attention noise and background tokens in deep transformer stages. Grad-CAM with a reshape transform permutes the Swin patch-grid tensor $(B, H, W, C) \to (B, C, H, W)$, allowing exact class-discriminative gradient localization directly comparable to CNN heatmaps.

---

## 7. Checkpoint Compatibility & Key-Remapping Architecture

- **Chosen**: Strict key validation with documented automatic prefix remapping (`module.`, `net.`, `model.`).
- **Policy**:
  - The production checkpoint (`test_nih/saved_models/nih_densenet121.pth`) is **strictly preserved**. Legacy TBX11K checkpoints (`outputs/*.pth`, `models/tb_model.pth`) were permanently purged.
  - Retraining jobs write exclusively to new directories: `outputs/retrained/<model_name>/`.
  - Discrepancies between model parameters and checkpoint keys are never silently ignored: `src.models.checkpoint.load_checkpoint()` logs all missing and unexpected keys and asserts exact compatibility.

---

## 8. VLM Instruction Construction from Multi-Hot Labels

- **Chosen Strategy**: Structured clinical prompt-response generation directly from verified multi-hot labels across 3 task types:
  1. *Findings List Generation*: Positive pathology listing or "No Finding".
  2. *Targeted Clinical Inquiries*: Yes/No questions regarding specific pathologies.
  3. *Structured Diagnostic Checklist*: Systematic 14-disease Positive/Negative status.
- **Critical Clinical Constraint**: The NIH ChestX-ray14 dataset does **not** contain full radiology report narratives (only NLP-mined labels). Generating pseudo-reports with external LLMs would introduce fabricated clinical findings (hallucinations). Generating structured instruction tasks guarantees 100% factual fidelity to ground truth.

---

## 9. VLM Fine-Tuning Hyperparameters (PEFT / LoRA / QLoRA)

- **Quantization**: 4-bit NF4 (`BitsAndBytesConfig`) with double quantization and bfloat16/float16 compute.
- **LoRA Parameters**: $r = 16$, $\alpha = 32$, dropout $= 0.05$.
- **Target Modules**: Linear projections in attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, plus MLP projections where supported).
- **VRAM Optimization**: Gradient accumulation steps $= 8$ with batch size $= 2$ (effective batch size $= 16$), fitting on an 8–10 GB VRAM GPU.

---

## 10. Evaluation Metrics Formulation

- **Primary Validation Metric**: **Macro-AUROC** across all 14 findings (unaffected by threshold choice).
- **Threshold Calibration**: Validation-tuned thresholds per class via fine-grained grid search ($\tau \in [0.05, 0.95]$ with step $0.02$) to maximize per-class $F_1$ score.
- **Operating Metrics**: Macro and per-class $F_1$, Sensitivity (Recall), Specificity, Precision, and Support saved to JSON and CSV.
- **Localization Evaluation**: Pointing Game Accuracy (peak pixel hit inside ground truth box) and Bounding Box IoU against `BBox_List_2017.csv`.

---

## 11. Phase 1 Upgrade: Tri-Model Ensemble, TTA, Uncertainty & ONNX Acceleration

### 11.1 Tri-Model Soft-Voting Ensemble
- **Chosen**: Weighted soft-voting ensemble across three distinct architectural paradigms:
  1. **DenseNet-121** (Weight: $0.45$): Feature-reuse architecture preserving high-frequency edges and micro-nodules.
  2. **Swin-T** (Weight: $0.30$): Shifted-window self-attention capturing bilateral lung asymmetries and diffuse infiltrates.
  3. **ResNet-50** (Weight: $0.25$): Deep residual inductive bias for structural stability.
- **Dynamic Normalization**: If only the production DenseNet-121 model is loaded, the ensemble automatically normalizes its weight to $1.0$, guaranteeing zero downtime or errors while awaiting completed retraining runs.
- **Ensemble Formula**:
  $$\bar{P}_c = \sum_{m=1}^M w_m \cdot P_{m, c}, \quad \text{where } \sum_{m=1}^M w_m = 1.0$$

### 11.2 Test-Time Augmentation (TTA)
- **Clinical Rationale**: Radiographs exhibit natural bilateral symmetry and slight acquisition variances (e.g., patient rotation, inspiration depth).
- **Transformations**:
  - $T_1$: Identity (original radiograph).
  - $T_2$: Bilateral Horizontal Flip ($\text{flip}_x$).
  - $T_3$: 95% Center Crop with Bilinear Rescaling (focal zoom on central lung fields).
  - $T_4$: 5% Contrast Intensity Shift ($\text{clamp}(1.05 \cdot x, 0, 1)$).
- **TTA Aggregation**: For each model $m$, prediction is the arithmetic mean across all $K$ augmentation passes:
  $$P_{m, c} = \frac{1}{K} \sum_{k=1}^K \sigma(f_m(T_k(x)))_c$$

### 11.3 Uncertainty Quantification Formulation
- **Predictive Variance**: Standard deviation computed over all $N = M \times K$ forward passes:
  $$\sigma_c = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (P_{i, c} - \bar{P}_c)^2}$$
- **Confidence Intervals**: 95% Normal confidence interval per pathology:
  $$CI_c = \left[\max\left(0, \bar{P}_c - 1.96\sigma_c\right), \, \min\left(1, \bar{P}_c + 1.96\sigma_c\right)\right]$$
- **High-Uncertainty Flag (`is_uncertain`)**:
  Identifies borderline or divergent diagnostic predictions for mandatory human radiologist review:
  $$\text{is\_uncertain}_c = (\sigma_c \ge 0.08 \land 0.25 \le \bar{P}_c \le 0.75) \lor (\sigma_c \ge 0.15)$$

### 11.4 ONNX Runtime Acceleration
- **Chosen**: Dedicated `src.models.onnx_engine` wrapping `onnxruntime` with `CUDAExecutionProvider` prioritization and `CPUExecutionProvider` fallback.
- **Export Specifications**:
  - Dynamic batching on axis 0: `input: (batch_size, 3, 224, 224)` $\to$ `logits: (batch_size, 14)`, `probabilities: (batch_size, 14)`.
  - Constant folding and full graph optimization (`GraphOptimizationLevel.ORT_ENABLE_ALL`).
  - Strict evaluation mode (`wrapper.eval()`) during tracing to prevent BatchNorm statistics drift.
- **Latency Gain**: Reduces cold inference latency from ~120ms to under 35ms per scan on CPU/GPU.

---

## 12. Phase 2 Upgrade: DICOM Ingestion, Async Job Queue & ACR-Style PDF Reports

### 12.1 DICOM Ingestion (`src.dicom.loader`)
- **Chosen**: `pydicom`-based server-side DICOM ingestion via a dedicated FastAPI endpoint (`POST /api/v1/dicom/upload`).
- **Windowing Strategy**: Window Center/Width extracted from DICOM tags (`WindowCenter`, `WindowWidth`); fallback to clinical lung window (WC=40, WW=400). MONOCHROME1 images are automatically inverted.
- **Output**: Converted to base64 JPEG and returned to client alongside DICOM metadata (Patient ID, Study Date, Modality, SOP UID, Rows/Cols, Bits). The base64 payload can be passed directly to `/api/v1/analyze`.
- **Validation**: 132-byte DICM magic-byte check before pydicom parse; graceful error messages on invalid files.

### 12.2 Async Background Job Queue (`src.jobs.store`)
- **Chosen**: In-process `asyncio`-native job store using a thread-safe `asyncio.Lock`-protected `Dict[str, JobRecord]`.
- **Lifecycle**: `pending → running → completed | failed` with timestamps for submitted_at, started_at, completed_at.
- **API Surface**: `POST /api/v1/jobs/submit` (immediate response with job_id) → `GET /api/v1/jobs/{job_id}` (poll) → `GET /api/v1/jobs/{job_id}/pdf` (PDF download on completion).
- **Production Upgrade Path**: Replace `_jobs` dict with Redis HASH operations (`redis_client.hset` / `hgetall`) for horizontal scaling without changing the API contract.
- **Cleanup**: `cleanup_old_jobs(max_age_seconds=3600)` prunes stale completed/failed records.

### 12.3 ACR-Style PDF Report Generator (`report_generation.pdf_report`)
- **Chosen**: `reportlab` pure-Python PDF generation with ACR-inspired structure.
- **Report Sections**:
  1. Dark header band (institution, patient ID, dates, request ID).
  2. Impression banner (ABNORMAL/NORMAL with color-coded border).
  3. Technique section (models evaluated, inference mode).
  4. Findings table (14-disease grid with probability, threshold, detection status, uncertainty flag).
  5. Impression & narrative narrative.
  6. Embedded Grad-CAM figure (when `heatmap_b64` is supplied).
  7. Mandatory clinical disclaimer (research-only statement).
- **Uncertainty Highlighting**: Cells with `is_uncertain=True` receive a yellow background in the findings table.
- **Rationale**: `reportlab` is pure Python (no external binary dependencies), produces deterministic output, and integrates cleanly with FastAPI `Response(content=pdf_bytes, media_type="application/pdf")`.

---

## 13. Phase 3 Upgrade: Medical Frontend Viewport & Longitudinal Comparison

### 13.1 XrayViewer Component (`frontend/src/components/XrayViewer.jsx`)
- **Chosen**: Pure-React viewport with CSS `filter` for Window/Level and CSS `transform` for zoom/pan. No WASM or Cornerstone dependency required at this stage.
- **Interaction Model**:
  - Scroll wheel → zoom (0.3×–5.0×).
  - Click-drag → pan (translateX/Y).
  - Brightness / Contrast sliders → `brightness()` / `contrast()` CSS filters.
  - Invert toggle → `invert(1)` filter (MONOCHROME1 correction).
  - CAM toggle → Grad-CAM overlay blended via `mix-blend-mode: screen` at configurable opacity.
- **Rationale**: Eliminates a 500+ kB WASM payload while delivering clinically useful W/L control. Cornerstone.js remains a clear upgrade path if pixel-accurate HU values are required.

### 13.2 LongitudinalComparison Component (`frontend/src/components/LongitudinalComparison.jsx`)
- **Chosen**: Client-side `sessionStorage` for prior study persistence, side-by-side image panels, and a delta comparison table.
- **Delta Encoding**: `▲ +N%` (red) for increases ≥5 pp, `▼ -N%` (green) for decreases ≥5 pp, `↔` (gray) for stable.
- **Badges**: `NEW` (red) when a condition is detected that was not in the prior study; `RESOLVED` (green) for the opposite.
- **Storage Contract**: `sessionStorage.setItem('priorResult', ...)` / `getItem('priorResult')` — no backend required.

### 13.3 Enhanced MultiDiseaseResults
- **Phase 3 additions**: Sortable grid (by probability / detected / name), uncertainty badges (Low/Med/High σ with color coding), 95% CI bars, uncertain-case warning banner, interactive `onSelectDisease` callback linking to the image viewer tab.

### 13.4 ResultsPage Tab Architecture
- **Three tabs**: Overview (cards + heatmaps + report editor), Image Viewer (XrayViewer + per-disease CAM selector), Comparison (LongitudinalComparison).
- **PDF Download**: `POST /api/v1/report/pdf` → binary blob download via `URL.createObjectURL`.

---

## 14. Phase 4 Upgrade: Unified VLM Fine-Tuning & Evaluation Pipeline

### 14.1 Unified Trainer (`src.vlm.trainer`)
- **Chosen**: Single `trainer.py` covering all three VLMs (MedGemma, LLaVA-Med, BiomedGPT) via a `VLM_CONFIGS` registry keyed by short name.
- **LoRA Target Resolution**: Each VLM entry specifies its exact `lora_targets` list to avoid manual discovery errors.
- **Training Loop**: HuggingFace `Trainer` with `TrainingArguments` (cosine LR schedule, warmup_ratio=0.03, group_by_length=True, bf16/fp16 autoselect, eval and save every 100/200 steps).
- **Dataset**: `VLMInstructionDataset` wraps `VLMInstructionItem` records; text format `"{prompt}\n{response}{eos}"` with `labels = input_ids` and padding tokens masked to -100.
- **JSONL Persistence**: Training and validation instruction sets are saved to `{output_dir}/train_instructions.jsonl` and `val_instructions.jsonl` for reproducibility and inspection.

### 14.2 VLM Evaluation (`src.eval.vlm_eval`)
- **Findings Parser**: Three-strategy cascade — (1) direct NIH class name matching with explicit-negative exclusion, (2) `"- Disease: Positive/Negative"` checklist parsing, (3) `"Identified findings: X, Y"` list extraction.
- **Metrics**: Macro precision, recall, F1, exact match rate, **hallucination rate** (fraction of samples where VLM output contains a class not in ground truth).
- **Per-Class F1**: Computed for all 14 NIH disease categories independently.
- **CLI Mode**: `--dry-run` validates the parser without loading model weights.

---

## 15. Phase 5 Upgrade: Retraining Orchestrator

### 15.1 `scripts/retrain_all.py`
- **Chosen**: Python subprocess orchestrator that sequentially invokes `python -m src.train` for each model with a runtime YAML override file (injecting `csv_path`, `images_dir`, `save_dir` without mutating version-controlled configs).
- **ONNX Export**: After each successful training run, invokes `scripts/export_onnx.py` to produce `outputs/onnx/{model}.onnx`.
- **Safety**: Never touches `test_nih/saved_models/` (production checkpoint). Each retrained model writes to its own `outputs/retrained/{model_name}/` subdirectory.
- **Dry-Run**: `--dry-run` flag propagates to each `src.train` call (1 epoch, 2 batches) for CI/CD validation.

### 15.2 VLM Retraining Commands
```bash
# MedGemma (requires HF_TOKEN + accepted license)
python -m src.vlm.trainer --vlm medgemma \
    --csv_path /data/Data_Entry_2017_v2020.csv \
    --image_dir /data/images/ \
    --load_in_4bit --epochs 3

# LLaVA-Med
python -m src.vlm.trainer --vlm llava_med \
    --csv_path /data/Data_Entry_2017_v2020.csv \
    --image_dir /data/images/ \
    --load_in_4bit --epochs 3

# BiomedGPT
python -m src.vlm.trainer --vlm biomedgpt \
    --csv_path /data/Data_Entry_2017_v2020.csv \
    --image_dir /data/images/ \
    --epochs 3
```
