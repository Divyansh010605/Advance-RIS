# NIH ChestX-ray14 AI Screening & Report Generation Service

Clean, reproducible multi-label classification, explainable AI (Grad-CAM / Grad-CAM++), and multimodal vision-language model (VLM) fine-tuning microservice centered exclusively on the **NIH ChestX-ray14** benchmark.

---

## 1. Setup & Installation

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA recommended for training; CPU supported for inference and dry-run tests.

```bash
cd ai-service
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

Download the **NIH ChestX-ray14** dataset from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) or [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data).

Place the extracted images and metadata in:
```
ai-service/
└── test_nih/
    ├── Data_Entry_2017_v2020.csv      # Metadata labels
    ├── images/                        # All extracted PNG images (e.g. 00000001_000.png)
    └── saved_models/
        └── nih_densenet121.pth        # Production checkpoint (~85 MB)
```
*(Nested image folders such as `images_001/images/` are also automatically indexed).*

---

## 3. Retraining Classifiers

All training is config-driven and writes checkpoints exclusively to `outputs/retrained/<model_name>/` (never overwriting existing weights).

### Run Individual Models
```bash
# DenseNet-121
python -m src.train --config configs/densenet121.yaml

# ResNet-50
python -m src.train --config configs/resnet50.yaml

# Swin Transformer Tiny
python -m src.train --config configs/swin_tiny.yaml
```

### Dry-Run Verification (Test Execution)
```bash
python -m src.train --config configs/densenet121.yaml --dry-run
```

### Retrain All Models Sequentially
- **Linux / macOS**:
  ```bash
  bash scripts/retrain_all.sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\scripts\retrain_all.ps1
  ```

---

## 4. Explainable AI (XAI) Evaluation

Evaluate localization accuracy (Pointing Game Accuracy and Bounding Box IoU) against `BBox_List_2017.csv`:

```bash
# Evaluate existing DenseNet-121 checkpoint
python -m src.eval.xai_eval \
    --model_path test_nih/saved_models/nih_densenet121.pth \
    --architecture densenet121 \
    --bbox_csv test_nih/BBox_List_2017.csv \
    --method gradcam++

# Dry-run test mode (creates synthetic samples if BBox CSV is not yet downloaded)
python -m src.eval.xai_eval --architecture densenet121 --dry-run
```

Results are saved to:
- `outputs/xai_eval/xai_evaluation.json`
- `outputs/xai_eval/xai_evaluation_per_disease.csv`
- `outputs/xai_eval/overlays/` (visual heatmaps with ground-truth green bounding boxes)

---

## 5. Vision-Language Models (VLM)

### Data Construction
Build clinical instruction pairs (Findings List, Targeted Yes/No Queries, Structured Summaries) directly from verified multi-hot labels:
```bash
python -c "
from src.data.splits import parse_nih_csv
from src.vlm.data_builder import build_vlm_instruction_dataset, save_vlm_dataset_jsonl
records = parse_nih_csv('test_nih/Data_Entry_2017_v2020.csv', 'test_nih/images')
items = build_vlm_instruction_dataset(records)
save_vlm_dataset_jsonl(items, 'data/processed/vlm_instructions.jsonl')
"
```

### Fine-Tuning Scripts
```bash
# MedGemma (QLoRA 4-bit)
python src/vlm/medgemma.py --load_in_4bit --dry-run

# LLaVA-Med (QLoRA 4-bit)
python src/vlm/llava_med.py --load_in_4bit --dry-run

# BiomedGPT
python src/vlm/biomedgpt.py --dry-run
```

### Evaluate VLM Generated Findings
```bash
python -m src.eval.vlm_eval --dry-run
```

---

## 6. Running the Production API

Start the FastAPI microservice:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive documentation available at `http://localhost:8000/docs`.

---

## 7. Documentation
- [Decision Log](Decision.md): In-depth justification of losses, splits, augmentations, and VLM design.
- [Architecture Guide](Architecture.md): Complete network specifications, XAI mechanics, and Mermaid diagrams.
