# BioMedCLIP Vision-Language Model (VLM) Setup Guide

This guide details the **BioMedCLIP** vision-language model integrated into Stage B of the radiology report generator, including direct Hugging Face model links, download commands, zero-shot prompt evaluation, and offline caching instructions.

---

## 1. BioMedCLIP Overview & Hugging Face Source

### Primary Model: BioMedCLIP (PubMedBERT + ViT-B/16)
- **Hugging Face Hub Link**: [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **Architecture**: Vision Transformer (ViT-B/16) image encoder combined with PubMedBERT (256 token length) text encoder.
- **Pretraining**: Trained on 15 million biomedical image-text pairs extracted from PubMed Central articles.
- **Role in System**: Performs zero-shot visual-textual matching between input chest X-rays (NIH ChestX-ray14) and clinical prompt candidates for grounded radiology report generation across all 14 thoracic pathology classes.

---

## 2. Automatic & Offline Downloading Options

### Option A: Automatic Online Download (Default)
When `ai-service` initializes, `open_clip` / `transformers` automatically fetches `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` into the local Hugging Face cache (`~/.cache/huggingface/hub/`).

### Option B: Pre-Downloading Model Weights for Offline Environments
To pre-cache model weights for air-gapped or offline deployment:

```bash
python -c "import open_clip; model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
```

---

## 3. Local Model Directory Placement

If downloading model files manually from Hugging Face:
Place the downloaded model files in the local weights folder inside `ai-service`:

```
ai-service/
├── report_generation/
│   ├── vlm_weights/
│   │   ├── config.json
│   │   ├── open_clip_config.json
│   │   ├── open_clip_pytorch_model.bin
│   │   └── text_config.json
```

Set the environment variable in `.env` or `docker-compose.yml`:
```env
VLM_MODEL_PATH=./report_generation/vlm_weights
```

---

## 4. Grounded Fallback Architecture
If BioMedCLIP weights or GPU memory are unavailable during runtime, the report generation pipeline automatically falls back to **Stage A: Template-Based Grounded Generator**, ensuring 100% operational uptime for clinical reporting.
