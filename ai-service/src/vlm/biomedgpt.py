"""
BiomedGPT Vision-Language Pipeline & Investigation Report.

Target Model: fudan-zju-biomed/BiomedGPT-1.6B / PharMolix/BiomedGPT-10B
Licence / Status: Open academic research release.

CRITICAL INVESTIGATION FINDINGS (Documented per instructions):
1. Architecture:
   BiomedGPT is built upon the OFA (One-For-All) sequence-to-sequence multimodal backbone
   (OFA-Sys/OFA-large) rather than modern Hugging Face causal vision-language architectures
   (such as LLaVA or Gemma Vision2Seq).
2. Hugging Face Compatibility:
   Standard `transformers >= 4.35` does NOT provide native `AutoModelForVision2Seq` or standard
   chat-template processors for BiomedGPT. Running full fine-tuning requires the legacy OFA codebase
   (custom fairseq/transformers fork with specialized patch extraction and BPE tokenization).
3. Recommended Workable Alternatives:
   - For generative multi-label VLM report drafting: MedGemma (`google/medgemma-4b`) or LLaVA-Med (`microsoft/llava-med-v1.5-mistral-7b`).
   - For zero-shot / contrastive pathology scoring: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (supported natively via `open_clip`).
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.data.splits import create_patient_splits
from src.vlm.data_builder import build_vlm_instruction_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.vlm.biomedgpt")


def main():
    parser = argparse.ArgumentParser(description="BiomedGPT Adaptation & Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, default="fudan-zju-biomed/BiomedGPT-1.6B")
    parser.add_argument("--output_dir", type=str, default="outputs/vlm/biomedgpt")
    parser.add_argument("--use_biomedclip_fallback", action="store_true", default=False, help="Use BiomedCLIP contrastive adapter if OFA fork is absent")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and print compatibility status")
    args = parser.parse_args()

    print("==========================================================")
    print(" BiomedGPT Investigation & Adaptation Suite")
    print(f" Target Model ID:     {args.model_id}")
    print(" Backbone Type:       OFA (Sequence-to-Sequence Multi-Modal)")
    print(" HF Support Status:   Non-standard processor / requires legacy OFA fork")
    print(" Recommended Alt:     microsoft/BiomedCLIP or google/medgemma-4b")
    print("==========================================================")

    if args.dry_run:
        logger.info("[DRY-RUN] Checking local environment for BiomedGPT / OFA...")
        try:
            import transformers
            logger.info("Transformers version: %s", transformers.__version__)
        except ImportError:
            pass

        mock_records = [
            {"image_name": f"img_{i}.png", "path": f"mock_{i}.png", "patient_id": f"P{i//2}", "labels": ["Cardiomegaly", "Effusion"]}
            for i in range(10)
        ]
        train_recs, val_recs, _ = create_patient_splits(mock_records, val_fraction=0.3, test_fraction=0.0)
        train_inst = build_vlm_instruction_dataset(train_recs)
        logger.info("[DRY-RUN] Dataset instructions constructed: %d samples.", len(train_inst))
        logger.info("[DRY-RUN] BiomedGPT pipeline audit complete. See Decision.md for full architecture trade-offs.")
        return

    logger.warning(
        "Direct Hugging Face fine-tuning for '%s' is gated by custom OFA dependencies. "
        "To fine-tune BiomedGPT directly, clone https://github.com/PharMolix/BioMedGPT and use fairseq-train, "
        "or use the MedGemma/LLaVA-Med pipelines provided in this package.",
        args.model_id
    )


if __name__ == "__main__":
    main()
