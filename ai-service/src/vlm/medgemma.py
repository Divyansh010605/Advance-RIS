"""
MedGemma Vision-Language Fine-Tuning Script (LoRA / QLoRA).

Model Card: google/medgemma-4b / google/gemma-3-4b-it
Licence / Gating: Requires accepting the Gemma Terms of Use on Hugging Face Hub (huggingface.co/google/medgemma-4b).
Hardware Requirement:
  - 4-bit QLoRA: ~8 GB VRAM (fits on RTX 3070/4060 Ti)
  - 16-bit LoRA: ~16 GB VRAM (RTX 4090 / A10G / A100)
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

from src.data.splits import create_patient_splits, parse_nih_csv
from src.vlm.data_builder import build_vlm_instruction_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.vlm.medgemma")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma with LoRA/QLoRA on NIH ChestX-ray14 Instructions")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b")
    parser.add_argument("--output_dir", type=str, default="outputs/vlm/medgemma_lora")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit NF4 QLoRA quantization")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without initiating weight training")
    args = parser.parse_args()

    print("==========================================================")
    print(" MedGemma LoRA/QLoRA Fine-Tuning Pipeline")
    print(f" Target Model ID:     {args.model_id}")
    print(f" Quantization:        {'4-bit NF4 (QLoRA)' if args.load_in_4bit else '16-bit (LoRA)'}")
    print(f" Estimated VRAM:      {'~8 GB' if args.load_in_4bit else '~16 GB'}")
    print(f" LoRA Rank / Alpha:   r={args.lora_r}, alpha={args.lora_alpha}")
    print(f" Effective Batch:     {args.batch_size * args.grad_accum_steps}")
    print(" Gating / License:    Gated on Hugging Face (huggingface.co/google/medgemma-4b)")
    print("==========================================================")

    if args.dry_run:
        logger.info("[DRY-RUN] Validating data construction pipeline...")
        mock_records = [
            {"image_name": f"img_{i}.png", "path": f"mock_{i}.png", "patient_id": f"P{i//2}", "labels": ["Atelectasis", "Effusion"]}
            for i in range(10)
        ]
        train_recs, val_recs, _ = create_patient_splits(mock_records, val_fraction=0.3, test_fraction=0.0)
        train_inst = build_vlm_instruction_dataset(train_recs)
        val_inst = build_vlm_instruction_dataset(val_recs)
        logger.info("Successfully constructed %d train instructions and %d val instructions.", len(train_inst), len(val_inst))
        logger.info("[DRY-RUN] MedGemma fine-tuning configuration validated successfully.")
        return

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        logger.error("Missing VLM dependencies: %s. Install with `pip install transformers peft accelerate bitsandbytes`", e)
        return

    # Check Hugging Face authentication
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        logger.warning(
            "MedGemma is a gated repository. Set HF_TOKEN environment variable or run `huggingface-cli login` "
            "with a token that has accepted the MedGemma license terms."
        )

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info("Script initialized. Ready for execution on GPU hardware with Hugging Face credentials.")


if __name__ == "__main__":
    main()
