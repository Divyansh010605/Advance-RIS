"""
VLM Fine-Tuning Trainer: Shared SFT (Supervised Fine-Tuning) loop for all three VLMs.

Supports:
  - MedGemma (google/medgemma-4b)
  - LLaVA-Med (microsoft/llava-med-v1.5-mistral-7b)
  - BiomedGPT (microsoft/BiomedNLP-BiomedBERT-base-uncased / custom)

Uses HuggingFace Trainer with LoRA / QLoRA via PEFT.
Data format: JSONL from src.vlm.data_builder (VLMInstructionItem).

Usage (run from ai-service/):
  python -m src.vlm.trainer --vlm medgemma --csv_path /data/Data_Entry_2017_v2020.csv \\
         --image_dir /data/images/ --output_dir outputs/vlm/medgemma_lora

Hardware:
  --load_in_4bit  → ~8-10 GB VRAM (RTX 3070/4060 Ti)
  default 16-bit  → ~16-24 GB VRAM (RTX 4090 / A100)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure ai-service root on path
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.data.splits import create_patient_splits, parse_nih_csv
from src.vlm.data_builder import (
    VLMInstructionItem,
    build_vlm_instruction_dataset,
    save_vlm_dataset_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("src.vlm.trainer")


# ─── VLM Registry ────────────────────────────────────────────────────────────

VLM_CONFIGS = {
    "medgemma": {
        "model_id": "google/medgemma-4b",
        "output_dir": "outputs/vlm/medgemma_lora",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "gated": True,
        "vram_4bit": "~8 GB",
        "vram_16bit": "~16 GB",
    },
    "llava_med": {
        "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        "output_dir": "outputs/vlm/llava_med_lora",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "gated": False,
        "vram_4bit": "~10 GB",
        "vram_16bit": "~20 GB",
    },
    "biomedgpt": {
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "output_dir": "outputs/vlm/biomedgpt_lora",
        "lora_targets": ["query", "key", "value"],
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoTokenizer",
        "gated": False,
        "vram_4bit": "~4 GB",
        "vram_16bit": "~8 GB",
    },
}


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VLMInstructionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping VLMInstructionItem records.
    Tokenizes prompt+response pairs for causal LM training.
    """

    MAX_LENGTH = 512

    def __init__(
        self,
        items: List[VLMInstructionItem],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        # Format: "<prompt>\n<response><eos>"
        text = f"{item.prompt}\n{item.response}{self.tokenizer.eos_token or ''}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # For causal LM: labels = input_ids (shifted inside model)
        labels = input_ids.clone()
        # Mask padding tokens from loss
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─── Training Functions ───────────────────────────────────────────────────────

def build_bnb_config(load_in_4bit: bool) -> Optional[Any]:
    """Build BitsAndBytesConfig for QLoRA if bitsandbytes is available."""
    if not load_in_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        logger.warning("bitsandbytes not available — falling back to full precision.")
        return None


def build_lora_config(vlm_key: str, lora_r: int, lora_alpha: int) -> Any:
    """Build LoraConfig for the specified VLM."""
    from peft import LoraConfig, TaskType  # type: ignore

    targets = VLM_CONFIGS[vlm_key]["lora_targets"]
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(
    vlm_key: str,
    bnb_config: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Load the VLM model and its tokenizer/processor.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # type: ignore

    cfg = VLM_CONFIGS[vlm_key]
    model_id = cfg["model_id"]
    logger.info("Loading model: %s", model_id)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_cls_name = cfg["model_class"]
    if model_cls_name == "LlavaForConditionalGeneration":
        from transformers import LlavaForConditionalGeneration  # type: ignore
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    proc_cls_name = cfg["processor_class"]
    if proc_cls_name == "AutoTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Many processors have a tokenizer attribute
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_peft_model(model: Any, lora_config: Any, load_in_4bit: bool) -> Any:
    """Apply PEFT LoRA to the model."""
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def run_training(
    model: Any,
    tokenizer: Any,
    train_items: List[VLMInstructionItem],
    val_items: List[VLMInstructionItem],
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    max_length: int = 512,
) -> None:
    """Execute HuggingFace Trainer-based SFT loop."""
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq  # type: ignore

    train_ds = VLMInstructionDataset(train_items, tokenizer, max_length)
    val_ds = VLMInstructionDataset(val_items, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        eval_steps=100,
        save_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        group_by_length=True,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Starting training: %d train / %d val instruction items.", len(train_ds), len(val_ds))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to: %s", output_dir)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified VLM Fine-Tuning Trainer for NIH ChestX-ray14 instruction tuning."
    )
    parser.add_argument(
        "--vlm",
        type=str,
        choices=list(VLM_CONFIGS.keys()),
        default="medgemma",
        help="Which VLM to fine-tune.",
    )
    parser.add_argument("--model_id", type=str, default=None, help="Override default model ID.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Data_Entry_2017_v2020.csv")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory of NIH images.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to VLM config.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit NF4 QLoRA.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_records", type=int, default=None, help="Cap dataset size for fast testing.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data pipeline and model loading without running training.",
    )
    args = parser.parse_args()

    cfg = VLM_CONFIGS[args.vlm]
    model_id = args.model_id or cfg["model_id"]
    output_dir = args.output_dir or cfg["output_dir"]

    print("=" * 62)
    print(f" VLM Fine-Tuning Trainer — {args.vlm.upper()}")
    print(f" Model:        {model_id}")
    print(f" Quant:        {'4-bit NF4 (QLoRA)' if args.load_in_4bit else '16-bit (LoRA)'}")
    print(f" VRAM est:     {cfg['vram_4bit'] if args.load_in_4bit else cfg['vram_16bit']}")
    print(f" LoRA r/α:     {args.lora_r} / {args.lora_alpha}")
    print(f" Eff. Batch:   {args.batch_size * args.grad_accum_steps}")
    print(f" Epochs:       {args.epochs}")
    print(f" Output:       {output_dir}")
    if cfg.get("gated"):
        print(" ⚠  Gated model — ensure HF_TOKEN env variable is set and license accepted.")
    print("=" * 62)

    # ── Build data pipeline ───────────────────────────────────────────
    logger.info("Loading NIH CSV from %s ...", args.csv_path)
    records = parse_nih_csv(args.csv_path, image_dir=args.image_dir)
    if args.max_records:
        records = records[: args.max_records]
    logger.info("Total records: %d", len(records))

    train_recs, val_recs, _ = create_patient_splits(
        records, val_fraction=args.val_fraction, test_fraction=0.0
    )
    train_items = build_vlm_instruction_dataset(train_recs)
    val_items = build_vlm_instruction_dataset(val_recs)

    logger.info(
        "Instruction dataset: %d train / %d val items (3 tasks × records).",
        len(train_items), len(val_items),
    )

    # Save JSONL datasets for reproducibility
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_vlm_dataset_jsonl(train_items, Path(output_dir) / "train_instructions.jsonl")
    save_vlm_dataset_jsonl(val_items, Path(output_dir) / "val_instructions.jsonl")

    if args.dry_run:
        logger.info("[DRY-RUN] Data pipeline validated. Skipping model loading and training.")
        return

    # ── Load model ────────────────────────────────────────────────────
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        logger.error("Missing dependencies: %s — install with: pip install transformers peft accelerate bitsandbytes", e)
        sys.exit(1)

    if cfg.get("gated") and "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        logger.warning(
            "Gated model detected. Set HF_TOKEN or run `huggingface-cli login` "
            "with a token that has accepted the model's license terms."
        )

    bnb_config = build_bnb_config(args.load_in_4bit)
    lora_config = build_lora_config(args.vlm, args.lora_r, args.lora_alpha)

    model, tokenizer = load_model_and_tokenizer(args.vlm, bnb_config)
    model = prepare_peft_model(model, lora_config, args.load_in_4bit)

    # ── Run training ──────────────────────────────────────────────────
    run_training(
        model=model,
        tokenizer=tokenizer,
        train_items=train_items,
        val_items=val_items,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        max_length=args.max_length,
    )
    logger.info("Training complete. Weights saved to: %s", output_dir)


if __name__ == "__main__":
    main()
