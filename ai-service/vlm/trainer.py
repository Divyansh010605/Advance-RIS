"""
VLM Fine-Tuning Trainer: Supervised Fine-Tuning loop for vision-language models.
Supports MedGemma, LLaVA-Med, and BiomedGPT.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm.trainer")

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
        "vram_16bit": "~24 GB",
    },
    "biomedgpt": {
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased",
        "output_dir": "outputs/vlm/biomedgpt_lora",
        "lora_targets": ["query", "key", "value"],
        "model_class": "AutoModelForSeq2SeqLM",
        "processor_class": "AutoTokenizer",
        "gated": False,
        "vram_4bit": "~4 GB",
        "vram_16bit": "~8 GB",
    },
}


@dataclass
class VLMTrainingArgs:
    vlm: str = "medgemma"
    csv_path: str = "data/Data_Entry_2017_v2020.csv"
    image_dir: str = "data/images/"
    output_dir: Optional[str] = None
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False
    val_split: float = 0.15
    seed: int = 42
    max_length: int = 512
    gradient_accumulation_steps: int = 4


__all__ = ["VLMTrainingArgs", "VLM_CONFIGS"]
