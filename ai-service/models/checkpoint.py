import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def remap_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model_keys: Set[str],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Safely remaps state_dict keys to match model keys if prefixes differ.
    Supported prefixes: 'module.', 'net.', 'model.'
    All modifications are explicitly logged.
    """
    remapped = {}
    log_messages = []

    # Check if exact match
    loaded_keys = set(state_dict.keys())
    if loaded_keys == model_keys:
        return state_dict, log_messages

    for key, val in state_dict.items():
        new_key = key
        if new_key not in model_keys:
            for prefix in ("module.", "net.", "model."):
                if new_key.startswith(prefix) and new_key[len(prefix):] in model_keys:
                    new_key = new_key[len(prefix):]
                    log_messages.append(f"Remapped prefix '{prefix}': {key} -> {new_key}")
                    break
        remapped[new_key] = val

    return remapped, log_messages


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads checkpoint into model with explicit key validation.
    Reports mismatches to stdout/logging instead of silently ignoring.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading checkpoint from: %s", ckpt_path)
    raw_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    metadata: Dict[str, Any] = {}
    if isinstance(raw_ckpt, dict):
        metadata = {k: v for k, v in raw_ckpt.items() if k not in ("model_state_dict", "state_dict", "optimizer_state_dict")}
        state_dict = raw_ckpt.get("model_state_dict") or raw_ckpt.get("state_dict") or raw_ckpt
    else:
        state_dict = raw_ckpt

    model_keys = set(model.state_dict().keys())
    remapped_state_dict, remap_logs = remap_state_dict_keys(state_dict, model_keys)

    for msg in remap_logs:
        logger.info(msg)

    # Detect discrepancies
    loaded_keys = set(remapped_state_dict.keys())
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys

    if missing_keys or unexpected_keys:
        msg = [
            f"Checkpoint key mismatch for {ckpt_path}:",
            f"  Missing keys ({len(missing_keys)}): {sorted(list(missing_keys))[:10]}...",
            f"  Unexpected keys ({len(unexpected_keys)}): {sorted(list(unexpected_keys))[:10]}...",
        ]
        report = "\n".join(msg)
        logger.warning(report)
        print(f"[MISMATCH WARNING] {report}")

        if strict:
            raise RuntimeError(f"Strict load failed for {ckpt_path} due to key discrepancies:\n{report}")

    model.load_state_dict(remapped_state_dict, strict=strict)
    logger.info("Successfully loaded checkpoint into %s (strict=%s)", model.__class__.__name__, strict)
    return model, metadata
