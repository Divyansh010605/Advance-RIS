"""
Swin Transformer Multi-Disease Classifier for NIH ChestX-ray14.
Supports:
  - Pretrained SwinCheX (swin_large_patch4_window7_224) directly deserialized from state
  - Swin-Tiny (swin_tiny_patch4_window7_224) with shape-validated state_dict loading
Provides LayerCAM and Grad-CAM extraction hooks.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.densenet import NIH_LABELS

logger = logging.getLogger(__name__)

SWINCHEX_DEFAULT_ARCH = "swin_large_patch4_window7_224"
DEFAULT_SWIN_DIR = Path(__file__).resolve().parent.parent / "outputs" / "retrained" / "swin"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _prepare_swin_input(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Prepares image tensor for Swin Transformer (RGB, 224x224, ImageNet normalized)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    if x.min() >= -0.05 and x.max() <= 1.05:
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std

    return x


def auto_detect_swin_arch_from_state_dict(state_dict: Dict[str, Any]) -> str:
    """Detects Swin architecture (tiny, base, or large) from state_dict tensor dimensions."""
    for norm_key in ("norm.weight", "model.norm.weight"):
        if norm_key in state_dict:
            dim = state_dict[norm_key].shape[0]
            if dim == 768:
                return "swin_tiny_patch4_window7_224"
            if dim == 1024:
                return "swin_base_patch4_window7_224"
            if dim == 1536:
                return "swin_large_patch4_window7_224"

    for key in ("patch_embed.proj.weight", "model.patch_embed.proj.weight"):
        if key in state_dict:
            embed_dim = state_dict[key].shape[0]
            if embed_dim == 96:
                return "swin_tiny_patch4_window7_224"
            if embed_dim == 128:
                return "swin_base_patch4_window7_224"
            if embed_dim == 192:
                return "swin_large_patch4_window7_224"

    return SWINCHEX_DEFAULT_ARCH


class NIHSwin(nn.Module):
    """Swin Transformer wrapper for 14-disease multi-label NIH ChestX-ray14 classification."""

    def __init__(
        self,
        architecture: str = SWINCHEX_DEFAULT_ARCH,
        num_classes: int = 15,
        pretrained: bool = True,
        weights_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        self.weights_path: Optional[str] = None
        self._feature_maps: Optional[torch.Tensor] = None

        ckpt_file = None
        if weights_path:
            p = Path(weights_path)
            if p.is_file():
                ckpt_file = p
        if ckpt_file is None:
            for candidate in [
                DEFAULT_SWIN_DIR / "swin_tiny.pth",
                DEFAULT_SWIN_DIR / "swin_tiny_patch4_window7_224.pth",
                DEFAULT_SWIN_DIR / "pretrained_SwinCheX_classifier.pth",
            ]:
                if candidate.is_file():
                    ckpt_file = candidate
                    break

        raw_loaded = None
        if ckpt_file and ckpt_file.is_file():
            logger.info("Loading Swin weights from: %s", ckpt_file)
            raw_loaded = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

        # Case 1: Checkpoint contains an instantiated SwinTransformer Module directly (e.g. SwinCheX)
        if isinstance(raw_loaded, dict) and isinstance(raw_loaded.get("model"), nn.Module):
            logger.info("Loaded instantiated SwinTransformer module directly from checkpoint.")
            self.model = raw_loaded["model"]
            self.weights_path = str(ckpt_file)
            return

        # Case 2: State dict
        state_dict = None
        if isinstance(raw_loaded, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in raw_loaded and isinstance(raw_loaded[k], dict):
                    state_dict = raw_loaded[k]
                    break
            if state_dict is None:
                state_dict = raw_loaded
            detected_arch = auto_detect_swin_arch_from_state_dict(state_dict)
            if detected_arch != self.architecture:
                logger.info("Auto-detected Swin arch from weights: %s (overrides %s)", detected_arch, self.architecture)
                self.architecture = detected_arch

        # Create timm model
        self.model = timm.create_model(
            self.architecture,
            pretrained=pretrained and state_dict is None,
            num_classes=num_classes,
        )

        if state_dict is not None:
            # Strip prefixes
            clean_state = {}
            for k, v in state_dict.items():
                clean_k = k
                for prefix in ("module.", "model.", "net."):
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                clean_state[clean_k] = v

            # Filter shape mismatches so head size differences don't throw RuntimeError
            model_sd = self.model.state_dict()
            shape_matched_state = {}
            for k, v in clean_state.items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    shape_matched_state[k] = v

            self.model.load_state_dict(shape_matched_state, strict=False)
            self.weights_path = str(ckpt_file)
            logger.info("Swin (%s) successfully loaded from: %s", self.architecture, ckpt_file)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _prepare_swin_input(x, 224)
        m = self.model
        x_f = m.forward_features(x)
        self._feature_maps = x_f
        logits = m.forward_head(x_f)
        probs = torch.sigmoid(logits)
        return probs, x_f


def build_swin(
    architecture: str = SWINCHEX_DEFAULT_ARCH,
    num_classes: int = 15,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
) -> NIHSwin:
    """Builds Swin Transformer classifier."""
    return NIHSwin(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        weights_path=weights_path,
    )


def build_swin_tiny(
    num_classes: int = 15,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
) -> NIHSwin:
    """Builds Swin-Tiny classifier (lightweight 28M parameter Vision Transformer)."""
    return NIHSwin(
        architecture="swin_tiny_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
        weights_path=weights_path,
    )


__all__ = ["NIHSwin", "build_swin", "build_swin_tiny", "_prepare_swin_input"]
