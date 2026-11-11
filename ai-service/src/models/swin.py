"""
Swin Transformer Multi-Disease Classifier for NIH ChestX-ray14.
Architecture reference: https://github.com/rajaatreja/SwinCheX_Classifier
Supports swin_large_patch4_window7_224 (SwinCheX default), swin_base_patch4_window7_224,
and swin_tiny_patch4_window7_224 with Grad-CAM and Grad-CAM++ hooks.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.densenet_multilabel import NIH_LABELS

logger = logging.getLogger(__name__)

SWINCHEX_DEFAULT_ARCH = "swin_large_patch4_window7_224"
DEFAULT_SWIN_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "retrained" / "swin"

# ImageNet normalization parameters
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _prepare_swin_input(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Prepares image tensor for Swin Transformer (RGB, 224x224, ImageNet normalized)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    # Replicate single-channel grayscale to 3 channels
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    # Resize if not 224x224
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    # Normalize if values appear to be in [0, 1] raw pixel range
    if x.min() >= -0.05 and x.max() <= 1.05:
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std

    return x


def auto_detect_swin_arch_from_state_dict(state_dict: Dict[str, Any]) -> str:
    """Infers Swin Transformer variant (large, base, tiny) from checkpoint tensor shapes."""
    for k, v in state_dict.items():
        if "layers.3" in k and hasattr(v, "shape") and len(v.shape) == 1:
            dim = v.shape[0]
            if dim == 1536:
                return "swin_large_patch4_window7_224"
            elif dim == 1024:
                return "swin_base_patch4_window7_224"
            elif dim == 768:
                return "swin_tiny_patch4_window7_224"
    return SWINCHEX_DEFAULT_ARCH


class NIHSwin(nn.Module):
    """
    Swin Transformer classifier for NIH ChestX-ray14.
    Based on SwinCheX (https://github.com/rajaatreja/SwinCheX_Classifier).
    Exposes final stage feature maps before pooling for transformer Grad-CAM.
    """

    def __init__(
        self,
        num_classes: int = 15,
        architecture: str = SWINCHEX_DEFAULT_ARCH,
        pretrained: bool = True,
        weights_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        self.weights_path: Optional[str] = None
        self._raw_num_classes = 14  # SwinCheX heads predict 14 pathologies

        # Locate candidate checkpoint
        ckpt_path = self._find_weights(weights_path)

        state_dict = None
        if ckpt_path and Path(ckpt_path).is_file():
            state_dict = self._load_raw_checkpoint(ckpt_path)
            if state_dict is not None:
                detected_arch = auto_detect_swin_arch_from_state_dict(state_dict)
                if detected_arch != self.architecture:
                    logger.info("Auto-detected Swin architecture '%s' from checkpoint", detected_arch)
                    self.architecture = detected_arch

        # Determine head size: 14 if training on 14 NIH diseases, 15 if trained with No_Finding
        head_classes = 14
        if state_dict is not None:
            for k in ("head.weight", "model.head.weight", "classifier.weight"):
                if k in state_dict:
                    head_classes = state_dict[k].shape[0]
                    break
        elif num_classes == 15:
            head_classes = 14

        self.head_classes = head_classes
        logger.info("Initializing Swin backbone: %s (head_classes=%d)", self.architecture, self.head_classes)

        try:
            self.model = timm.create_model(
                self.architecture,
                pretrained=pretrained if state_dict is None else False,
                num_classes=self.head_classes,
            )
        except Exception as e:
            logger.warning("Failed to create %s (%s), falling back to swin_base_patch4_window7_224", self.architecture, e)
            self.architecture = "swin_base_patch4_window7_224"
            self.model = timm.create_model(
                self.architecture,
                pretrained=pretrained if state_dict is None else False,
                num_classes=self.head_classes,
            )

        if state_dict is not None:
            self._apply_state_dict(state_dict)
            self.weights_path = str(ckpt_path)

        # Grad-CAM hook storage
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def _find_weights(self, weights_path: Optional[str | Path]) -> Optional[Path]:
        if weights_path and Path(weights_path).is_file():
            return Path(weights_path)

        # Search standard output directories
        candidates = [
            DEFAULT_SWIN_DIR / "pretrained_SwinCheX_classifier.pth",
            DEFAULT_SWIN_DIR / "swin_base_cxr14.pth",
            DEFAULT_SWIN_DIR / "best_model.pth",
            Path("outputs/retrained/swin_tiny/best_model.pth"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _load_raw_checkpoint(self, path: Path) -> Optional[Dict[str, Any]]:
        logger.info("Loading Swin weights from: %s", path)
        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Could not read checkpoint from %s: %s", path, e)
            return None

        if isinstance(raw, dict):
            if "model" in raw:
                inner = raw["model"]
                return inner.state_dict() if isinstance(inner, nn.Module) else inner
            if "state_dict" in raw:
                return raw["state_dict"]
            if "model_state_dict" in raw:
                return raw["model_state_dict"]
            return raw
        elif isinstance(raw, nn.Module):
            return raw.state_dict()
        return None

    def _apply_state_dict(self, state_dict: Dict[str, Any]) -> None:
        clean_sd = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in ("module.", "model.", "net."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            clean_sd[clean_k] = v

        model_sd = self.model.state_dict()
        filtered_sd = {k: v for k, v in clean_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        missing, unexpected = self.model.load_state_dict(filtered_sd, strict=False)
        logger.info(
            "Loaded Swin state dict: matched=%d keys, missing=%d keys, unexpected=%d keys",
            len(filtered_sd), len(missing), len(unexpected)
        )

    def save_gradient(self, target_idx: int):
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            scores: (B, num_classes) probabilities (with No_Finding complement if num_classes=15)
            fmaps:  (B, 7, 7, C) spatial feature maps from last Swin stage
        """
        x = _prepare_swin_input(x, 224)
        fmaps = self.model.forward_features(x)
        self._feature_maps = fmaps
        logits = self.model.forward_head(fmaps)
        probs = torch.sigmoid(logits)

        # If model outputs 14 diseases and service expects 15 classes, compute No_Finding complement
        if self.num_classes == 15 and probs.shape[-1] == 14:
            no_finding = (1.0 - probs).clamp(0, 1).prod(dim=1, keepdim=True)
            scores = torch.cat((probs, no_finding), dim=1)
        else:
            scores = probs

        return scores, fmaps

    def predict_with_gradients(self, image_tensor: torch.Tensor, target_indices: List[int]):
        """Runs forward pass with autograd hook to capture target class gradients on feature maps."""
        device = next(self.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        self._gradients.clear()
        with torch.enable_grad():
            scores, feature_maps = self.forward(image_tensor)
            feature_maps.retain_grad()
        for target_idx in target_indices:
            handle = feature_maps.register_hook(self.save_gradient(target_idx))
            self.zero_grad()
            scores[0, target_idx].backward(retain_graph=True)
            handle.remove()
        return scores.detach(), feature_maps.detach(), self._gradients


def build_swin(
    num_classes: int = 15,
    architecture: str = SWINCHEX_DEFAULT_ARCH,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
) -> NIHSwin:
    """Builds Swin Transformer classifier for NIH ChestX-ray14."""
    return NIHSwin(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained,
        weights_path=weights_path,
    )


def build_swin_tiny(num_classes: int = 15, pretrained: bool = True, weights_path: Optional[str | Path] = None) -> NIHSwin:
    """Backward-compatible builder for Swin-Tiny."""
    return build_swin(
        num_classes=num_classes,
        architecture="swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        weights_path=weights_path,
    )


__all__ = [
    "NIHSwin",
    "build_swin",
    "build_swin_tiny",
    "SWINCHEX_DEFAULT_ARCH",
    "_prepare_swin_input",
]
