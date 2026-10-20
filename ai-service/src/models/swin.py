import logging
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NIHSwin(nn.Module):
    """
    Swin Transformer Tiny classifier for NIH ChestX-ray14.
    Architecture: swin_tiny_patch4_window7_224 (timm).
    Exposes final stage feature maps before pooling for transformer Grad-CAM.
    Feature map shape: (B, 7, 7, 768).
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Grad-CAM hook storage
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def save_gradient(self, target_idx: int):
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes)
            fmaps:  (B, 7, 7, 768) - raw spatial features from last Swin stage
        """
        fmaps = self.model.forward_features(x)
        self._feature_maps = fmaps
        logits = self.model.forward_head(fmaps)
        return logits, fmaps


def build_swin_tiny(num_classes: int = 14, pretrained: bool = True) -> NIHSwin:
    """Builds Swin-T classifier for NIH ChestX-ray14."""
    return NIHSwin(num_classes=num_classes, pretrained=pretrained)


__all__ = ["NIHSwin", "build_swin_tiny"]
