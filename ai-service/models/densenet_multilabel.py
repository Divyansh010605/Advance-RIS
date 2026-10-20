"""
DenseNet-121 Multi-Label Model for NIH ChestX-ray14.

Trained with plain torchvision DenseNet-121 (ImageNet-pretrained backbone,
replaced classifier head).  This module intentionally does NOT depend on
torchxrayvision so the saved checkpoint architecture matches exactly.
"""
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NIHDenseNet(nn.Module):
    """
    DenseNet-121 backbone with a custom linear classifier head.

    Architecture mirrors the training script in test_nih/train_nih.py:
      - torchvision densenet121 features
      - nn.Linear(1024, num_classes)  (no extra Dropout layer so weights load cleanly)

    Grad-CAM hooks capture gradients at the final feature map.
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        self.features = base.features          # DenseNet feature extractor
        in_features = base.classifier.in_features  # 1024

        # Plain linear head – identical to train_nih.py
        self.classifier = nn.Linear(in_features, num_classes)

        # Grad-CAM storage
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Grad-CAM helpers
    # ------------------------------------------------------------------
    def save_gradient(self, target_idx: int):
        """Returns a hook that stores the gradient for *target_idx*."""
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits      – (B, num_classes)
            feature_maps – (B, C_f, H_f, W_f)  before global pooling
        """
        fmaps = F.relu(self.features(x), inplace=True)
        self._feature_maps = fmaps

        pooled = F.adaptive_avg_pool2d(fmaps, (1, 1))
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits, fmaps


class MultiDiseaseModel:
    """High-level wrapper: load weights, run inference, expose Grad-CAM data."""

    def __init__(self, num_classes: int, model_path: str = None, pretrained: bool = False):
        self.device = DEVICE
        self.num_classes = num_classes
        self.model_path = model_path
        self.pretrained = pretrained
        self.net: NIHDenseNet | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self):
        self.net = NIHDenseNet(num_classes=self.num_classes, pretrained=self.pretrained).to(self.device)

        if self.model_path:
            import os
            if os.path.isfile(self.model_path):
                logger.info(f"Loading NIH DenseNet-121 weights from {self.model_path}")
                try:
                    checkpoint = torch.load(
                        self.model_path, map_location=self.device, weights_only=False
                    )
                    # Accept both raw state-dict and checkpoint dicts
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        state = checkpoint["model_state_dict"]
                    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        state = checkpoint["state_dict"]
                    else:
                        state = checkpoint
                    self.net.load_state_dict(state, strict=True)
                    logger.info("NIH model weights loaded successfully.")
                except Exception as exc:
                    logger.error(f"Weight loading failed: {exc}")
                    raise
            else:
                logger.warning(
                    f"Model file not found at {self.model_path}. "
                    "Running with random (untrained) weights."
                )

        self.net.eval()

    # ------------------------------------------------------------------
    # Inference with Grad-CAM gradients
    # ------------------------------------------------------------------
    def predict_with_gradients(
        self,
        image_tensor: torch.Tensor,
        target_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass + per-class backward passes for Grad-CAM.

        Args:
            image_tensor  – (1, 3, 224, 224)
            target_indices – class indices to generate gradients for

        Returns:
            probabilities  – (1, num_classes), detached
            feature_maps   – (1, C_f, H_f, W_f), detached
            gradients_dict – {target_idx: gradient tensor}
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        image_tensor = image_tensor.to(self.device)
        self.net._gradients.clear()

        with torch.enable_grad():
            feat = F.relu(self.net.features(image_tensor), inplace=False)
            feat.retain_grad()

            pooled = F.adaptive_avg_pool2d(feat, (1, 1))
            flat = torch.flatten(pooled, 1)
            logits = self.net.classifier(flat)
            probs = torch.sigmoid(logits)

        for target_idx in target_indices:
            hook_handle = feat.register_hook(self.net.save_gradient(target_idx))
            self.net.zero_grad()
            logits[0, target_idx].backward(retain_graph=True)
            hook_handle.remove()

        return probs.detach(), feat.detach(), self.net._gradients
