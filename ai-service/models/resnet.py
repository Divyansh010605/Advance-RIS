"""
ResNet-50 Multi-Disease Classifier for NIH ChestX-ray14 (512x512 resolution).
TorchXRayVision ResNet-50 adapter for the service ensemble.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

from models.densenet import NIH_LABELS, _to_xrv_input

logger = logging.getLogger(__name__)

XRV_RESNET_WEIGHTS = "resnet50-res512-all"
DEFAULT_RESNET_DIR = Path(__file__).resolve().parent.parent / "outputs" / "retrained" / "resnet50"
DEFAULT_RESNET_CKPT = DEFAULT_RESNET_DIR / "resnet50_cxr.pth"


class NIHResNet50(nn.Module):
    """TorchXRayVision ResNet-50 wrapper with LayerCAM activation extraction."""

    def __init__(self, num_classes: int = 15, pretrained: bool = True, weights_path: Optional[str | Path] = None):
        super().__init__()
        if num_classes not in (14, 15):
            raise ValueError(f"Expected 14 or 15 NIH outputs, got {num_classes}")
        self.num_classes = num_classes
        self.xrv_probabilities = True

        self.xrv_model = xrv.models.ResNet(weights=XRV_RESNET_WEIGHTS)
        self.weights_path = XRV_RESNET_WEIGHTS

        ckpt_file = None
        if weights_path:
            p = Path(weights_path)
            if p.is_file():
                ckpt_file = p
        elif pretrained:
            for candidate in [
                DEFAULT_RESNET_DIR / "resnet50-res512-all.pth",
                DEFAULT_RESNET_CKPT,
            ]:
                if candidate.is_file():
                    ckpt_file = candidate
                    break

        if ckpt_file and ckpt_file.is_file():
            logger.info("Loading retrained ResNet-50 weights from: %s", ckpt_file)
            state_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.xrv_model.load_state_dict(state_dict)
            self.weights_path = str(ckpt_file)

        self.model = self.xrv_model.model
        self.pathology_indices = [self.xrv_model.pathologies.index(label) for label in NIH_LABELS[:-1]]
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def save_gradient(self, target_idx: int):
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _to_xrv_input(x, 512)
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        fmaps = m.layer4(x)
        self._feature_maps = fmaps
        pooled = m.avgpool(fmaps).flatten(1)
        raw = torch.sigmoid(m.fc(pooled))
        scores = xrv.models.op_norm(raw, self.xrv_model.op_threshs)
        selected = scores[:, self.pathology_indices]
        if self.num_classes == 15:
            no_finding = (1.0 - selected).clamp(0, 1).prod(dim=1, keepdim=True)
            output = torch.cat([selected, no_finding], dim=1)
        else:
            output = selected
        return output, fmaps


def build_resnet50(
    num_classes: int = 15,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
) -> NIHResNet50:
    """Builds ResNet-50 (512x512) classifier for NIH ChestX-ray14."""
    return NIHResNet50(num_classes=num_classes, pretrained=pretrained, weights_path=weights_path)


__all__ = ["NIHResNet50", "build_resnet50", "XRV_RESNET_WEIGHTS"]
