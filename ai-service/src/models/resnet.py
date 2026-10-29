"""TorchXRayVision ResNet-50 adapter for the service ensemble."""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

from models.densenet_multilabel import NIH_LABELS, _to_xrv_input

XRV_RESNET_WEIGHTS = "resnet50-res512-all"


class NIHResNet50(nn.Module):
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super().__init__()
        if num_classes != len(NIH_LABELS):
            raise ValueError(f"Expected {len(NIH_LABELS)} NIH outputs, got {num_classes}")
        self.num_classes = num_classes
        self.xrv_probabilities = True
        # This service is configured to use the requested pretrained XRV weights.
        self.xrv_model = xrv.models.ResNet(weights=XRV_RESNET_WEIGHTS)
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
        no_finding = (1.0 - selected).clamp(0, 1).prod(dim=1, keepdim=True)
        return torch.cat((selected, no_finding), dim=1), fmaps


def build_resnet50(num_classes: int = 15, pretrained: bool = True) -> NIHResNet50:
    return NIHResNet50(num_classes=num_classes, pretrained=pretrained)


__all__ = ["NIHResNet50", "build_resnet50", "XRV_RESNET_WEIGHTS"]
