import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class NIHResNet50(nn.Module):
    """
    ResNet-50 multi-label classifier for NIH ChestX-ray14.
    Replaces final fc layer with nn.Linear(2048, num_classes).
    Grad-CAM hooks target layer4 (output channels: 2048, spatial: 7x7 at 224x224 input).
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        # Backbone layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

        # Grad-CAM hooks storage
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def save_gradient(self, target_idx: int):
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmaps = self.layer4(x)
        self._feature_maps = fmaps

        pooled = self.avgpool(fmaps)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits, fmaps


def build_resnet50(num_classes: int = 14, pretrained: bool = True) -> NIHResNet50:
    """Builds ResNet-50 classifier for NIH ChestX-ray14."""
    return NIHResNet50(num_classes=num_classes, pretrained=pretrained)


__all__ = ["NIHResNet50", "build_resnet50"]
