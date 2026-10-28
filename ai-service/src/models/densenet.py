import torch
import torch.nn as nn
from models.densenet_multilabel import NIHDenseNet, MultiDiseaseModel


def build_densenet121(num_classes: int = 15, pretrained: bool = True) -> NIHDenseNet:
    """Builds DenseNet-121 classifier for NIH ChestX-ray14."""
    return NIHDenseNet(num_classes=num_classes, pretrained=pretrained)


__all__ = ["NIHDenseNet", "MultiDiseaseModel", "build_densenet121"]
