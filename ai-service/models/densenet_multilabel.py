"""TorchXRayVision DenseNet-121 inference adapter for NIH label order."""
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XRV_DENSENET_WEIGHTS = "densenet121-res224-all"
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)


def _to_xrv_input(x: torch.Tensor, size: int) -> torch.Tensor:
    """Convert the service's ImageNet-normalized RGB tensor to XRV grayscale."""
    mean = x.new_tensor(_RGB_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_RGB_STD).view(1, 3, 1, 1)
    rgb = (x * std + mean).clamp(0, 1)
    gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    gray = gray * 2048.0 - 1024.0
    if gray.shape[-2:] != (size, size):
        gray = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)
    return gray


class NIHDenseNet(nn.Module):
    """XRV DenseNet wrapper preserving the service's `(scores, feature_maps)` API."""
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super().__init__()
        if num_classes != len(NIH_LABELS):
            raise ValueError(f"Expected {len(NIH_LABELS)} NIH outputs, got {num_classes}")
        self.num_classes = num_classes
        # This service is configured to use the requested pretrained XRV weights.
        self.xrv_model = xrv.models.DenseNet(weights=XRV_DENSENET_WEIGHTS, apply_sigmoid=False)
        self.xrv_probabilities = True
        self.features = self.xrv_model.features
        self.classifier = self.xrv_model.classifier
        self.pathology_indices = [self.xrv_model.pathologies.index(label) for label in NIH_LABELS[:-1]]
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def save_gradient(self, target_idx: int):
        return lambda grad: self._gradients.__setitem__(target_idx, grad.detach())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _to_xrv_input(x, 224)
        fmaps = F.relu(self.features(x), inplace=False)
        self._feature_maps = fmaps
        pooled = F.adaptive_avg_pool2d(fmaps, (1, 1)).flatten(1)
        raw = torch.sigmoid(self.classifier(pooled))
        # Match TorchXRayVision's normal forward path, including its trained thresholds.
        scores = xrv.models.op_norm(raw, self.xrv_model.op_threshs)
        selected = scores[:, self.pathology_indices]
        # TorchXRayVision does not train a "No Finding" output; expose a derived complement.
        no_finding = (1.0 - selected).clamp(0, 1).prod(dim=1, keepdim=True)
        return torch.cat((selected, no_finding), dim=1), fmaps


class MultiDiseaseModel:
    """Load XRV pretrained weights and expose inference/Grad-CAM helpers."""
    def __init__(self, num_classes: int, model_path: str = None, pretrained: bool = True):
        self.device = DEVICE
        self.num_classes = num_classes
        self.model_path = model_path
        self.pretrained = pretrained
        self.net: NIHDenseNet | None = None

    def load(self):
        self.net = NIHDenseNet(self.num_classes, pretrained=self.pretrained).to(self.device).eval()
        if self.model_path:
            logger.warning("Ignoring incompatible local checkpoint %s; using TorchXRayVision %s", self.model_path, XRV_DENSENET_WEIGHTS)

    def predict_with_gradients(self, image_tensor: torch.Tensor, target_indices: List[int]):
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        self.net._gradients.clear()
        with torch.enable_grad():
            scores, feature_maps = self.net(image_tensor)
            feature_maps.retain_grad()
        for target_idx in target_indices:
            handle = feature_maps.register_hook(self.net.save_gradient(target_idx))
            self.net.zero_grad()
            scores[0, target_idx].backward(retain_graph=True)
            handle.remove()
        return scores.detach(), feature_maps.detach(), self.net._gradients
