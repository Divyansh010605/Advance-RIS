"""
DenseNet-121 Multi-Label Model for NIH ChestX-ray14.
TorchXRayVision DenseNet-121 inference adapter and model manager.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XRV_DENSENET_WEIGHTS = "densenet121-res224-all"
DEFAULT_DENSENET_DIR = Path(__file__).resolve().parent.parent / "outputs" / "retrained" / "densenet121"
DEFAULT_DENSENET_CKPT = DEFAULT_DENSENET_DIR / "densenet121_cxr.pth"

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)

CONFIG_DIR = os.getenv("CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "configs"))
CLASSES_FILE = os.path.join(CONFIG_DIR, "classes.json")
THRESHOLDS_FILE = os.path.join(CONFIG_DIR, "thresholds.json")


def _to_xrv_input(x: torch.Tensor, size: int) -> torch.Tensor:
    """Convert the service's ImageNet-normalized RGB tensor to XRV grayscale [-1024, 1024]."""
    mean = x.new_tensor(_RGB_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_RGB_STD).view(1, 3, 1, 1)
    rgb = (x * std + mean).clamp(0, 1)
    gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    gray = gray * 2048.0 - 1024.0
    if gray.shape[-2:] != (size, size):
        gray = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)
    return gray


class NIHDenseNet(nn.Module):
    """XRV DenseNet wrapper supporting retrained weights and preserving (scores, feature_maps) API."""

    def __init__(self, num_classes: int = 15, pretrained: bool = True, weights_path: Optional[str | Path] = None):
        super().__init__()
        if num_classes not in (14, 15):
            raise ValueError(f"Expected 14 or 15 NIH outputs, got {num_classes}")
        self.num_classes = num_classes
        self.xrv_probabilities = True

        ckpt_file = None
        if weights_path:
            p = Path(weights_path)
            if p.is_file():
                ckpt_file = p
        if ckpt_file is None:
            for candidate in [
                DEFAULT_DENSENET_CKPT,
                DEFAULT_DENSENET_DIR / "nih_densenet121.pth",
            ]:
                if candidate.is_file():
                    ckpt_file = candidate
                    break

        if ckpt_file and ckpt_file.is_file():
            logger.info("Loading retrained DenseNet-121 weights from: %s", ckpt_file)
            self.xrv_model = xrv.models.DenseNet(weights=None, op_threshs=torch.empty(18), apply_sigmoid=False)
            self.xrv_model.pathologies = xrv.models.model_urls[XRV_DENSENET_WEIGHTS]["labels"]
            state_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.xrv_model.load_state_dict(state_dict)
            self.weights_path = str(ckpt_file)
        else:
            logger.info("Using default TorchXRayVision DenseNet-121 weights: %s", XRV_DENSENET_WEIGHTS)
            self.xrv_model = xrv.models.DenseNet(weights=XRV_DENSENET_WEIGHTS)
            self.weights_path = XRV_DENSENET_WEIGHTS

        self.features = self.xrv_model.features
        self.classifier = self.xrv_model.classifier
        self.pathology_indices = [self.xrv_model.pathologies.index(label) for label in NIH_LABELS[:-1]]
        self._gradients: Dict[int, torch.Tensor] = {}
        self._feature_maps: torch.Tensor | None = None

    def save_gradient(self, target_idx: int):
        def hook(grad: torch.Tensor):
            self._gradients[target_idx] = grad.detach()
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _to_xrv_input(x, 224)
        f = F.relu(self.features(x), inplace=False)
        self._feature_maps = f
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        raw = torch.sigmoid(self.classifier(pooled))
        scores = xrv.models.op_norm(raw, self.xrv_model.op_threshs)
        selected = scores[:, self.pathology_indices]
        if self.num_classes == 15:
            no_finding = (1.0 - selected).clamp(0, 1).prod(dim=1, keepdim=True)
            output = torch.cat([selected, no_finding], dim=1)
        else:
            output = selected
        return output, f

    def forward_with_gradients(self, x: torch.Tensor, target_class_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _to_xrv_input(x, 224)
        self.zero_grad()
        f = F.relu(self.features(x), inplace=False)
        self._feature_maps = f
        if f.requires_grad:
            f.register_hook(self.save_gradient(target_class_idx))
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        raw = torch.sigmoid(self.classifier(pooled))
        scores = xrv.models.op_norm(raw, self.xrv_model.op_threshs)
        selected = scores[:, self.pathology_indices]
        if self.num_classes == 15:
            no_finding = (1.0 - selected).clamp(0, 1).prod(dim=1, keepdim=True)
            output = torch.cat([selected, no_finding], dim=1)
        else:
            output = selected

        target_score = output[0, target_class_idx]
        target_score.backward(retain_graph=True)
        return output, f.detach(), self._gradients.get(target_class_idx, torch.zeros_like(f))


class MultiDiseaseModel:
    """Wrapper around NIHDenseNet matching the legacy prediction API."""

    def __init__(self, device: torch.device = DEVICE, weights_path: Optional[str | Path] = None):
        self.device = device
        self.model = NIHDenseNet(num_classes=15, pretrained=True, weights_path=weights_path).to(device)
        self.model.eval()

    def predict_multi(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            scores, feat_maps = self.model(image_tensor)
        return scores.cpu(), feat_maps.cpu()

    def predict_multi_with_gradients(
        self,
        image_tensor: torch.Tensor,
        target_class_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        image_tensor = image_tensor.to(self.device)
        grad_dict: Dict[int, torch.Tensor] = {}
        for idx in target_class_indices:
            scores, feat_maps, grad = self.model.forward_with_gradients(image_tensor, idx)
            grad_dict[idx] = grad.cpu()
        return scores.cpu(), feat_maps.cpu(), grad_dict


class TBModel:
    """
    Unified manager serving the NIH ChestX-ray14 DenseNet-121 model.
    Provides multi-label screening and backward-compatible gradient methods.
    """

    def __init__(self):
        self.device = DEVICE
        self.classes: List[str] = self._load_classes()
        self.thresholds: Dict[str, float] = self._load_thresholds()
        self.num_classes = len(self.classes)
        self._model: Optional[MultiDiseaseModel] = None

    def _load_classes(self) -> List[str]:
        if os.path.exists(CLASSES_FILE):
            with open(CLASSES_FILE, "r") as f:
                return json.load(f).get("classes", NIH_LABELS)
        return NIH_LABELS

    def _load_thresholds(self) -> Dict[str, float]:
        if os.path.exists(THRESHOLDS_FILE):
            with open(THRESHOLDS_FILE, "r") as f:
                data = json.load(f)
                return data.get("thresholds", data)
        return {c: 0.50 for c in NIH_LABELS}

    def load(self, weights_path: Optional[str | Path] = None):
        logger.info("Loading NIHDenseNet for multi-disease screening...")
        self._model = MultiDiseaseModel(device=self.device, weights_path=weights_path)
        logger.info("NIHDenseNet ready on %s.", self.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def net(self) -> Optional[NIHDenseNet]:
        return self._model.model if self._model else None

    def predict_multi(self, image_tensor: torch.Tensor) -> Tuple[List[Dict[str, Any]], Any, Dict[int, Any]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        scores_t, feat_maps = self._model.predict_multi(image_tensor)
        scores = scores_t[0].numpy()
        results = []
        target_indices = []
        for i, disease in enumerate(self.classes):
            prob = float(scores[i])
            thresh = float(self.thresholds.get(disease, 0.50))
            detected = prob >= thresh
            if detected and i < 14:
                target_indices.append(i)
            results.append({
                "disease": disease,
                "probability": round(prob, 4),
                "threshold": thresh,
                "detected": detected,
            })
        if not target_indices:
            top_idx = int(scores[:14].argmax())
            target_indices.append(top_idx)

        _, _, grad_dict = self._model.predict_multi_with_gradients(image_tensor, target_indices)
        return results, feat_maps.numpy(), {k: v.numpy() for k, v in grad_dict.items()}

    def predict_with_gradients(self, image_tensor: torch.Tensor) -> Tuple[float, Any, Any]:
        """Legacy single-score interface."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        scores_t, feat_maps, grad_dict = self._model.predict_multi_with_gradients(image_tensor, [0])
        score = float(scores_t[0, 0])
        return score, feat_maps.numpy(), grad_dict[0].numpy()


def build_densenet121(
    num_classes: int = 15,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
) -> NIHDenseNet:
    """Builds DenseNet-121 classifier for NIH ChestX-ray14."""
    return NIHDenseNet(num_classes=num_classes, pretrained=pretrained, weights_path=weights_path)


__all__ = [
    "NIHDenseNet",
    "MultiDiseaseModel",
    "TBModel",
    "build_densenet121",
    "NIH_LABELS",
    "_to_xrv_input",
    "_RGB_MEAN",
    "_RGB_STD",
]
