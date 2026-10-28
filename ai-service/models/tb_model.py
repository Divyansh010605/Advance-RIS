"""
Unified Model Manager for NIH ChestX-ray14 Multi-Disease Chest X-Ray Screening.

Loads the DenseNet-121 model trained on NIH ChestX-ray14 (14 pathology classes)
and class-level detection thresholds.
"""
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import torch

from models.densenet_multilabel import MultiDiseaseModel

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
# Default model path points at the NIH trained checkpoint.
# Override at runtime via the MODEL_PATH environment variable.
_HERE = os.path.dirname(__file__)
_DEFAULT_MODEL = os.path.join(
    _HERE, "..", "test_nih", "saved_models", "nih_densenet121.pth"
)
MODEL_PATH   = os.getenv("MODEL_PATH", _DEFAULT_MODEL)
CONFIG_DIR   = os.getenv("CONFIG_DIR", os.path.join(_HERE, "..", "configs"))
CLASSES_FILE    = os.path.join(CONFIG_DIR, "classes.json")
THRESHOLDS_FILE = os.path.join(CONFIG_DIR, "thresholds.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TBModel:
    """
    Serves the NIH ChestX-ray14 DenseNet-121 model.

    Provides:
      - predict_multi()            – 14-class multi-label inference + Grad-CAM
      - predict_with_gradients()   – backward-compatible single-score method
    """

    def __init__(self):
        self.device = DEVICE
        self.classes: List[str]        = self._load_classes()
        self.thresholds: Dict[str, float] = self._load_thresholds()
        self.num_classes               = len(self.classes)
        self._model: MultiDiseaseModel | None = None

    # ── Config loaders ─────────────────────────────────────────────────────────
    def _load_classes(self) -> List[str]:
        if os.path.exists(CLASSES_FILE):
            with open(CLASSES_FILE, "r") as f:
                return json.load(f).get("classes", [])
        # Hardcoded NIH fallback so service still starts without configs/
        return [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia", "No_Finding",
        ]

    def _load_thresholds(self) -> Dict[str, float]:
        if os.path.exists(THRESHOLDS_FILE):
            with open(THRESHOLDS_FILE, "r") as f:
                return json.load(f)
        return {cls: 0.45 for cls in self.classes}

    # ── Model load ─────────────────────────────────────────────────────────────
    def load(self):
        logger.info(f"Loading NIH DenseNet-121 from: {MODEL_PATH}")
        self._model = MultiDiseaseModel(
            num_classes=self.num_classes,
            model_path=MODEL_PATH,
        )
        self._model.load()

    # ── net property (kept for Grad-CAM utilities that access model.net) ───────
    @property
    def net(self):
        return self._model.net if self._model else None

    # ── Multi-label inference ──────────────────────────────────────────────────
    def predict_multi(
        self,
        image_tensor: torch.Tensor,
        target_classes: List[str] = None,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Runs multi-label inference over all 14 NIH classes.

        Returns:
            predictions   – list of {disease, probability, threshold, detected}
            feature_maps  – Tensor (1, C_f, H_f, W_f)
            gradients_dict – {class_index: gradient tensor}
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        if target_classes is None:
            target_classes = self.classes

        target_indices = [
            self.classes.index(c) for c in target_classes if c in self.classes
        ]
        if not target_indices:
            target_indices = list(range(self.num_classes))

        probs, feature_maps, gradients_dict = self._model.predict_with_gradients(
            image_tensor, target_indices
        )
        probs_np = probs.squeeze(0).cpu().numpy()

        predictions: List[Dict[str, Any]] = []
        for idx, cls in enumerate(self.classes):
            prob   = float(probs_np[idx])
            thresh = float(self.thresholds.get(cls, 0.45))
            predictions.append({
                "disease":     cls,
                "probability": round(prob, 4),
                "threshold":   thresh,
                "detected":    bool(prob >= thresh),
            })

        return predictions, feature_maps, gradients_dict

    # ── Backward-compatible single prediction ──────────────────────────────────
    def predict_with_gradients(
        self, image_tensor: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Legacy endpoint: returns the probability of the highest-confidence
        detected class (or the top class overall if none exceed threshold).
        """
        predictions, feat, grad_dict = self.predict_multi(image_tensor)

        # Pick the highest-probability detected class, else overall top
        detected = [p for p in predictions if p["detected"]]
        top_pred = max(detected or predictions, key=lambda p: p["probability"])
        top_idx  = self.classes.index(top_pred["disease"])

        gradients = grad_dict.get(top_idx, torch.zeros_like(feat))
        return top_pred["probability"], feat, gradients
