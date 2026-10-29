"""
Multi-model Ensemble & Test-Time Augmentation (TTA) with Uncertainty Quantification.
Supports DenseNet-121, ResNet-50, and Swin-T multi-label classifiers for NIH ChestX-ray14.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import NIH_CLASSES
from src.models.checkpoint import load_checkpoint
from src.models.densenet import build_densenet121
from src.models.resnet import NIHResNet50
from src.models.swin import build_swin_tiny

logger = logging.getLogger("src.models.ensemble")


class TestTimeAugmentation:
    """
    Applies clinical test-time augmentations (TTA) to input tensors:
      1. Identity (original scan)
      2. Horizontal Flip (anatomical bilateral symmetry)
      3. Subtle Scale/Crop (95% center crop upscaled to 224x224)
      4. Subtle Intensity Shift (±5% contrast shift)
    """

    @staticmethod
    def generate_augmented_tensors(x: torch.Tensor, steps: int = 2) -> List[Tuple[torch.Tensor, str]]:
        """
        Returns list of (augmented_tensor, transform_name).
        steps=1: [Identity]
        steps=2: [Identity, HFlip]
        steps=3: [Identity, HFlip, CenterScale]
        steps=4: [Identity, HFlip, CenterScale, ContrastShift]
        """
        B, C, H, W = x.shape
        transforms: List[Tuple[torch.Tensor, str]] = [(x, "identity")]

        if steps >= 2:
            hflip = torch.flip(x, dims=[-1])
            transforms.append((hflip, "horizontal_flip"))

        if steps >= 3:
            # 95% center crop resized back to original
            crop_h, crop_w = int(H * 0.95), int(W * 0.95)
            start_y, start_x = (H - crop_h) // 2, (W - crop_w) // 2
            cropped = x[:, :, start_y : start_y + crop_h, start_x : start_x + crop_w]
            scaled = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
            transforms.append((scaled, "center_crop_scale"))

        if steps >= 4:
            # Subtle contrast shift
            contrast = torch.clamp(x * 1.05, 0.0, 1.0)
            transforms.append((contrast, "contrast_scale"))

        return transforms[:steps]


class MultiModelEnsemble:
    """
    Orchestrates weighted soft-voting across registered DenseNet-121, ResNet-50, and Swin-T models
    with Test-Time Augmentation (TTA) and Monte-Carlo/ensemble uncertainty estimation.
    """

    def __init__(
        self,
        num_classes: int = 15,
        classes: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.classes = classes or NIH_CLASSES
        self.num_classes = len(self.classes) if classes is not None else num_classes
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Default model weighting based on validation macro-AUROC hierarchy
        # Raw defaults are normalized over whichever models are registered.
        self.raw_weights = weights or {
            "densenet121": 0.45,
            "swin_tiny": 0.30,
            "resnet50": 0.25,
        }

        self.models: Dict[str, nn.Module] = {}
        self.normalized_weights: Dict[str, float] = {}

    def register_model(self, name: str, model: nn.Module) -> None:
        """Registers a model backbone into the ensemble."""
        model.eval()
        model.to(self.device)
        self.models[name] = model
        self._recompute_weights()
        logger.info("Registered model '%s' into ensemble. Active models: %s", name, list(self.models.keys()))

    def load_from_checkpoints(
        self,
        densenet_path: Optional[str] = None,
        resnet_path: Optional[str] = None,
        swin_path: Optional[str] = None,
    ) -> None:
        """
        Loads models from provided checkpoint paths.
        Gracefully loads whatever is available.
        """
        # 1. DenseNet-121
        if densenet_path and Path(densenet_path).is_file():
            try:
                dense = build_densenet121(num_classes=self.num_classes, pretrained=False)
                load_checkpoint(dense, densenet_path, map_location=str(self.device))
                self.register_model("densenet121", dense)
            except Exception as e:
                logger.warning("Failed to load DenseNet checkpoint from %s: %s", densenet_path, e)

        # 2. ResNet-50
        if resnet_path and Path(resnet_path).is_file():
            try:
                res = NIHResNet50(num_classes=self.num_classes, pretrained=False)
                load_checkpoint(res, resnet_path, map_location=str(self.device))
                self.register_model("resnet50", res)
            except Exception as e:
                logger.warning("Failed to load ResNet checkpoint from %s: %s", resnet_path, e)

        # 3. Swin-T
        if swin_path and Path(swin_path).is_file():
            try:
                swin = build_swin_tiny(num_classes=self.num_classes, pretrained=False)
                load_checkpoint(swin, swin_path, map_location=str(self.device))
                self.register_model("swin_tiny", swin)
            except Exception as e:
                logger.warning("Failed to load Swin-T checkpoint from %s: %s", swin_path, e)

    def _recompute_weights(self) -> None:
        """Re-normalizes weights so active registered models sum to 1.0."""
        active_names = list(self.models.keys())
        if not active_names:
            self.normalized_weights = {}
            return

        total_weight = sum(self.raw_weights.get(name, 1.0) for name in active_names)
        self.normalized_weights = {
            name: self.raw_weights.get(name, 1.0) / total_weight
            for name in active_names
        }

    def predict(
        self,
        x: torch.Tensor,
        use_tta: bool = True,
        tta_steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Performs ensemble forward pass with TTA and uncertainty quantification.

        Returns:
            {
                "mean_probabilities": (B, num_classes) numpy array,
                "uncertainties": (B, num_classes) numpy array (standard deviation),
                "confidence_intervals": (B, num_classes, 2) [lower, upper] bounds,
                "is_uncertain": (B, num_classes) boolean array,
                "per_model_probabilities": {model_name: (B, num_classes)},
                "primary_feature_maps": tensor from primary model for Grad-CAM,
                "primary_model_name": name of primary model,
            }
        """
        if not self.models:
            raise RuntimeError("No models registered in MultiModelEnsemble.")

        x = x.to(self.device)
        B = x.shape[0]

        # Determine augmentations
        aug_list = TestTimeAugmentation.generate_augmented_tensors(x, steps=tta_steps if use_tta else 1)
        num_passes = len(self.models) * len(aug_list)

        # Storage for all passes to compute mean and variance: shape (num_passes, B, num_classes)
        all_pass_probs: List[torch.Tensor] = []
        per_model_probs: Dict[str, List[torch.Tensor]] = {name: [] for name in self.models}

        primary_fmaps: Optional[torch.Tensor] = None
        primary_model_name = "densenet121" if "densenet121" in self.models else list(self.models.keys())[0]

        with torch.no_grad():
            for aug_tensor, transform_name in aug_list:
                for model_name, model in self.models.items():
                    logits, fmaps = model(aug_tensor)
                    # TorchXRayVision adapters already return probability scores after
                    # applying XRV's operating-point normalization.
                    probs = logits if getattr(model, "xrv_probabilities", False) else torch.sigmoid(logits)

                    all_pass_probs.append(probs)
                    per_model_probs[model_name].append(probs)

                    # Capture feature maps from identity pass of primary model
                    if transform_name == "identity" and model_name == primary_model_name:
                        primary_fmaps = fmaps

        # Compute per-model average across TTA
        averaged_model_probs: Dict[str, torch.Tensor] = {}
        weighted_ensemble_prob = torch.zeros((B, self.num_classes), device=self.device)

        for model_name, probs_list in per_model_probs.items():
            model_mean = torch.stack(probs_list, dim=0).mean(dim=0)
            averaged_model_probs[model_name] = model_mean
            weight = self.normalized_weights[model_name]
            weighted_ensemble_prob += weight * model_mean

        # Stack all passes to compute predictive uncertainty (sample standard deviation)
        stacked_passes = torch.stack(all_pass_probs, dim=0)  # (num_passes, B, num_classes)

        if num_passes > 1:
            uncertainty = torch.std(stacked_passes, dim=0)  # (B, num_classes)
        else:
            # Single model, single pass -> uncertainty is estimated from distance to decision boundary
            # Lower confidence when probability is near 0.5
            dist_to_boundary = torch.abs(weighted_ensemble_prob - 0.5)
            uncertainty = torch.clamp((0.5 - dist_to_boundary) * 0.4, min=0.0)

        # 95% Confidence Interval (Normal approximation: mean ± 1.96 * std)
        ci_lower = torch.clamp(weighted_ensemble_prob - 1.96 * uncertainty, min=0.0, max=1.0)
        ci_upper = torch.clamp(weighted_ensemble_prob + 1.96 * uncertainty, min=0.0, max=1.0)
        ci_stack = torch.stack([ci_lower, ci_upper], dim=-1)  # (B, num_classes, 2)

        # Flag high-uncertainty detections
        # High uncertainty if std > 0.08 and probability in borderline zone [0.25, 0.75],
        # or extreme variance (std > 0.15)
        is_borderline = (weighted_ensemble_prob >= 0.25) & (weighted_ensemble_prob <= 0.75)
        is_uncertain = ((uncertainty >= 0.08) & is_borderline) | (uncertainty >= 0.15)

        return {
            "mean_probabilities": weighted_ensemble_prob.cpu().numpy(),
            "uncertainties": uncertainty.cpu().numpy(),
            "confidence_intervals": ci_stack.cpu().numpy(),
            "is_uncertain": is_uncertain.cpu().numpy(),
            "per_model_probabilities": {
                k: v.cpu().numpy() for k, v in averaged_model_probs.items()
            },
            "primary_feature_maps": primary_fmaps,
            "primary_model_name": primary_model_name,
            "tta_steps_evaluated": len(aug_list),
            "total_passes": num_passes,
        }

    def predict_multi_structured(
        self,
        x: torch.Tensor,
        thresholds: Dict[str, float],
        use_tta: bool = True,
        tta_steps: int = 2,
    ) -> Tuple[List[Dict[str, Any]], Optional[torch.Tensor], Dict[str, Any]]:
        """
        High-level helper for FastAPI /api/v1/analyze.
        Returns:
            (predictions_list, primary_feature_maps, metadata)
        """
        results = self.predict(x, use_tta=use_tta, tta_steps=tta_steps)
        probs = results["mean_probabilities"][0]
        stds = results["uncertainties"][0]
        cis = results["confidence_intervals"][0]
        uncertain_flags = results["is_uncertain"][0]

        predictions_list = []
        for i, disease in enumerate(self.classes):
            prob = float(probs[i])
            thresh = float(thresholds.get(disease, 0.5))
            uncertainty = float(stds[i])
            ci = [float(cis[i][0]), float(cis[i][1])]
            is_unc = bool(uncertain_flags[i])

            predictions_list.append({
                "disease": disease,
                "probability": round(prob, 4),
                "threshold": thresh,
                "detected": prob >= thresh,
                "uncertainty": round(uncertainty, 4),
                "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
                "is_uncertain": is_unc,
            })

        metadata = {
            "models_evaluated": list(self.models.keys()),
            "normalized_weights": self.normalized_weights,
            "tta_enabled": use_tta,
            "tta_steps": results["tta_steps_evaluated"],
            "total_passes": results["total_passes"],
            "mean_uncertainty": round(float(np.mean(stds)), 4),
        }

        return predictions_list, results["primary_feature_maps"], metadata
