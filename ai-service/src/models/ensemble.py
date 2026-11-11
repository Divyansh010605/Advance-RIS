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
from src.models.resnet import build_resnet50, NIHResNet50
from src.models.swin import build_swin, NIHSwin

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
        if classes is not None:
            self.classes = classes
            self.num_classes = len(classes)
        else:
            self.num_classes = num_classes
            self.classes = NIH_CLASSES[:num_classes]
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Default model weighting based on validation macro-AUROC hierarchy
        # Raw defaults are normalized over whichever models are registered.
        self.raw_weights = weights or {
            "densenet121": 0.40,
            "swin": 0.35,
            "swin_tiny": 0.35,
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
        Loads models from provided checkpoint paths or default retrained paths.
        Gracefully loads whatever is available.
        """
        retrained_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "retrained"

        # 1. DenseNet-121
        if "densenet121" not in self.models:
            try:
                d_path = densenet_path or str(retrained_dir / "densenet121" / "densenet121_cxr.pth")
                dense = build_densenet121(
                    num_classes=self.num_classes,
                    pretrained=False,
                    weights_path=d_path if Path(d_path).is_file() else None,
                )
                self.register_model("densenet121", dense)
            except Exception as e:
                logger.warning("Failed to initialize DenseNet: %s", e)

        # 2. ResNet-50
        try:
            r_path = resnet_path or str(retrained_dir / "resnet50" / "resnet50_cxr.pth")
            res = build_resnet50(
                num_classes=self.num_classes,
                pretrained=False,
                weights_path=r_path if Path(r_path).is_file() else None,
            )
            self.register_model("resnet50", res)
        except Exception as e:
            logger.warning("Failed to initialize ResNet-50: %s", e)

        # 3. Swin Transformer (SwinCheX)
        try:
            swin = build_swin(
                num_classes=self.num_classes,
                pretrained=True,
                weights_path=swin_path,
            )
            self.register_model("swin", swin)
        except Exception as e:
            logger.warning("Failed to initialize Swin Transformer: %s", e)

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
                    # All models (NIHDenseNet, NIHResNet50, NIHSwin) output probabilities directly
                    probs = logits if (getattr(model, "xrv_probabilities", False) or isinstance(model, NIHSwin)) else torch.sigmoid(logits)

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

            per_model = {
                m_name: round(float(results["per_model_probabilities"][m_name][0][i]), 4)
                for m_name in self.models
                if m_name in results.get("per_model_probabilities", {})
            }
            agreeing = [m for m, p in per_model.items() if p >= thresh]
            consensus_str = f"{len(agreeing)}/{len(self.models)}"

            predictions_list.append({
                "disease": disease,
                "probability": round(prob, 4),
                "threshold": thresh,
                "detected": prob >= thresh,
                "uncertainty": round(uncertainty, 4),
                "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
                "is_uncertain": is_unc,
                "model_predictions": per_model,
                "models_agreeing": agreeing,
                "consensus": consensus_str,
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

    def generate_multiclass_cams(
        self,
        image_tensor: torch.Tensor,
        target_classes: List[str],
        original_np: np.ndarray,
        alpha: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """
        Generates high-precision, anatomically grounded LayerCAM heatmaps for target diseases.
        Uses pixel-wise gradient-activation weighting (LayerCAM, Jiang et al. 2021) across:
          - SwinCheX (Stage 4 Transformer norm2 block)
          - DenseNet-121 (DenseBlock 4 / norm5 features)
          - ResNet-50 (Layer 4 residual block)
        Features:
          - Eliminates edge/corner zero-padding artifacts
          - Anatomical thoracic super-gaussian windowing (removes scanner labels and borders)
          - Noise floor suppression and smooth Gaussian refinement
          - Weighted tri-model ensemble fusion (Swin 45% + DenseNet 35% + ResNet 20%)
        """
        import cv2
        from io import BytesIO
        from PIL import Image
        import base64

        def _b64(arr: np.ndarray) -> str:
            buf = BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        def _refine_cam(cam_raw: np.ndarray, h: int, w: int, noise_pct: float = 0.20) -> np.ndarray:
            """Refine raw CAM: interpolate, apply thoracic mask, suppress noise, and smooth."""
            cam = cv2.resize(cam_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Anatomical super-gaussian window (flat over thoracic cavity, smooth rolloff at image margins)
            y, x = np.ogrid[:h, :w]
            cy, cx = 0.48 * h, 0.50 * w
            sy, sx = 0.44 * h, 0.44 * w
            dist = ((x - cx) / sx) ** 4 + ((y - cy) / sy) ** 4
            thoracic_mask = np.exp(-0.5 * dist).astype(np.float32)
            cam = cam * thoracic_mask

            # Background noise floor suppression
            if cam.max() > 0:
                thresh = noise_pct * cam.max()
                cam = np.maximum(0.0, cam - thresh)
                if cam.max() > 0:
                    cam = cam / cam.max()
            cam = cv2.GaussianBlur(cam, (15, 15), 0)
            return np.clip(cam, 0.0, 1.0)

        def _cam_to_items(cam: np.ndarray, disease: str, model_name: str,
                          orig: np.ndarray, a: float) -> Dict[str, Any]:
            c_uint8 = np.uint8(255 * cam)
            hm = cv2.cvtColor(cv2.applyColorMap(c_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            ov = cv2.addWeighted(orig, 1 - a, hm, a, 0)
            return {"disease": disease, "heatmap": _b64(hm), "overlay": _b64(ov), "model_name": model_name}

        h_orig, w_orig = original_np.shape[:2]
        output_items: List[Dict[str, Any]] = []

        for disease in target_classes:
            if disease not in self.classes:
                continue
            idx = self.classes.index(disease)
            # Skip No_Finding — no localized anatomical pathology exists
            if idx >= 14:
                continue

            model_cams: Dict[str, np.ndarray] = {}

            # ── 1. Swin Transformer LayerCAM ──────────────────────────────────
            if "swin" in self.models:
                swin_model = self.models["swin"]
                try:
                    from src.models.swin import _prepare_swin_input
                    x_swin = _prepare_swin_input(image_tensor.clone(), 224).to(self.device)

                    acts_s, grads_s = [], []
                    target_s = swin_model.model.layers[-1].blocks[-1].norm2
                    h1 = target_s.register_forward_hook(lambda m, i, o: acts_s.append(o))
                    h2 = target_s.register_full_backward_hook(lambda m, gi, go: grads_s.append(go[0]))

                    swin_model.zero_grad()
                    with torch.enable_grad():
                        f = swin_model.model.forward_features(x_swin)
                        logits = swin_model.model.forward_head(f)
                        target_idx = min(idx, logits.shape[-1] - 1)
                        logits[0, target_idx].backward()

                    h1.remove()
                    h2.remove()

                    if acts_s and grads_s:
                        act, grad = acts_s[0].detach(), grads_s[0].detach()
                        if act.dim() == 4 and act.shape[1] == 7 and act.shape[2] == 7:
                            act = act.permute(0, 3, 1, 2)
                            grad = grad.permute(0, 3, 1, 2)
                        elif act.dim() == 3:
                            B, N, C = act.shape
                            s = int(N ** 0.5)
                            act = act.permute(0, 2, 1).reshape(B, C, s, s)
                            grad = grad.permute(0, 2, 1).reshape(B, C, s, s)

                        # Pixel-wise LayerCAM
                        weights = torch.relu(grad)
                        cam_s = (weights * act).sum(dim=1).squeeze()
                        cam_s = torch.relu(cam_s).cpu().numpy()
                        if cam_s.max() > 0:
                            model_cams["swin"] = cam_s / cam_s.max()
                except Exception as e_swin:
                    logger.warning("Swin LayerCAM failed for %s: %s", disease, e_swin)

            # ── 2. DenseNet-121 LayerCAM ───────────────────────────────────────
            if "densenet121" in self.models:
                dense_model = self.models["densenet121"]
                try:
                    from models.densenet_multilabel import _to_xrv_input
                    x_dense = _to_xrv_input(image_tensor.clone(), 224).to(self.device)

                    acts_d, grads_d = [], []
                    target_d = dense_model.features.norm5
                    h1 = target_d.register_forward_hook(lambda m, i, o: acts_d.append(o))
                    h2 = target_d.register_full_backward_hook(lambda m, gi, go: grads_d.append(go[0]))

                    dense_model.zero_grad()
                    with torch.enable_grad():
                        f = F.relu(dense_model.features(x_dense), inplace=False)
                        p = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
                        raw_logits = dense_model.classifier(p)
                        target_idx = dense_model.pathology_indices[idx]
                        raw_logits[0, target_idx].backward()

                    h1.remove()
                    h2.remove()

                    if acts_d and grads_d:
                        act = F.relu(acts_d[0].detach(), inplace=False)
                        grad = grads_d[0].detach()

                        # Pixel-wise LayerCAM
                        weights = torch.relu(grad)
                        cam_d = (weights * act).sum(dim=1).squeeze()
                        cam_d = torch.relu(cam_d).cpu().numpy()
                        if cam_d.max() > 0:
                            model_cams["densenet121"] = cam_d / cam_d.max()
                except Exception as e_dense:
                    logger.warning("DenseNet LayerCAM failed for %s: %s", disease, e_dense)

            # ── 3. ResNet-50 LayerCAM ──────────────────────────────────────────
            if "resnet50" in self.models:
                res_model = self.models["resnet50"]
                try:
                    from models.densenet_multilabel import _to_xrv_input
                    x_res = _to_xrv_input(image_tensor.clone(), 512).to(self.device)

                    acts_r, grads_r = [], []
                    target_r = res_model.model.layer4
                    h1 = target_r.register_forward_hook(lambda m, i, o: acts_r.append(o))
                    h2 = target_r.register_full_backward_hook(lambda m, gi, go: grads_r.append(go[0]))

                    res_model.zero_grad()
                    with torch.enable_grad():
                        m_r = res_model.model
                        out = m_r.conv1(x_res)
                        out = m_r.bn1(out)
                        out = m_r.relu(out)
                        out = m_r.maxpool(out)
                        out = m_r.layer1(out)
                        out = m_r.layer2(out)
                        out = m_r.layer3(out)
                        out = m_r.layer4(out)
                        p = m_r.avgpool(out).flatten(1)
                        raw_logits = m_r.fc(p)
                        target_idx = res_model.pathology_indices[idx]
                        raw_logits[0, target_idx].backward()

                    h1.remove()
                    h2.remove()

                    if acts_r and grads_r:
                        act = acts_r[0].detach()
                        grad = grads_r[0].detach()

                        # Pixel-wise LayerCAM
                        weights = torch.relu(grad)
                        cam_r = (weights * act).sum(dim=1).squeeze()
                        cam_r = torch.relu(cam_r).cpu().numpy()
                        if cam_r.max() > 0:
                            model_cams["resnet50"] = cam_r / cam_r.max()
                except Exception as e_res:
                    logger.warning("ResNet LayerCAM failed for %s: %s", disease, e_res)

            # ── 4. Generate fused ensemble CAM + individual model CAM items ────
            if model_cams:
                logger.info("LayerCAMs computed for %s: %s", disease, list(model_cams.keys()))

                # Refine individual model CAMs with anatomical window and noise cutoff
                refined: Dict[str, np.ndarray] = {}
                for m_name, raw_cam in model_cams.items():
                    refined[m_name] = _refine_cam(raw_cam, h_orig, w_orig, noise_pct=0.20)

                # Fused ensemble CAM (weighted combination)
                fused = np.zeros((h_orig, w_orig), dtype=np.float32)
                total_w = 0.0
                cam_weights = {"swin": 0.45, "densenet121": 0.35, "resnet50": 0.20}
                for m_name, r_cam in refined.items():
                    w_m = cam_weights.get(m_name, 1.0)
                    fused += w_m * r_cam
                    total_w += w_m
                if total_w > 0:
                    fused /= total_w

                if fused.max() > 0:
                    fused = fused / fused.max()
                fused = cv2.GaussianBlur(fused, (11, 11), 0)
                fused = np.clip(fused, 0.0, 1.0)

                # Emit Ensemble CAM
                output_items.append(_cam_to_items(fused, disease, "ensemble", original_np, alpha))

                # Emit Individual model CAMs for each backbone so clinicians can review each
                for m_name, r_cam in refined.items():
                    output_items.append(_cam_to_items(r_cam, disease, m_name, original_np, alpha))
            else:
                logger.warning("No CAMs could be generated for %s", disease)

        return output_items
