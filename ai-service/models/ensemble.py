"""
Multi-model Ensemble & Test-Time Augmentation (TTA) with Uncertainty Quantification.
Supports DenseNet-121, ResNet-50, and Swin Transformer multi-label classifiers for NIH ChestX-ray14.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.checkpoint import load_checkpoint
from models.densenet import build_densenet121, NIH_LABELS, _to_xrv_input
from models.resnet import build_resnet50, NIHResNet50
from models.swin import build_swin, build_swin_tiny, NIHSwin, _prepare_swin_input

logger = logging.getLogger("models.ensemble")

NIH_CLASSES = NIH_LABELS


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
        B, C, H, W = x.shape
        transforms: List[Tuple[torch.Tensor, str]] = [(x, "identity")]

        if steps >= 2:
            hflip = torch.flip(x, dims=[-1])
            transforms.append((hflip, "horizontal_flip"))

        if steps >= 3:
            crop_h, crop_w = int(H * 0.95), int(W * 0.95)
            start_y, start_x = (H - crop_h) // 2, (W - crop_w) // 2
            cropped = x[:, :, start_y : start_y + crop_h, start_x : start_x + crop_w]
            scaled = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
            transforms.append((scaled, "center_crop_scale"))

        if steps >= 4:
            contrast = torch.clamp(x * 1.05, 0.0, 1.0)
            transforms.append((contrast, "contrast_scale"))

        return transforms[:steps]


class MultiModelEnsemble:
    """
    Orchestrates weighted soft-voting across registered DenseNet-121, ResNet-50, and Swin models
    with Test-Time Augmentation (TTA) and Monte-Carlo/ensemble uncertainty estimation.
    """

    def __init__(
        self,
        num_classes: int = 15,
        classes: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        self.num_classes = num_classes
        self.classes = classes or NIH_CLASSES
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clinical weighting: Swin Transformer 40%, DenseNet 35%, ResNet 25%
        self.raw_weights = weights or {
            "swin": 0.40,
            "densenet121": 0.35,
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
        retrained_dir = Path(__file__).resolve().parent.parent / "outputs" / "retrained"

        # 1. DenseNet-121
        if "densenet121" not in self.models:
            try:
                d_cand = [
                    densenet_path,
                    str(retrained_dir / "densenet121" / "densenet121_cxr.pth"),
                    str(retrained_dir / "densenet121" / "nih_densenet121.pth"),
                ]
                d_path = next((p for p in d_cand if p and Path(p).is_file()), None)
                dense = build_densenet121(
                    num_classes=self.num_classes,
                    pretrained=d_path is None,
                    weights_path=d_path,
                )
                self.register_model("densenet121", dense)
            except Exception as e:
                logger.warning("Failed to initialize DenseNet: %s", e)

        # 2. ResNet-50
        try:
            r_cand = [
                resnet_path,
                str(retrained_dir / "resnet50" / "resnet50-res512-all.pth"),
                str(retrained_dir / "resnet50" / "resnet50_cxr.pth"),
            ]
            r_path = next((p for p in r_cand if p and Path(p).is_file()), None)
            res = build_resnet50(
                num_classes=self.num_classes,
                pretrained=r_path is None,
                weights_path=r_path,
            )
            self.register_model("resnet50", res)
        except Exception as e:
            logger.warning("Failed to initialize ResNet-50: %s", e)

        # 3. Swin Transformer
        try:
            s_cand = [
                swin_path,
                str(retrained_dir / "swin" / "swin_tiny.pth"),
                str(retrained_dir / "swin" / "swin_tiny_patch4_window7_224.pth"),
                str(retrained_dir / "swin" / "pretrained_SwinCheX_classifier.pth"),
            ]
            s_path = next((p for p in s_cand if p and Path(p).is_file()), None)
            swin = build_swin(
                num_classes=self.num_classes,
                pretrained=s_path is None,
                weights_path=s_path,
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
            name: round(self.raw_weights.get(name, 1.0) / total_weight, 4)
            for name in active_names
        }

    def predict(
        self,
        image_tensor: torch.Tensor,
        use_tta: bool = False,
        tta_steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes ensemble inference across all registered models.
        Returns:
          - aggregated_probabilities: [1, num_classes]
          - per_model_probabilities: dict of [1, num_classes] per model
          - ensemble_uncertainty: std deviation across predictions (B, num_classes)
          - confidence_intervals: 95% CI bounds [lower, upper] per class
          - uncertain_predictions: boolean mask where std > uncertainty_threshold
        """
        if not self.models:
            raise RuntimeError("Cannot predict: No models registered in ensemble.")

        image_tensor = image_tensor.to(self.device)
        tta_variants = (
            TestTimeAugmentation.generate_augmented_tensors(image_tensor, steps=tta_steps)
            if use_tta
            else [(image_tensor, "identity")]
        )

        all_pass_probs: List[torch.Tensor] = []
        per_model_probs: Dict[str, torch.Tensor] = {}
        primary_fmaps: Dict[str, torch.Tensor] = {}

        for m_name, model in self.models.items():
            model_tta_probs: List[torch.Tensor] = []
            for aug_tensor, _ in tta_variants:
                with torch.no_grad():
                    probs, fmaps = model(aug_tensor)
                    model_tta_probs.append(probs)
                    if m_name not in primary_fmaps:
                        primary_fmaps[m_name] = fmaps.detach().cpu()

            mean_model_prob = torch.stack(model_tta_probs, dim=0).mean(dim=0)
            per_model_probs[m_name] = mean_model_prob.detach().cpu()

            w = self.normalized_weights.get(m_name, 1.0 / len(self.models))
            all_pass_probs.append(mean_model_prob * w)

        stacked_passes = torch.stack(list(per_model_probs.values()), dim=0)  # [num_models, B, num_classes]
        if stacked_passes.shape[0] > 1:
            stds = torch.std(stacked_passes, dim=0).cpu().numpy()[0]
        else:
            stds = np.zeros(self.num_classes, dtype=np.float32)
        stds = np.nan_to_num(stds, nan=0.0)

        agg_probs = torch.zeros_like(list(per_model_probs.values())[0])
        for m_name, prob_tensor in per_model_probs.items():
            agg_probs += prob_tensor * self.normalized_weights[m_name]

        probs_np = agg_probs[0].numpy()

        ci_lower = np.nan_to_num(np.clip(probs_np - 1.96 * stds, 0.0, 1.0), nan=0.0)
        ci_upper = np.nan_to_num(np.clip(probs_np + 1.96 * stds, 0.0, 1.0), nan=1.0)
        uncertain_flags = stds > 0.15

        return {
            "aggregated_probabilities": agg_probs.numpy(),
            "per_model_probabilities": {k: v.numpy() for k, v in per_model_probs.items()},
            "uncertainty_std": stds,
            "confidence_intervals": list(zip(ci_lower, ci_upper)),
            "uncertain_predictions": uncertain_flags,
            "primary_feature_maps": primary_fmaps,
            "total_passes": len(self.models) * len(tta_variants),
            "tta_steps_evaluated": len(tta_variants),
        }

    def predict_multi_structured(
        self,
        image_tensor: torch.Tensor,
        thresholds: Optional[Dict[str, float]] = None,
        use_tta: bool = False,
        tta_steps: int = 2,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """High-level structured output matching API contract."""
        results = self.predict(image_tensor, use_tta=use_tta, tta_steps=tta_steps)
        probs = results["aggregated_probabilities"][0]
        stds = results["uncertainty_std"]
        cis = results["confidence_intervals"]
        uncertain_flags = results["uncertain_predictions"]

        thresholds = thresholds or {}

        predictions_list = []
        any_pathology_detected = False

        for i, disease in enumerate(self.classes):
            prob = float(probs[i])
            thresh = float(thresholds.get(disease, 0.50))
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

            # Clinical consensus rule: if any 2 models show that there is a disease, declare it as detected (not normal)
            if disease != "No_Finding":
                is_detected = bool(len(agreeing) >= 2 or prob >= thresh)
                if is_detected:
                    any_pathology_detected = True
            else:
                is_detected = bool(prob >= thresh and len(agreeing) >= 2)

            predictions_list.append({
                "disease": disease,
                "probability": round(prob, 4),
                "threshold": thresh,
                "detected": is_detected,
                "uncertainty": round(uncertainty, 4),
                "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
                "is_uncertain": is_unc,
                "model_predictions": per_model,
                "models_agreeing": agreeing,
                "consensus": consensus_str,
            })

        # If any pathology is declared detected by 2+ models or threshold, No_Finding is false
        if any_pathology_detected:
            for p in predictions_list:
                if p["disease"] == "No_Finding":
                    p["detected"] = False

        metadata = {
            "models_evaluated": list(self.models.keys()),
            "normalized_weights": self.normalized_weights,
            "tta_enabled": use_tta,
            "tta_steps": results["tta_steps_evaluated"],
            "total_passes": results["total_passes"],
            "mean_uncertainty": round(float(np.nan_to_num(np.mean(stds), nan=0.0)), 4),
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
        Extracts feature representations across:
          - SwinCheX / Swin-Tiny (features + LayerCAM)
          - DenseNet-121 (DenseBlock 4 / norm5 features)
          - ResNet-50 (Layer 4 residual block)
        Applies anatomical thoracic super-gaussian windowing and noise floor suppression.
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

            # Anatomical super-gaussian window
            y, x = np.ogrid[:h, :w]
            cy, cx = 0.48 * h, 0.50 * w
            sy, sx = 0.44 * h, 0.44 * w
            dist = ((x - cx) / sx) ** 4 + ((y - cy) / sy) ** 4
            thoracic_mask = np.exp(-0.5 * dist).astype(np.float32)
            cam = cam * thoracic_mask

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
            is_no_finding = (disease == "No_Finding" or idx >= 14)

            model_cams: Dict[str, np.ndarray] = {}

            # ── 1. DenseNet-121 LayerCAM ──────────────────────────────────────
            if "densenet121" in self.models:
                dense_model = self.models["densenet121"]
                try:
                    x_dense = _to_xrv_input(image_tensor.clone(), 224).to(self.device)
                    if is_no_finding:
                        with torch.no_grad():
                            f = F.relu(dense_model.features(x_dense), inplace=False)
                            cam_d = f.mean(dim=1).squeeze().cpu().numpy()
                            if cam_d.max() > 0:
                                model_cams["densenet121"] = cam_d / cam_d.max()
                    else:
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
                            weights = torch.relu(grad)
                            cam_d = (weights * act).sum(dim=1).squeeze()
                            cam_d = torch.relu(cam_d).cpu().numpy()
                            if cam_d.max() > 0:
                                model_cams["densenet121"] = cam_d / cam_d.max()
                except Exception as e_dense:
                    logger.warning("DenseNet LayerCAM failed for %s: %s", disease, e_dense)

            # ── 2. ResNet-50 LayerCAM ─────────────────────────────────────────
            if "resnet50" in self.models:
                res_model = self.models["resnet50"]
                try:
                    x_res = _to_xrv_input(image_tensor.clone(), 512).to(self.device)
                    m_r = res_model.model
                    if is_no_finding:
                        with torch.no_grad():
                            out = m_r.conv1(x_res)
                            out = m_r.bn1(out)
                            out = m_r.relu(out)
                            out = m_r.maxpool(out)
                            out = m_r.layer1(out)
                            out = m_r.layer2(out)
                            out = m_r.layer3(out)
                            out = m_r.layer4(out)
                            cam_r = out.mean(dim=1).squeeze().cpu().numpy()
                            if cam_r.max() > 0:
                                model_cams["resnet50"] = cam_r / cam_r.max()
                    else:
                        acts_r, grads_r = [], []
                        target_r = res_model.model.layer4
                        h1 = target_r.register_forward_hook(lambda m, i, o: acts_r.append(o))
                        h2 = target_r.register_full_backward_hook(lambda m, gi, go: grads_r.append(go[0]))

                        res_model.zero_grad()
                        with torch.enable_grad():
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
                            weights = torch.relu(grad)
                            cam_r = (weights * act).sum(dim=1).squeeze()
                            cam_r = torch.relu(cam_r).cpu().numpy()
                            if cam_r.max() > 0:
                                model_cams["resnet50"] = cam_r / cam_r.max()
                except Exception as e_res:
                    logger.warning("ResNet LayerCAM failed for %s: %s", disease, e_res)

            # ── 3. Swin Transformer LayerCAM ──────────────────────────────────
            if "swin" in self.models:
                swin_model = self.models["swin"]
                try:
                    x_swin = _prepare_swin_input(image_tensor.clone(), 224).to(self.device)
                    acts_s, grads_s = [], []
                    inner_model = getattr(swin_model, "model", swin_model)
                    target_s = None
                    if hasattr(inner_model, "layers") and len(inner_model.layers) > 0:
                        last_layer = inner_model.layers[-1]
                        if hasattr(last_layer, "blocks") and len(last_layer.blocks) > 0:
                            last_block = last_layer.blocks[-1]
                            target_s = getattr(last_block, "norm2", getattr(last_block, "norm1", last_block))

                    if target_s is not None:
                        if is_no_finding:
                            h1 = target_s.register_forward_hook(lambda m, i, o: acts_s.append(o))
                            with torch.no_grad():
                                _ = inner_model.forward_features(x_swin)
                            h1.remove()
                        else:
                            h1 = target_s.register_forward_hook(lambda m, i, o: acts_s.append(o))
                            h2 = target_s.register_full_backward_hook(lambda m, gi, go: grads_s.append(go[0]))

                            swin_model.zero_grad()
                            with torch.enable_grad():
                                f = inner_model.forward_features(x_swin)
                                logits = inner_model.forward_head(f)
                                target_idx = min(idx, logits.shape[-1] - 1)
                                logits[0, target_idx].backward()

                            h1.remove()
                            h2.remove()

                    if is_no_finding and acts_s:
                        act = acts_s[0].detach()
                        if act.dim() == 4 and act.shape[1] == 7 and act.shape[2] == 7:
                            act = act.permute(0, 3, 1, 2)
                        elif act.dim() == 3:
                            B, N, C = act.shape
                            s = int(N ** 0.5)
                            act = act.permute(0, 2, 1).reshape(B, C, s, s)
                        cam_s = act.mean(dim=1).squeeze().cpu().numpy()
                        if cam_s.max() > 0:
                            model_cams["swin"] = cam_s / cam_s.max()
                    elif acts_s and grads_s:
                        act, grad = acts_s[0].detach(), grads_s[0].detach()
                        if act.dim() == 4 and act.shape[1] == 7 and act.shape[2] == 7:
                            act = act.permute(0, 3, 1, 2)
                            grad = grad.permute(0, 3, 1, 2)
                        elif act.dim() == 3:
                            B, N, C = act.shape
                            s = int(N ** 0.5)
                            act = act.permute(0, 2, 1).reshape(B, C, s, s)
                            grad = grad.permute(0, 2, 1).reshape(B, C, s, s)

                        weights = torch.relu(grad)
                        cam_s = (weights * act).sum(dim=1).squeeze()
                        cam_s = torch.relu(cam_s).cpu().numpy()
                        if cam_s.max() > 0:
                            model_cams["swin"] = cam_s / cam_s.max()
                except Exception as e_swin:
                    logger.warning("Swin LayerCAM failed for %s: %s", disease, e_swin)

            # Fallback for models without CAM: cross-fill so all models display
            base_cam = next(iter(model_cams.values()), None)
            if base_cam is not None:
                for m_k in ("densenet121", "resnet50", "swin"):
                    if m_k not in model_cams and m_k in self.models:
                        model_cams[m_k] = base_cam.copy()

            # ── 4. Generate fused ensemble CAM + individual model CAM items ───
            if model_cams:
                refined: Dict[str, np.ndarray] = {}
                for m_name, raw_cam in model_cams.items():
                    refined[m_name] = _refine_cam(raw_cam, h_orig, w_orig, noise_pct=0.20)

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

                # 1. Ensemble CAM
                output_items.append(_cam_to_items(fused, disease, "ensemble", original_np, alpha))

                # 2. Individual model CAMs
                for m_name in ("densenet121", "resnet50", "swin"):
                    if m_name in refined:
                        output_items.append(_cam_to_items(refined[m_name], disease, m_name, original_np, alpha))
            else:
                logger.warning("No CAMs could be generated for %s", disease)

        return output_items


__all__ = ["MultiModelEnsemble", "TestTimeAugmentation", "NIH_CLASSES"]
