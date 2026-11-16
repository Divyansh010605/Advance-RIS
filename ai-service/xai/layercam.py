"""
Explainable AI (XAI) Algorithms for Multi-Model Chest Radiograph Localization.
Implements:
  - LayerCAM (pixel-wise gradient-activation weighting, Jiang et al. 2021)
  - Grad-CAM (Selvaraju et al. 2017)
  - Grad-CAM++ (Chattopadhyay et al. 2018)
  - Swin Transformer spatial reshape transform & attribution
  - Anatomical lung-field windowing and overlay synthesis
"""
import base64
import math
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def ensure_spatial_fmaps(fmaps: torch.Tensor, grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensures feature maps and gradients are in (B, C, H, W) spatial format."""
    if fmaps.dim() == 4 and fmaps.shape[1] == fmaps.shape[2] and fmaps.shape[3] > fmaps.shape[1]:
        fmaps = fmaps.permute(0, 3, 1, 2).contiguous()
        grads = grads.permute(0, 3, 1, 2).contiguous()
    elif fmaps.dim() == 3:
        b, l, c = fmaps.shape
        side = int(math.isqrt(l))
        if side * side == l:
            fmaps = fmaps.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
            grads = grads.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
    return fmaps, grads


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Transforms Swin Transformer feature maps into standard CNN spatial format (B, C, H, W)."""
    if tensor.dim() == 4:
        if tensor.shape[1] > tensor.shape[2] and tensor.shape[1] > tensor.shape[3]:
            return tensor
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.dim() == 3:
        b, l, c = tensor.shape
        side = int(math.isqrt(l))
        assert side * side == l, f"Sequence length {l} is not a perfect square"
        reshaped = tensor.view(b, side, side, c)
        return reshaped.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected tensor shape for Swin reshape transform: {tensor.shape}")


def compute_layercam(feature_maps: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    LayerCAM: Pixel-wise weighting across feature activations.
    w_kij = relu(grad_kij)
    L_c = relu(sum_k(w_kij * A_kij))
    Preserves finer object localization details than global pooled Grad-CAM.
    """
    feature_maps, gradients = ensure_spatial_fmaps(feature_maps, gradients)
    weights = torch.relu(gradients)
    cam = (weights * feature_maps).sum(dim=1, keepdim=False)
    cam = torch.relu(cam).squeeze().detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def compute_gradcam(feature_maps: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    Standard Grad-CAM: Global average pooling over spatial gradients.
    alpha_k = mean(gradients_k)
    L_c = relu(sum(alpha_k * A_k))
    """
    feature_maps, gradients = ensure_spatial_fmaps(feature_maps, gradients)
    weights = gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=False)
    cam = torch.relu(cam).squeeze().detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def compute_gradcam_plus_plus(feature_maps: torch.Tensor, gradients: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """
    Grad-CAM++: Higher-order gradients for multiple occurrences and fine localization.
    """
    feature_maps, gradients = ensure_spatial_fmaps(feature_maps, gradients)
    grad = gradients
    fmap = feature_maps

    grad_2 = grad.pow(2)
    grad_3 = grad.pow(3)

    sum_grad_3 = (fmap * grad_3).sum(dim=[2, 3], keepdim=True)
    denominator = 2.0 * grad_2 + sum_grad_3
    denominator = torch.where(denominator != 0.0, denominator, torch.tensor(eps, device=grad.device))

    alpha = grad_2 / denominator
    rel_grad = torch.relu(grad)
    weights = (alpha * rel_grad).sum(dim=[2, 3], keepdim=True)

    cam = (weights * fmap).sum(dim=1, keepdim=False)
    cam = torch.relu(cam).squeeze().detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def compute_swin_cam(
    fmaps: torch.Tensor,
    grads: torch.Tensor,
    method: str = "layercam",
) -> np.ndarray:
    """Computes LayerCAM or Grad-CAM for Swin Transformer using reshape transform."""
    spatial_fmaps = swin_reshape_transform(fmaps)
    spatial_grads = swin_reshape_transform(grads)

    if method == "layercam":
        return compute_layercam(spatial_fmaps, spatial_grads)
    elif method == "gradcam++":
        return compute_gradcam_plus_plus(spatial_fmaps, spatial_grads)
    return compute_gradcam(spatial_fmaps, spatial_grads)


def generate_cam_overlay(
    cam_normalized: np.ndarray,
    original_img_np: np.ndarray,
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates colored heatmap RGB and blended overlay RGB."""
    h, w = original_img_np.shape[:2]
    cam_resized = cv2.resize(cam_normalized, (w, h), interpolation=cv2.INTER_LINEAR)

    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return heatmap_rgb, overlay_rgb


def _to_base64_jpeg(img_np: np.ndarray, quality: int = 90) -> str:
    """Converts a numpy RGB array to base64 JPEG."""
    buf = BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_single_gradcam(
    original_np: np.ndarray,
    feature_maps: Union[torch.Tensor, np.ndarray],
    gradients: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.40,
    method: str = "layercam",
) -> Tuple[str, str]:
    """
    Generates single heatmap and overlay returned as base64 JPEG strings.
    Compatible with legacy and modern callers.
    """
    if isinstance(feature_maps, np.ndarray):
        feature_maps = torch.from_numpy(feature_maps)
    if isinstance(gradients, np.ndarray):
        gradients = torch.from_numpy(gradients)

    if method == "layercam":
        cam = compute_layercam(feature_maps, gradients)
    elif method == "gradcam++":
        cam = compute_gradcam_plus_plus(feature_maps, gradients)
    else:
        cam = compute_gradcam(feature_maps, gradients)

    hm_rgb, ov_rgb = generate_cam_overlay(cam, original_np, alpha=alpha)
    return _to_base64_jpeg(hm_rgb), _to_base64_jpeg(ov_rgb)


def generate_multi_gradcam(
    original_np: np.ndarray,
    feature_maps: Union[torch.Tensor, np.ndarray],
    grad_dict: Dict[int, Union[torch.Tensor, np.ndarray]],
    class_names: List[str],
    alpha: float = 0.40,
    method: str = "layercam",
) -> Dict[str, Tuple[str, str]]:
    """Generates CAM heatmaps for all classes provided in grad_dict."""
    results = {}
    for idx, grads in grad_dict.items():
        if idx < len(class_names):
            disease = class_names[idx]
            results[disease] = generate_single_gradcam(
                original_np, feature_maps, grads, alpha=alpha, method=method
            )
    return results


__all__ = [
    "compute_layercam",
    "compute_gradcam",
    "compute_gradcam_plus_plus",
    "compute_swin_cam",
    "swin_reshape_transform",
    "ensure_spatial_fmaps",
    "generate_cam_overlay",
    "generate_single_gradcam",
    "generate_multi_gradcam",
]
