"""
Per-Disease Grad-CAM heatmap generation.
Produces coloured heatmaps and overlays for each requested disease class,
returned as base64-encoded JPEG strings.
"""
import base64
import logging
from io import BytesIO
from typing import Tuple, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _ensure_spatial_fmaps(fmaps: torch.Tensor, grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensures feature maps and gradients are in (B, C, H, W) spatial format."""
    # If 4D and channels are last (e.g. Swin (B, 7, 7, C))
    if fmaps.dim() == 4 and fmaps.shape[1] == fmaps.shape[2] and fmaps.shape[3] > fmaps.shape[1]:
        fmaps = fmaps.permute(0, 3, 1, 2).contiguous()
        grads = grads.permute(0, 3, 1, 2).contiguous()
    elif fmaps.dim() == 3:  # (B, L, C)
        import math
        b, l, c = fmaps.shape
        side = int(math.isqrt(l))
        if side * side == l:
            fmaps = fmaps.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
            grads = grads.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
    return fmaps, grads


def generate_single_gradcam(
    original_np: np.ndarray,
    feature_maps: torch.Tensor,
    gradients: torch.Tensor,
    alpha: float = 0.4,
    method: str = "gradcam",
) -> Tuple[str, str]:
    """
    Generates Grad-CAM or Grad-CAM++ heatmap and blended overlay image.
    Supports DenseNet-121, ResNet-50, and Swin Transformer backbones.

    Args:
        original_np:   uint8 numpy array (H, W, 3) – original image.
        feature_maps:  Tensor (1, C, H', W') or (1, H', W', C) – feature maps from backbone.
        gradients:     Tensor – gradients w.r.t feature maps.
        alpha:         Blend factor for overlay.
        method:        'gradcam' or 'gradcam++'.

    Returns:
        (heatmap_b64, overlay_b64)
    """
    feature_maps, gradients = _ensure_spatial_fmaps(feature_maps, gradients)

    if method == "gradcam++":
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        sum_grad_3 = (feature_maps * grad_3).sum(dim=[2, 3], keepdim=True)
        denominator = 2.0 * grad_2 + sum_grad_3
        denominator = torch.where(denominator != 0.0, denominator, torch.tensor(1e-8, device=gradients.device))
        alpha_coeff = grad_2 / denominator
        rel_grad = torch.relu(gradients)
        weights = (alpha_coeff * rel_grad).sum(dim=[2, 3], keepdim=True)
    else:
        # Standard Grad-CAM: Global average pool of gradients -> channel weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

    # Positive-influence filtered weighted combination of feature maps
    cam = (weights * feature_maps).sum(dim=1, keepdim=False)  # (1, H', W')
    cam = torch.relu(cam).squeeze().detach().cpu().numpy()     # (H', W')

    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)

    h, w = original_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply JET colormap
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay_rgb = cv2.addWeighted(original_np, 1 - alpha, heatmap_rgb, alpha, 0)

    # Encode to base64
    heatmap_b64 = _numpy_to_b64_jpeg(heatmap_rgb)
    overlay_b64 = _numpy_to_b64_jpeg(overlay_rgb)

    return heatmap_b64, overlay_b64


def generate_multi_gradcam(
    original_np: np.ndarray,
    feature_maps: torch.Tensor,
    gradients_dict: Dict[int, torch.Tensor],
    classes: List[str],
    alpha: float = 0.4,
    method: str = "gradcam",
) -> Dict[str, Tuple[str, str]]:
    """
    Generates Grad-CAM or Grad-CAM++ heatmaps for multiple target diseases.
    
    Returns:
        Dict[disease_name, (heatmap_b64, overlay_b64)]
    """
    results = {}
    for target_idx, gradients in gradients_dict.items():
        if target_idx < len(classes):
            disease_name = classes[target_idx]
            heatmap_b64, overlay_b64 = generate_single_gradcam(
                original_np, feature_maps, gradients, alpha=alpha, method=method
            )
            results[disease_name] = (heatmap_b64, overlay_b64)
    return results


def _numpy_to_b64_jpeg(arr: np.ndarray, quality: int = 90) -> str:
    """Converts an (H, W, 3) uint8 numpy array to a base64-encoded JPEG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Backward-compatibility alias
generate_gradcam = generate_single_gradcam
