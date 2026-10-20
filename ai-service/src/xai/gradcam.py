import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def compute_gradcam(feature_maps: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    Standard Grad-CAM:
    alpha_k = mean(gradients_k)
    L_c = relu(sum(alpha_k * A_k))
    """
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
    Grad-CAM++:
    Uses higher-order gradients for multiple occurrences and finer localization.
    alpha_ij^kc = (grad_ij^2) / (2 * grad_ij^2 + sum_ab(A_ab^k * grad_ab^3) + eps)
    w_k = sum_ij(alpha_ij^kc * relu(grad_ij))
    L_c = relu(sum(w_k * A_k))
    """
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


def generate_cam_overlay(
    image_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[str, str, np.ndarray]:
    """
    Resizes 2D CAM to image dimensions, applies colormap, and blends with original image.
    Returns (heatmap_b64, overlay_b64, raw_overlay_np).
    """
    if cam.ndim == 3:
        cam = cam[0]
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0.0, 1.0))

    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)

    heatmap_b64 = _numpy_to_b64(heatmap_rgb)
    overlay_b64 = _numpy_to_b64(overlay_rgb)
    return heatmap_b64, overlay_b64, overlay_rgb


def _numpy_to_b64(arr: np.ndarray, quality: int = 90) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
