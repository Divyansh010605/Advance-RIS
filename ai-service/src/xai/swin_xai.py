import math
from typing import Tuple

import numpy as np
import torch

from src.xai.gradcam import compute_gradcam, compute_gradcam_plus_plus, generate_cam_overlay


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transforms Swin Transformer feature maps into standard CNN spatial format (B, C, H, W).

    Handles both:
      - (B, H, W, C) -> (B, C, H, W)
      - (B, L, C)    -> (B, sqrt(L), sqrt(L), C) -> (B, C, sqrt(L), sqrt(L))
    """
    if tensor.dim() == 4:
        # (B, H, W, C) -> (B, C, H, W)
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.dim() == 3:
        # (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
        b, l, c = tensor.shape
        side = int(math.sqrt(l))
        assert side * side == l, f"Sequence length {l} is not a perfect square"
        reshaped = tensor.view(b, side, side, c)
        return reshaped.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected tensor shape for Swin reshape transform: {tensor.shape}")


def compute_swin_cam(
    fmaps: torch.Tensor,
    grads: torch.Tensor,
    method: str = "gradcam",
) -> np.ndarray:
    """
    Computes spatial Grad-CAM or Grad-CAM++ for Swin Transformer using reshape transform.
    """
    spatial_fmaps = swin_reshape_transform(fmaps)
    spatial_grads = swin_reshape_transform(grads)

    if method == "gradcam++":
        return compute_gradcam_plus_plus(spatial_fmaps, spatial_grads)
    return compute_gradcam(spatial_fmaps, spatial_grads)


__all__ = ["swin_reshape_transform", "compute_swin_cam", "generate_cam_overlay"]
