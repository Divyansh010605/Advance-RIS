"""
Explainable AI (XAI) and Image Processing module for NIH ChestX-ray14 screening.
"""
from xai.image_processor import (
    preprocess_image,
    resize_with_padding,
    apply_clahe,
    TARGET_SIZE,
)
from xai.layercam import (
    compute_layercam,
    compute_gradcam,
    compute_gradcam_plus_plus,
    compute_swin_cam,
    swin_reshape_transform,
    ensure_spatial_fmaps,
    generate_cam_overlay,
    generate_single_gradcam,
    generate_multi_gradcam,
)

__all__ = [
    "preprocess_image",
    "resize_with_padding",
    "apply_clahe",
    "TARGET_SIZE",
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
