"""
Image preprocessing utilities for NIH ChestX-ray14 AI Service.
Decodes base64 / binary images, applies CLAHE, resizing, and ImageNet normalizations.
"""
import base64
import logging
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)


def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Maintain aspect ratio and pad with black borders to target_size RGB."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(
        img_copy,
        (
            (target_size[0] - img_copy.size[0]) // 2,
            (target_size[1] - img_copy.size[1]) // 2,
        ),
    )
    return new_img


def apply_clahe(img_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to thoracic X-ray."""
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img_np)


_transform = transforms.Compose([
    transforms.Lambda(resize_with_padding),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(
    image_b64: str, mime_type: str = "image/jpeg"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Decodes a base64 image string, converts to RGB, and prepares for inference.

    Args:
        image_b64: Base64-encoded image (with or without data URI prefix).
        mime_type: MIME type of the uploaded image.

    Returns:
        Tuple of:
          - image_tensor: torch.Tensor of shape (1, 3, 224, 224)
          - original_np:  uint8 numpy array of shape (H, W, 3) in RGB
    """
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        missing_padding = len(image_b64) % 4
        if missing_padding:
            image_b64 += "=" * (4 - missing_padding)

        image_data = base64.b64decode(image_b64)
        pil_image = Image.open(BytesIO(image_data))

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        original_np = np.array(pil_image)

        image_tensor = _transform(pil_image).unsqueeze(0)
        return image_tensor, original_np

    except Exception as exc:
        logger.error(f"Image preprocessing failed: {exc}")
        raise ValueError(f"Invalid image data: {exc}") from exc


__all__ = [
    "preprocess_image",
    "resize_with_padding",
    "apply_clahe",
    "TARGET_SIZE",
]
