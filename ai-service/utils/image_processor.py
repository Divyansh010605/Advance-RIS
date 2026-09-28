"""
Image preprocessing utilities.
Decodes base64 image, resizes and normalizes for DenseNet-121.
"""
import base64
import logging
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)


def resize_with_padding(img: Image.Image) -> Image.Image:
    """Maintain aspect ratio and pad with black borders to 224x224 RGB."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)
    new_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    new_img.paste(
        img,
        (
            (TARGET_SIZE[0] - img.size[0]) // 2,
            (TARGET_SIZE[1] - img.size[1]) // 2,
        ),
    )
    return new_img


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
        image_b64: Base64-encoded image (no data URI prefix needed, but tolerated).
        mime_type:  MIME type hint (used for logging only).

    Returns:
        image_tensor: Normalized float32 tensor of shape (1, 3, 224, 224).
        original_np:  Uint8 numpy array (H, W, 3) for Grad-CAM overlay.
    """
    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc

    try:
        pil_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    logger.debug(f"Loaded image: {pil_image.size}, mode={pil_image.mode}, type={mime_type}")

    # Keep a resized uint8 copy (RGB) for overlay generation
    original_resized = resize_with_padding(pil_image)
    original_np = np.array(original_resized, dtype=np.uint8)

    # Apply transforms
    tensor = _transform(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

    # Safeguard channel dimension for 3-channel Conv2d models
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor, original_np
