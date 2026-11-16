"""
DICOM Loader for NIH ChestX-ray14 AI Service.

Handles DICOM file ingestion, pixel extraction, window/level normalization,
and conversion to RGB PIL images ready for the classification pipeline.

Supported DICOM formats:
  - Standard chest PA/AP radiographs (MONOCHROME1/2)
  - Presentation-state aware windowing (WC/WW tags)
  - Photometric interpretation correction
"""
import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# NIH ChestX-ray14 standard window defaults (lung window)
NIH_WINDOW_CENTER_DEFAULT = 40
NIH_WINDOW_WIDTH_DEFAULT = 400


def load_dicom_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    Parse raw DICOM bytes and return (pixel_array_rgb, metadata_dict).

    Args:
        raw_bytes: Raw bytes from a .dcm file or HTTP multipart upload.

    Returns:
        Tuple of:
          - pixel_rgb: np.ndarray shape (H, W, 3) dtype uint8
          - metadata: dict with patient_id, study_date, modality, wc, ww
    """
    try:
        import pydicom  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "pydicom is required for DICOM ingestion. "
            "Install with: pip install pydicom"
        ) from e

    ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
    pixel_array = _extract_pixel_array(ds)
    pixel_rgb = _to_rgb_uint8(pixel_array)

    metadata = {
        "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "modality": str(getattr(ds, "Modality", "CR")),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "rows": int(getattr(ds, "Rows", pixel_array.shape[0])),
        "columns": int(getattr(ds, "Columns", pixel_array.shape[1])),
        "bits_stored": int(getattr(ds, "BitsStored", 8)),
        "photometric_interpretation": str(
            getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        ),
        "window_center": float(
            getattr(ds, "WindowCenter", NIH_WINDOW_CENTER_DEFAULT)
            if not isinstance(getattr(ds, "WindowCenter", None), pydicom.multival.MultiValue)
            else ds.WindowCenter[0]
        ),
        "window_width": float(
            getattr(ds, "WindowWidth", NIH_WINDOW_WIDTH_DEFAULT)
            if not isinstance(getattr(ds, "WindowWidth", None), pydicom.multival.MultiValue)
            else ds.WindowWidth[0]
        ),
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
    }

    logger.info(
        "DICOM loaded: patient=%s, shape=%s, photometric=%s",
        metadata["patient_id"],
        pixel_array.shape,
        metadata["photometric_interpretation"],
    )
    return pixel_rgb, metadata


def load_dicom_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """Load a DICOM file from disk. Delegates to load_dicom_bytes."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    return load_dicom_bytes(path.read_bytes())


def dicom_bytes_to_pil(raw_bytes: bytes) -> Tuple[Image.Image, dict]:
    """Load DICOM bytes and return a PIL RGB image alongside metadata."""
    pixel_rgb, metadata = load_dicom_bytes(raw_bytes)
    pil_img = Image.fromarray(pixel_rgb, mode="RGB")
    return pil_img, metadata


def dicom_to_base64_jpeg(raw_bytes: bytes, quality: int = 90) -> Tuple[str, dict]:
    """
    Convert DICOM bytes to a base64-encoded JPEG string.
    Suitable for passing directly to the PredictionRequest.image field.
    """
    pil_img, metadata = dicom_bytes_to_pil(raw_bytes)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, metadata


# ─── Internal helpers ───────────────────────────────────────────────────────


def _extract_pixel_array(ds) -> np.ndarray:
    """
    Extract and window-level normalize the DICOM pixel array.
    Returns float32 array normalized to [0.0, 1.0].
    """
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to extract DICOM pixel data: {e}") from e

    # Apply window/level normalization
    try:
        wc = float(ds.WindowCenter[0] if hasattr(ds.WindowCenter, "__iter__") else ds.WindowCenter)
        ww = float(ds.WindowWidth[0] if hasattr(ds.WindowWidth, "__iter__") else ds.WindowWidth)
    except AttributeError:
        wc = NIH_WINDOW_CENTER_DEFAULT
        ww = NIH_WINDOW_WIDTH_DEFAULT

    img_min = wc - ww / 2.0
    img_max = wc + ww / 2.0
    pixel_array = np.clip(pixel_array, img_min, img_max)
    pixel_array = (pixel_array - img_min) / (img_max - img_min)

    # Handle MONOCHROME1 (invert: 0=white in DICOM -> 0=black in display)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        pixel_array = 1.0 - pixel_array

    # Rescale slope/intercept (before windowing in full pipeline, but handle here for robustness)
    # Already handled via pixel_array extraction above; slope/intercept applied by pydicom

    return pixel_array


def _to_rgb_uint8(pixel_float: np.ndarray) -> np.ndarray:
    """
    Convert a [0.0, 1.0] float array of shape (H, W) or (H, W, C)
    to a uint8 RGB (H, W, 3) array.
    """
    pixel_uint8 = np.clip(pixel_float * 255.0, 0, 255).astype(np.uint8)

    if pixel_uint8.ndim == 2:
        # Grayscale → RGB
        return np.stack([pixel_uint8] * 3, axis=-1)
    elif pixel_uint8.ndim == 3:
        if pixel_uint8.shape[-1] == 1:
            return np.concatenate([pixel_uint8] * 3, axis=-1)
        elif pixel_uint8.shape[-1] == 3:
            return pixel_uint8
        elif pixel_uint8.shape[-1] == 4:
            return pixel_uint8[..., :3]  # Drop alpha
    raise ValueError(f"Unexpected pixel array shape: {pixel_uint8.shape}")


def validate_dicom_bytes(raw_bytes: bytes) -> bool:
    """
    Lightweight validation: checks DICOM magic bytes (DICM at offset 128).
    Returns True if the bytes look like a valid DICOM file.
    """
    if len(raw_bytes) < 132:
        return False
    return raw_bytes[128:132] == b"DICM"
