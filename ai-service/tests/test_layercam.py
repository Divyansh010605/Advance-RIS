"""
Unit tests for LayerCAM, Grad-CAM, image preprocessing, and schemas.
"""
import base64
import io
from typing import Tuple
import numpy as np
import pytest
import torch
from PIL import Image

from xai.image_processor import preprocess_image, resize_with_padding, TARGET_SIZE
from xai.layercam import (
    generate_single_gradcam,
    generate_multi_gradcam,
    compute_layercam,
    compute_gradcam,
    compute_gradcam_plus_plus,
    compute_swin_cam,
)
from schemas.prediction import PredictionRequest, PredictionResponse


@pytest.fixture
def dummy_image_b64() -> str:
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def dummy_spatial_fmaps() -> Tuple[torch.Tensor, torch.Tensor]:
    fmaps = torch.randn(1, 1024, 7, 7)
    grads = torch.randn(1, 1024, 7, 7)
    return fmaps, grads


class TestImageProcessor:
    def test_preprocess_returns_correct_tensor_shape(self, dummy_image_b64):
        tensor, original_np = preprocess_image(dummy_image_b64)
        assert tensor.shape == (1, 3, 224, 224)
        assert original_np.shape == (300, 300, 3)
        assert tensor.dtype == torch.float32

    def test_preprocess_invalid_b64_raises_value_error(self):
        with pytest.raises(ValueError):
            preprocess_image("not-a-valid-base64-string!!!")

    def test_preprocess_strips_data_uri_prefix(self, dummy_image_b64):
        prefixed = f"data:image/jpeg;base64,{dummy_image_b64}"
        tensor, _ = preprocess_image(prefixed)
        assert tensor.shape == (1, 3, 224, 224)

    def test_resize_with_padding_aspect_ratio(self):
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        padded = resize_with_padding(img, TARGET_SIZE)
        assert padded.size == TARGET_SIZE


class TestGradCAM:
    def test_gradcam_returns_base64_strings(self, dummy_spatial_fmaps):
        fmaps, grads = dummy_spatial_fmaps
        original_np = np.zeros((224, 224, 3), dtype=np.uint8)

        heatmap_b64, overlay_b64 = generate_single_gradcam(original_np, fmaps, grads, method="layercam")
        assert isinstance(heatmap_b64, str)
        assert isinstance(overlay_b64, str)
        assert len(heatmap_b64) > 100
        assert len(overlay_b64) > 100

        # Verify they are valid JPEGs
        hm_bytes = base64.b64decode(heatmap_b64)
        ov_bytes = base64.b64decode(overlay_b64)
        assert hm_bytes[:2] == b"\xff\xd8"
        assert ov_bytes[:2] == b"\xff\xd8"

    def test_gradcam_handles_zero_gradients(self):
        original_np = np.zeros((224, 224, 3), dtype=np.uint8)
        fmaps = torch.randn(1, 1024, 7, 7)
        grads = torch.zeros(1, 1024, 7, 7)

        heatmap_b64, overlay_b64 = generate_single_gradcam(original_np, fmaps, grads)
        assert isinstance(heatmap_b64, str)
        assert isinstance(overlay_b64, str)

    def test_multi_gradcam_dictionary_output(self, dummy_spatial_fmaps):
        fmaps, grads = dummy_spatial_fmaps
        original_np = np.zeros((224, 224, 3), dtype=np.uint8)
        grad_dict = {0: grads, 1: grads}
        classes = ["Atelectasis", "Cardiomegaly"]

        res = generate_multi_gradcam(original_np, fmaps, grad_dict, classes)
        assert "Atelectasis" in res
        assert "Cardiomegaly" in res
        assert len(res["Atelectasis"][0]) > 0

    def test_swin_cam_reshape_transform(self):
        # Swin feature maps shape (B, 49, 768)
        fmaps = torch.randn(1, 49, 768)
        grads = torch.randn(1, 49, 768)
        cam = compute_swin_cam(fmaps, grads, method="layercam")
        assert cam.shape == (7, 7)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0


class TestPredictionRequest:
    def test_valid_request(self, dummy_image_b64):
        req = PredictionRequest(image=dummy_image_b64, mime_type="image/jpeg")
        assert req.mime_type == "image/jpeg"
        assert req.use_ensemble is True
        assert req.use_tta is False

    def test_invalid_mime_type_raises(self, dummy_image_b64):
        with pytest.raises(Exception):
            PredictionRequest(image=dummy_image_b64, mime_type="image/gif")

    def test_response_label_is_set(self):
        resp = PredictionResponse(prediction=0.85, heatmap="abc", overlay="def")
        assert resp.label == "TB Positive"

        resp_neg = PredictionResponse(prediction=0.10, heatmap="abc", overlay="def")
        assert resp_neg.label == "TB Negative"
