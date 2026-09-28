"""
Pytest tests for the AI service.
"""
import base64
import os
import sys
import numpy as np
import pytest
import torch
from io import BytesIO
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_dummy_b64_image(width: int = 224, height: int = 224, color: tuple = (128, 128, 128)) -> str:
    """Creates a solid-colour base64 JPEG image."""
    img = Image.fromarray(np.full((height, width, 3), color, dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestTBModel:
    def test_model_loads_without_weights_file(self):
        """Model should load with ImageNet weights when no checkpoint exists."""
        from models.tb_model import TBModel
        m = TBModel()
        m.load()
        assert m.net is not None

    def test_model_returns_valid_prediction_score(self):
        """predict_with_gradients should return a float in [0, 1]."""
        from models.tb_model import TBModel
        from utils.image_processor import preprocess_image

        m = TBModel()
        m.load()

        dummy_b64 = make_dummy_b64_image()
        tensor, _ = preprocess_image(dummy_b64)

        score, fmaps, grads = m.predict_with_gradients(tensor)

        assert isinstance(score, float), "score should be float"
        assert 0.0 <= score <= 1.0, f"score out of range: {score}"
        assert fmaps is not None, "feature maps should not be None"

    def test_model_output_shapes(self):
        """Feature maps should have shape (1, 1024, 7, 7) for DenseNet-121."""
        from models.tb_model import TBModel
        from utils.image_processor import preprocess_image

        m = TBModel()
        m.load()

        dummy_b64 = make_dummy_b64_image()
        tensor, _ = preprocess_image(dummy_b64)
        _, fmaps, _ = m.predict_with_gradients(tensor)

        assert fmaps.shape[1] == 1024, f"Expected 1024 channels, got {fmaps.shape[1]}"


# ─── Image processor tests ───────────────────────────────────────────────────

class TestImageProcessor:
    def test_preprocess_returns_correct_tensor_shape(self):
        from utils.image_processor import preprocess_image
        b64 = make_dummy_b64_image()
        tensor, orig_np = preprocess_image(b64)
        assert tensor.shape == (1, 3, 224, 224)
        assert orig_np.shape == (224, 224, 3)

    def test_preprocess_invalid_b64_raises_value_error(self):
        from utils.image_processor import preprocess_image
        with pytest.raises(ValueError, match="Invalid base64"):
            preprocess_image("not_valid_base64!!!", "image/jpeg")

    def test_preprocess_strips_data_uri_prefix(self):
        from utils.image_processor import preprocess_image
        b64 = make_dummy_b64_image()
        with_prefix = f"data:image/jpeg;base64,{b64}"
        tensor, _ = preprocess_image(with_prefix)
        assert tensor.shape == (1, 3, 224, 224)


# ─── Grad-CAM tests ───────────────────────────────────────────────────────────

class TestGradCAM:
    def test_gradcam_returns_base64_strings(self):
        from models.tb_model import TBModel
        from utils.image_processor import preprocess_image
        from utils.gradcam import generate_gradcam

        m = TBModel()
        m.load()

        b64 = make_dummy_b64_image()
        tensor, orig_np = preprocess_image(b64)
        _, fmaps, grads = m.predict_with_gradients(tensor)

        heatmap_b64, overlay_b64 = generate_gradcam(orig_np, fmaps, grads)

        assert isinstance(heatmap_b64, str), "heatmap should be str"
        assert isinstance(overlay_b64, str), "overlay should be str"
        # Valid base64 should be decodable
        base64.b64decode(heatmap_b64)
        base64.b64decode(overlay_b64)

    def test_gradcam_handles_zero_gradients(self):
        """Should not raise when gradients are all zero."""
        from utils.gradcam import generate_gradcam
        orig = np.full((224, 224, 3), 128, dtype=np.uint8)
        fmaps = torch.zeros(1, 2048, 7, 7)
        grads = torch.zeros(1, 2048, 7, 7)
        hm, ov = generate_gradcam(orig, fmaps, grads)
        assert isinstance(hm, str)
        assert isinstance(ov, str)


# ─── Pydantic schema tests ────────────────────────────────────────────────────

class TestPredictionRequest:
    def test_valid_request(self):
        from schemas.prediction import PredictionRequest
        b64 = make_dummy_b64_image()
        req = PredictionRequest(image=b64, mime_type="image/jpeg")
        assert req.mime_type == "image/jpeg"

    def test_invalid_mime_type_raises(self):
        from schemas.prediction import PredictionRequest
        import pydantic
        b64 = make_dummy_b64_image()
        with pytest.raises(pydantic.ValidationError):
            PredictionRequest(image=b64, mime_type="application/pdf")

    def test_response_label_is_set(self):
        from schemas.prediction import PredictionResponse
        hm = base64.b64encode(b"fake").decode()
        ov = base64.b64encode(b"fake").decode()
        resp = PredictionResponse(prediction=0.8, heatmap=hm, overlay=ov)
        assert resp.label == "TB Positive"
        assert resp.confidence == 80

        resp2 = PredictionResponse(prediction=0.3, heatmap=hm, overlay=ov)
        assert resp2.label == "TB Negative"
        assert resp2.confidence == 70
