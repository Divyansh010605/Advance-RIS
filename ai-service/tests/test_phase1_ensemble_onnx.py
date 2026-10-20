"""
Unit tests for Phase 1: Tri-Model Ensemble, TTA, Uncertainty Quantification, and ONNX Runtime Engine.
"""
import base64
import io
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from src.models.densenet import build_densenet121
from src.models.resnet import NIHResNet50
from src.models.swin import build_swin_tiny
from src.models.ensemble import MultiModelEnsemble, TestTimeAugmentation
from src.models.onnx_engine import export_model_to_onnx, ONNXInferenceEngine


@pytest.fixture
def dummy_xray_tensor():
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_image_b64():
    """Generates a base64 encoded synthetic 224x224 RGB image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestTTA:
    def test_tta_transform_steps(self, dummy_xray_tensor):
        for steps in [1, 2, 3, 4]:
            aug_list = TestTimeAugmentation.generate_augmented_tensors(dummy_xray_tensor, steps=steps)
            assert len(aug_list) == steps
            for tensor, name in aug_list:
                assert tensor.shape == dummy_xray_tensor.shape
                assert isinstance(name, str)


class TestMultiModelEnsemble:
    def test_single_model_ensemble(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=14)
        model = build_densenet121(num_classes=14, pretrained=False)
        ensemble.register_model("densenet121", model)

        assert len(ensemble.models) == 1
        assert ensemble.normalized_weights["densenet121"] == 1.0

        res = ensemble.predict(dummy_xray_tensor, use_tta=True, tta_steps=2)
        assert res["mean_probabilities"].shape == (1, 14)
        assert np.all(res["mean_probabilities"] >= 0.0)
        assert np.all(res["mean_probabilities"] <= 1.0)
        assert res["uncertainties"].shape == (1, 14)
        assert np.all(res["uncertainties"] >= 0.0)
        assert res["confidence_intervals"].shape == (1, 14, 2)
        assert res["is_uncertain"].shape == (1, 14)

    def test_tri_model_ensemble(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=14)
        m1 = build_densenet121(num_classes=14, pretrained=False)
        m2 = NIHResNet50(num_classes=14, pretrained=False)
        m3 = build_swin_tiny(num_classes=14, pretrained=False)

        ensemble.register_model("densenet121", m1)
        ensemble.register_model("resnet50", m2)
        ensemble.register_model("swin_tiny", m3)

        assert len(ensemble.models) == 3
        total_w = sum(ensemble.normalized_weights.values())
        assert abs(total_w - 1.0) < 1e-5

        res = ensemble.predict(dummy_xray_tensor, use_tta=True, tta_steps=2)
        assert res["total_passes"] == 6  # 3 models * 2 TTA passes
        assert res["mean_probabilities"].shape == (1, 14)
        assert res["uncertainties"].shape == (1, 14)

        # Check CI lower <= upper
        ci = res["confidence_intervals"]
        assert np.all(ci[:, :, 0] <= ci[:, :, 1])

    def test_structured_prediction_helper(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=14)
        model = build_densenet121(num_classes=14, pretrained=False)
        ensemble.register_model("densenet121", model)

        thresholds = {c: 0.5 for c in ensemble.classes}
        preds, fmaps, meta = ensemble.predict_multi_structured(dummy_xray_tensor, thresholds, use_tta=True)

        assert len(preds) == 14
        first_item = preds[0]
        assert "disease" in first_item
        assert "probability" in first_item
        assert "uncertainty" in first_item
        assert "confidence_interval" in first_item
        assert "is_uncertain" in first_item
        assert meta["tta_enabled"] is True
        assert "models_evaluated" in meta


class TestONNXEngine:
    def test_onnx_export_and_inference(self, tmp_path, dummy_xray_tensor):
        model = build_densenet121(num_classes=14, pretrained=False)
        model.eval()

        onnx_file = tmp_path / "test_dense.onnx"
        export_model_to_onnx(model, onnx_file, input_shape=(1, 3, 224, 224), dynamic_batch=True)
        assert onnx_file.is_file()
        assert onnx_file.stat().st_size > 1000

        engine = ONNXInferenceEngine(onnx_file, prefer_gpu=False)
        assert engine.active_provider == "CPUExecutionProvider"

        # Compare PyTorch vs ONNX numerical parity
        with torch.no_grad():
            pt_logits, _ = model(dummy_xray_tensor)
            pt_probs = torch.sigmoid(pt_logits).numpy()

        onnx_logits, onnx_probs = engine.predict(dummy_xray_tensor)
        assert onnx_probs.shape == (1, 14)
        np.testing.assert_allclose(pt_probs, onnx_probs, rtol=1e-3, atol=1e-3)


class TestAPIEnsembleEndpoint:
    def test_api_analyze_with_ensemble_and_tta(self, dummy_image_b64):
        from unittest.mock import patch
        from starlette.testclient import TestClient
        from app import app

        with patch("report_generation.vlm_report.VLMReportGenerator.load_vlm", return_value=None):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/analyze",
                    json={
                        "image": dummy_image_b64,
                        "mime_type": "image/jpeg",
                        "use_ensemble": True,
                        "use_tta": True,
                        "tta_steps": 2,
                    },
                )
                assert resp.status_code == 200
            data = resp.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 14

            # Verify Phase 1 uncertainty output in predictions
            sample_pred = data["predictions"][0]
            assert "uncertainty" in sample_pred
            assert sample_pred["uncertainty"] is not None
            assert "confidence_interval" in sample_pred
            assert "is_uncertain" in sample_pred

            # Verify metadata
            assert data["metadata"]["ensemble_enabled"] is True
            assert data["metadata"]["tta_enabled"] is True
            assert "models_evaluated" in data["metadata"]
