"""
Unit and integration tests for Multi-Model Ensemble, TTA, Uncertainty, and ONNX engine.
"""
import base64
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from models.densenet import TBModel, build_densenet121
from models.resnet import build_resnet50, NIHResNet50
from models.swin import build_swin_tiny, NIHSwin
from models.ensemble import MultiModelEnsemble, TestTimeAugmentation
from models.onnx_engine import export_model_to_onnx, ONNXInferenceEngine


@pytest.fixture
def dummy_image_b64() -> str:
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def dummy_xray_tensor() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)


class TestTBModel:
    def test_model_loads_without_weights_file(self):
        m = TBModel()
        m.load()
        assert m.is_loaded is True
        assert m.num_classes == 15
        assert len(m.classes) == 15

    def test_model_returns_valid_prediction_score(self, dummy_xray_tensor):
        m = TBModel()
        m.load()
        score, feat_maps, grad = m.predict_with_gradients(dummy_xray_tensor)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_model_output_shapes(self, dummy_xray_tensor):
        m = TBModel()
        m.load()
        results, feat_maps, grad_dict = m.predict_multi(dummy_xray_tensor)
        assert len(results) == 15
        assert isinstance(results[0]["probability"], float)
        assert feat_maps.shape[-2:] == (7, 7)


class TestTTA:
    def test_tta_transform_steps(self, dummy_xray_tensor):
        transforms = TestTimeAugmentation.generate_augmented_tensors(dummy_xray_tensor, steps=2)
        assert len(transforms) == 2
        assert transforms[0][1] == "identity"
        assert transforms[1][1] == "horizontal_flip"

        transforms_4 = TestTimeAugmentation.generate_augmented_tensors(dummy_xray_tensor, steps=4)
        assert len(transforms_4) == 4


class TestMultiModelEnsemble:
    def test_single_model_ensemble(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=15)
        dense = build_densenet121(num_classes=15, pretrained=False)
        ensemble.register_model("densenet121", dense)

        results = ensemble.predict(dummy_xray_tensor, use_tta=False)
        assert "aggregated_probabilities" in results
        assert results["aggregated_probabilities"].shape == (1, 15)
        assert "densenet121" in results["per_model_probabilities"]

    def test_tri_model_ensemble(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=15)
        dense = build_densenet121(num_classes=15, pretrained=False)
        res = build_resnet50(num_classes=15, pretrained=False)
        swin = build_swin_tiny(num_classes=15, pretrained=False)

        ensemble.register_model("densenet121", dense)
        ensemble.register_model("resnet50", res)
        ensemble.register_model("swin", swin)

        results = ensemble.predict(dummy_xray_tensor, use_tta=True, tta_steps=2)
        assert results["aggregated_probabilities"].shape == (1, 15)
        assert len(results["per_model_probabilities"]) == 3
        assert len(results["uncertainty_std"]) == 15
        assert len(results["confidence_intervals"]) == 15

    def test_structured_prediction_helper(self, dummy_xray_tensor):
        ensemble = MultiModelEnsemble(num_classes=15)
        dense = build_densenet121(num_classes=15, pretrained=False)
        ensemble.register_model("densenet121", dense)

        preds, fmaps, meta = ensemble.predict_multi_structured(dummy_xray_tensor)
        assert len(preds) == 15
        assert "disease" in preds[0]
        assert "probability" in preds[0]
        assert "uncertainty" in preds[0]
        assert "model_predictions" in preds[0]
        assert meta["models_evaluated"] == ["densenet121"]


class TestONNXEngine:
    def test_onnx_export_and_inference(self, tmp_path, dummy_xray_tensor):
        dense = build_densenet121(num_classes=15, pretrained=False)
        onnx_file = tmp_path / "test_densenet.onnx"

        export_model_to_onnx(dense, onnx_file, input_shape=(1, 3, 224, 224))
        assert onnx_file.exists()

        engine = ONNXInferenceEngine(onnx_file)
        out = engine.predict(dummy_xray_tensor)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (1, 15)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)


class TestAPIEnsembleEndpoint:
    @pytest.mark.asyncio
    async def test_api_analyze_with_ensemble_and_tta(self, dummy_image_b64):
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        # Ensure model and ensemble are initialized for testing
        if app_module.model is None:
            app_module.model = TBModel()
            app_module.model.load()
        if app_module.ensemble is None:
            app_module.ensemble = MultiModelEnsemble(num_classes=15, classes=app_module.model.classes)
            if app_module.model.net is not None:
                app_module.ensemble.register_model("densenet121", app_module.model.net)
        if app_module.vlm_generator is None:
            from vlm.generator import VLMReportGenerator
            app_module.vlm_generator = VLMReportGenerator(target_classes=app_module.model.classes)

        with patch("vlm.generator.VLMReportGenerator.load_vlm", return_value=None):
            transport = ASGITransport(app=app_module.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
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
                assert len(data["predictions"]) in (14, 15)
                assert "heatmaps" in data
                assert "report" in data
                assert "metadata" in data
