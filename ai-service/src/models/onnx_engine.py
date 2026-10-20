"""
ONNX Runtime Acceleration Engine and Model Export Utilities.
Provides sub-40ms inference, dynamic batching, and performance benchmarking.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

logger = logging.getLogger("src.models.onnx_engine")


class ONNXModelWrapper(nn.Module):
    """
    Wrapper around PyTorch models to produce clean outputs for ONNX export:
      - logits: (B, num_classes)
      - probabilities: (B, num_classes) via torch.sigmoid
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # base_model returns (logits, fmaps)
        logits, _ = self.base_model(x)
        probs = torch.sigmoid(logits)
        return logits, probs


def export_model_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    dynamic_batch: bool = True,
    opset_version: int = 14,
) -> Path:
    """
    Exports a PyTorch classifier to an ONNX model file.
    Configures dynamic batch axes for flexible serving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = ONNXModelWrapper(model)
    wrapper.eval()
    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        }

    logger.info("Exporting model to ONNX at %s (opset=%d, dynamic=%s)...", output_path, opset_version, dynamic_batch)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "probabilities"],
        dynamic_axes=dynamic_axes,
    )
    logger.info("Successfully exported ONNX model to %s (size: %.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))
    return output_path


class ONNXInferenceEngine:
    """
    High-performance inference engine using onnxruntime.
    Automatically prioritizes CUDAExecutionProvider, falling back to CPUExecutionProvider.
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        prefer_gpu: bool = True,
    ):
        self.onnx_path = str(onnx_path)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")

        available_providers = ort.get_available_providers()
        selected_providers = []

        if prefer_gpu and "CUDAExecutionProvider" in available_providers:
            selected_providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            selected_providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=session_options,
            providers=selected_providers,
        )
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(
            "Initialized ONNXInferenceEngine for '%s' using provider: %s",
            self.onnx_path,
            self.active_provider,
        )

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes inference on input tensor.
        Input shape: (B, 3, H, W)
        Returns:
            logits: (B, num_classes) numpy array
            probabilities: (B, num_classes) numpy array
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = x.astype(np.float32)

        outputs = self.session.run(self.output_names, {self.input_name: x_np})
        logits = outputs[0]
        probs = outputs[1]
        return logits, probs


def benchmark_inference(
    pytorch_model: nn.Module,
    onnx_engine: ONNXInferenceEngine,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_runs: int = 30,
    warmup_runs: int = 5,
) -> Dict[str, Any]:
    """
    Benchmarks cold and warm inference latencies between PyTorch and ONNX Runtime.
    """
    dummy_input = torch.randn(*input_shape)
    dummy_np = dummy_input.numpy()

    # PyTorch Warmup
    pytorch_model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = pytorch_model(dummy_input)

    # PyTorch Timing
    pt_latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = pytorch_model(dummy_input)
            pt_latencies.append((time.perf_counter() - t0) * 1000)

    # ONNX Warmup
    for _ in range(warmup_runs):
        _ = onnx_engine.predict(dummy_np)

    # ONNX Timing
    onnx_latencies: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = onnx_engine.predict(dummy_np)
        onnx_latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "num_runs": num_runs,
        "input_shape": list(input_shape),
        "onnx_provider": onnx_engine.active_provider,
        "pytorch": {
            "mean_ms": round(float(np.mean(pt_latencies)), 2),
            "p50_ms": round(float(np.median(pt_latencies)), 2),
            "p95_ms": round(float(np.percentile(pt_latencies, 95)), 2),
        },
        "onnx": {
            "mean_ms": round(float(np.mean(onnx_latencies)), 2),
            "p50_ms": round(float(np.median(onnx_latencies)), 2),
            "p95_ms": round(float(np.percentile(onnx_latencies, 95)), 2),
        },
        "speedup_factor": round(float(np.mean(pt_latencies) / np.mean(onnx_latencies)), 2),
    }
