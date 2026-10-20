"""
CLI Script to export PyTorch models (DenseNet-121, ResNet-50, Swin-T) to ONNX format.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.models.checkpoint import load_checkpoint
from src.models.densenet import build_densenet121
from src.models.resnet import NIHResNet50
from src.models.swin import build_swin_tiny
from src.models.onnx_engine import export_model_to_onnx, ONNXInferenceEngine, benchmark_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scripts.export_onnx")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch NIH-14 models to ONNX format.")
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=["densenet121", "resnet50", "swin_tiny", "all"],
        help="Model architecture to export.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="test_nih/saved_models/nih_densenet121.pth",
        help="Path to trained PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx",
        help="Destination directory for exported .onnx models.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run inference latency benchmark (PyTorch vs ONNX Runtime).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_export = ["densenet121", "resnet50", "swin_tiny"] if args.model == "all" else [args.model]

    for m_name in models_to_export:
        print(f"\n=======================================================")
        print(f" Exporting {m_name.upper()} to ONNX Runtime format")
        print(f"=======================================================")

        if m_name == "densenet121":
            net = build_densenet121(num_classes=14, pretrained=False)
            ckpt_path = args.checkpoint if Path(args.checkpoint).is_file() else None
        elif m_name == "resnet50":
            net = NIHResNet50(num_classes=14, pretrained=False)
            res_path = Path("outputs/retrained/resnet50/best_model.pth")
            ckpt_path = str(res_path) if res_path.is_file() else None
        else:  # swin_tiny
            net = build_swin_tiny(num_classes=14, pretrained=False)
            swin_path = Path("outputs/retrained/swin_tiny/best_model.pth")
            ckpt_path = str(swin_path) if swin_path.is_file() else None

        if ckpt_path and Path(ckpt_path).is_file():
            logger.info("Loading weights from %s...", ckpt_path)
            load_checkpoint(net, ckpt_path, map_location="cpu")
        else:
            logger.warning("No checkpoint found for %s, exporting architecture with initialized weights.", m_name)

        onnx_file = out_dir / f"{m_name}.onnx"
        export_model_to_onnx(net, onnx_file)

        if args.benchmark:
            logger.info("Benchmarking PyTorch vs. ONNX Runtime...")
            engine = ONNXInferenceEngine(onnx_file)
            stats = benchmark_inference(net, engine)
            print("\nBenchmark Results:")
            print(f"  PyTorch Latency: {stats['pytorch']['mean_ms']} ms (p95: {stats['pytorch']['p95_ms']} ms)")
            print(f"  ONNX Latency:    {stats['onnx']['mean_ms']} ms (p95: {stats['onnx']['p95_ms']} ms)")
            print(f"  Speedup Factor:  {stats['speedup_factor']}x")


if __name__ == "__main__":
    main()
