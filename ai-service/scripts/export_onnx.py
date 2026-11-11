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
from src.models.resnet import build_resnet50, NIHResNet50
from src.models.swin import build_swin, NIHSwin
from src.models.onnx_engine import export_model_to_onnx, ONNXInferenceEngine, benchmark_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scripts.export_onnx")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch NIH-14 models to ONNX format.")
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=["densenet121", "resnet50", "swin", "swin_tiny", "all"],
        help="Model architecture to export.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
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

    models_to_export = ["densenet121", "resnet50", "swin"] if args.model == "all" else [args.model]

    for m_name in models_to_export:
        print(f"\n=======================================================")
        print(f" Exporting {m_name.upper()} to ONNX Runtime format")
        print(f"=======================================================")

        if m_name == "densenet121":
            default_ckpt = Path("outputs/retrained/densenet121/densenet121_cxr.pth")
            ckpt_path = args.checkpoint or (str(default_ckpt) if default_ckpt.is_file() else None)
            net = build_densenet121(num_classes=14, pretrained=False, weights_path=ckpt_path)
        elif m_name == "resnet50":
            default_ckpt = Path("outputs/retrained/resnet50/resnet50_cxr.pth")
            ckpt_path = args.checkpoint or (str(default_ckpt) if default_ckpt.is_file() else None)
            net = build_resnet50(num_classes=14, pretrained=False, weights_path=ckpt_path)
        else:  # swin
            candidates = [
                Path("outputs/retrained/swin/pretrained_SwinCheX_classifier.pth"),
                Path("outputs/retrained/swin/swin_base_cxr14.pth"),
                Path("outputs/retrained/swin/best_model.pth"),
            ]
            ckpt_path = args.checkpoint
            if not ckpt_path:
                for c in candidates:
                    if c.is_file():
                        ckpt_path = str(c)
                        break
            net = build_swin(num_classes=14, pretrained=True, weights_path=ckpt_path)

        if ckpt_path and Path(ckpt_path).is_file():
            logger.info("Exporting %s using checkpoint: %s", m_name, ckpt_path)
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
