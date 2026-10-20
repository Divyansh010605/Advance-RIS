#!/usr/bin/env python3
"""
NIH ChestX-ray14 Complete Retraining Script — Phase 5

Retrains all three NIH classifiers (DenseNet-121, ResNet-50, Swin-T) sequentially
and exports the best checkpoint of each to ONNX. Safe for unattended overnight runs.

Usage:
  cd ai-service/
  python scripts/retrain_all.py \\
      --csv_path /path/to/Data_Entry_2017_v2020.csv \\
      --image_dir /path/to/images/ \\
      --output_dir outputs/retrained/ \\
      [--dry-run]

Prerequisites:
  pip install -r requirements.txt
  # Dataset must be downloaded; see DATASET_SETUP.md

NEVER overwrites existing checkpoints in test_nih/saved_models/.
Each model writes to its own subdirectory under --output_dir.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retrain_all")

# ── Config paths to use for each model ────────────────────────────────────────
MODEL_CONFIGS = {
    "densenet121": "configs/densenet121.yaml",
    "resnet50":    "configs/resnet50.yaml",
    "swin_tiny":   "configs/swin_tiny.yaml",
}

ONNX_EXPORTS = {
    "densenet121": ("densenet121", "outputs/onnx/densenet121.onnx"),
    "resnet50":    ("resnet50",    "outputs/onnx/resnet50.onnx"),
    "swin_tiny":   ("swin_tiny",   "outputs/onnx/swin_tiny.onnx"),
}


def run_step(label: str, cmd: list[str], cwd: Path) -> bool:
    """Run a subprocess step, log output, return True on success."""
    logger.info("─── %s ─────────────────────────────────────────", label)
    logger.info("Command: %s", " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        logger.error("FAILED: %s (exit=%d, %.1fs)", label, result.returncode, elapsed)
        return False
    logger.info("OK: %s (%.1fs)", label, elapsed)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NIH ChestX-ray14 Full Retraining Pipeline — all 3 models."
    )
    parser.add_argument("--csv_path", required=True, help="Path to Data_Entry_2017_v2020.csv")
    parser.add_argument("--image_dir", required=True, help="Root directory of NIH chest X-ray images.")
    parser.add_argument("--output_dir", default="outputs/retrained", help="Root output directory.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                        default=list(MODEL_CONFIGS.keys()), help="Models to retrain.")
    parser.add_argument("--skip_onnx", action="store_true", help="Skip ONNX export step.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run each model for 1 epoch / 2 batches to validate pipeline.")
    args = parser.parse_args()

    ai_service_root = Path(__file__).resolve().parent.parent
    python = sys.executable
    failures = []

    print("\n" + "=" * 65)
    print("  NIH ChestX-ray14 Retraining Pipeline")
    print(f"  Models:    {', '.join(args.models)}")
    print(f"  CSV:       {args.csv_path}")
    print(f"  Images:    {args.image_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Dry-run:   {args.dry_run}")
    print("=" * 65 + "\n")

    # ── Step 0: Patch configs with runtime paths ─────────────────────────────
    # Write override YAML files so --csv_path and --image_dir are injected
    # without modifying the checked-in config files.
    override_dir = ai_service_root / "outputs" / "retrained" / "_runtime_overrides"
    override_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        model_output = (Path(args.output_dir) / model_name).resolve()
        model_output.mkdir(parents=True, exist_ok=True)

        override_cfg = {
            "base_config": str(ai_service_root / MODEL_CONFIGS[model_name]),
            "data": {
                "csv_path": str(Path(args.csv_path).resolve()),
                "images_dir": str(Path(args.image_dir).resolve()),
            },
            "output": {
                "save_dir": str(model_output),
            },
        }
        import yaml  # noqa: PLC0415
        override_path = override_dir / f"{model_name}_override.yaml"
        with override_path.open("w") as f:
            yaml.safe_dump(override_cfg, f)
        logger.info("Runtime override config written: %s", override_path)

    # ── Step 1: Retrain each model ────────────────────────────────────────────
    for model_name in args.models:
        override_path = override_dir / f"{model_name}_override.yaml"
        cmd = [python, "-m", "src.train", "--config", str(override_path)]
        if args.dry_run:
            cmd.append("--dry-run")

        label = f"TRAIN {model_name.upper()}"
        ok = run_step(label, cmd, ai_service_root)
        if not ok:
            failures.append(f"Training: {model_name}")

    # ── Step 2: ONNX export ───────────────────────────────────────────────────
    if not args.skip_onnx:
        onnx_root = ai_service_root / "outputs" / "onnx"
        onnx_root.mkdir(parents=True, exist_ok=True)

        for model_name in args.models:
            arch_key, onnx_path = ONNX_EXPORTS[model_name]
            best_ckpt = ai_service_root / args.output_dir / model_name / "best_model.pth"
            if not best_ckpt.is_file():
                logger.warning("No best_model.pth found for %s; skipping ONNX export.", model_name)
                continue

            onnx_out = (ai_service_root / onnx_path).resolve()
            cmd = [
                python, "scripts/export_onnx.py",
                "--arch", arch_key,
                "--checkpoint", str(best_ckpt),
                "--output", str(onnx_out),
            ]
            label = f"ONNX EXPORT {model_name.upper()}"
            ok = run_step(label, cmd, ai_service_root)
            if not ok:
                failures.append(f"ONNX export: {model_name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if failures:
        print(f"  ⚠  COMPLETED WITH {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"     • {f}")
    else:
        print("  ✓  ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 65 + "\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
