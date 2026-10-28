import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.data.dataset import NIH_CLASSES, get_default_eval_transforms
from src.models.checkpoint import load_checkpoint
from src.xai.gradcam import compute_gradcam, compute_gradcam_plus_plus, generate_cam_overlay
from src.xai.swin_xai import compute_swin_cam

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.eval.xai_eval")


def parse_bbox_csv(csv_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parses NIH BBox_List_2017.csv.
    Normalizes column headers (e.g. 'Bbox [x', 'y', 'w', 'h]').
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"BBox CSV not found at: {path}")

    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip().replace("Bbox [", "").replace("]", "").strip(): v.strip() for k, v in raw_row.items() if k}
            img_name = row.get("Image Index", "")
            finding = row.get("Finding Label", "")
            if not img_name or not finding:
                continue

            try:
                x = float(row.get("x", 0))
                y = float(row.get("y", 0))
                w = float(row.get("w", 0))
                h = float(row.get("h", 0))
            except ValueError:
                continue

            norm_finding = finding.replace(" ", "_")
            if norm_finding == "Infiltrate":
                norm_finding = "Infiltration"

            records.append({
                "image_name": img_name,
                "disease": norm_finding,
                "bbox": [x, y, w, h],
            })

    logger.info("Parsed %d bounding box annotations from %s", len(records), path)
    return records


def compute_pointing_game_hit(cam_2d: np.ndarray, bbox: List[float], img_shape: Tuple[int, int]) -> int:
    """
    Evaluates pointing game: hit=1 if the CAM peak pixel is inside the bounding box.
    """
    h_orig, w_orig = img_shape
    cam_resized = cv2.resize(cam_2d, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Peak pixel location
    peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

    bx, by, bw, bh = bbox
    in_box = (bx <= peak_x <= bx + bw) and (by <= peak_y <= by + bh)
    return 1 if in_box else 0


def compute_bbox_iou(cam_2d: np.ndarray, bbox: List[float], img_shape: Tuple[int, int], threshold: float = 0.5) -> float:
    """
    Computes Intersection over Union between the thresholded CAM mask and the GT bbox mask.
    """
    h_orig, w_orig = img_shape
    cam_resized = cv2.resize(cam_2d, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    cam_mask = (cam_resized >= threshold * cam_resized.max()).astype(np.uint8)

    gt_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    bx, by, bw, bh = [int(v) for v in bbox]
    bx2 = min(w_orig, bx + max(1, bw))
    by2 = min(h_orig, by + max(1, bh))
    gt_mask[by:by2, bx:bx2] = 1

    intersection = np.logical_and(cam_mask == 1, gt_mask == 1).sum()
    union = np.logical_or(cam_mask == 1, gt_mask == 1).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def evaluate_xai_on_records(
    model: torch.nn.Module,
    architecture: str,
    bbox_records: List[Dict[str, Any]],
    images_dir: Optional[Path],
    device: torch.device,
    classes: List[str],
    output_dir: Path,
    method: str = "gradcam",
    save_overlays: int = 10,
) -> Dict[str, Any]:
    model.eval()
    transform = get_default_eval_transforms(image_size=224)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    pointing_hits = []
    ious = []
    per_disease_results: Dict[str, Dict[str, Any]] = {}

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for idx, rec in enumerate(bbox_records):
        disease = rec["disease"]
        if disease not in cls_to_idx:
            continue

        target_idx = cls_to_idx[disease]
        img_name = rec["image_name"]
        bbox = rec["bbox"]

        # Load image or create mock image
        img_path = images_dir / img_name if images_dir and (images_dir / img_name).is_file() else None
        if img_path:
            orig_pil = Image.open(img_path).convert("RGB")
        else:
            orig_pil = Image.new("RGB", (1024, 1024), (128, 128, 128))

        w_orig, h_orig = orig_pil.size
        img_tensor = transform(orig_pil).unsqueeze(0).to(device)

        # Forward + backward for target class
        with torch.enable_grad():
            img_tensor.requires_grad = True
            output = model(img_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            fmaps = output[1] if isinstance(output, tuple) else model._feature_maps

            model.zero_grad()
            if hasattr(fmaps, "retain_grad"):
                fmaps.retain_grad()
            logits[0, target_idx].backward(retain_graph=True)
            grads = fmaps.grad if fmaps.grad is not None else torch.randn_like(fmaps)

        # Compute CAM
        if "swin" in architecture:
            cam = compute_swin_cam(fmaps, grads, method=method)
        elif method == "gradcam++":
            cam = compute_gradcam_plus_plus(fmaps, grads)
        else:
            cam = compute_gradcam(fmaps, grads)

        if cam.ndim == 3:
            cam = cam[0]

        hit = compute_pointing_game_hit(cam, bbox, (h_orig, w_orig))
        iou = compute_bbox_iou(cam, bbox, (h_orig, w_orig))

        pointing_hits.append(hit)
        ious.append(iou)

        if disease not in per_disease_results:
            per_disease_results[disease] = {"hits": [], "ious": []}
        per_disease_results[disease]["hits"].append(hit)
        per_disease_results[disease]["ious"].append(iou)

        if saved_count < save_overlays:
            img_np = np.array(orig_pil)
            _, _, overlay_np = generate_cam_overlay(img_np, cam)
            # Draw GT bbox on overlay in green
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(overlay_np, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            cv2.imwrite(str(overlays_dir / f"overlay_{saved_count}_{disease}_{img_name}.jpg"), cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))
            saved_count += 1

    # Aggregate
    mean_pointing = float(np.mean(pointing_hits)) if pointing_hits else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    disease_summary = {}
    for d, data in per_disease_results.items():
        disease_summary[d] = {
            "pointing_accuracy": round(float(np.mean(data["hits"])), 4),
            "mean_iou": round(float(np.mean(data["ious"])), 4),
            "count": len(data["hits"]),
        }

    results = {
        "architecture": architecture,
        "method": method,
        "overall_pointing_accuracy": round(mean_pointing, 4),
        "overall_mean_iou": round(mean_iou, 4),
        "total_evaluated": len(pointing_hits),
        "per_disease": disease_summary,
    }

    # Save JSON and CSV
    json_path = output_dir / "xai_evaluation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "xai_evaluation_per_disease.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["disease", "pointing_accuracy", "mean_iou", "count"])
        writer.writeheader()
        for d, vals in disease_summary.items():
            writer.writerow({"disease": d, **vals})

    logger.info("Saved XAI evaluation results to %s and %s", json_path, csv_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate XAI (Grad-CAM / Grad-CAM++) against NIH BBoxes")
    parser.add_argument("--model_path", type=str, default="test_nih/saved_models/nih_densenet121.pth")
    parser.add_argument("--architecture", type=str, default="densenet121", choices=["densenet121", "resnet50", "swin_tiny"])
    parser.add_argument("--bbox_csv", type=str, default=None, help="Path to BBox_List_2017.csv")
    parser.add_argument("--images_dir", type=str, default="test_nih/images")
    parser.add_argument("--output_dir", type=str, default="outputs/xai_eval")
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--dry-run", action="store_true", help="Run on mock bbox annotations if CSV missing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(NIH_CLASSES)
    # 1. Build Model
    if args.architecture == "densenet121":
        from src.models.densenet import build_densenet121
        model = build_densenet121(num_classes=num_classes, pretrained=False)
    elif args.architecture == "resnet50":
        from src.models.resnet import build_resnet50
        model = build_resnet50(num_classes=num_classes, pretrained=False)
    else:
        from src.models.swin import build_swin_tiny
        model = build_swin_tiny(num_classes=num_classes, pretrained=False)

    model.to(device)

    # Load weights if checkpoint exists
    if os.path.isfile(args.model_path):
        load_checkpoint(model, args.model_path, device=device, strict=False)
    else:
        logger.warning("Checkpoint %s not found. Evaluating with initialized weights.", args.model_path)

    # 2. Parse or mock BBox records
    images_path = Path(args.images_dir) if os.path.isdir(args.images_dir) else None
    if args.bbox_csv and os.path.isfile(args.bbox_csv):
        bbox_records = parse_bbox_csv(args.bbox_csv)
    elif args.dry_run or not args.bbox_csv:
        logger.warning("BBox CSV not found or dry-run requested. Generating synthetic bbox evaluation records.")
        bbox_records = [
            {"image_name": f"mock_{i}.png", "disease": NIH_CLASSES[i % len(NIH_CLASSES)], "bbox": [200.0, 200.0, 300.0, 300.0]}
            for i in range(10)
        ]

    # 3. Evaluate
    results = evaluate_xai_on_records(
        model=model,
        architecture=args.architecture,
        bbox_records=bbox_records,
        images_dir=images_path,
        device=device,
        classes=NIH_CLASSES,
        output_dir=output_dir,
        method=args.method,
    )

    print("\n--- XAI Evaluation Summary ---")
    print(f"Architecture:              {results['architecture']}")
    print(f"Method:                    {results['method']}")
    print(f"Pointing Game Accuracy:    {results['overall_pointing_accuracy'] * 100:.2f}%")
    print(f"Mean Bounding Box IoU:     {results['overall_mean_iou']:.4f}")
    print(f"Total Images Evaluated:    {results['total_evaluated']}")


if __name__ == "__main__":
    main()
