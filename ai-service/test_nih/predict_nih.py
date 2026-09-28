import os
import json
import argparse
from pathlib import Path

import torch
from torch import nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True

from nih_dataset import NIH_CLASSES

TEST_NIH_DIR = Path(__file__).resolve().parent


def predict_single_image(image_path: str, model_path: str, classes_config: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if classes_config and os.path.exists(classes_config):
        with open(classes_config, "r") as f:
            classes = json.load(f)["classes"]
    else:
        classes = NIH_CLASSES

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}. Train the model first.")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", classes)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded NIH model checkpoint from: {model_path}")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    print(f"\n--- NIH 14-Class Predictions for {Path(image_path).name} ---")
    results = []
    for cls_name, prob in zip(classes, probs):
        detected = prob >= 0.45
        results.append({"class": cls_name, "probability": round(float(prob), 4), "detected": detected})
        status = "Positive" if detected else "Negative"
        print(f"  [{status}] {cls_name:<20}: {prob*100:.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NIH 14-class DenseNet model on a single X-ray image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image")
    parser.add_argument("--model_path", type=str, default=str(TEST_NIH_DIR / "saved_models" / "nih_densenet121.pth"), help="Path to NIH model weights")
    args = parser.parse_args()

    predict_single_image(args.image, args.model_path)
