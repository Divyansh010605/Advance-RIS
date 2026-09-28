"""Train a DenseNet-121 classifier on NIH ChestX-ray14 and save its checkpoint."""
import argparse
import json
import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from nih_dataset import NIH_CLASSES, NIHDensenedDataset, parse_nih_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_nih")


def split_by_patient(rows, validation_fraction, seed):
    patients = sorted({row["patient_id"] for row in rows})
    random.Random(seed).shuffle(patients)
    count = max(1, round(len(patients) * validation_fraction)) if len(patients) > 1 else 0
    validation_patients = set(patients[:count])
    train = [row for row in rows if row["patient_id"] not in validation_patients]
    valid = [row for row in rows if row["patient_id"] in validation_patients]
    if not train or not valid:
        raise ValueError("Need at least two distinct patients to create train and validation sets.")
    return train, valid


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on NIH ChestX-ray14 (14 findings).")
    default_csv = (
        here / "Data_Entry_2017_v2020.csv"
        if (here / "Data_Entry_2017_v2020.csv").exists()
        else here / "Data_Entry_2017.csv"
        if (here / "Data_Entry_2017.csv").exists()
        else here / "data/nih/Data_Entry_2017.csv"
    )
    default_images = here / "images" if (here / "images").is_dir() else here / "data/nih/images"

    parser.add_argument("--csv_path", type=Path, default=default_csv)
    parser.add_argument("--images_dir", type=Path, default=default_images)
    parser.add_argument("--classes_config", type=Path, default=here / "nih_classes.json")
    parser.add_argument("--save_dir", type=Path, default=here / "saved_models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Initialize from ImageNet weights (downloads weights if needed).")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.val_fraction < 1:
        parser.error("epochs and batch_size must be positive; val_fraction must be between 0 and 1")
    with args.classes_config.open(encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    if classes != NIH_CLASSES:
        raise ValueError("classes_config must list the 14 ChestX-ray14 findings in NIH order.")

    rows = parse_nih_csv(str(args.csv_path), str(args.images_dir))
    train_rows, val_rows = split_by_patient(rows, args.val_fraction, args.seed)
    logger.info("Images: %d | Train: %d | Validation: %d", len(rows), len(train_rows), len(val_rows))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    train_loader = DataLoader(
        NIHDensenedDataset(train_rows, classes), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NIHDensenedDataset(val_rows, classes), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    weights = models.DenseNet121_Weights.DEFAULT if args.pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.save_dir / "nih_densenet121.pth"
    best_loss = float("inf")

    total_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_count = 0.0, 0
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)

            if batch_idx % 100 == 0 or batch_idx == total_batches:
                logger.info(
                    "Epoch %d/%d [Batch %d/%d (%.1f%%)] - Running Train Loss: %.4f",
                    epoch, args.epochs, batch_idx, total_batches, (batch_idx / total_batches) * 100,
                    train_loss / train_count
                )

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                loss = criterion(model(images), targets)
                val_loss += loss.item() * images.size(0)
                val_count += images.size(0)
        train_loss /= train_count
        val_loss /= val_count
        logger.info("Epoch %d/%d - train loss %.4f - val loss %.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "architecture": "densenet121", "num_classes": len(classes), "classes": classes,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "image_size": 224,
            }, checkpoint_path)
            logger.info("Saved best checkpoint to %s", checkpoint_path)

    logger.info("Training complete. Best validation loss %.4f; model: %s", best_loss, checkpoint_path)


if __name__ == "__main__":
    main()
