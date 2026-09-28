import csv
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class NIHDensenedDataset(Dataset):
    """
    Dedicated dataset loader for NIH ChestX-ray14 dataset (14 pathology classes).
    Supports multi-label multi-class classification.
    """
    def __init__(self, data_list: list[dict[str, Any]], classes: list[str] | None = None, transform=None):
        self.data_list = data_list
        self.classes = classes if classes is not None else NIH_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def _get_multi_hot_labels(self, labels: list[str]) -> torch.Tensor:
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in labels:
            # Map variations like "Pleural Thickening" or "Pleural_Thickening"
            norm_label = label.replace(" ", "_")
            if norm_label in self.class_to_idx:
                target[self.class_to_idx[norm_label]] = 1.0
            elif label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        img_path = item["path"]
        labels = item.get("labels", [])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading NIH image {img_path}: {e}. Falling back to blank image.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        target = self._get_multi_hot_labels(labels)
        return image, target


def parse_nih_csv(csv_path: str, images_dir: str) -> list[dict[str, Any]]:
    """
    Parses NIH Data_Entry_2017.csv file into structured data_list format.
    CSV columns: Image Index, Finding Labels, Patient ID, etc.
    Finding Labels are pipe-separated (e.g., "Atelectasis|Effusion").
    """
    csv_file = Path(csv_path)
    image_root = Path(images_dir)
    if not csv_file.is_file():
        raise FileNotFoundError(f"NIH CSV file not found: {csv_file}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"NIH image directory not found: {image_root}")

    # NIH downloads commonly place images in images_001/.../images rather than
    # directly under one directory. Index once so either layout works.
    image_index = {p.name: p for p in image_root.rglob("*.png")}
    image_index.update({p.name: p for p in image_root.rglob("*.jpg")})
    parsed_data = []
    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Image Index", "Finding Labels", "Patient ID"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            img_name = row.get("Image Index", "").strip()
            findings_str = row.get("Finding Labels", "").strip()
            patient_id = row.get("Patient ID", "").strip()

            if not img_name:
                continue

            img_path = image_index.get(img_name)
            if img_path is None:
                direct_path = image_root / img_name
                if direct_path.is_file():
                    img_path = direct_path
                else:
                    continue

            labels = [
                label.strip().replace(" ", "_")
                for label in findings_str.split("|")
                if label.strip() and label.strip() != "No Finding"
            ]

            parsed_data.append({
                "path": str(img_path),
                "labels": labels,
                "patient_id": patient_id
            })

    if not parsed_data:
        raise ValueError(f"No matching images found under {image_root} for {csv_file}")
    return parsed_data
