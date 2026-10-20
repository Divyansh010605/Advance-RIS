import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

NIH_CLASSES: List[str] = [
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard clinical chest X-ray augmentations: slight scale, rotate, flip, jitter."""
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_default_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic validation/test transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class NIHChestXrayDataset(Dataset):
    """
    Production-grade Dataset loader for NIH ChestX-ray14.
    Produces (3, H, W) normalized tensor and 14-dim multi-hot float32 target vector.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False,
        image_size: int = 224,
        cache_in_memory: bool = False,
    ):
        self.records = records
        self.classes = classes if classes is not None else NIH_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.is_training = is_training
        self.image_size = image_size
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, Image.Image] = {}

        if transform is not None:
            self.transform = transform
        elif is_training:
            self.transform = get_default_train_transforms(image_size)
        else:
            self.transform = get_default_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def _get_multi_hot_target(self, labels: List[str]) -> torch.Tensor:
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for lbl in labels:
            norm_lbl = lbl.replace(" ", "_")
            if norm_lbl in self.class_to_idx:
                target[self.class_to_idx[norm_lbl]] = 1.0
            elif lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1.0
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        img_path = record["path"]

        if self.cache_in_memory and idx in self._cache:
            img = self._cache[idx]
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning("Failed loading image %s (%s); replacing with zero image", img_path, exc)
                img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            if self.cache_in_memory:
                self._cache[idx] = img

        tensor = self.transform(img)
        target = self._get_multi_hot_target(record.get("labels", []))
        return tensor, target
