import os
import sys
import json
import argparse
import logging
from typing import List
from pathlib import Path

# Add root of ai-service and current directory to sys.path so modules are found regardless of how python is invoked
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

cwd = Path.cwd()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.densenet_multilabel import MultiLabelDenseNet
from datasets.unified_dataset import UnifiedChestXrayDataset
from training.trainer import MultiLabelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_list(path: str) -> list:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/processed/train.json")
    parser.add_argument("--val_data", type=str, default="data/processed/val.json")
    parser.add_argument("--classes_config", type=str, default="configs/classes.json")
    parser.add_argument("--save_dir", type=str, default="outputs/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, choices=["bce", "focal"], default="bce")
    
    args = parser.parse_args()
    
    with open(args.classes_config, 'r') as f:
        classes = json.load(f)["classes"]
        
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_list = load_data_list(args.train_data)
    val_list = load_data_list(args.val_data)
    
    if not train_list:
        logger.warning(f"No training data found at {args.train_data}. Creating dummy data for validation purposes.")
        train_list = [{"path": "dummy", "labels": [classes[0]]} for _ in range(10)]
        val_list = [{"path": "dummy", "labels": [classes[0]]} for _ in range(10)]

    train_dataset = UnifiedChestXrayDataset(train_list, classes)
    val_dataset = UnifiedChestXrayDataset(val_list, classes)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = MultiLabelDenseNet(num_classes=num_classes, init_weights=True).to(device)
    
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() 
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        classes=classes,
        scheduler=scheduler
    )
    
    trainer.fit(num_epochs=args.epochs)
    
if __name__ == "__main__":
    main()
