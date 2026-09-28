import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from training.metrics import calculate_metrics, find_optimal_thresholds

logger = logging.getLogger(__name__)

class MultiLabelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str,
        classes: list,
        scheduler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.classes = classes
        self.scheduler = scheduler
        
        self.best_macro_f1 = 0.0
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                logits, _ = self.model(images)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        
        # Optimize thresholds on validation set
        thresholds = find_optimal_thresholds(all_targets, all_probs)
        
        metrics = calculate_metrics(all_targets, all_probs, thresholds)
        metrics["val_loss"] = total_loss / len(self.val_loader)
        metrics["thresholds"] = thresholds.tolist()
        
        return metrics

    def save_checkpoint(self, is_best: bool, metrics: dict, epoch: int):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "classes": self.classes
        }
        
        last_path = os.path.join(self.save_dir, "last_model.pth")
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(state, best_path)
            
            # Save thresholds for inference
            thresh_dict = {cls: thresh for cls, thresh in zip(self.classes, metrics["thresholds"])}
            with open(os.path.join(self.save_dir, "thresholds.json"), "w") as f:
                json.dump(thresh_dict, f, indent=4)
                
            logger.info(f"Saved new best model with Macro F1: {metrics['macro']['f1']:.4f}")

    def fit(self, num_epochs: int, patience: int = 5):
        no_improve_epochs = 0
        
        for epoch in range(num_epochs):
            logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]
            macro_f1 = val_metrics["macro"]["f1"]
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
                
            is_best = macro_f1 > self.best_macro_f1
            if is_best:
                self.best_macro_f1 = macro_f1
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                
            self.save_checkpoint(is_best, val_metrics, epoch)
            
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
