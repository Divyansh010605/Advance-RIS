"""
Multi-Epoch PyTorch Training Loop for NIH ChestX-ray14 Backbones.
Supports AMP, warmup + cosine learning rate scheduler, gradient clipping,
and macro-AUROC early stopping.
"""
import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import compute_multilabel_metrics, find_optimal_thresholds, save_metrics

logger = logging.getLogger(__name__)


class MultiLabelTrainer:
    """
    High-performance multi-label trainer with AMP, warmup + cosine LR schedule,
    gradient clipping, macro-AUROC early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str | Path,
        classes: List[str],
        test_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        warmup_epochs: int = 2,
        min_lr: float = 1e-6,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 5,
        amp: bool = True,
        channels_last: bool = True,
        dry_run: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.amp = amp and torch.cuda.is_available()
        self.channels_last = channels_last
        self.dry_run = dry_run

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.best_macro_auroc = -1.0
        self.best_model_weights = None
        self.history: List[Dict[str, Any]] = []

    def _adjust_lr(self, epoch: int) -> float:
        """Warmup + Cosine Annealing."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * ((epoch + 1) / max(1, self.warmup_epochs))
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(images)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if self.dry_run and batch_idx >= 2:
                break

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(images)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(logits, targets)
                probs = torch.sigmoid(logits)

            total_loss += loss.item()
            num_batches += 1
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

            if self.dry_run and batch_idx >= 2:
                break

        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)

        metrics = compute_multilabel_metrics(y_true, y_pred, self.classes)
        return total_loss / max(1, num_batches), metrics

    def fit(self) -> Dict[str, Any]:
        patience_counter = 0

        for epoch in range(self.epochs):
            lr = self._adjust_lr(epoch)
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.evaluate(self.val_loader)

            val_auroc = val_metrics["macro_auroc"]
            logger.info(
                "Epoch %02d/%02d [lr=%.2e] Train Loss: %.4f | Val Loss: %.4f | Val Macro AUROC: %.4f",
                epoch + 1, self.epochs, lr, train_loss, val_loss, val_auroc,
            )

            if val_auroc > self.best_macro_auroc:
                self.best_macro_auroc = val_auroc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save best checkpoint
                best_ckpt_path = self.save_dir / "best_checkpoint.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": self.best_model_weights,
                    "val_macro_auroc": val_auroc,
                    "val_metrics": val_metrics,
                }, best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        return {
            "best_macro_auroc": self.best_macro_auroc,
            "best_checkpoint": str(self.save_dir / "best_checkpoint.pth"),
        }


__all__ = ["MultiLabelTrainer"]
