import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_pos_weights(
    targets_or_records: List[dict] | torch.Tensor,
    num_classes: int,
    classes: Optional[List[str]] = None,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """
    Computes per-class positive class weights: pos_weight = (N_total - N_pos) / max(N_pos, 1).
    Clamped to max_weight to prevent gradient explosion on extremely rare classes (e.g. Hernia).
    """
    if isinstance(targets_or_records, torch.Tensor):
        total = targets_or_records.shape[0]
        pos_counts = targets_or_records.sum(dim=0)
    else:
        assert classes is not None, "classes must be provided when records are passed"
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        pos_counts = torch.zeros(num_classes, dtype=torch.float32)
        total = len(targets_or_records)

        for rec in targets_or_records:
            for lbl in rec.get("labels", []):
                norm_lbl = lbl.replace(" ", "_")
                if norm_lbl in cls_to_idx:
                    pos_counts[cls_to_idx[norm_lbl]] += 1.0

    neg_counts = float(total) - pos_counts
    weights = neg_counts / torch.clamp(pos_counts, min=1.0)
    weights = torch.clamp(weights, min=1.0, max=max_weight)

    logger.info("Computed per-class pos_weights (min=%.2f, max=%.2f)", weights.min().item(), weights.max().item())
    return weights


class FocalLoss(nn.Module):
    """
    Multi-label focal loss for class imbalance:
    FL = - alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        loss = focal_weight * bce_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(
    loss_type: str = "bce_pos_weight",
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Instantiates the multi-label loss function."""
    if device is not None and pos_weight is not None:
        pos_weight = pos_weight.to(device)

    if loss_type == "bce_pos_weight":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
