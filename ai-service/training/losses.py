"""
Loss functions for multi-label classification on imbalanced medical datasets.
Implements:
  - Class-imbalanced weighted BCE (BCEWithLogitsLoss with pos_weight)
  - Multi-label Focal Loss (Lin et al. 2017)
  - Asymmetric Loss (Ridnik et al. 2021)
"""
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_pos_weights(
    targets_or_records: Union[List[dict], torch.Tensor],
    num_classes: int,
    classes: Optional[List[str]] = None,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """
    Computes per-class positive class weights: pos_weight = (N_total - N_pos) / max(N_pos, 1).
    Clamped to max_weight to prevent gradient explosion on rare classes (e.g. Hernia).
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
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma

        loss = focal_factor * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification (Ridnik et al. 2021).
    Discards easy negatives via margin clipping and dynamically balances positive and negative gradients.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


__all__ = ["compute_pos_weights", "FocalLoss", "AsymmetricLoss"]
