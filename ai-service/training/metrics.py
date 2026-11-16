"""
Multi-Label Metrics Evaluation for NIH ChestX-ray14.
Computes macro/micro AUROC, sensitivity, specificity, F1, and calibrated thresholds.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _compute_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Computes binary classification metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "sensitivity": round(float(sensitivity), 4),
        "recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC AUC score robustly."""
    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        u = sum(np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores) for p in pos_scores)
        return float(u / (n_pos * n_neg))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    classes: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-label evaluation metrics."""
    thresholds = thresholds or {c: 0.5 for c in classes}
    per_class = {}
    aurocs = []

    for i, c in enumerate(classes):
        yt = y_true[:, i]
        ys = y_pred_probs[:, i]
        thresh = thresholds.get(c, 0.5)

        m = _compute_binary_metrics(yt, ys, thresh)
        auc = _compute_auroc(yt, ys)
        m["auroc"] = round(auc, 4) if not np.isnan(auc) else None
        if not np.isnan(auc):
            aurocs.append(auc)
        per_class[c] = m

    macro_auroc = round(float(np.mean(aurocs)), 4) if aurocs else 0.0
    macro_f1 = round(float(np.mean([m["f1"] for m in per_class.values()])), 4)
    macro_sensitivity = round(float(np.mean([m["sensitivity"] for m in per_class.values()])), 4)
    macro_specificity = round(float(np.mean([m["specificity"] for m in per_class.values()])), 4)

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "per_class": per_class,
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    classes: List[str],
    metric: str = "f1",
) -> Dict[str, float]:
    """Finds optimal decision thresholds per pathology using grid search."""
    thresholds = {}
    candidates = np.linspace(0.1, 0.9, 17)

    for i, c in enumerate(classes):
        yt = y_true[:, i]
        ys = y_pred_probs[:, i]
        best_val = -1.0
        best_th = 0.50

        for th in candidates:
            m = _compute_binary_metrics(yt, ys, th)
            val = m.get(metric, 0.0)
            if val > best_val:
                best_val = val
                best_th = float(th)

        thresholds[c] = round(best_th, 4)

    return thresholds


def save_metrics(metrics: Dict[str, Any], output_path: str | Path) -> None:
    """Saves metrics to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


__all__ = [
    "compute_multilabel_metrics",
    "find_optimal_thresholds",
    "save_metrics",
]
