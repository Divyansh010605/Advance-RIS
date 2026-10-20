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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
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
    """Computes ROC AUC score robustly without external dependencies."""
    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Fast Mann-Whitney U statistic fallback if sklearn not available
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # Rank all scores
        all_scores = np.concatenate([pos_scores, neg_scores])
        ranks = np.argsort(np.argsort(all_scores)) + 1
        rank_sum_pos = np.sum(ranks[:n_pos])
        u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    classes: List[str],
    step: float = 0.02,
) -> Dict[str, float]:
    """
    Finds per-class probability thresholds that maximize F1 on validation data.
    Sweeps thresholds in [0.05, 0.95].
    """
    optimal_thresholds: Dict[str, float] = {}
    threshold_candidates = np.arange(0.05, 0.95, step)

    for idx, cls in enumerate(classes):
        target = y_true[:, idx]
        probs = y_probs[:, idx]

        best_f1 = -1.0
        best_thresh = 0.5  # fallback default

        if np.sum(target == 1) == 0:
            optimal_thresholds[cls] = 0.5
            continue

        for thresh in threshold_candidates:
            metrics = _compute_binary_metrics(target, probs, float(thresh))
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_thresh = float(thresh)

        optimal_thresholds[cls] = round(best_thresh, 4)

    return optimal_thresholds


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    classes: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes per-class and macro-averaged metrics: AUROC, F1, sensitivity, specificity.
    """
    if thresholds is None:
        thresholds = {cls: 0.5 for cls in classes}

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    aurocs: List[float] = []
    f1s: List[float] = []
    sensitivities: List[float] = []
    specificities: List[float] = []

    for idx, cls in enumerate(classes):
        target = y_true[:, idx]
        probs = y_probs[:, idx]
        thresh = thresholds.get(cls, 0.5)

        auc = _compute_auroc(target, probs)
        bin_metrics = _compute_binary_metrics(target, probs, thresh)

        class_dict = {
            "auroc": round(auc, 4) if not np.isnan(auc) else None,
            "threshold": round(thresh, 4),
            "f1": bin_metrics["f1"],
            "sensitivity": bin_metrics["sensitivity"],
            "specificity": bin_metrics["specificity"],
            "precision": bin_metrics["precision"],
            "support": int(np.sum(target == 1)),
        }
        per_class_metrics[cls] = class_dict

        if not np.isnan(auc):
            aurocs.append(auc)
        f1s.append(bin_metrics["f1"])
        sensitivities.append(bin_metrics["sensitivity"])
        specificities.append(bin_metrics["specificity"])

    macro_auroc = round(float(np.mean(aurocs)), 4) if aurocs else 0.0
    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    macro_sens = round(float(np.mean(sensitivities)), 4) if sensitivities else 0.0
    macro_spec = round(float(np.mean(specificities)), 4) if specificities else 0.0

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sens,
        "macro_specificity": macro_spec,
        "per_class": per_class_metrics,
        "thresholds": thresholds,
    }


def save_metrics(metrics: Dict[str, Any], output_dir: str | Path, prefix: str = "metrics") -> Tuple[Path, Path]:
    """Saves metrics dictionary to both JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f"{prefix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = out_path / f"{prefix}_per_class.csv"
    per_class = metrics.get("per_class", {})
    if per_class:
        fieldnames = ["class", "auroc", "threshold", "f1", "sensitivity", "specificity", "precision", "support"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cls_name, vals in per_class.items():
                row = {"class": cls_name, **vals}
                writer.writerow(row)

    logger.info("Saved metrics to %s and %s", json_path, csv_path)
    return json_path, csv_path
