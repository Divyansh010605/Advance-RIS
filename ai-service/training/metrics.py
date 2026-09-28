import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, thresholds: np.ndarray = None) -> dict:
    """
    Calculate per-class and aggregated metrics.
    y_true: (N, C)
    y_pred_probs: (N, C)
    thresholds: (C,)
    """
    num_classes = y_true.shape[1]
    if thresholds is None:
        thresholds = np.full(num_classes, 0.5)
        
    y_pred_binary = (y_pred_probs >= thresholds).astype(int)
    
    metrics = {
        "per_class": {},
        "macro": {},
        "micro": {}
    }
    
    # Per-class metrics
    for i in range(num_classes):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            auc = 0.0 # Handled case where only one class is present in true labels
            
        try:
            ap = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            ap = 0.0
            
        f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        prec = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        
        metrics["per_class"][i] = {
            "auroc": float(auc),
            "ap": float(ap),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec)
        }
        
    # Macro averages
    metrics["macro"] = {
        "auroc": float(np.mean([m["auroc"] for m in metrics["per_class"].values()])),
        "ap": float(np.mean([m["ap"] for m in metrics["per_class"].values()])),
        "f1": float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_binary, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, average="macro", zero_division=0))
    }
    
    return metrics

def find_optimal_thresholds(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    """
    Find optimal thresholds per class by maximizing F1 score.
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    
    for i in range(num_classes):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            pred_bin = (y_pred_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(y_true[:, i], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds[i] = best_thresh
        
    return optimal_thresholds
