from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    return classification_report(y_true, y_pred, target_names=["No Pneumonia", "Pneumonia"], output_dict=True, zero_division=0)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def max_pair_iou(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    return float(max(box_iou(p, g) for p in pred_boxes for g in gt_boxes))


def greedy_mean_iou(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> float:
    """Greedy one-to-one box matching, returning mean matched IoU.

    Unmatched ground-truth boxes contribute 0, which makes missed regions visible in the metric.
    """
    if not gt_boxes:
        return 0.0
    remaining_pred = set(range(len(pred_boxes)))
    matches: list[float] = []
    for gt in gt_boxes:
        if not remaining_pred:
            matches.append(0.0)
            continue
        best_idx = max(remaining_pred, key=lambda i: box_iou(pred_boxes[i], gt))
        matches.append(box_iou(pred_boxes[best_idx], gt))
        remaining_pred.remove(best_idx)
    return float(np.mean(matches))


def localization_match_iou(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> float:
    """Consistent localization score used by CAM and R-CNN evaluation."""
    return greedy_mean_iou(pred_boxes, gt_boxes)


def localization_summary(iou_values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(iou_values), dtype=float)
    if values.size == 0:
        return {"mean_iou": float("nan"), "median_iou": float("nan"), "nonzero_iou_rate": float("nan")}
    return {
        "mean_iou": float(np.mean(values)),
        "median_iou": float(np.median(values)),
        "nonzero_iou_rate": float(np.mean(values > 0.0)),
    }
