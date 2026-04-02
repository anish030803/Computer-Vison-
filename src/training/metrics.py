"""Training metrics: QWK, per-class precision/recall/F1, AUC.

QWK (Quadratic Weighted Kappa) is the primary metric for DR grading.
It measures inter-rater agreement accounting for ordinal distance.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("dr_detection")


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa (QWK).

    Primary evaluation metric for DR grading. Penalizes predictions
    that are far from the true label more heavily.

    Args:
        y_true: True labels (0-4).
        y_pred: Predicted labels (0-4).

    Returns:
        QWK score in [-1, 1]. Higher is better, 1.0 = perfect agreement.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute comprehensive metrics for DR classification.

    Args:
        y_true: True labels (N,).
        y_pred: Predicted labels (N,).
        y_probs: Predicted probabilities (N, num_classes). Optional.
        class_names: Names for each class.

    Returns:
        Dict of metric name -> value.
    """
    if class_names is None:
        class_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

    num_classes = len(class_names)
    metrics: dict[str, float] = {}

    # Primary: QWK
    metrics["qwk"] = quadratic_weighted_kappa(y_true, y_pred)

    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class recall (sensitivity) — critical for Severe & Proliferative
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    for i, name in enumerate(class_names):
        metrics[f"recall_{name.replace(' ', '_').lower()}"] = float(per_class_recall[i]) if i < len(per_class_recall) else 0.0

    # Per-class precision
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    for i, name in enumerate(class_names):
        metrics[f"precision_{name.replace(' ', '_').lower()}"] = float(per_class_precision[i]) if i < len(per_class_precision) else 0.0

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    for i, name in enumerate(class_names):
        metrics[f"f1_{name.replace(' ', '_').lower()}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

    # Macro averages
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC metrics (require probability predictions)
    if y_probs is not None and y_probs.shape[1] == num_classes:
        try:
            # One-vs-rest AUC-ROC
            metrics["auc_roc_macro"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro"
            )

            # Per-class AUC-PR (especially important for minority classes)
            for i, name in enumerate(class_names):
                binary_true = (y_true == i).astype(int)
                if binary_true.sum() > 0:
                    metrics[f"auc_pr_{name.replace(' ', '_').lower()}"] = average_precision_score(
                        binary_true, y_probs[:, i]
                    )
        except ValueError as e:
            logger.debug("AUC computation failed: %s", e)

    return metrics


class RunningMetrics:
    """Accumulate predictions across batches for epoch-level metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.all_preds: list[np.ndarray] = []
        self.all_labels: list[np.ndarray] = []
        self.all_probs: list[np.ndarray] = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: Optional[np.ndarray] = None,
        loss: float = 0.0,
    ) -> None:
        """Add a batch of predictions.

        Args:
            preds: Predicted class indices (B,).
            labels: True labels (B,).
            probs: Predicted probabilities (B, num_classes).
            loss: Batch loss value.
        """
        self.all_preds.append(preds)
        self.all_labels.append(labels)
        if probs is not None:
            self.all_probs.append(probs)
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> dict[str, float]:
        """Compute epoch-level metrics from accumulated batches."""
        y_true = np.concatenate(self.all_preds)
        y_pred = np.concatenate(self.all_labels)
        y_probs = np.concatenate(self.all_probs) if self.all_probs else None

        # Note: preds and labels are intentionally swapped in naming
        # all_preds stores model predictions, all_labels stores ground truth
        metrics = compute_all_metrics(y_pred, y_true, y_probs)
        metrics["loss"] = self.total_loss / max(self.num_batches, 1)

        return metrics
