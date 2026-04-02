"""Weighted ensemble of multiple DR models.

Combines predictions from EfficientNet-B4, DINOv2, and/or ResNet-50
using optimizable weights. Supports test-time augmentation (TTA).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("dr_detection")


class ModelEnsemble(nn.Module):
    """Weighted ensemble of multiple models.

    Final prediction = sum(model_i_probs * weight_i) for all models.
    Weights are optimized on the validation set.
    """

    def __init__(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None,
        model_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.models = nn.ModuleList(models)
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        assert len(weights) == len(models), "Weights must match number of models"
        self.weights = weights

        logger.info(
            "Ensemble initialized: %d models, weights=%s",
            len(models), dict(zip(self.model_names, self.weights)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of model predictions.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Averaged logits (B, num_classes).
        """
        all_probs = []

        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                output = model(x)
                # Handle DINOv2 dict output
                if isinstance(output, dict):
                    output = output["logits"]
                probs = F.softmax(output, dim=-1)
                all_probs.append(probs * weight)

        # Weighted average
        ensemble_probs = torch.stack(all_probs).sum(dim=0)

        return ensemble_probs

    def set_weights(self, weights: list[float]) -> None:
        """Update ensemble weights."""
        assert len(weights) == len(self.models)
        self.weights = weights
        logger.info("Updated ensemble weights: %s", dict(zip(self.model_names, weights)))


def optimize_ensemble_weights(
    models: list[nn.Module],
    val_loader: "DataLoader",
    device: str = "cuda",
    num_classes: int = 5,
) -> list[float]:
    """Find optimal ensemble weights on validation set via grid search.

    Args:
        models: List of trained models.
        val_loader: Validation DataLoader.
        device: Device for inference.
        num_classes: Number of classes.

    Returns:
        Optimized weights summing to 1.0.
    """
    from sklearn.metrics import cohen_kappa_score

    # Get predictions from all models
    all_model_probs = []

    for model in models:
        model.eval()
        model_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                output = model(images)
                if isinstance(output, dict):
                    output = output["logits"]
                probs = F.softmax(output, dim=-1).cpu().numpy()
                model_probs.append(probs)
                all_labels.append(labels.numpy())

        all_model_probs.append(np.concatenate(model_probs, axis=0))

    true_labels = np.concatenate(all_labels, axis=0)

    # Grid search over weight combinations
    best_qwk = -1.0
    best_weights = [1.0 / len(models)] * len(models)
    step = 0.1

    n_models = len(models)
    if n_models == 2:
        for w0 in np.arange(0, 1.01, step):
            w1 = 1.0 - w0
            weights = [w0, w1]
            combined = sum(p * w for p, w in zip(all_model_probs, weights))
            preds = combined.argmax(axis=1)
            qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
            if qwk > best_qwk:
                best_qwk = qwk
                best_weights = weights

    elif n_models == 3:
        for w0 in np.arange(0, 1.01, step):
            for w1 in np.arange(0, 1.01 - w0, step):
                w2 = 1.0 - w0 - w1
                weights = [w0, w1, w2]
                combined = sum(p * w for p, w in zip(all_model_probs, weights))
                preds = combined.argmax(axis=1)
                qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_weights = weights

    logger.info("Optimized ensemble weights: %s (QWK=%.4f)", best_weights, best_qwk)
    return best_weights
