"""Loss functions: weighted cross-entropy, focal loss, MixUp loss.

All losses support label smoothing and class weights as configured
in the training YAML configs.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("dr_detection")


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights and label smoothing."""

    def __init__(
        self,
        class_weights: Optional[list[float]] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the loss for well-classified examples, focusing training
    on hard, misclassified samples (especially minority classes).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[list[float]] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs).pow(self.gamma)

        # Weighted focal loss
        loss = -focal_weight * smooth_targets * log_probs

        # Apply class weights
        if self.class_weights is not None:
            weight = self.class_weights[targets].unsqueeze(1)
            loss = loss * weight

        return loss.sum(dim=-1).mean()


class CombinedLoss(nn.Module):
    """Weighted combination of cross-entropy and focal loss."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
        class_weights: Optional[list[float]] = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.ce_loss = WeightedCrossEntropyLoss(class_weights, label_smoothing)
        self.focal_loss = FocalLoss(focal_gamma, class_weights, label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.ce_loss(logits, targets)
            + self.focal_weight * self.focal_loss(logits, targets)
        )


def build_loss(loss_config, class_weights: Optional[list[float]] = None) -> nn.Module:
    """Build loss function from config.

    Args:
        loss_config: Loss section of training config.
        class_weights: Dynamic class weights (overrides config if provided).

    Returns:
        Loss module.
    """
    name = loss_config.name
    label_smoothing = loss_config.get("label_smoothing", 0.0)

    # Use provided class_weights or from config
    weights = class_weights
    if weights is None:
        weights = loss_config.get("class_weights")
    if isinstance(weights, dict):
        weights = [weights[i] for i in sorted(weights.keys())]

    if name == "cross_entropy":
        loss = WeightedCrossEntropyLoss(weights, label_smoothing)
    elif name == "focal_loss":
        gamma = loss_config.get("focal_loss_gamma", 2.0)
        loss = FocalLoss(gamma, weights, label_smoothing)
    elif name == "combined":
        gamma = loss_config.get("focal_loss_gamma", 2.0)
        loss = CombinedLoss(0.5, 0.5, weights, label_smoothing, gamma)
    else:
        raise ValueError(f"Unknown loss: {name}")

    logger.info("Built loss: %s (smoothing=%.2f, weights=%s)", name, label_smoothing, weights is not None)
    return loss
