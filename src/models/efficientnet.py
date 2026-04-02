"""EfficientNet-B4 with custom classification head.

Architecture per TRD §4.1:
- Backbone: EfficientNet-B4 (ImageNet pretrained via timm)
- Head: GAP → BN → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3) → Dense(5)
- Input: 380×380×3
"""

from __future__ import annotations

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn

from src.models.heads import ClassificationHead
from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class EfficientNetB4(nn.Module):
    """EfficientNet-B4 for DR classification."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout_pool: float = 0.4,
        dropout_hidden: float = 0.3,
        stochastic_depth_rate: float = 0.2,
    ) -> None:
        super().__init__()

        # Load pretrained backbone (no classifier)
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool="avg",
            drop_path_rate=stochastic_depth_rate,
        )

        # Get feature dimension
        in_features = self.backbone.num_features

        # Custom classification head
        self.head = ClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_pool=dropout_pool,
            dropout_hidden=dropout_hidden,
            use_batchnorm=True,
            activation="relu",
        )

        logger.info(
            "EfficientNet-B4 initialized: features=%d, classes=%d, pretrained=%s",
            in_features, num_classes, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 380, 380).
        Returns:
            Classification logits (B, num_classes).
        """
        features = self.backbone(x)  # (B, in_features) after GAP
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("EfficientNet-B4 backbone FROZEN")

    def unfreeze_backbone(self, ratio: float = 0.3) -> None:
        """Unfreeze top `ratio` of backbone layers (Phase 2 training).

        Args:
            ratio: Fraction of layers to unfreeze from the top.
        """
        all_params = list(self.backbone.named_parameters())
        num_to_unfreeze = max(1, int(len(all_params) * ratio))
        cutoff = len(all_params) - num_to_unfreeze

        for i, (name, param) in enumerate(all_params):
            param.requires_grad = i >= cutoff

        unfrozen = sum(1 for _, p in all_params if p.requires_grad)
        logger.info(
            "EfficientNet-B4 backbone: unfrozen %d/%d params (top %.0f%%)",
            unfrozen, len(all_params), ratio * 100,
        )

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def build_efficientnet(config: Config) -> EfficientNetB4:
    """Build EfficientNet-B4 from config.

    Args:
        config: Training config with model section.

    Returns:
        Initialized EfficientNetB4 model.
    """
    model_cfg = config.model
    head_cfg = model_cfg.head

    model = EfficientNetB4(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        hidden_dim=head_cfg.hidden_dim,
        dropout_pool=head_cfg.dropout_post_pool,
        dropout_hidden=head_cfg.dropout_post_hidden,
        stochastic_depth_rate=model_cfg.get("stochastic_depth_rate", 0.2),
    )

    logger.info(
        "Built EfficientNet-B4: total=%dM, trainable=%dM",
        model.get_total_params() // 1_000_000,
        model.get_trainable_params() // 1_000_000,
    )

    return model
