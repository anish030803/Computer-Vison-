"""ResNet-50 baseline for comparative evaluation.

Uses same classification head structure as EfficientNet-B4 for fair comparison.
Architecture per TRD §4.3.
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from src.models.heads import ClassificationHead
from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class ResNet50Baseline(nn.Module):
    """ResNet-50 baseline for DR classification."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout_pool: float = 0.4,
        dropout_hidden: float = 0.3,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        in_features = self.backbone.num_features

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
            "ResNet-50 initialized: features=%d, classes=%d, pretrained=%s",
            in_features, num_classes, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("ResNet-50 backbone FROZEN")

    def unfreeze_backbone(self, ratio: float = 0.3) -> None:
        all_params = list(self.backbone.named_parameters())
        num_to_unfreeze = max(1, int(len(all_params) * ratio))
        cutoff = len(all_params) - num_to_unfreeze

        for i, (name, param) in enumerate(all_params):
            param.requires_grad = i >= cutoff

        unfrozen = sum(1 for _, p in all_params if p.requires_grad)
        logger.info(
            "ResNet-50 backbone: unfrozen %d/%d params (top %.0f%%)",
            unfrozen, len(all_params), ratio * 100,
        )

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet(config: Config) -> ResNet50Baseline:
    """Build ResNet-50 from config."""
    model_cfg = config.model
    head_cfg = model_cfg.head

    model = ResNet50Baseline(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        hidden_dim=head_cfg.hidden_dim,
        dropout_pool=head_cfg.dropout_post_pool,
        dropout_hidden=head_cfg.dropout_post_hidden,
    )

    logger.info(
        "Built ResNet-50: total=%dM, trainable=%dM",
        model.get_total_params() // 1_000_000,
        model.get_trainable_params() // 1_000_000,
    )

    return model
