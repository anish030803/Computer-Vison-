"""DINOv2 ViT-L/14 for DR classification and optional segmentation.

Architecture per TRD §4.2:
- Backbone: DINOv2 ViT-L/14 (self-supervised pretrained)
- Classification head: [CLS] → LayerNorm → Linear(1024→512, GELU) → Dropout(0.3) → Linear(5)
- Optional segmentation head using patch tokens
- Input: 518×518×3 (DINOv2 native resolution)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.heads import SegmentationHead, ViTClassificationHead
from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class DINOv2Classifier(nn.Module):
    """DINOv2 ViT-L/14 for DR classification."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        enable_segmentation: bool = False,
        image_size: int = 518,
    ) -> None:
        super().__init__()

        self.enable_segmentation = enable_segmentation

        # Load DINOv2 backbone
        if pretrained:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False
            )

        in_features = self.backbone.embed_dim  # 1024 for ViT-L

        # Classification head
        self.cls_head = ViTClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Optional segmentation head
        if enable_segmentation:
            self.seg_head = SegmentationHead(
                in_features=in_features,
                num_classes=num_classes,
                patch_size=14,
                image_size=image_size,
            )

        logger.info(
            "DINOv2 ViT-L/14 initialized: features=%d, classes=%d, segmentation=%s",
            in_features, num_classes, enable_segmentation,
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input images (B, 3, 518, 518).
            return_features: If True, also return raw CLS and patch tokens.

        Returns:
            Dict with keys: 'logits', optionally 'segmentation', 'cls_token', 'patch_tokens'.
        """
        # Get intermediate features from DINOv2
        features = self.backbone.forward_features(x)

        # DINOv2 returns dict with keys: 'x_norm_clstoken', 'x_norm_patchtokens', etc.
        # Or a tensor depending on the version. Handle both cases.
        if isinstance(features, dict):
            cls_token = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]
        else:
            # Fallback: first token is CLS, rest are patch tokens
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]

        # Classification
        logits = self.cls_head(cls_token)

        output = {"logits": logits}

        # Segmentation
        if self.enable_segmentation and hasattr(self, "seg_head"):
            output["segmentation"] = self.seg_head(patch_tokens)

        if return_features:
            output["cls_token"] = cls_token
            output["patch_tokens"] = patch_tokens

        return output

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1: linear probe)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("DINOv2 backbone FROZEN")

    def unfreeze_backbone(self, last_n_blocks: int = 6) -> None:
        """Unfreeze the last N transformer blocks (Phase 2: fine-tuning).

        Args:
            last_n_blocks: Number of transformer blocks to unfreeze from the end.
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks
        total_blocks = len(self.backbone.blocks)
        start_block = max(0, total_blocks - last_n_blocks)

        for i in range(start_block, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Also unfreeze the norm layer
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        unfrozen = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        logger.info(
            "DINOv2 backbone: unfrozen blocks %d-%d (%d/%d params)",
            start_block, total_blocks - 1, unfrozen, total,
        )

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_dinov2(config: Config) -> DINOv2Classifier:
    """Build DINOv2 model from config.

    Args:
        config: Training config with model section.

    Returns:
        Initialized DINOv2Classifier.
    """
    model_cfg = config.model
    head_cfg = model_cfg.head

    model = DINOv2Classifier(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        hidden_dim=head_cfg.hidden_dim,
        dropout=head_cfg.dropout,
        image_size=model_cfg.image_size,
    )

    logger.info(
        "Built DINOv2: total=%dM, trainable=%dM",
        model.get_total_params() // 1_000_000,
        model.get_trainable_params() // 1_000_000,
    )

    return model
