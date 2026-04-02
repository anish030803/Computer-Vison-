"""Shared classification and segmentation heads for all models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head: GAP → BN → Dropout → Dense → Dropout → Dense.

    Used by EfficientNet-B4 and ResNet-50. Configurable hidden dim,
    dropout rates, and activation.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout_pool: float = 0.4,
        dropout_hidden: float = 0.3,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(in_features))

        layers.append(nn.Dropout(dropout_pool))
        layers.append(nn.Linear(in_features, hidden_dim))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        layers.append(nn.Dropout(dropout_hidden))
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ViTClassificationHead(nn.Module):
    """Classification head for Vision Transformers (DINOv2).

    Architecture: LayerNorm → Linear → GELU → Dropout → Linear
    Takes the [CLS] token output.
    """

    def __init__(
        self,
        in_features: int = 1024,
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SegmentationHead(nn.Module):
    """Simple segmentation head using patch tokens from ViT.

    Projects patch tokens to per-pixel class predictions and upsamples.
    """

    def __init__(
        self,
        in_features: int = 1024,
        num_classes: int = 5,
        patch_size: int = 14,
        image_size: int = 518,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size

        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

        self.upsample = nn.Upsample(
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, D) from ViT backbone.

        Returns:
            (B, num_classes, H, W) segmentation logits.
        """
        B = patch_tokens.shape[0]

        # Project to class logits
        x = self.projection(patch_tokens)  # (B, num_patches, num_classes)

        # Reshape to spatial grid
        x = x.transpose(1, 2)  # (B, num_classes, num_patches)
        x = x.reshape(B, -1, self.num_patches_side, self.num_patches_side)

        # Upsample to full resolution
        x = self.upsample(x)

        return x
