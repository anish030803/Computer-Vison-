"""Training augmentations using albumentations.

Provides clinically calibrated augmentations that do not alter
pathology presentation. Includes geometric, color, and advanced
augmentations (MixUp).
"""

from __future__ import annotations

import logging
from typing import Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from src.utils.config import Config

logger = logging.getLogger("dr_detection")


def get_train_transforms(
    config: Config,
    image_size: int = 380,
) -> A.Compose:
    """Get training augmentation pipeline.

    Augmentations from PRD FR-14:
    - Geometric: H/V flip, rotation ±36°, zoom 90-110%
    - Color: brightness ±10%, contrast ±10%
    - Normalize with ImageNet stats
    - Convert to tensor

    Args:
        config: Augmentation config section.
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    aug_cfg = config.train

    transforms = [
        A.Resize(image_size, image_size),
    ]

    # Geometric augmentations
    if aug_cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))

    rotation_limit = aug_cfg.get("rotation_limit", 0)
    if rotation_limit > 0:
        transforms.append(A.Rotate(limit=rotation_limit, p=0.5, border_mode=0))

    zoom_range = aug_cfg.get("zoom_range")
    if zoom_range is not None:
        scale_min = zoom_range[0] if isinstance(zoom_range, list) else 0.9
        scale_max = zoom_range[1] if isinstance(zoom_range, list) else 1.1
        transforms.append(
            A.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=0.5)
        )
        # Resize back after scale
        transforms.append(A.Resize(image_size, image_size))

    # Color augmentations
    brightness_limit = aug_cfg.get("brightness_limit", 0)
    contrast_limit = aug_cfg.get("contrast_limit", 0)
    if brightness_limit > 0 or contrast_limit > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            )
        )

    # Normalize
    normalize_cfg = aug_cfg.get("normalize")
    if normalize_cfg is not None:
        mean = normalize_cfg.mean if hasattr(normalize_cfg, "mean") else [0.485, 0.456, 0.406]
        std = normalize_cfg.std if hasattr(normalize_cfg, "std") else [0.229, 0.224, 0.225]
        transforms.append(A.Normalize(mean=mean, std=std))

    # To tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_transforms(
    config: Config,
    image_size: int = 380,
) -> A.Compose:
    """Get validation/test augmentation pipeline (no augmentation).

    Only resize, normalize, and convert to tensor.
    """
    val_cfg = config.val

    transforms = [A.Resize(image_size, image_size)]

    normalize_cfg = val_cfg.get("normalize")
    if normalize_cfg is not None:
        mean = normalize_cfg.mean if hasattr(normalize_cfg, "mean") else [0.485, 0.456, 0.406]
        std = normalize_cfg.std if hasattr(normalize_cfg, "std") else [0.229, 0.224, 0.225]
        transforms.append(A.Normalize(mean=mean, std=std))

    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def mixup_data(
    images: "torch.Tensor",
    labels: "torch.Tensor",
    alpha: float = 0.2,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float]:
    """Apply MixUp augmentation to a batch.

    MixUp creates convex combinations of pairs of examples and their labels,
    which is effective for improving calibration and reducing overfitting
    on minority classes.

    Args:
        images: Batch of images (B, C, H, W).
        labels: Batch of labels (B,) as class indices.
        alpha: MixUp interpolation strength. Higher = more mixing.

    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam) where lam is
        the mixing coefficient.
    """
    import torch

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(
    criterion: "torch.nn.Module",
    pred: "torch.Tensor",
    labels_a: "torch.Tensor",
    labels_b: "torch.Tensor",
    lam: float,
) -> "torch.Tensor":
    """Compute MixUp loss as weighted combination.

    Args:
        criterion: Loss function.
        pred: Model predictions.
        labels_a: First set of labels.
        labels_b: Second set of labels (shuffled).
        lam: Mixing coefficient.

    Returns:
        Mixed loss value.
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
