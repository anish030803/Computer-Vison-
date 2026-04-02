"""PyTorch Dataset and DataLoader for DR detection.

Supports loading from preprocessed .npy cache or raw images
with on-the-fly augmentation. Includes stratified splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class DRDataset(Dataset):
    """PyTorch Dataset for Diabetic Retinopathy images.

    Supports two modes:
    1. Preprocessed cache (.npy files) — fast loading
    2. Raw images with on-the-fly preprocessing — flexible
    """

    def __init__(
        self,
        image_ids: list[str],
        labels: list[int],
        images_dir: str | Path,
        transform=None,
        use_cache: bool = True,
    ) -> None:
        """
        Args:
            image_ids: List of image identifiers (filename stems).
            labels: Corresponding severity labels (0-4).
            images_dir: Directory containing images (.npy or raw).
            transform: Albumentations transform pipeline.
            use_cache: If True, load .npy files; otherwise raw images.
        """
        self.image_ids = image_ids
        self.labels = labels
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        if self.use_cache:
            img = self._load_cached(image_id)
        else:
            img = self._load_raw(image_id)

        if img is None:
            # Return a black image as fallback
            img = np.zeros((380, 380, 3), dtype=np.float32)

        # Apply augmentations
        if self.transform is not None:
            # Albumentations expects uint8 or float32 HWC
            if img.dtype != np.uint8 and img.max() <= 1.0:
                img_for_aug = (img * 255).astype(np.uint8)
            else:
                img_for_aug = img.astype(np.uint8)

            augmented = self.transform(image=img_for_aug)
            img_tensor = augmented["image"]
        else:
            # Manual conversion to tensor (CHW float32)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        return img_tensor, label

    def _load_cached(self, image_id: str) -> Optional[np.ndarray]:
        """Load a preprocessed .npy file."""
        npy_path = self.images_dir / f"{image_id}.npy"
        if npy_path.exists():
            return np.load(npy_path)
        # Fallback to raw
        return self._load_raw(image_id)

    def _load_raw(self, image_id: str) -> Optional[np.ndarray]:
        """Load a raw image file."""
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            img_path = self.images_dir / f"{image_id}{ext}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None


def create_splits(
    labels_path: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits.

    Args:
        labels_path: Path to labels.csv (columns: image_id, label).
        train_ratio: Training set proportion.
        val_ratio: Validation set proportion.
        test_ratio: Test set proportion.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = pd.read_csv(labels_path)

    # First split: train+val vs test
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_val_idx, test_idx = next(splitter1.split(df, df["label"]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Second split: train vs val (adjust ratio for remaining data)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    train_idx, val_idx = next(splitter2.split(train_val_df, train_val_df["label"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Split: train=%d, val=%d, test=%d (total=%d)",
        len(train_df), len(val_df), len(test_df), len(df),
    )

    # Log class distributions per split
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts().sort_index().to_dict()
        logger.info("  %s distribution: %s", name, dist)

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str | Path,
    train_transform=None,
    val_transform=None,
    batch_size: int = 64,
    num_workers: int = 16,
    pin_memory: bool = True,
    use_cache: bool = True,
    class_weights: Optional[dict[int, float]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test sets.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        images_dir: Directory containing images.
        train_transform: Training augmentation pipeline.
        val_transform: Validation transform pipeline.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.
        use_cache: Use preprocessed .npy cache.
        class_weights: Per-class weights for weighted sampling.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = DRDataset(
        image_ids=train_df["image_id"].astype(str).tolist(),
        labels=train_df["label"].tolist(),
        images_dir=images_dir,
        transform=train_transform,
        use_cache=use_cache,
    )

    val_dataset = DRDataset(
        image_ids=val_df["image_id"].astype(str).tolist(),
        labels=val_df["label"].tolist(),
        images_dir=images_dir,
        transform=val_transform,
        use_cache=use_cache,
    )

    test_dataset = DRDataset(
        image_ids=test_df["image_id"].astype(str).tolist(),
        labels=test_df["label"].tolist(),
        images_dir=images_dir,
        transform=val_transform,
        use_cache=use_cache,
    )

    # Weighted random sampling for imbalanced training data
    train_sampler = None
    train_shuffle = True
    if class_weights is not None:
        sample_weights = [class_weights.get(label, 1.0) for label in train_df["label"]]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles shuffling

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader
