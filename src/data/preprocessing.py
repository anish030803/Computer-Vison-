"""Preprocessing pipeline: Ben Graham's method, CLAHE, caching.

Ben Graham's method (from Kaggle DR competition):
1. Resize to target_size x target_size
2. Apply circular crop (remove black borders)
3. Local average color subtraction (Gaussian blur)
4. Add back constant to maintain contrast
5. Apply circular mask
6. Normalize to [0, 1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import Config

logger = logging.getLogger("dr_detection")


def ben_graham_preprocess(
    image: np.ndarray,
    target_size: int = 380,
    sigma_ratio: float = 0.1,
    add_back: int = 128,
) -> np.ndarray:
    """Apply Ben Graham's preprocessing to a single fundus image.

    Args:
        image: Input BGR image (numpy array).
        target_size: Target dimension (square).
        sigma_ratio: Gaussian sigma as ratio of image width.
        add_back: Value added back after local average subtraction.

    Returns:
        Preprocessed image as float32 array in [0, 1].
    """
    # Resize
    img = cv2.resize(image, (target_size, target_size))

    # Create circular mask
    mask = _create_circular_mask(target_size)

    # Gaussian blur for local average color
    sigma = int(target_size * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1  # Kernel size must be odd
    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    # Subtract local average, add back constant
    img = cv2.addWeighted(img, 4, blur, -4, add_back)

    # Apply circular mask
    img = cv2.bitwise_and(img, img, mask=mask)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def clahe_preprocess(
    image: np.ndarray,
    target_size: int = 380,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE preprocessing for enhanced contrast.

    Particularly useful for visualizing microaneurysms.

    Args:
        image: Input BGR image.
        target_size: Target dimension.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Preprocessed image as float32 in [0, 1].
    """
    img = cv2.resize(image, (target_size, target_size))

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge and convert back
    lab = cv2.merge([l_enhanced, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img


def green_channel_extract(
    image: np.ndarray,
    target_size: int = 380,
) -> np.ndarray:
    """Extract green channel (highest contrast for retinal lesions).

    Args:
        image: Input BGR image.
        target_size: Target dimension.

    Returns:
        Single-channel float32 image in [0, 1], expanded to 3 channels.
    """
    img = cv2.resize(image, (target_size, target_size))
    green = img[:, :, 1]  # BGR -> green channel
    green = green.astype(np.float32) / 255.0
    # Stack to 3 channels for model compatibility
    return np.stack([green, green, green], axis=-1)


def preprocess_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    config: Config,
    target_size: int = 380,
    method: str = "ben_graham",
) -> Path:
    """Preprocess all images in a dataset and cache to disk.

    Args:
        dataset_path: Path to cleaned dataset (with images/ dir).
        output_path: Where to save preprocessed arrays.
        config: Data config with preprocessing settings.
        target_size: Target image size.
        method: Preprocessing method ("ben_graham", "clahe", "green_channel").

    Returns:
        Path to output directory containing .npy files.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    images_dir = dataset_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)

    prep_cfg = config.preprocessing

    # Select preprocessing function
    if method == "ben_graham":
        sigma_ratio = prep_cfg.ben_graham.gaussian_sigma_ratio
        add_back = prep_cfg.ben_graham.add_back_value
        preprocess_fn = lambda img: ben_graham_preprocess(img, target_size, sigma_ratio, add_back)
    elif method == "clahe":
        clip_limit = prep_cfg.clahe.clip_limit
        tile_grid = tuple(prep_cfg.clahe.tile_grid_size)
        preprocess_fn = lambda img: clahe_preprocess(img, target_size, clip_limit, tile_grid)
    elif method == "green_channel":
        preprocess_fn = lambda img: green_channel_extract(img, target_size)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    # Process all images
    image_files = sorted(images_dir.iterdir())
    processed = 0
    skipped = 0

    for img_path in tqdm(image_files, desc=f"Preprocessing ({method})"):
        if not img_path.is_file():
            continue

        out_file = output_path / f"{img_path.stem}.npy"

        # Skip if already cached
        if out_file.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read: %s", img_path)
                continue

            preprocessed = preprocess_fn(img)
            np.save(out_file, preprocessed)
            processed += 1
        except Exception as e:
            logger.warning("Failed to preprocess %s: %s", img_path.name, e)

    # Copy labels file
    labels_src = dataset_path / "labels.csv"
    labels_dst = output_path / "labels.csv"
    if labels_src.exists() and not labels_dst.exists():
        import shutil
        shutil.copy2(labels_src, labels_dst)

    logger.info(
        "Preprocessing complete: processed=%d, skipped=%d, method=%s, size=%d",
        processed, skipped, method, target_size,
    )

    return output_path


def verify_preprocessed_cache(cache_path: str | Path, expected_shape: tuple[int, ...]) -> dict:
    """Verify integrity of preprocessed cache.

    Args:
        cache_path: Path to directory with .npy files.
        expected_shape: Expected shape of each array (H, W, C).

    Returns:
        Verification results.
    """
    cache_path = Path(cache_path)
    npy_files = sorted(cache_path.glob("*.npy"))

    total = len(npy_files)
    valid = 0
    invalid = 0
    shape_mismatches = 0

    for npy_file in npy_files:
        try:
            arr = np.load(npy_file)
            if arr.shape == expected_shape:
                valid += 1
            else:
                shape_mismatches += 1
                logger.warning(
                    "Shape mismatch: %s has %s, expected %s",
                    npy_file.name, arr.shape, expected_shape,
                )
        except Exception as e:
            invalid += 1
            logger.warning("Invalid cache file: %s (%s)", npy_file.name, e)

    result = {
        "total_files": total,
        "valid": valid,
        "invalid": invalid,
        "shape_mismatches": shape_mismatches,
        "all_valid": valid == total and total > 0,
    }

    logger.info("Cache verification: %s", result)
    return result


def _create_circular_mask(size: int) -> np.ndarray:
    """Create a circular mask for cropping fundus images."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = int(size * 0.45)  # Slightly less than half to crop border
    cv2.circle(mask, (center, center), radius, 255, -1)
    return mask
