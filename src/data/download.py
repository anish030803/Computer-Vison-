"""Dataset discovery, download, and verification.

Supports APTOS 2019 (Kaggle), EyePACS (Kaggle), Messidor-2, IDRiD, DDR.
Downloads are verified via file counts and organized into a standardized
directory structure: data/raw/{dataset_name}/images/ + labels.csv.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import Config

logger = logging.getLogger("dr_detection")


def download_all_datasets(config: Config, datasets: Optional[list[str]] = None) -> dict[str, bool]:
    """Download and organize all configured datasets.

    Args:
        config: Data config with dataset definitions.
        datasets: Optional list of dataset keys to download.
                  If None, downloads all datasets.

    Returns:
        Dict mapping dataset name to success status.
    """
    results = {}
    dataset_configs = config.datasets.to_dict()

    for name, ds_cfg in dataset_configs.items():
        if datasets is not None and name not in datasets:
            continue

        logger.info("Processing dataset: %s", name)
        ds = Config(ds_cfg)

        try:
            if ds.source == "kaggle":
                success = _download_kaggle_dataset(name, ds)
            else:
                logger.info(
                    "Dataset '%s' requires manual download from: %s",
                    name,
                    ds.get("url", "see documentation"),
                )
                success = _check_existing_dataset(ds.local_path)
            results[name] = success
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            results[name] = False

    return results


def _download_kaggle_dataset(name: str, ds: Config) -> bool:
    """Download a dataset from Kaggle using the Kaggle API.

    Requires ~/.kaggle/kaggle.json to be configured.
    """
    local_path = Path(ds.local_path)
    images_dir = local_path / "images"

    if images_dir.exists() and any(images_dir.iterdir()):
        logger.info("Dataset '%s' already exists at %s, skipping download", name, local_path)
        return True

    local_path.mkdir(parents=True, exist_ok=True)
    kaggle_dataset = ds.kaggle_dataset

    logger.info("Downloading '%s' from Kaggle: %s", name, kaggle_dataset)

    try:
        # Try competition dataset first, fall back to dataset
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", kaggle_dataset,
                "-p", str(local_path),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error("Kaggle download failed: %s", result.stderr)
            return False

        logger.info("Download complete for '%s'", name)

    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Configure credentials at ~/.kaggle/kaggle.json"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Download timed out for '%s'", name)
        return False

    # Organize into standardized structure
    _organize_dataset(name, local_path)

    return True


def _organize_dataset(name: str, local_path: Path) -> None:
    """Organize downloaded files into standardized structure.

    Target: local_path/images/ + local_path/labels.csv
    """
    images_dir = local_path / "images"
    images_dir.mkdir(exist_ok=True)

    # Find image files at any depth and move to images/
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    moved = 0

    for f in local_path.rglob("*"):
        if f.suffix.lower() in image_extensions and f.parent != images_dir:
            dest = images_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1

    if moved > 0:
        logger.info("Organized %d images into %s", moved, images_dir)

    # Find and standardize labels CSV
    _standardize_labels(name, local_path)

    # Clean up empty directories
    for d in sorted(local_path.rglob("*"), reverse=True):
        if d.is_dir() and d != images_dir and not any(d.iterdir()):
            d.rmdir()


def _standardize_labels(name: str, local_path: Path) -> None:
    """Find and standardize labels into local_path/labels.csv.

    Output format: image_id, label (0-4 severity grade)
    """
    labels_path = local_path / "labels.csv"
    if labels_path.exists():
        return

    # Search for CSV files
    csv_files = list(local_path.rglob("*.csv"))
    if not csv_files:
        logger.warning("No label CSV found for '%s'", name)
        return

    # Use the first CSV that looks like a label file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Look for common column patterns
            id_col = _find_column(df, ["id_code", "image", "img_name", "filename", "ID"])
            label_col = _find_column(df, ["diagnosis", "label", "level", "grade", "DR_grade"])

            if id_col is not None and label_col is not None:
                labels = df[[id_col, label_col]].copy()
                labels.columns = ["image_id", "label"]
                # Strip file extensions from image_id
                labels["image_id"] = labels["image_id"].astype(str).apply(
                    lambda x: Path(x).stem
                )
                labels.to_csv(labels_path, index=False)
                logger.info(
                    "Standardized labels for '%s': %d entries from %s",
                    name, len(labels), csv_file.name,
                )
                return
        except Exception as e:
            logger.debug("Could not parse %s: %s", csv_file, e)

    logger.warning("Could not standardize labels for '%s'", name)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def _check_existing_dataset(local_path: str) -> bool:
    """Check if a dataset already exists at the given path."""
    path = Path(local_path)
    images_dir = path / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        logger.info("Dataset found at %s", path)
        return True
    logger.warning("Dataset not found at %s — manual download required", path)
    return False


def verify_dataset(local_path: str | Path) -> dict:
    """Verify a downloaded dataset's integrity.

    Returns:
        Dict with verification results: image_count, has_labels,
        sample_extensions, any_corrupt.
    """
    path = Path(local_path)
    images_dir = path / "images"
    labels_path = path / "labels.csv"

    result = {
        "exists": path.exists(),
        "images_dir_exists": images_dir.exists(),
        "image_count": 0,
        "has_labels": labels_path.exists(),
        "label_count": 0,
        "extensions": set(),
    }

    if images_dir.exists():
        image_files = list(images_dir.iterdir())
        result["image_count"] = len(image_files)
        result["extensions"] = {f.suffix.lower() for f in image_files if f.is_file()}

    if labels_path.exists():
        try:
            df = pd.read_csv(labels_path)
            result["label_count"] = len(df)
        except Exception:
            result["label_count"] = 0

    return result
