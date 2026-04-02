"""Post-cleaning data validation and dynamic hyperparameter computation.

Generates validation reports with class distributions, quality stats,
and auto-computes class weights, focal loss gamma, and batch size.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import Config, load_config

logger = logging.getLogger("dr_detection")


def validate_cleaned_dataset(
    dataset_path: str | Path,
    config: Config,
) -> dict[str, Any]:
    """Run comprehensive validation on a cleaned dataset.

    Generates:
    - Class distribution analysis
    - Image quality statistics
    - Resolution statistics
    - Sample visualization grid
    - Validation report (JSON)

    Returns:
        Validation results dict.
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_path = dataset_path / "labels.csv"
    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"dataset": str(dataset_path)}

    # Load labels
    if not labels_path.exists():
        logger.error("No labels.csv found at %s", labels_path)
        return results

    df = pd.read_csv(labels_path)
    class_names = config.classes.names

    # --- Class Distribution ---
    dist = df["label"].value_counts().sort_index()
    results["class_distribution"] = dist.to_dict()
    results["total_images"] = len(df)
    results["class_names"] = class_names

    logger.info("Class distribution:")
    for label, count in dist.items():
        pct = count / len(df) * 100
        name = class_names[label] if label < len(class_names) else f"Class {label}"
        logger.info("  %s (grade %d): %d (%.1f%%)", name, label, count, pct)

    # --- Class Distribution Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(dist)), dist.values, color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"])
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels([class_names[i] if i < len(class_names) else f"Class {i}" for i in dist.index], rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution After Cleaning")
    for bar, count in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(reports_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # --- Quality Statistics ---
    quality_stats = _compute_quality_stats(images_dir)
    results["quality_stats"] = quality_stats

    # --- Quality Histogram ---
    if quality_stats["sharpness_values"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(quality_stats["sharpness_values"], bins=50, color="#3498db", alpha=0.7)
        axes[0].set_title("Sharpness Distribution")
        axes[0].set_xlabel("Laplacian Variance")
        axes[1].hist(quality_stats["brightness_values"], bins=50, color="#2ecc71", alpha=0.7)
        axes[1].set_title("Brightness Distribution")
        axes[1].set_xlabel("Mean Pixel Value")
        axes[2].hist(quality_stats["contrast_values"], bins=50, color="#e74c3c", alpha=0.7)
        axes[2].set_title("Contrast Distribution")
        axes[2].set_xlabel("Pixel Std Dev")
        plt.tight_layout()
        fig.savefig(reports_dir / "quality_histograms.png", dpi=150)
        plt.close(fig)

    # Remove raw arrays from saved results
    stats_summary = {
        k: v for k, v in quality_stats.items()
        if not k.endswith("_values")
    }
    results["quality_stats"] = stats_summary

    # --- Sample Grid ---
    _generate_sample_grid(images_dir, df, class_names, reports_dir)

    # --- Save Report ---
    report_path = reports_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)

    return results


def compute_dynamic_hyperparameters(
    dataset_path: str | Path,
    config: Config,
    gpu_memory_gb: float = 80.0,
) -> dict[str, Any]:
    """Compute hyperparameters from cleaned dataset statistics.

    Computes:
    - Class weights (inverse frequency via sklearn)
    - Recommended batch size (based on dataset size + GPU memory)
    - Focal loss gamma (based on imbalance ratio)
    - Augmentation intensity (based on minority class count)

    Args:
        dataset_path: Path to cleaned dataset.
        config: Data config.
        gpu_memory_gb: Available GPU memory in GB.

    Returns:
        Dict of computed hyperparameters.
    """
    dataset_path = Path(dataset_path)
    labels_path = dataset_path / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pd.read_csv(labels_path)
    labels = df["label"].values

    # --- Class Weights (inverse frequency) ---
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    class_weights_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weights)}

    # --- Imbalance Ratio ---
    class_counts = pd.Series(labels).value_counts().sort_index()
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # --- Focal Loss Gamma ---
    # Higher gamma for more imbalanced datasets
    if imbalance_ratio > 20:
        focal_gamma = 3.0
    elif imbalance_ratio > 10:
        focal_gamma = 2.5
    elif imbalance_ratio > 5:
        focal_gamma = 2.0
    else:
        focal_gamma = 1.5

    # --- Batch Size (based on dataset size and GPU memory) ---
    # These are reasonable defaults for H200 (80GB)
    image_size = config.get("image_sizes", Config({"efficientnet_b4": 380}))
    default_size = 380
    if hasattr(image_size, "efficientnet_b4"):
        default_size = image_size.efficientnet_b4

    # Rough memory estimate: larger images or more data doesn't change per-batch memory
    # but we cap batch size so training isn't too slow on small datasets
    total_images = len(df)
    if total_images < 1000:
        recommended_batch = 32
    elif total_images < 5000:
        recommended_batch = 64
    else:
        recommended_batch = 64  # H200 can handle 64 easily at 380x380

    # --- Augmentation Intensity ---
    # More aggressive augmentation for minority classes
    if min_count < 100:
        aug_intensity = "aggressive"
    elif min_count < 500:
        aug_intensity = "moderate"
    else:
        aug_intensity = "light"

    hyperparams = {
        "class_weights": class_weights_dict,
        "class_distribution": class_counts.to_dict(),
        "total_train_images": int(total_images),
        "imbalance_ratio": float(imbalance_ratio),
        "focal_loss_gamma": focal_gamma,
        "recommended_batch_size": recommended_batch,
        "augmentation_intensity": aug_intensity,
    }

    logger.info("Dynamic hyperparameters computed:")
    for k, v in hyperparams.items():
        logger.info("  %s: %s", k, v)

    # Save to file
    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    hp_path = reports_dir / "dynamic_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(hyperparams, f, indent=2, default=str)
    logger.info("Saved dynamic hyperparameters to %s", hp_path)

    return hyperparams


def _compute_quality_stats(images_dir: Path) -> dict[str, Any]:
    """Compute quality statistics for all images in a directory."""
    sharpness_values = []
    brightness_values = []
    contrast_values = []
    widths = []
    heights = []

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness_values.append(float(gray.mean()))
            contrast_values.append(float(gray.std()))
        except Exception:
            continue

    return {
        "num_images": len(sharpness_values),
        "sharpness_values": sharpness_values,
        "brightness_values": brightness_values,
        "contrast_values": contrast_values,
        "mean_sharpness": float(np.mean(sharpness_values)) if sharpness_values else 0,
        "mean_brightness": float(np.mean(brightness_values)) if brightness_values else 0,
        "mean_contrast": float(np.mean(contrast_values)) if contrast_values else 0,
        "mean_width": float(np.mean(widths)) if widths else 0,
        "mean_height": float(np.mean(heights)) if heights else 0,
        "min_width": int(min(widths)) if widths else 0,
        "min_height": int(min(heights)) if heights else 0,
    }


def _generate_sample_grid(
    images_dir: Path,
    labels_df: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
    samples_per_class: int = 5,
) -> None:
    """Generate a grid of sample images for each class."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(3 * samples_per_class, 3 * num_classes))

    if num_classes == 1:
        axes = axes[np.newaxis, :]

    for class_idx in range(num_classes):
        class_samples = labels_df[labels_df["label"] == class_idx]["image_id"].values
        selected = class_samples[:samples_per_class]

        for col in range(samples_per_class):
            ax = axes[class_idx, col]
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(class_names[class_idx], fontsize=10, rotation=0, labelpad=60, va="center")

            if col < len(selected):
                img_id = selected[col]
                # Find the image file
                img_path = None
                for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                    candidate = images_dir / f"{img_id}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

                if img_path is not None:
                    try:
                        img = cv2.imread(str(img_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(f"Grade {class_idx}", fontsize=8)
                    except Exception:
                        ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle("Sample Images by Severity Grade", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample grid saved to %s", output_dir / "sample_grid.png")
