"""Prediction threshold optimization for clinical deployment.

Optimizes sensitivity for Severe and Proliferative DR grades
at the cost of specificity — missing severe cases is the worst
failure mode in clinical screening.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("dr_detection")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def optimize_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_classes: list[int] = [3, 4],
    target_sensitivity: float = 0.90,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Find optimal thresholds for high-sensitivity clinical detection.

    Sweeps probability thresholds for target classes (Severe, Proliferative DR)
    to find operating points that maximize sensitivity.

    Args:
        y_true: True labels (N,).
        y_probs: Predicted probabilities (N, num_classes).
        target_classes: Classes to optimize (default: Severe + Proliferative).
        target_sensitivity: Minimum target sensitivity.
        output_dir: Where to save plots.

    Returns:
        Dict with optimal thresholds, sensitivity/specificity at each.
    """
    results = {}

    for cls_idx in target_classes:
        cls_name = CLASS_NAMES[cls_idx]
        binary_true = (y_true == cls_idx).astype(int)
        cls_probs = y_probs[:, cls_idx]

        # Skip if no positive samples
        if binary_true.sum() == 0:
            logger.warning("No positive samples for class %s", cls_name)
            continue

        thresholds = np.arange(0.01, 1.0, 0.01)
        sensitivities = []
        specificities = []

        for thresh in thresholds:
            predictions = (cls_probs >= thresh).astype(int)

            tp = ((predictions == 1) & (binary_true == 1)).sum()
            fn = ((predictions == 0) & (binary_true == 1)).sum()
            tn = ((predictions == 0) & (binary_true == 0)).sum()
            fp = ((predictions == 1) & (binary_true == 0)).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            sensitivities.append(sensitivity)
            specificities.append(specificity)

        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)

        # Find threshold achieving target sensitivity
        valid_mask = sensitivities >= target_sensitivity
        if valid_mask.any():
            # Among thresholds meeting sensitivity target, pick highest specificity
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(specificities[valid_indices])]
            optimal_threshold = thresholds[best_idx]
        else:
            # Fall back to threshold maximizing Youden's J statistic
            j_stat = sensitivities + specificities - 1
            best_idx = np.argmax(j_stat)
            optimal_threshold = thresholds[best_idx]

        results[cls_name] = {
            "class_index": cls_idx,
            "optimal_threshold": float(optimal_threshold),
            "sensitivity_at_threshold": float(sensitivities[best_idx]),
            "specificity_at_threshold": float(specificities[best_idx]),
            "target_sensitivity_met": bool(sensitivities[best_idx] >= target_sensitivity),
        }

        logger.info(
            "%s: threshold=%.3f, sensitivity=%.3f, specificity=%.3f",
            cls_name, optimal_threshold,
            sensitivities[best_idx], specificities[best_idx],
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_dir / "threshold_optimization.json", "w") as f:
            json.dump(results, f, indent=2)

        # Plot sensitivity-specificity curves
        _plot_threshold_curves(
            y_true, y_probs, target_classes, results, output_dir
        )

    return results


def _plot_threshold_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_classes: list[int],
    results: dict,
    output_dir: Path,
) -> None:
    """Plot sensitivity/specificity vs threshold curves."""
    fig, axes = plt.subplots(1, len(target_classes), figsize=(7 * len(target_classes), 5))

    if len(target_classes) == 1:
        axes = [axes]

    for ax, cls_idx in zip(axes, target_classes):
        cls_name = CLASS_NAMES[cls_idx]
        binary_true = (y_true == cls_idx).astype(int)
        cls_probs = y_probs[:, cls_idx]

        thresholds = np.arange(0.01, 1.0, 0.01)
        sensitivities = []
        specificities = []

        for thresh in thresholds:
            preds = (cls_probs >= thresh).astype(int)
            tp = ((preds == 1) & (binary_true == 1)).sum()
            fn = ((preds == 0) & (binary_true == 1)).sum()
            tn = ((preds == 0) & (binary_true == 0)).sum()
            fp = ((preds == 1) & (binary_true == 0)).sum()

            sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        ax.plot(thresholds, sensitivities, "r-", label="Sensitivity", linewidth=2)
        ax.plot(thresholds, specificities, "b-", label="Specificity", linewidth=2)

        if cls_name in results:
            opt_thresh = results[cls_name]["optimal_threshold"]
            ax.axvline(opt_thresh, color="green", linestyle="--", label=f"Optimal ({opt_thresh:.3f})")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title(f"{cls_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Threshold Optimization: Sensitivity vs Specificity")
    plt.tight_layout()
    fig.savefig(output_dir / "threshold_curves.png", dpi=150)
    plt.close(fig)
