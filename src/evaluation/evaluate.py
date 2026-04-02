"""Full evaluation suite for trained DR models.

Generates confusion matrices, per-class metrics, QWK, AUC-ROC/PR,
and model comparison tables.
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
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.metrics import compute_all_metrics

logger = logging.getLogger("dr_detection")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    output_dir: Optional[str | Path] = None,
    model_name: str = "model",
) -> dict:
    """Run full evaluation on a test set.

    Generates:
    - Confusion matrix (raw + normalized)
    - Per-class precision/recall/F1
    - QWK, AUC-ROC, AUC-PR
    - All results saved to output_dir

    Returns:
        Dict of all metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output = model(images)
            if isinstance(output, dict):
                output = output["logits"]

            probs = F.softmax(output, dim=-1).cpu().numpy()
            preds = output.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs, CLASS_NAMES)

    logger.info("=== Evaluation Results: %s ===", model_name)
    logger.info("QWK: %.4f", metrics["qwk"])
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])

    # Clinically critical: Severe + Proliferative recall
    severe_recall = metrics.get("recall_severe_npdr", 0)
    prolif_recall = metrics.get("recall_proliferative_dr", 0)
    logger.info("Severe DR Recall: %.4f", severe_recall)
    logger.info("Proliferative DR Recall: %.4f", prolif_recall)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Generate confusion matrices
        _plot_confusion_matrix(y_true, y_pred, output_dir, model_name, normalize=False)
        _plot_confusion_matrix(y_true, y_pred, output_dir, model_name, normalize=True)

        # Per-class probability distribution
        _plot_probability_distribution(y_probs, y_true, output_dir, model_name)

        logger.info("Evaluation artifacts saved to %s", output_dir)

    return metrics


def compare_models(
    model_metrics: dict[str, dict],
    output_dir: Optional[str | Path] = None,
) -> str:
    """Generate a comparison table of multiple models.

    Args:
        model_metrics: Dict mapping model name to metrics dict.
        output_dir: Where to save comparison.

    Returns:
        Formatted comparison table as string.
    """
    key_metrics = ["qwk", "accuracy", "macro_f1", "recall_severe_npdr", "recall_proliferative_dr"]
    headers = ["Model", "QWK", "Accuracy", "Macro F1", "Severe Recall", "Prolif. Recall"]

    rows = []
    for model_name, metrics in model_metrics.items():
        row = [model_name]
        for key in key_metrics:
            row.append(f"{metrics.get(key, 0):.4f}")
        rows.append(row)

    # Format as table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-|-".join("-" * w for w in col_widths)

    table = [header_line, separator]
    for row in rows:
        table.append(" | ".join(v.ljust(w) for v, w in zip(row, col_widths)))

    table_str = "\n".join(table)
    logger.info("\n%s", table_str)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_comparison.txt", "w") as f:
            f.write(table_str)

        # Save as JSON too
        with open(output_dir / "model_comparison.json", "w") as f:
            json.dump(model_metrics, f, indent=2, default=str)

    return table_str


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    normalize: bool = False,
) -> None:
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(5))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
        suffix = "normalized"
    else:
        fmt = "d"
        suffix = "raw"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({model_name}, {suffix})")
    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_probability_distribution(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """Plot per-class probability distributions."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        mask = y_true == i
        if mask.sum() > 0:
            for j, cls_name in enumerate(CLASS_NAMES):
                ax.hist(
                    y_probs[mask, j], bins=20, alpha=0.5,
                    label=cls_name, density=True,
                )
        ax.set_title(f"True: {name}")
        ax.set_xlabel("Probability")
        if i == 0:
            ax.legend(fontsize=6)

    plt.suptitle(f"Prediction Probability Distributions ({model_name})")
    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_prob_distribution.png", dpi=150)
    plt.close(fig)
