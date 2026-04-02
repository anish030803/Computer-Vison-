"""Cross-dataset evaluation for domain shift analysis.

Evaluates trained model (APTOS 2019) on Messidor-2 to measure
generalization and document performance degradation.
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
import torch
from torch.utils.data import DataLoader

from src.evaluation.evaluate import evaluate_model
from src.training.metrics import compute_all_metrics

logger = logging.getLogger("dr_detection")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def evaluate_cross_dataset(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: str = "cuda",
    source_name: str = "APTOS 2019",
    target_name: str = "Messidor-2",
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Evaluate a model on both source and target datasets.

    Compares metrics to quantify domain shift.

    Args:
        model: Trained model.
        source_loader: DataLoader for source dataset (e.g., APTOS 2019).
        target_loader: DataLoader for target dataset (e.g., Messidor-2).
        device: Device.
        source_name: Name of source dataset.
        target_name: Name of target dataset.
        output_dir: Where to save analysis.

    Returns:
        Dict with source metrics, target metrics, and degradation analysis.
    """
    output_dir = Path(output_dir) if output_dir else None

    # Evaluate on source
    logger.info("Evaluating on source: %s", source_name)
    source_metrics = evaluate_model(
        model, source_loader, device,
        output_dir=output_dir / source_name.lower().replace(" ", "_") if output_dir else None,
        model_name=f"source_{source_name}",
    )

    # Evaluate on target
    logger.info("Evaluating on target: %s", target_name)
    target_metrics = evaluate_model(
        model, target_loader, device,
        output_dir=output_dir / target_name.lower().replace(" ", "_") if output_dir else None,
        model_name=f"target_{target_name}",
    )

    # Compute degradation
    degradation = {}
    key_metrics = ["qwk", "accuracy", "macro_f1", "recall_severe_npdr", "recall_proliferative_dr"]

    for metric in key_metrics:
        src_val = source_metrics.get(metric, 0)
        tgt_val = target_metrics.get(metric, 0)
        degradation[metric] = {
            "source": src_val,
            "target": tgt_val,
            "absolute_drop": src_val - tgt_val,
            "relative_drop_pct": ((src_val - tgt_val) / src_val * 100) if src_val > 0 else 0,
        }

    results = {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "source_metrics": source_metrics,
        "target_metrics": target_metrics,
        "degradation": degradation,
    }

    # Log summary
    logger.info("=== Cross-Dataset Analysis: %s → %s ===", source_name, target_name)
    for metric, vals in degradation.items():
        logger.info(
            "  %s: %.4f → %.4f (drop: %.4f / %.1f%%)",
            metric, vals["source"], vals["target"],
            vals["absolute_drop"], vals["relative_drop_pct"],
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "cross_dataset_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        _plot_comparison(degradation, source_name, target_name, output_dir)

    return results


def _plot_comparison(
    degradation: dict,
    source_name: str,
    target_name: str,
    output_dir: Path,
) -> None:
    """Plot side-by-side metric comparison."""
    metrics = list(degradation.keys())
    source_vals = [degradation[m]["source"] for m in metrics]
    target_vals = [degradation[m]["target"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, source_vals, width, label=source_name, color="#3498db")
    bars2 = ax.bar(x + width / 2, target_vals, width, label=target_name, color="#e74c3c")

    ax.set_ylabel("Score")
    ax.set_title("Cross-Dataset Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "cross_dataset_comparison.png", dpi=150)
    plt.close(fig)
