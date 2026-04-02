"""Checkpoint save/load with full training state and metadata."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("dr_detection")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    metric_name: str = "val_qwk",
    config: Optional[dict] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint with full metadata.

    Saves model weights, optimizer/scheduler state, epoch, best metric,
    config, and arbitrary extra metadata. Supports HPC job resume.

    Returns:
        Path where checkpoint was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "config": config,
        "extra": extra or {},
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    logger.info(
        "Saved checkpoint: %s (epoch=%d, %s=%.4f)",
        path,
        epoch,
        metric_name,
        best_metric if best_metric is not None else 0.0,
    )

    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: Path to the .pt checkpoint file.
        model: Model to load weights into (modified in-place).
        optimizer: Optimizer to restore (modified in-place, optional).
        scheduler: Scheduler to restore (modified in-place, optional).
        device: Device to map tensors to.

    Returns:
        Metadata dict with keys: epoch, best_metric, metric_name,
        config, extra, timestamp.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint: %s (epoch=%d, %s=%.4f)",
        path,
        checkpoint.get("epoch", 0),
        checkpoint.get("metric_name", "val_qwk"),
        checkpoint.get("best_metric", 0.0) or 0.0,
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric"),
        "metric_name": checkpoint.get("metric_name", "val_qwk"),
        "config": checkpoint.get("config"),
        "extra": checkpoint.get("extra", {}),
        "timestamp": checkpoint.get("timestamp"),
    }


def find_best_checkpoint(
    checkpoint_dir: str | Path,
    metric_name: str = "val_qwk",
) -> Optional[Path]:
    """Find the checkpoint with the best metric value in a directory.

    Loads metadata from each .pt file to compare metric values.

    Returns:
        Path to the best checkpoint, or None if directory is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    best_path = None
    best_value = float("-inf")

    for ckpt_path in sorted(checkpoint_dir.glob("*.pt")):
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            value = data.get("best_metric")
            if value is not None and value > best_value:
                best_value = value
                best_path = ckpt_path
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s", ckpt_path, e)

    return best_path


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Find the most recent checkpoint by epoch number.

    Useful for resuming interrupted HPC jobs.

    Returns:
        Path to the latest checkpoint, or None if directory is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    latest_path = None
    latest_epoch = -1

    for ckpt_path in sorted(checkpoint_dir.glob("*.pt")):
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            epoch = data.get("epoch", -1)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = ckpt_path
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s", ckpt_path, e)

    return latest_path
