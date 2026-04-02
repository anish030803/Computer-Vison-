"""Training callbacks: early stopping, checkpointing, logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger("dr_detection")


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to monitor.
        patience: Epochs to wait after last improvement.
        mode: 'max' for metrics like QWK/accuracy, 'min' for loss.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_qwk",
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Dict of metric values for this epoch.

        Returns:
            True if training should stop.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: %s did not improve for %d epochs "
                    "(best=%.4f, current=%.4f)",
                    self.monitor, self.patience, self.best_value, value,
                )
                return True

        return False

    def reset(self) -> None:
        """Reset state (e.g., between training phases)."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpointer:
    """Save model checkpoints based on metric improvement.

    Saves both best model and periodic checkpoints (every epoch by default).
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_qwk",
        mode: str = "max",
        save_every_epoch: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_every_epoch = save_every_epoch
        self.best_value: Optional[float] = None

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: dict[str, float],
        config: Optional[dict] = None,
    ) -> Optional[Path]:
        """Save checkpoint if appropriate.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        value = metrics.get(self.monitor, 0.0)
        is_best = False

        if self.best_value is None:
            self.best_value = value
            is_best = True
        elif self.mode == "max" and value > self.best_value:
            self.best_value = value
            is_best = True
        elif self.mode == "min" and value < self.best_value:
            self.best_value = value
            is_best = True

        saved_path = None

        # Save best model
        if is_best:
            best_path = self.save_dir / "best.pt"
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=value,
                metric_name=self.monitor,
                config=config,
            )
            saved_path = best_path
            logger.info("New best model saved: %s=%.4f", self.monitor, value)

        # Save epoch checkpoint
        if self.save_every_epoch:
            epoch_path = self.save_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=self.best_value,
                metric_name=self.monitor,
                config=config,
            )
            saved_path = epoch_path

        return saved_path

    def reset(self) -> None:
        """Reset best tracking (e.g., between phases)."""
        self.best_value = None


class MetricLogger:
    """Log metrics to TensorBoard and/or console."""

    def __init__(
        self,
        log_dir: Optional[str | Path] = None,
        use_tensorboard: bool = True,
    ) -> None:
        self.writer = None
        if use_tensorboard and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(log_dir))
                logger.info("TensorBoard logging enabled: %s", log_dir)
            except ImportError:
                logger.warning("TensorBoard not available, skipping")

    def log_epoch(
        self,
        phase: str,
        epoch: int,
        metrics: dict[str, float],
        lr: float,
    ) -> None:
        """Log metrics for an epoch.

        Args:
            phase: Training phase name (e.g., 'phase1', 'phase2').
            epoch: Epoch number.
            metrics: Dict of metric values.
            lr: Current learning rate.
        """
        # Console logging
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(
            "[%s] Epoch %d | lr=%.2e | %s",
            phase, epoch, lr, metrics_str,
        )

        # TensorBoard logging
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{name}", value, epoch)
            self.writer.add_scalar(f"{phase}/learning_rate", lr, epoch)
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
