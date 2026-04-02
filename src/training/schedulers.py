"""Learning rate schedulers: warmup, cosine annealing, combined.

Supports the two-phase training strategy:
- Phase 1: linear warmup → cosine annealing
- Phase 2: cosine annealing with warm restarts
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    ReduceLROnPlateau,
    SequentialLR,
)

logger = logging.getLogger("dr_detection")


class LinearWarmupCosineScheduler:
    """Linear warmup followed by cosine annealing.

    Used in Phase 1 training: warms up LR linearly for `warmup_epochs`,
    then decays with cosine schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # Warmup: linear from 0 to base_lr
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) if warmup_epochs > 0 else 1.0,
        )

        # Cosine: decay from base_lr to min_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    def step(self, epoch: Optional[int] = None) -> None:
        self.scheduler.step()

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


class CosineWarmRestartsScheduler:
    """Cosine annealing with warm restarts (Loshchilov & Hutter, 2016).

    Used in Phase 2 fine-tuning. Periodically restarts the learning rate
    to escape local minima.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 5,
        T_mult: int = 2,
        min_lr: float = 1e-7,
    ) -> None:
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
        )

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


def build_scheduler(
    scheduler_config,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
) -> object:
    """Build LR scheduler from config.

    Args:
        scheduler_config: Scheduler section of training phase config.
        optimizer: The optimizer to schedule.
        total_epochs: Total epochs for this training phase.

    Returns:
        Scheduler instance.
    """
    name = scheduler_config.name

    if name == "cosine_with_warmup":
        warmup_epochs = scheduler_config.get("warmup_epochs", 5)
        min_lr = scheduler_config.get("min_lr", 1e-6)
        scheduler = LinearWarmupCosineScheduler(
            optimizer, warmup_epochs, total_epochs, min_lr
        )
    elif name == "cosine_annealing_warm_restarts":
        T_0 = scheduler_config.get("T_0", 5)
        T_mult = scheduler_config.get("T_mult", 2)
        min_lr = scheduler_config.get("min_lr", 1e-7)
        scheduler = CosineWarmRestartsScheduler(optimizer, T_0, T_mult, min_lr)
    elif name == "reduce_lr_on_plateau":
        factor = scheduler_config.get("factor", 0.5)
        patience = scheduler_config.get("patience", 3)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    logger.info("Built scheduler: %s (epochs=%d)", name, total_epochs)
    return scheduler
