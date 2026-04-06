"""Main training loop with two-phase strategy.

Phase 1: Frozen backbone → train classification head
Phase 2: Unfreeze top layers → fine-tune with lower LR

Supports mixed precision (BF16), gradient clipping, checkpoint resume,
early stopping, and comprehensive per-epoch logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import mixup_criterion, mixup_data
from src.training.callbacks import EarlyStopping, MetricLogger, ModelCheckpointer
from src.training.losses import build_loss
from src.training.metrics import RunningMetrics
from src.training.schedulers import build_scheduler
from src.utils.checkpoint import find_latest_checkpoint, load_checkpoint
from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class Trainer:
    """Two-phase trainer for DR classification models.

    Handles the complete training lifecycle:
    1. Phase 1: Head warmup (frozen backbone)
    2. Phase 2: Fine-tuning (partially unfrozen backbone)
    3. Checkpoint save/resume for HPC resilience
    4. Mixed precision training (BF16 on H200)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        class_weights: Optional[list[float]] = None,
        resume_checkpoint: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights

        # Mixed precision
        hw_cfg = config.get("hardware", Config({"mixed_precision": False, "precision": "fp32"}))
        self.use_amp = hw_cfg.get("mixed_precision", False) and torch.cuda.is_available()
        precision = hw_cfg.get("precision", "fp32")
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Monitoring
        tb_cfg = config.get("monitoring", Config({"tensorboard": Config({"enabled": False})}))
        tb_enabled = tb_cfg.get("tensorboard", Config({"enabled": False})).get("enabled", False)
        tb_log_dir = tb_cfg.get("tensorboard", Config({"log_dir": None})).get("log_dir")
        self.metric_logger = MetricLogger(
            log_dir=tb_log_dir if tb_enabled else None,
            use_tensorboard=tb_enabled,
        )

        # Checkpointing
        ckpt_cfg = config.get("checkpointing", Config({
            "save_dir": "checkpoints", "save_every_epoch": True,
            "monitor": "val_qwk", "mode": "max",
        }))
        self.checkpointer = ModelCheckpointer(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            monitor=ckpt_cfg.get("monitor", "val_qwk"),
            mode=ckpt_cfg.get("mode", "max"),
            save_every_epoch=ckpt_cfg.get("save_every_epoch", True),
        )

        # Regularization
        reg_cfg = config.get("regularization", Config({}))
        es_cfg = reg_cfg.get("early_stopping", Config({"monitor": "val_qwk", "patience": 5, "mode": "max"}))
        self.early_stopping = EarlyStopping(
            monitor=es_cfg.get("monitor", "val_qwk"),
            patience=es_cfg.get("patience", 5),
            mode=es_cfg.get("mode", "max"),
        )

        # Resume state
        self.start_epoch = 0
        self.best_metric = None
        self.current_phase = 1

        if resume_checkpoint:
            self._resume_from_checkpoint(resume_checkpoint)

    def train(self) -> dict[str, Any]:
        """Run the complete two-phase training.

        Returns:
            Dict with training history and final metrics.
        """
        history: dict[str, list] = {"phase1": [], "phase2": []}

        # --- Phase 1: Head Warmup ---
        logger.info("=" * 60)
        logger.info("Phase 1: Head Warmup (backbone frozen)")
        logger.info("=" * 60)

        phase1_cfg = self.config.training.phase1
        self.current_phase = 1

        # Freeze backbone
        self.model.freeze_backbone()

        # Build optimizer, scheduler, loss for Phase 1
        optimizer = self._build_optimizer(phase1_cfg)
        scheduler = build_scheduler(phase1_cfg.scheduler, optimizer, phase1_cfg.epochs)
        criterion = build_loss(phase1_cfg.loss, self.class_weights).to(self.device)

        phase1_history = self._train_phase(
            phase_name="phase1",
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=phase1_cfg.epochs,
            use_mixup=False,  # No MixUp in Phase 1
            gradient_clip=None,
        )
        history["phase1"] = phase1_history

        # --- Phase 2: Fine-tuning ---
        logger.info("=" * 60)
        logger.info("Phase 2: Fine-tuning (backbone partially unfrozen)")
        logger.info("=" * 60)

        phase2_cfg = self.config.training.phase2
        self.current_phase = 2

        # Unfreeze backbone
        if hasattr(phase2_cfg, "unfreeze_ratio"):
            self.model.unfreeze_backbone(ratio=phase2_cfg.unfreeze_ratio)
        elif hasattr(phase2_cfg, "unfreeze_last_n_blocks"):
            self.model.unfreeze_backbone(last_n_blocks=phase2_cfg.unfreeze_last_n_blocks)
        else:
            self.model.unfreeze_backbone()

        # Reset early stopping and checkpointer for Phase 2
        self.early_stopping.reset()
        self.checkpointer.reset()

        # Build new optimizer (different LR), scheduler, loss for Phase 2
        optimizer = self._build_optimizer(phase2_cfg)
        scheduler = build_scheduler(phase2_cfg.scheduler, optimizer, phase2_cfg.epochs)
        criterion = build_loss(phase2_cfg.loss, self.class_weights).to(self.device)

        # MixUp and gradient clipping config
        mixup_cfg = phase2_cfg.get("mixup", Config({"enabled": False}))
        use_mixup = mixup_cfg.get("enabled", False)
        mixup_alpha = mixup_cfg.get("alpha", 0.2) if use_mixup else 0.0

        grad_clip_cfg = phase2_cfg.get("gradient_clipping", Config({"max_norm": None}))
        gradient_clip = grad_clip_cfg.get("max_norm")

        phase2_history = self._train_phase(
            phase_name="phase2",
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=phase2_cfg.epochs,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            gradient_clip=gradient_clip,
        )
        history["phase2"] = phase2_history

        self.metric_logger.close()
        return history

    def _train_phase(
        self,
        phase_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        epochs: int,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        gradient_clip: Optional[float] = None,
    ) -> list[dict[str, float]]:
        """Train for a single phase.

        Returns:
            List of per-epoch metric dicts.
        """
        history = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch(
                optimizer, criterion, use_mixup, mixup_alpha, gradient_clip
            )

            # Validate
            val_metrics = self._validate_epoch(criterion)

            # Combine metrics
            all_metrics = {}
            for k, v in train_metrics.items():
                all_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                all_metrics[f"val_{k}"] = v

            elapsed = time.time() - epoch_start
            all_metrics["epoch_time"] = elapsed

            # Get current LR
            if hasattr(scheduler, "get_last_lr"):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Log
            self.metric_logger.log_epoch(phase_name, epoch, all_metrics, current_lr)

            # Checkpoint
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else None
            self.checkpointer(
                self.model, optimizer, scheduler, epoch, all_metrics, config_dict
            )

            # Step scheduler
            if hasattr(scheduler, "step"):
                scheduler.step()

            # Early stopping
            if self.early_stopping(all_metrics):
                logger.info("Stopping early at epoch %d", epoch)
                break

            history.append(all_metrics)

        return history

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        use_mixup: bool,
        mixup_alpha: float,
        gradient_clip: Optional[float],
    ) -> dict[str, float]:
        """Train for a single epoch."""
        self.model.train()
        running = RunningMetrics()

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if use_mixup and mixup_alpha > 0:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                    output = self.model(images)
                    if isinstance(output, dict):
                        output = output["logits"]
                    loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
                else:
                    output = self.model(images)
                    if isinstance(output, dict):
                        output = output["logits"]
                    loss = criterion(output, labels)

            # Backward
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                if gradient_clip:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()

            # Track metrics
            with torch.no_grad():
                probs = F.softmax(output, dim=-1).cpu().numpy()
                preds = output.argmax(dim=-1).cpu().numpy()
                true_labels = labels.cpu().numpy() if not use_mixup else labels.cpu().numpy()
                running.update(preds, true_labels, probs, loss.item())

        return running.compute()

    @torch.no_grad()
    def _validate_epoch(self, criterion: nn.Module) -> dict[str, float]:
        """Validate for a single epoch."""
        self.model.eval()
        running = RunningMetrics()

        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(images)
                if isinstance(output, dict):
                    output = output["logits"]
                loss = criterion(output, labels)

            probs = F.softmax(output, dim=-1).cpu().numpy()
            preds = output.argmax(dim=-1).cpu().numpy()
            running.update(preds, labels.cpu().numpy(), probs, loss.item())

        return running.compute()

    def _build_optimizer(self, phase_config) -> torch.optim.Optimizer:
        """Build optimizer from phase config."""
        opt_cfg = phase_config.optimizer
        name = opt_cfg.name
        lr = opt_cfg.lr
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info("Optimizer: %s, lr=%.2e, params=%d", name, lr, len(trainable_params))

        if name == "adamw":
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif name == "adam":
            return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif name == "sgd":
            return torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        path = Path(checkpoint_path)

        # If path is a directory, find the latest checkpoint
        if path.is_dir():
            path = find_latest_checkpoint(path)
            if path is None:
                logger.warning("No checkpoints found in %s", checkpoint_path)
                return

        logger.info("Resuming from checkpoint: %s", path)
        meta = load_checkpoint(path, self.model, device=self.device)
        self.start_epoch = meta["epoch"]
        self.best_metric = meta["best_metric"]
        logger.info("Resumed at epoch %d, best %s=%.4f",
                     self.start_epoch, meta["metric_name"], self.best_metric or 0.0)
