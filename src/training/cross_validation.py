"""K-fold cross-validation trainer.

Runs the full two-phase Trainer across all folds, aggregates per-fold
metrics, and reports mean +/- std for each metric. Evaluates the
held-out test set with the best model from each fold.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import DRDataset, create_kfold_splits
from src.evaluation.evaluate import evaluate_model
from src.training.metrics import compute_all_metrics
from src.training.trainer import Trainer
from src.utils.checkpoint import load_checkpoint
from src.utils.config import Config

logger = logging.getLogger("dr_detection")


class CrossValidationTrainer:
    """Run k-fold cross-validation with the two-phase Trainer.

    For each fold:
    1. Build fresh model (pretrained backbone, untrained head)
    2. Run two-phase training (Phase 1: frozen, Phase 2: unfrozen)
    3. Evaluate on fold's validation set
    4. Save best checkpoint per fold

    After all folds: aggregate metrics, evaluate on held-out test set.
    """

    def __init__(
        self,
        config: Config,
        labels_path: str | Path,
        images_dir: str | Path,
        n_folds: int = 5,
        device: str = "cuda",
        class_weights: Optional[list[float]] = None,
        use_cache: bool = True,
        output_dir: str | Path = "outputs/cross_validation",
    ) -> None:
        self.config = config
        self.labels_path = labels_path
        self.images_dir = images_dir
        self.n_folds = n_folds
        self.device = device
        self.class_weights = class_weights
        self.use_cache = use_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Run full k-fold cross-validation.

        Returns:
            Dict with per-fold metrics, aggregated stats, and test metrics.
        """
        seed = self.config.get("seed", 42)
        image_size = self.config.model.image_size
        batch_size = self.config.training.phase1.batch_size
        hw_cfg = self.config.get("hardware", Config({"num_workers": 4}))
        num_workers = hw_cfg.get("num_workers", 4)

        # Create folds
        folds, test_df = create_kfold_splits(
            self.labels_path, n_folds=self.n_folds, seed=seed
        )

        # Transforms
        aug_config = self.config.augmentation
        train_transform = get_train_transforms(aug_config, image_size)
        val_transform = get_val_transforms(aug_config, image_size)

        fold_metrics: list[dict[str, float]] = []
        best_fold_qwk = -1.0
        best_fold_idx = 0

        for fold_idx, (train_df, val_df) in enumerate(folds):
            fold_num = fold_idx + 1
            logger.info("=" * 60)
            logger.info("FOLD %d/%d", fold_num, self.n_folds)
            logger.info("=" * 60)

            fold_dir = self.output_dir / f"fold_{fold_num}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Build fresh model for this fold
            model = self._build_model()

            # Create dataloaders for this fold
            train_loader = self._make_loader(
                train_df, train_transform, batch_size, num_workers, shuffle=True
            )
            val_loader = self._make_loader(
                val_df, val_transform, batch_size, num_workers, shuffle=False
            )

            # Override checkpoint dir for this fold — save only best to conserve disk
            fold_config = copy.deepcopy(self.config)
            fold_config._data["checkpointing"] = {
                "save_dir": str(fold_dir / "checkpoints"),
                "save_every_epoch": False,
                "save_best_only": True,
                "monitor": "val_qwk",
                "mode": "max",
            }
            fold_config._data["monitoring"] = {
                "tensorboard": {"enabled": False},
                "wandb": {"enabled": False},
                "log_every_n_steps": 50,
            }

            # Train
            trainer = Trainer(
                model=model,
                config=fold_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                class_weights=self.class_weights,
            )
            history = trainer.train()

            # Save fold history
            with open(fold_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)

            # Evaluate best model on validation set
            best_ckpt = fold_dir / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                load_checkpoint(best_ckpt, model, device=self.device)

            model.to(self.device)
            val_metrics = evaluate_model(
                model, val_loader, self.device,
                output_dir=fold_dir / "evaluation",
                model_name=f"fold_{fold_num}",
            )
            fold_metrics.append(val_metrics)

            logger.info(
                "Fold %d results: QWK=%.4f, Acc=%.4f, Macro-F1=%.4f",
                fold_num, val_metrics["qwk"], val_metrics["accuracy"],
                val_metrics["macro_f1"],
            )

            if val_metrics["qwk"] > best_fold_qwk:
                best_fold_qwk = val_metrics["qwk"]
                best_fold_idx = fold_idx

        # Aggregate metrics across folds
        aggregated = self._aggregate_metrics(fold_metrics)

        logger.info("=" * 60)
        logger.info("CROSS-VALIDATION SUMMARY (%d folds)", self.n_folds)
        logger.info("=" * 60)
        for metric, stats in aggregated.items():
            logger.info(
                "  %s: %.4f +/- %.4f (min=%.4f, max=%.4f)",
                metric, stats["mean"], stats["std"], stats["min"], stats["max"],
            )

        # Evaluate best fold model on held-out test set
        test_metrics = None
        if len(test_df) > 0:
            logger.info("Evaluating best fold (%d) on held-out test set...", best_fold_idx + 1)
            best_model = self._build_model()
            best_ckpt = self.output_dir / f"fold_{best_fold_idx + 1}" / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                load_checkpoint(best_ckpt, best_model, device=self.device)
            best_model.to(self.device)

            test_loader = self._make_loader(
                test_df, val_transform, batch_size, num_workers, shuffle=False
            )
            test_metrics = evaluate_model(
                best_model, test_loader, self.device,
                output_dir=self.output_dir / "test_evaluation",
                model_name="best_fold_on_test",
            )
            logger.info(
                "Test set: QWK=%.4f, Acc=%.4f", test_metrics["qwk"], test_metrics["accuracy"]
            )

        # Save full results
        results = {
            "n_folds": self.n_folds,
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
            "best_fold": best_fold_idx + 1,
            "best_fold_qwk": best_fold_qwk,
            "test_metrics": test_metrics,
        }
        with open(self.output_dir / "cv_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def _build_model(self) -> torch.nn.Module:
        """Build a fresh model from config."""
        model_name = self.config.model.name
        if "efficientnet" in model_name:
            from src.models.efficientnet import build_efficientnet
            return build_efficientnet(self.config)
        elif "dinov2" in model_name:
            from src.models.dinov2 import build_dinov2
            return build_dinov2(self.config)
        elif "resnet" in model_name:
            from src.models.resnet_baseline import build_resnet
            return build_resnet(self.config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _make_loader(
        self,
        df: pd.DataFrame,
        transform,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create a DataLoader from a DataFrame split."""
        dataset = DRDataset(
            image_ids=df["image_id"].astype(str).tolist(),
            labels=df["label"].tolist(),
            images_dir=self.images_dir,
            transform=transform,
            use_cache=self.use_cache,
        )

        sampler = None
        if shuffle and self.class_weights is not None:
            sample_weights = [
                self.class_weights[label] if isinstance(self.class_weights, list)
                else self.class_weights.get(label, 1.0)
                for label in df["label"]
            ]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    @staticmethod
    def _aggregate_metrics(
        fold_metrics: list[dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean, std, min, max across folds for each metric."""
        all_keys = set()
        for fm in fold_metrics:
            all_keys.update(fm.keys())

        aggregated = {}
        for key in sorted(all_keys):
            values = [fm.get(key, 0.0) for fm in fold_metrics]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "per_fold": values,
            }

        return aggregated
