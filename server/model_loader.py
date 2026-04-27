"""Model loading utilities for the inference server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, Config

logger = logging.getLogger("dr_detection")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


class ModelManager:
    """Manages model lifecycle: loading, inference, and cleanup."""

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.config: Optional[Config] = None
        self.model_name: Optional[str] = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(
        self,
        config_path: str = "configs/train_efficientnet.yaml",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Load a model from config and checkpoint.

        Args:
            config_path: Path to training config YAML.
            checkpoint_path: Path to .pt checkpoint file.
        """
        self.config = load_config(config_path)
        self.model_name = self.config.model.name

        # Build model
        if "efficientnet" in self.model_name:
            from src.models.efficientnet import build_efficientnet
            self.model = build_efficientnet(self.config)
        elif "dinov2" in self.model_name:
            from src.models.dinov2 import build_dinov2
            self.model = build_dinov2(self.config)
        elif "resnet" in self.model_name:
            from src.models.resnet_baseline import build_resnet
            self.model = build_resnet(self.config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Load checkpoint weights
        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(checkpoint_path, self.model, device=self.device)
            logger.info("Loaded checkpoint: %s", checkpoint_path)
        else:
            logger.warning("No checkpoint loaded — using pretrained weights only")

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model ready: %s on %s", self.model_name, self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def image_size(self) -> int:
        if self.config is None:
            return 380
        return self.config.model.image_size
