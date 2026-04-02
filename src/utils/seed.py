"""Reproducibility utilities — seed all sources of randomness."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("dr_detection")


def seed_everything(seed: int = 42) -> None:
    """Set random seed for full reproducibility.

    Seeds Python's random, NumPy, PyTorch (CPU + all CUDA devices),
    and sets deterministic cuDNN flags.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Seeded everything with seed=%d", seed)
