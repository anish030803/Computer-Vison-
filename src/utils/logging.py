"""Structured logging with console and file handlers."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_CONSOLE_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "dr_detection",
    log_dir: Optional[str | Path] = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    filename: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Args:
        name: Logger name (hierarchical, e.g. "dr_detection.training").
        log_dir: Directory for log files. Created if needed.
        level: Logging level.
        console: Add a console (stderr) handler.
        file: Add a file handler.
        filename: Custom log filename. Defaults to "{name}_{timestamp}.log".

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT)
        )
        logger.addHandler(console_handler)

    if file and log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_path / filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "dr_detection") -> logging.Logger:
    """Get an existing logger by name.

    If not yet configured via setup_logger(), returns a basic logger.
    """
    return logging.getLogger(name)
