"""Lightweight logger factory."""

from __future__ import annotations

import datetime
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = False,
) -> logging.Logger:
    """Create a process-safe logger with optional file logging."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    if enable_file_logging and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_dir / f"pairwise_bo_{datetime.date.today()}.log",
            maxBytes=10_000_000,
            backupCount=5,
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
