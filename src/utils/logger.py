"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
