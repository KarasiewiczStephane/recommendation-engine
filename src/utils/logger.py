"""Structured logging utilities for the recommendation engine.

Provides a centralized logging configuration with consistent formatting
across all modules.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a named logger with standard formatting.

    Args:
        name: The logger name, typically __name__ of the calling module.
        level: The logging level. Defaults to INFO.

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
