"""
logger - Shared logging configuration utility.

Provides :func:`configure_logger` which attaches file and console handlers to
a named logger while preventing duplicate handlers on repeated calls.
"""

import logging
import sys
from pathlib import Path


def configure_logger(
    name: str, filepath: Path, level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure a logger with file and console handlers.

    The function is idempotent: calling it again for the same *name* when
    handlers are already attached is a no-op.

    Args:
        name (str): Name of the logger.
        filepath (Path): Path to the log file (parent directories are created
            automatically).
        level (int): Logging level for both handlers (default: ``logging.DEBUG``).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.error("Error creating log directory: %s", exc)
        return logger

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
