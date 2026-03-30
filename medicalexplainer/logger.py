"""
logger - Shared logging configuration utility.

Provides :func:`configure_logger` which attaches file and console handlers to
a named logger while preventing duplicate handlers on repeated calls.

When ``rich`` is available the console handler uses
:class:`rich.logging.RichHandler` so that log messages are rendered cleanly
alongside any active :class:`rich.progress.Progress` bars (rich coordinates
all writes through a shared ``Console`` instance, preventing interleaving).
"""

import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler
    from rich.console import Console as _Console

    _rich_console = _Console(stderr=False)
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


def configure_logger(
    name: str, filepath: Path, level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure a logger with file and console handlers.

    The function is idempotent: calling it again for the same *name* when
    handlers are already attached is a no-op.

    When ``rich`` is installed the console handler is a
    :class:`rich.logging.RichHandler` which cooperates gracefully with
    :class:`rich.progress.Progress` bars rendered at the same time.

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

    file_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    file_formatter = logging.Formatter(file_fmt, datefmt=date_fmt)

    # --- File handler (always plain text) ---
    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console handler ---
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            console=_rich_console,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            log_time_format=date_fmt,
            level=level,
        )
        # RichHandler formats its own markup; suppress duplicate level/name prefix
        console_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
