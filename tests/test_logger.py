"""Tests for the logger configuration utility."""

import logging
import os
import tempfile
from pathlib import Path

from medicalexplainer.logger import configure_logger


def test_configure_logger_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "subdir" / "test.log"
    logger = configure_logger(name="test_logger_file", filepath=log_file)

    assert log_file.exists(), "Log file should be created"
    assert isinstance(logger, logging.Logger)


def test_configure_logger_returns_existing_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "dup.log"
    logger1 = configure_logger(name="test_logger_dup", filepath=log_file)
    logger2 = configure_logger(name="test_logger_dup", filepath=log_file)

    # Same object, no duplicate handlers
    assert logger1 is logger2
    # Handlers are only added once
    assert len(logger1.handlers) == 2  # file + console


def test_configure_logger_level(tmp_path: Path) -> None:
    log_file = tmp_path / "level.log"
    logger = configure_logger(
        name="test_logger_level", filepath=log_file, level=logging.WARNING
    )
    assert logger.level == logging.WARNING
