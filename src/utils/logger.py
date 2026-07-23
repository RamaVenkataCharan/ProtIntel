"""Structured logging utility for ProtIntel.

Provides a consistent, configurable logging interface used throughout
the project. Supports both console and file output with structured formatting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_CONFIGURED = False


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Get or create a configured logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (e.g., ``logging.DEBUG``, ``"INFO"``).
        log_file: Optional path to a log file. If provided, logs are
            written to both console and the file.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if not _CONFIGURED:
        _configure_root_logger(level, log_file)
        _CONFIGURED = True

    return logger


def _configure_root_logger(
    level: int,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level for all handlers.
        log_file: Optional path to a log file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevent duplicate handlers on re-initialization
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
