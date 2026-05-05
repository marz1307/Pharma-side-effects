"""Logging helpers used across the package."""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once with a consistent format.

    Args:
        level: Logging level (int or string name). Defaults to INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root logger if needed.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard library logger.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
