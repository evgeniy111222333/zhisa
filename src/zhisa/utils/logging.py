"""Lightweight logging helper built on the standard library."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _setup_root() -> None:
    global _initialized
    if _initialized:
        return
    level = os.environ.get("ZHISA_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger. Idempotent and cheap to call."""
    _setup_root()
    return logging.getLogger(name if name else "zhisa")
