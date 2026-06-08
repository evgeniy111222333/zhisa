"""Deterministic seeding for Python, NumPy and PyTorch."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


_DEFAULT_SEED = 42


def set_seed(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Also sets PYTHONHASHSEED for dict ordering determinism and
    enables deterministic CuDNN where possible.
    """
    if seed is None:
        seed = _DEFAULT_SEED
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_seed() -> int:
    """Return the active integer seed (from env if set)."""
    s = os.environ.get("PYTHONHASHSEED")
    if s is None:
        return _DEFAULT_SEED
    try:
        return int(s)
    except (TypeError, ValueError):
        return _DEFAULT_SEED
