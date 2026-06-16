"""Centralised DataLoader configuration.

Why this exists
---------------
Across the codebase there are ~20 ``DataLoader(...)`` call sites. They were
created ad-hoc and have inconsistent defaults: most set ``num_workers``
explicitly but few enable ``pin_memory`` or ``persistent_workers``. On a
GTX 1080-class GPU this leaves the model waiting on the CPU for the next
batch.

This module provides a single ``build_dataloader`` factory that picks
sensible defaults based on the dataset size, device, and whether the
dataset already precomputes expensive samples (like ``MarketDataset`` with
``cache_charts=True``). It is the recommended way to construct a loader
for any trainer; legacy call sites can be migrated gradually without
changing their public API.

Key choices
-----------
* ``pin_memory=True`` whenever CUDA is in use (zero-copy H2D is free).
* ``persistent_workers=True`` whenever ``num_workers > 0`` (avoids
  re-spawning workers every epoch — major win for the matplotlib
  ``MarketDataset._chart`` lazy path, harmless for the precompute path).
* ``num_workers`` heuristic:
    - If the dataset has a ``__fast_getitem__`` marker (e.g. precomputed
      charts), default to ``0`` — the IPC overhead is larger than the
      work per item.
    - Otherwise default to ``min(4, os.cpu_count() or 1)`` capped at a
      safe fraction.
* All defaults are overridable via keyword arguments and via the
  ``ZHISA_NUM_WORKERS`` / ``ZHISA_FORCE_PIN_MEMORY`` env vars.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_num_workers(requested: Optional[int], ds: Dataset) -> int:
    if requested is not None and int(requested) >= 0:
        return int(requested)
    env = os.environ.get("ZHISA_NUM_WORKERS")
    if env is not None and env.strip():
        return int(env)
    # Heuristic: if the dataset advertises a fast path, prefer 0.
    if getattr(ds, "__fast_getitem__", False):
        return 0
    cpu = os.cpu_count() or 1
    # Cap at 4 by default — beyond that the IPC overhead usually wins
    # for tabular / numeric datasets.
    return min(4, cpu)


def _resolve_pin_memory(requested: Optional[bool]) -> bool:
    if requested is not None:
        return bool(requested)
    env = os.environ.get("ZHISA_FORCE_PIN_MEMORY")
    if env is not None and env.strip():
        return env.strip().lower() in ("1", "true", "yes")
    return bool(torch.cuda.is_available())


def _resolve_persistent_workers(num_workers: int, requested: Optional[bool]) -> bool:
    if requested is not None:
        return bool(requested)
    return num_workers > 0


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
    collate_fn=None,
    prefetch_factor: int = 2,
    **kwargs,
) -> DataLoader:
    """Construct a DataLoader with sensible defaults.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int
    shuffle : bool
    num_workers : int or None
        ``None`` picks a default from the heuristic above.
    pin_memory : bool or None
        ``None`` enables it automatically on CUDA.
    persistent_workers : bool or None
        ``None`` enables it whenever ``num_workers > 0``.
    drop_last, collate_fn, **kwargs
        Forwarded to :class:`torch.utils.data.DataLoader`.
    """
    nw = _resolve_num_workers(num_workers, dataset)
    pm = _resolve_pin_memory(pin_memory)
    pw = _resolve_persistent_workers(nw, persistent_workers)
    # prefetch_factor is only meaningful with workers
    pf = prefetch_factor if nw > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw,
        drop_last=drop_last,
        collate_fn=collate_fn,
        prefetch_factor=pf,
        **kwargs,
    )
    return loader


__all__ = ["build_dataloader"]
