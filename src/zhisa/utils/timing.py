"""Timing and rate-limiting helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional


class Timer:
    """Wall-clock timer with start/stop/elapsed and pause/resume support."""

    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self._accum: float = 0.0
        self._running: bool = False

    def start(self) -> "Timer":
        if self._running:
            return self
        self._t0 = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> float:
        if not self._running:
            return self._accum
        self._accum += time.perf_counter() - (self._t0 or 0.0)
        self._t0 = None
        self._running = False
        return self._accum

    @property
    def elapsed(self) -> float:
        if self._running and self._t0 is not None:
            return self._accum + (time.perf_counter() - self._t0)
        return self._accum

    def reset(self) -> None:
        self._t0 = None
        self._accum = 0.0
        self._running = False

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def rate_limit(max_per_sec: float) -> Generator[None, None, None]:
    """Block the caller so the surrounding block runs at <= max_per_sec.

    Useful for capping data-emission rates in synthetic generators and feeds.
    """
    if max_per_sec <= 0:
        yield
        return
    interval = 1.0 / max_per_sec
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed < interval:
        time.sleep(interval - elapsed)
