"""Common benchmark helpers and registration."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional


@contextmanager
def bench_timer(name: str = "op") -> Generator[dict, None, None]:
    record = {"name": name, "elapsed_s": 0.0, "n": 0}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_s"] = time.perf_counter() - t0


def report(record: dict, *, n: Optional[int] = None, unit: str = "ops/s") -> str:
    elapsed = record["elapsed_s"]
    count = n or record.get("n") or 0
    if elapsed <= 0 or count <= 0:
        return f"{record['name']}: elapsed={elapsed:.4f}s"
    rate = count / elapsed
    return f"{record['name']}: {count} in {elapsed:.4f}s -> {rate:.2f} {unit}"
