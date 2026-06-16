"""Micro-benchmark for the smaller / less-tested datasets.

Tests RegimeOutcomeDataset and RegimeSupervisionDataset to see if they
are worth the optimisation effort.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("ZHISA_FAST_RENDER", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from zhisa.data.synthetic import MarketConfig, generate_market  # noqa: E402
from zhisa.regime.learned import RegimeOutcomeDataset, RegimeOutcomeDatasetConfig  # noqa: E402
from zhisa.regime.dataset import RegimeSupervisionDataset, RegimeSupervisionConfig  # noqa: E402
from zhisa.utils.seeding import set_seed  # noqa: E402


def make_df(n_bars: int = 2000) -> pd.DataFrame:
    set_seed(42)
    return generate_market(MarketConfig(n_bars=n_bars, freq="5min", seed=42))


def time_getitem(ds, indices, repeat: int = 2) -> tuple[float, float, float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for i in indices:
            _ = ds[i]
        times.append(time.perf_counter() - t0)
    cold = times[0]
    warm = min(times[1:]) if len(times) > 1 else cold
    per_cold = cold / max(1, len(indices)) * 1000.0
    per_warm = warm / max(1, len(indices)) * 1000.0
    return cold, warm, per_cold, per_warm


def bench_regime_outcome(n_bars: int = 2000) -> None:
    df = make_df(n_bars)
    set_seed(42)
    cfg = RegimeOutcomeDatasetConfig(horizons=(6, 12), stride=10, min_history=64)
    ds = RegimeOutcomeDataset(df, cfg=cfg)
    n = len(ds)
    print(f"\nRegimeOutcomeDataset: bars={n_bars} len(ds)={n} "
          f"horizons={cfg.horizons} stride={cfg.stride}")
    # Skip the first 3 to warm caches
    indices = list(range(3, n))
    cold, warm, per_cold, per_warm = time_getitem(ds, indices)
    print(f"  cold  : {cold:.3f}s | {per_cold:.2f} ms/sample | {n/max(cold,1e-9):.1f}/s")
    print(f"  warm  : {warm:.3f}s | {per_warm:.2f} ms/sample | {n/max(warm,1e-9):.1f}/s")


def bench_regime_supervision(n_bars: int = 2000) -> None:
    df = make_df(n_bars)
    set_seed(42)
    cfg = RegimeSupervisionConfig(horizon=12, stride=10, min_history=64)
    ds = RegimeSupervisionDataset(df, cfg=cfg)
    n = len(ds)
    print(f"\nRegimeSupervisionDataset: bars={n_bars} len(ds)={n} "
          f"horizon={cfg.horizon} stride={cfg.stride}")
    indices = list(range(3, n))
    cold, warm, per_cold, per_warm = time_getitem(ds, indices)
    print(f"  cold  : {cold:.3f}s | {per_cold:.2f} ms/sample | {n/max(cold,1e-9):.1f}/s")
    print(f"  warm  : {warm:.3f}s | {per_warm:.2f} ms/sample | {n/max(warm,1e-9):.1f}/s")


def main() -> int:
    bench_regime_outcome(2000)
    bench_regime_supervision(2000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
