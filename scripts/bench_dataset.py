"""Quick micro-benchmark for MarketDataset.__getitem__.

Run: python scripts/bench_dataset.py [bars]

Measures:
  - cold __getitem__ time (no cache, first touch of every sample)
  - warm __getitem__ time (with --cache-charts)
  - fast_render vs full_render

We do NOT depend on pytest or test suite.
"""
from __future__ import annotations

import argparse
import os
import time

# Force the same render mode the tests use
os.environ.setdefault("ZHISA_FAST_RENDER", "1")
os.environ.setdefault("ZHISA_TEST_DEVICE", "auto")

import numpy as np
import pandas as pd  # noqa: E402

from zhisa.data.dataset import MarketDataset, SampleSpec  # noqa: E402
from zhisa.data.synthetic import MarketConfig, generate_market  # noqa: E402
from zhisa.utils.seeding import set_seed  # noqa: E402


def make_df(n_bars: int) -> pd.DataFrame:
    set_seed(1234)
    return generate_market(MarketConfig(n_bars=n_bars, freq="5min", seed=1234))


def time_getitem(ds: MarketDataset, indices: list[int]) -> float:
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--fast-render", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--repeat", type=int, default=2,
                    help="How many times to iterate the dataset (for warm-cache timing).")
    args = ap.parse_args()

    if args.fast_render:
        os.environ["ZHISA_FAST_RENDER"] = "1"

    print(f"bars={args.bars} window={args.window} image={args.image_size} "
          f"workers={args.workers} fast_render={bool(os.environ.get('ZHISA_FAST_RENDER'))} "
          f"cache={args.cache}")

    df = make_df(args.bars)
    spec = SampleSpec(chart_window=args.window, feature_window=args.window,
                      image_size=args.image_size, horizons=(4, 16, 64))
    ds = MarketDataset(df, spec=spec, cache_charts=args.cache)

    n = len(ds)
    indices = list(range(n))
    print(f"len(ds) = {n}")

    # Cold (no cache, just first pass)
    cold = time_getitem(ds, indices)
    cold_per = cold / max(1, n) * 1000.0
    print(f"cold  : {cold:.3f}s total | {cold_per:.2f} ms/sample | {n/max(cold,1e-9):.1f} samples/s")

    # Warm repeats
    warm_times = []
    for r in range(args.repeat):
        wt = time_getitem(ds, indices)
        warm_times.append(wt)
        print(f"warm{r+1}: {wt:.3f}s total | {wt/n*1000.0:.2f} ms/sample | {n/wt:.1f} samples/s")

    # Simulate training: take small batches (sequential reads across the dataset)
    batch_size = 32
    seq_indices = indices  # DataLoader with shuffle=False approximates this
    t0 = time.perf_counter()
    n_batches = 0
    for i in range(0, n - batch_size, batch_size):
        for j in range(i, i + batch_size):
            _ = ds[j]
        n_batches += 1
    seq_time = time.perf_counter() - t0
    print(f"seq batches={n_batches} bs={batch_size} : {seq_time:.3f}s | "
          f"{n_batches*batch_size/seq_time:.1f} samples/s | "
          f"{seq_time/max(1,n_batches)*1000.0:.2f} ms/batch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
