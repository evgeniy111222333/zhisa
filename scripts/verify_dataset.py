"""Bit-exactness and benchmark for the new MarketDataset.

Compares the optimised __getitem__ against a reference (re-implemented
inline, identical to the *old* pandas-based logic) on the same indices.

Run: python scripts/verify_dataset.py [--bars 2000] [--fast-render]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("ZHISA_FAST_RENDER", "1")

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.features.ohlcv import compute_ohlcv_features, normalize_feature_window
from zhisa.features.time import compute_time_features
from zhisa.rendering.chart_renderer import render_chart
from zhisa.data.labeling import (
    TripleBarrierConfig,
    hmm_regime_labels,
    realized_volatility,
    triple_barrier,
)
from zhisa.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Reference (OLD) __getitem__ — must produce identical results to the new one
# ---------------------------------------------------------------------------
class ReferenceMarketDataset:
    def __init__(self, df, spec):
        self.df = df
        self.spec = spec
        self._features = compute_ohlcv_features(
            df, include_volume=spec.include_volume, include_indicators=spec.include_indicators
        )
        self._time_features = compute_time_features(df)
        primary = spec.horizons[len(spec.horizons) // 2] if spec.horizons else 16
        cfg = TripleBarrierConfig(max_holding=primary)
        self._tb = triple_barrier(df, cfg)
        self._vol = realized_volatility(df, horizon=primary)
        self._regime = hmm_regime_labels(df, n_states=spec.n_regime_states, lookback=256)

    def __len__(self):
        h = max(self.spec.horizons) if self.spec.horizons else 0
        return max(0, len(self.df) - self.spec.chart_window - h - 1)

    def __getitem__(self, t):
        spec = self.spec
        start = t
        end = t + spec.chart_window
        feature_window = self._features.iloc[start:end].to_numpy(dtype=np.float32)
        hist_start = max(0, t - 256)
        history_window = self._features.iloc[hist_start:end].to_numpy(dtype=np.float32)
        num = normalize_feature_window(feature_window, history_window)
        window_df = self.df.iloc[start:end]
        chart = render_chart(window_df, size=spec.image_size)
        primary_idx = end - 1
        ctx = self._time_features.iloc[primary_idx].to_numpy(dtype=np.float32)
        lbl_dir = int(self._tb["label"].iloc[primary_idx])
        lbl_ret = float(self._tb["ret"].iloc[primary_idx])
        v = float(self._vol.iloc[primary_idx])
        lbl_vol = 0.0 if np.isnan(v) else v
        lbl_risk = max(-lbl_ret, 0.0) + max(lbl_vol, 0.0)
        lbl_regime = int(self._regime.iloc[primary_idx])
        return {
            "chart": chart,
            "numeric": torch.from_numpy(num),
            "context": torch.from_numpy(ctx),
            "label_dir": torch.tensor(lbl_dir, dtype=torch.long),
            "label_ret": torch.tensor(lbl_ret, dtype=torch.float32),
            "label_vol": torch.tensor(lbl_vol, dtype=torch.float32),
            "label_risk": torch.tensor(lbl_risk, dtype=torch.float32),
            "label_regime": torch.tensor(lbl_regime, dtype=torch.long),
            "mask": torch.ones(spec.chart_window, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def compare_samples(s_new, s_ref, tol=1e-5):
    """Return list of (field, diff) for fields that differ."""
    diffs = []
    for k in ("chart", "numeric", "context"):
        a = s_new[k]
        b = s_ref[k]
        if isinstance(a, torch.Tensor):
            a_np = a.detach().cpu().numpy()
        else:
            a_np = np.asarray(a)
        if isinstance(b, torch.Tensor):
            b_np = b.detach().cpu().numpy()
        else:
            b_np = np.asarray(b)
        d = float(np.max(np.abs(a_np - b_np)))
        if d > tol:
            diffs.append((k, d))
    for k in ("label_dir", "label_regime"):
        a = int(s_new[k].item() if hasattr(s_new[k], "item") else s_new[k])
        b = int(s_ref[k].item() if hasattr(s_ref[k], "item") else s_ref[k])
        if a != b:
            diffs.append((k, f"{a} != {b}"))
    for k in ("label_ret", "label_vol", "label_risk"):
        a = float(s_new[k].item() if hasattr(s_new[k], "item") else s_new[k])
        b = float(s_ref[k].item() if hasattr(s_ref[k], "item") else s_ref[k])
        d = abs(a - b)
        if d > tol:
            diffs.append((k, f"{a} != {b}  (diff={d})"))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--fast-render", action="store_true")
    ap.add_argument("--check-every", type=int, default=17,
                    help="Check every Nth sample for bit-exactness.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Compare non-caching (lazy) path against the reference.")
    args = ap.parse_args()

    if args.fast_render:
        os.environ["ZHISA_FAST_RENDER"] = "1"

    set_seed(1234)
    df = generate_market(MarketConfig(n_bars=args.bars, freq="5min", seed=1234))
    spec = SampleSpec(
        chart_window=args.window, feature_window=args.window,
        image_size=args.image_size, horizons=(4, 16, 64),
    )

    cache_charts = not args.no_cache
    print(f"bars={args.bars} window={args.window} cache_charts={cache_charts}")

    # Build datasets with the *same* base df and seed
    set_seed(1234)
    new_ds = MarketDataset(df, spec=spec, cache_charts=cache_charts)
    set_seed(1234)
    ref_ds = ReferenceMarketDataset(df, spec)

    n = min(len(new_ds), len(ref_ds))
    print(f"len(new)={len(new_ds)}  len(ref)={len(ref_ds)}  -> checking {n}")

    # Bit-exactness on every Nth sample
    bad = 0
    samples_checked = 0
    for t in range(0, n, args.check_every):
        s_new = new_ds[t]
        s_ref = ref_ds[t]
        diffs = compare_samples(s_new, s_ref)
        samples_checked += 1
        if diffs:
            print(f"  DIFF at t={t}: {diffs[:5]}")
            bad += 1
            if bad >= 3:
                print("  ...stopping after 3 mismatches")
                break
    if bad == 0:
        print(f"  PASS: {samples_checked} samples bit-exact (tol=1e-5)")
    else:
        print(f"  FAIL: {bad}/{samples_checked} samples had differences")
        return 1

    # Performance: time __getitem__ across the whole dataset
    indices = list(range(n))
    t0 = time.perf_counter()
    for i in indices:
        _ = new_ds[i]
    elapsed = time.perf_counter() - t0
    print(f"NEW: {n} samples in {elapsed:.3f}s | {n/elapsed:.1f} samples/s "
          f"| {elapsed/n*1000:.2f} ms/sample")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
