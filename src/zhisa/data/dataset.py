"""Multimodal PyTorch Dataset for trading.

Each sample at index ``t`` contains:
    - chart image of the last ``chart_window`` bars
    - numeric feature tensor over the same window
    - context vector (time, instrument, position)
    - labels (triple-barrier, realized vol, regime) for supervision

The dataset supports the typical ``torch.utils.data.Dataset`` protocol
and exposes a ``collate`` function that pads/batches multimodal samples
into ``MultimodalBatch`` namedtuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zhisa.data.labeling import (
    TripleBarrierConfig,
    hmm_regime_labels,
    realized_volatility,
    triple_barrier,
)
from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.features.time import compute_time_features
from zhisa.rendering.chart_renderer import render_chart


@dataclass
class SampleSpec:
    """What every sample must contain."""

    chart_window: int = 64
    feature_window: int = 64
    horizons: Sequence[int] = (4, 16, 64)  # for triple-barrier labels
    image_size: int = 64
    include_volume: bool = True
    include_indicators: bool = True
    n_regime_states: int = 4


@dataclass
class MultimodalBatch:
    """A batched collection of multimodal samples."""

    chart: torch.Tensor         # (B, 3, H, W)
    numeric: torch.Tensor       # (B, T, F)
    context: torch.Tensor       # (B, C)
    label_dir: torch.Tensor     # (B,) long  -- direction label in {-1, 0, +1}
    label_vol: torch.Tensor     # (B,) float -- forward realised vol
    label_regime: torch.Tensor  # (B,) long
    label_ret: torch.Tensor     # (B,) float -- forward return
    mask: torch.Tensor          # (B, T) bool -- which time steps are valid
    meta: list                  # per-sample metadata (symbol, ts, ...)


class MarketDataset(Dataset):
    """Multimodal trading dataset over a single OHLCV DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        spec: Optional[SampleSpec] = None,
        triple_barrier_cfg: Optional[TripleBarrierConfig] = None,
        cache_charts: bool = False,
    ) -> None:
        spec = spec or SampleSpec()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if len(df) < spec.chart_window + 2:
            raise ValueError("df is too short for the requested window")
        self.df = df
        self.spec = spec
        self.tb_cfg = triple_barrier_cfg or TripleBarrierConfig()

        self._features = compute_ohlcv_features(
            df, include_volume=spec.include_volume, include_indicators=spec.include_indicators
        )
        self._time_features = compute_time_features(df)
        # Primary triple-barrier at smallest horizon; use horizon 16 by default
        primary = spec.horizons[len(spec.horizons) // 2] if spec.horizons else 16
        self._tb_cfg_primary = TripleBarrierConfig(
            tp_atr_mult=self.tb_cfg.tp_atr_mult,
            sl_atr_mult=self.tb_cfg.sl_atr_mult,
            max_holding=primary,
            atr_window=self.tb_cfg.atr_window,
        )
        self._tb = triple_barrier(df, self._tb_cfg_primary)
        self._vol = realized_volatility(df, horizon=primary)
        self._regime = hmm_regime_labels(
            df, n_states=spec.n_regime_states, lookback=256, prefer_sklearn=False
        )

        self._cache_charts = cache_charts
        self._chart_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        # Leave room for the largest horizon
        horizon_max = max(self.spec.horizons) if self.spec.horizons else 0
        return max(0, len(self.df) - self.spec.chart_window - horizon_max - 1)

    def _chart(self, t: int) -> torch.Tensor:
        if self._cache_charts and t in self._chart_cache:
            return self._chart_cache[t]
        start = t
        end = t + self.spec.chart_window
        window_df = self.df.iloc[start:end]
        img = render_chart(window_df, size=self.spec.image_size)
        if self._cache_charts:
            self._chart_cache[t] = img
        return img

    def __getitem__(self, t: int) -> dict:
        spec = self.spec
        start = t
        end = t + spec.chart_window
        feature_window = self._features.iloc[start:end]
        time_window = self._time_features.iloc[start:end]

        # Combine numeric features (NaN -> 0, robust fill)
        num = np.concatenate([feature_window.values, time_window.values], axis=1)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        num = num.astype(np.float32)
        # z-score per feature using trailing window stats (avoid look-ahead).
        # Use a wider trailing window (up to t) for the mean/std.
        hist_start = max(0, t - 256)
        hist = self._features.iloc[hist_start:end]
        # ``nanmean``/``nanstd`` ignore NaN values; if a column is all-NaN
        # (early samples where long-lookback features like ``logret_16``
        # or ``sma_slope_50`` are not yet valid) we still want finite
        # statistics. Use ``np.nanmean`` with a NaN-to-0 fallback.
        hist_arr = np.nan_to_num(hist.to_numpy(dtype=np.float32), nan=0.0)
        mu = hist_arr.mean(axis=0)
        sd = hist_arr.std(axis=0) + 1e-6
        feat_dim = self._features.shape[1]
        num[:, :feat_dim] = (num[:, :feat_dim] - mu) / sd
        # Final safety net: any residual NaN/Inf becomes 0.
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)

        chart = self._chart(t)
        ctx = self._time_features.iloc[end - 1].to_numpy(dtype=np.float32)
        primary_idx = end - 1
        lbl_dir = int(self._tb["label"].iloc[primary_idx])
        lbl_ret = float(self._tb["ret"].iloc[primary_idx])
        lbl_vol = float(self._vol.iloc[primary_idx]) if not np.isnan(self._vol.iloc[primary_idx]) else 0.0
        lbl_regime = int(self._regime.iloc[primary_idx])

        mask = np.ones(spec.chart_window, dtype=bool)
        return {
            "chart": chart,
            "numeric": torch.from_numpy(num),
            "context": torch.from_numpy(ctx),
            "label_dir": torch.tensor(lbl_dir, dtype=torch.long),
            "label_ret": torch.tensor(lbl_ret, dtype=torch.float32),
            "label_vol": torch.tensor(lbl_vol, dtype=torch.float32),
            "label_regime": torch.tensor(lbl_regime, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "meta": {
                "ts": str(self.df.index[primary_idx]),
                "t": int(primary_idx),
                "instrument": getattr(self.df, "name", "unknown"),
            },
        }


def multimodal_collate(batch: Sequence[dict]) -> MultimodalBatch:
    """Default collate: stack all tensors, list metas."""
    keys_tensor = ("chart", "numeric", "context", "label_dir", "label_ret",
                   "label_vol", "label_regime", "mask")
    out: dict = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return MultimodalBatch(**out)
