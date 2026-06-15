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
from zhisa.features.ohlcv import compute_ohlcv_features, normalize_feature_window
from zhisa.features.time import compute_time_features
from zhisa.rendering.chart_renderer import render_chart
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


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
    label_risk: torch.Tensor    # (B,) float -- downside return + realised vol
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

        logger.info(f"MarketDataset Init: Processing {len(df)} bars. Step 1/5: Computing OHLCV features...")
        self._features = compute_ohlcv_features(
            df, include_volume=spec.include_volume, include_indicators=spec.include_indicators
        )
        
        logger.info("MarketDataset Init: Step 2/5: Computing Time embeddings...")
        self._time_features = compute_time_features(df)
        
        # Primary triple-barrier at smallest horizon; use horizon 16 by default
        primary = spec.horizons[len(spec.horizons) // 2] if spec.horizons else 16
        self._tb_cfg_primary = TripleBarrierConfig(
            tp_atr_mult=self.tb_cfg.tp_atr_mult,
            sl_atr_mult=self.tb_cfg.sl_atr_mult,
            max_holding=primary,
            atr_window=self.tb_cfg.atr_window,
        )
        
        logger.info("MarketDataset Init: Step 3/5: Computing Triple Barrier Labels (Returns/Directions)...")
        self._tb = triple_barrier(df, self._tb_cfg_primary)
        
        logger.info("MarketDataset Init: Step 4/5: Computing Realized Volatility...")
        self._vol = realized_volatility(df, horizon=primary)
        
        logger.info("MarketDataset Init: Step 5/5: Computing HMM Regime Labels (Macro States)...")
        self._regime = hmm_regime_labels(
            df, n_states=spec.n_regime_states, lookback=256, prefer_sklearn=False
        )

        logger.info("MarketDataset Init: All tables processed and ready for DataLoader!")
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
        feature_window = self._features.iloc[start:end].to_numpy(dtype=np.float32)
        hist_start = max(0, t - 256)
        history_window = self._features.iloc[hist_start:end].to_numpy(dtype=np.float32)

        # Numeric features only (NaN -> 0, robust fill). The cyclic time
        # embeddings are placed in ``context`` (last bar) so the dataset's
        # ``numeric`` shape matches the env's contract — and the policy's
        # ``in_numeric_features`` default (32) is wire-compatible.
        num = normalize_feature_window(feature_window, history_window)

        chart = self._chart(t)
        ctx = self._time_features.iloc[end - 1].to_numpy(dtype=np.float32)
        primary_idx = end - 1
        lbl_dir = int(self._tb["label"].iloc[primary_idx])
        lbl_ret = float(self._tb["ret"].iloc[primary_idx])
        lbl_vol = float(self._vol.iloc[primary_idx]) if not np.isnan(self._vol.iloc[primary_idx]) else 0.0
        lbl_risk = max(-lbl_ret, 0.0) + max(lbl_vol, 0.0)
        lbl_regime = int(self._regime.iloc[primary_idx])

        mask = np.ones(spec.chart_window, dtype=bool)
        return {
            "chart": chart,
            "numeric": torch.from_numpy(num),
            "context": torch.from_numpy(ctx),
            "label_dir": torch.tensor(lbl_dir, dtype=torch.long),
            "label_ret": torch.tensor(lbl_ret, dtype=torch.float32),
            "label_vol": torch.tensor(lbl_vol, dtype=torch.float32),
            "label_risk": torch.tensor(lbl_risk, dtype=torch.float32),
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
                   "label_vol", "label_risk", "label_regime", "mask")
    out: dict = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return MultimodalBatch(**out)
