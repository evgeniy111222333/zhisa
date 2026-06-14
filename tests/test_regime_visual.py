"""Tests for visual regime intelligence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from zhisa.regime import (
    PLAYBOOK_NAMES,
    RegimeIntelligenceConfig,
    RegimeSupervisionConfig,
    VisualRegimeConfig,
    VisualRegimeEncoder,
    VisualRegimeLoss,
    VisualRegimeSupervisionConfig,
    VisualRegimeSupervisionDataset,
    fuse_regime_embeddings,
    visual_regime_collate,
)


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    if np.isscalar(volume):
        vol = np.full(close.size, float(volume))
    else:
        vol = np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def _mixed_df(n: int = 180) -> pd.DataFrame:
    x = np.arange(n)
    close = 100.0 + 0.08 * x + 2.0 * np.sin(x / 8)
    close[n // 2 :] -= np.linspace(0.0, 18.0, n - n // 2)
    volume = 120.0 + 40.0 * (np.sin(x / 5) > 0).astype(float)
    return _ohlcv_from_close(close, volume=volume)


def _base_cfg() -> RegimeSupervisionConfig:
    return RegimeSupervisionConfig(
        horizon=5,
        stride=8,
        min_history=64,
        symbol="BTC/USDT",
        analyzer=RegimeIntelligenceConfig(timeframes=("5m", "15m")),
    )


def test_visual_regime_dataset_aligns_chart_with_regime_supervision() -> None:
    ds = VisualRegimeSupervisionDataset(
        _mixed_df(),
        VisualRegimeSupervisionConfig(chart_window=48, image_size=32, base=_base_cfg()),
    )
    item = ds[0]
    batch = visual_regime_collate([ds[0], ds[1], ds[2]])

    assert len(ds) > 0
    assert item.chart.shape == (3, 32, 32)
    assert torch.isfinite(item.chart).all()
    assert item.supervision.meta["t"] >= _base_cfg().min_history - 1
    assert batch.chart.shape == (3, 3, 32, 32)
    assert batch.supervision.x.shape[0] == 3


def test_visual_regime_encoder_heads_loss_and_gradients() -> None:
    ds = VisualRegimeSupervisionDataset(
        _mixed_df(190),
        VisualRegimeSupervisionConfig(chart_window=48, image_size=32, base=_base_cfg()),
    )
    batch = visual_regime_collate([ds[0], ds[1], ds[2], ds[3]])
    model = VisualRegimeEncoder(
        VisualRegimeConfig(
            image_size=32,
            chart_window=48,
            vision_dim=24,
            embed_dim=12,
            hidden_dim=24,
            vision_channels=(8, 16),
            dropout=0.0,
        )
    )
    loss_fn = VisualRegimeLoss()

    out = model(batch.chart)
    losses = loss_fn(out, batch.supervision)
    losses["total"].backward()

    assert out["embedding"].shape == (4, 12)
    assert out["macro_logits"].shape[0] == 4
    assert out["playbook_logits"].shape == (4, len(PLAYBOOK_NAMES))
    assert out["tradeability"].shape == (4,)
    assert out["transition_risk"].shape == (4,)
    assert out["visual_uncertainty"].shape == (4,)
    assert losses["total"].item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_visual_regime_encoder_fuses_tabular_embedding() -> None:
    model = VisualRegimeEncoder(
        VisualRegimeConfig(
            image_size=16,
            chart_window=32,
            vision_dim=16,
            embed_dim=8,
            hidden_dim=16,
            vision_channels=(8, 16),
            dropout=0.0,
            tabular_embed_dim=6,
        )
    )
    chart = torch.rand(3, 3, 16, 16)
    tab = torch.randn(3, 6)

    out = model(chart, tabular_embedding=tab)

    assert out["embedding"].shape == (3, 8)
    assert out["visual_embedding"].shape == (3, 8)
    assert out["tabular_embedding"].shape == (3, 8)
    assert out["fusion_gate"].shape == (3,)
    assert torch.all((out["fusion_gate"] >= 0.0) & (out["fusion_gate"] <= 1.0))


def test_fuse_regime_embeddings_obeys_gate_extremes() -> None:
    structured = torch.ones(2, 4)
    visual = torch.zeros(2, 4)

    torch.testing.assert_close(fuse_regime_embeddings(structured, visual, 1.0), structured)
    torch.testing.assert_close(fuse_regime_embeddings(structured, visual, 0.0), visual)
    torch.testing.assert_close(fuse_regime_embeddings(structured, visual, 0.25), torch.full((2, 4), 0.25))
