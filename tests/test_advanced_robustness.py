"""Advanced robustness, leakage, and validation tests for the ZHISA system."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.actions import DiscreteAction
from zhisa.env.execution import ExecutionConfig
from zhisa.rendering.chart_renderer import render_chart
from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig


def test_anti_look_ahead_bias_truncation(small_market):
    """Verify that calculating features on a truncated time series yields

    identical results at the truncation point as calculating on the full series.
    This guarantees no look-ahead/leakage in indicators and features.
    """
    # 1. Calculate features on the full market dataset
    feats_full = compute_ohlcv_features(small_market)

    # 2. Pick several truncation points and verify
    truncation_points = [100, 250, 500, 1000]
    for t in truncation_points:
        if t >= len(small_market):
            continue
        # Slice the market up to t (inclusive of index t-1)
        market_trunc = small_market.iloc[:t]
        feats_trunc = compute_ohlcv_features(market_trunc)

        # Retrieve the feature vector at the last step
        row_full = feats_full.iloc[t - 1]
        row_trunc = feats_trunc.iloc[-1]

        # Ignore columns where values are NaN in both (e.g. initial rolling windows)
        valid_mask = row_trunc.notna()
        if not valid_mask.any():
            continue

        # Check values
        np.testing.assert_allclose(
            row_full[valid_mask].values.astype(np.float32),
            row_trunc[valid_mask].values.astype(np.float32),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Look-ahead bias detected in features at truncation step t={t}",
        )


def test_conservation_of_money_flat_market():
    """Verify that in a flat market (no price change), trading ONLY loses

    the exact transaction fees. No equity leaks or arbitrary creation of money.
    """
    # Create 100 bars of a completely flat market
    idx = pd.date_range("2026-06-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10.0
    }, index=idx)

    # Fee is 5 bps (0.0005)
    fee_bps = 5.0
    cfg = EnvConfig(
        initial_equity=10.0,
        fee_bps=fee_bps,
        slippage_bps_per_unit=0.0, # no slippage for now
        max_leverage=2.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        kill_on_drawdown=False,
        window=8,
    )

    env = TradingEnv(df, cfg=cfg)
    obs, info = env.reset(seed=0)
    eq_start = env._equity # Read internal equity directly as info on reset is empty

    # 1. Open LONG_100 (100% position, meaning target size = 1.0, notionally multiplied by max_leverage)
    # Size in units of base asset = target_size * leverage / price = 1.0 * 2.0 / 100.0 = 0.02 units
    # Taker fee = 0.02 * 100.0 * (5 / 10000) = 2.0 * 0.0005 = 0.001
    obs, r, term, trunc, info = env.step(int(DiscreteAction.LONG_100))
    eq_after_open = info["equity"]
    
    fee_open = info["fee"]
    assert math.isclose(fee_open, 0.001, rel_tol=1e-5)
    assert math.isclose(eq_after_open, eq_start - fee_open, rel_tol=1e-5)

    # 2. Hold position: repeat LONG_100 (target size = 1.0, same as self._position, so no trade executed)
    obs, r, term, trunc, info = env.step(int(DiscreteAction.LONG_100))
    assert math.isclose(info["equity"], eq_after_open, rel_tol=1e-5)
    assert info["fee"] == 0.0

    # 3. Close position
    # Close should sell the 0.02 units, paying fee on close.
    # Taker fee on close = 0.02 * 100.0 * 0.0005 = 0.001
    obs, r, term, trunc, info = env.step(int(DiscreteAction.CLOSE))
    eq_after_close = info["equity"]
    fee_close = info["fee"]
    
    assert math.isclose(fee_close, 0.001, rel_tol=1e-5)
    assert math.isclose(eq_after_close, eq_start - fee_open - fee_close, rel_tol=1e-5)


def test_pnl_to_price_delta_leverage():
    """Verify that price delta translates exactly to position PnL scaling

    with the leverage factor.
    """
    idx = pd.date_range("2026-06-01", periods=100, freq="5min")
    # Step-by-step price: starts at 100, then jumps to 105 on step 2 (index 18)
    prices = [100.0] * 18 + [105.0] * 82
    df = pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [10.0] * 100
    }, index=idx)

    # Max leverage is 3.0, zero fees, zero slippage, window=8
    cfg = EnvConfig(
        initial_equity=1.0,
        fee_bps=0.0,
        slippage_bps_per_unit=0.0,
        max_leverage=3.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        kill_on_drawdown=False,
        window=8,
    )
    env = TradingEnv(df, cfg=cfg)
    env.reset(seed=0)

    # Bar 1 (t=8 is window start, so we must advance from self._t = window)
    # Open LONG_100 (target size = 1.0)
    # Price close at t=8 is 100.0. Position is opened at 100.0.
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    eq_open = info["equity"]
    assert eq_open == 1.0  # no fees

    # Fast forward self._t to 17 (just before the price jump).
    # Since step() advances time by 1, we execute SKIP until self._t = 17.
    while env._t < 17:
        _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))

    # At self._t = 17, the price close is still 100.0.
    # Now execute step where self._t moves to 18 (close is 105.0).
    # return = (105.0 / 100.0) - 1.0 = +5%.
    # PnL = position * leverage * return = 1.0 * 3.0 * 5% = +15%.
    # New equity = 1.0 + 0.15 = 1.15.
    _, _, _, _, info = env.step(int(DiscreteAction.LONG_100))
    eq_high = info["equity"]
    assert math.isclose(eq_high, 1.15, rel_tol=1e-5)


def test_slippage_and_latency_impact():
    """Verify that transaction parameters (slippage, latency) are correctly

    integrated and scale execution costs.
    """
    idx = pd.date_range("2026-06-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10.0
    }, index=idx)

    # Case A: Low slippage
    cfg_low = EnvConfig(
        initial_equity=1.0,
        fee_bps=0.0,
        slippage_bps_per_unit=1.0,
        market_depth_units=10.0,
        max_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        kill_on_drawdown=False,
        window=8,
    )
    env_low = TradingEnv(df, cfg=cfg_low)
    env_low.reset(seed=42)
    _, _, _, _, info_low = env_low.step(int(DiscreteAction.LONG_100))
    slip_low = info_low["slippage_bps"]

    # Case B: High slippage
    cfg_high = EnvConfig(
        initial_equity=1.0,
        fee_bps=0.0,
        slippage_bps_per_unit=10.0,  # 10x higher coefficient
        market_depth_units=10.0,
        max_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        kill_on_drawdown=False,
        window=8,
    )
    env_high = TradingEnv(df, cfg=cfg_high)
    env_high.reset(seed=42)
    _, _, _, _, info_high = env_high.step(int(DiscreteAction.LONG_100))
    slip_high = info_high["slippage_bps"]

    assert slip_high > slip_low
    assert info_high["fill_price"] > info_low["fill_price"]


def test_visual_chart_style_invariant():
    """Verify that the VisionEncoder produces similar embeddings for the same

    chart even under background color or minor visual perturbations.
    """
    # Create simple dataframe for chart rendering
    idx = pd.date_range("2026-06-01", periods=64, freq="5min")
    prices = np.sin(np.linspace(0, 5, 64)) * 10 + 100
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices + 0.5,
        "volume": np.random.rand(64) * 100
    }, index=idx)

    # 1. Render standard chart (default bg=0.05)
    img_orig = render_chart(df, size=64) # shape (3, 64, 64)

    # 2. Perturb chart (add small Gaussian noise to the image tensor)
    # This simulates minor visual noise, contrast shifts, or color perturbations
    noise = torch.randn_like(img_orig) * 0.02
    img_perturbed = torch.clamp(img_orig + noise, 0.0, 1.0)

    # 3. Pass both through VisionEncoder
    cfg = VisionEncoderConfig(image_size=64, out_dim=64)
    encoder = VisionEncoder(cfg)
    encoder.eval()

    with torch.no_grad():
        emb_orig = encoder(img_orig.unsqueeze(0))      # (1, 64)
        emb_perturbed = encoder(img_perturbed.unsqueeze(0))  # (1, 64)

    # 4. Measure cosine similarity
    cos = torch.nn.functional.cosine_similarity(emb_orig, emb_perturbed, dim=1)
    # Since the encoder is randomly initialized, a small perturbation
    # should still keep the cosine similarity high (e.g. > 0.85) because of the local spatial
    # pooling and Conv filters preserving general shape activations.
    assert cos.item() > 0.85, f"Vision encoder is too sensitive to visual noise: cos={cos.item():.4f}"
