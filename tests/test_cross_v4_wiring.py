"""Tests: v4 wiring into preparation/CLI + legacy portfolio bias + PortfolioEnv.corr_matrix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, r"D:\zhisa\src")
sys.path.insert(0, r"D:\zhisa")

from zhisa.data.preparation import PrepareConfig, enrich_prepared_root
from zhisa.env.portfolio_env import PortfolioConfig, PortfolioEnv
from zhisa.models.portfolio_policy import PortfolioPolicyConfig, PortfolioPolicyNetwork


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": rng.uniform(1, 2, n)}, index=idx)


def _base_root(path: Path) -> Path:
    root = path / "base"
    (root / "symbols").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    frames = {}
    for sym in ("A", "B"):
        df = _frame(seed=1 if sym == "A" else 2)
        df.to_parquet(root / "symbols" / f"{sym}.parquet")
        frames[sym] = df
    pd.concat([frames["A"].iloc[:200].assign(symbol="A"),
               frames["B"].iloc[:200].assign(symbol="B")]).sort_index().to_parquet(
        root / "splits" / "train.parquet")
    (root / "manifest.json").write_text(json.dumps({
        "version": "v1", "timeframe": "1h", "symbols": ["A", "B"],
        "rows_total": 800, "rows_per_symbol": {"A": 400, "B": 400},
        "feature_columns": ["open", "high", "low", "close", "volume"],
        "output_checksum": "base123",
    }))
    return root


def test_enrich_from_v4_flags_wire_columns(tmp_path):
    base = _base_root(tmp_path)
    out = tmp_path / "out_v4"
    summary = enrich_prepared_root(
        base, out, with_regime_betas=True, with_volume_ratios=True,
        with_resid_alpha=True, with_vol_index=True,
        lead_lag_lags=(0, 1),
    )
    assert summary["output_checksum"]
    cols = pd.read_parquet(out / "symbols" / "A.parquet").columns
    for c in ("resid_alpha_64", "resid_alpha_256", "resid_alpha_64",
              "market_index_vw_logret", "rel_logret_vw_1", "beta_vw_64",
              "leadlag_avg_0", "leadlag_avg_1", "leadlag_B_1"):
        assert c in cols, c
    # OHLCV byte-identity preserved
    ohlcv = ["open", "high", "low", "close", "volume"]
    a = pd.read_parquet(base / "symbols" / "A.parquet")[ohlcv].to_numpy(np.float64)
    b = pd.read_parquet(out / "symbols" / "A.parquet")[ohlcv].to_numpy(np.float64)
    assert np.array_equal(a, b)


def test_prepare_config_fields_exist_and_defaults():
    cfg = PrepareConfig(
        tsdb_root=Path("x"), out_root=Path("y"), symbols=["A"], timeframe="1h")
    assert cfg.cross_asset_resid_alpha is False
    assert cfg.cross_asset_vol_index is False
    assert cfg.cross_asset_lead_lag_lags == ()


def test_prepared_defaults_stay_v31_compatible(tmp_path):
    base = _base_root(tmp_path)
    out = tmp_path / "out_default"
    enrich_prepared_root(base, out, with_regime_betas=True)
    cols = pd.read_parquet(out / "symbols" / "A.parquet").columns
    for c in ("resid_alpha_64", "market_index_vw_logret", "leadlag_avg_1"):
        assert c not in cols, c


# ---------------------------------------------------------------------------
# legacy PortfolioPolicyNetwork bias support
# ---------------------------------------------------------------------------


def _legacy(B=2, N=3, **kw) -> PortfolioPolicyNetwork:
    cfg = PortfolioPolicyConfig(n_instruments=N, in_numeric_features=32,
                                window=16, image_size=32, embed_dim=32,
                                portfolio_dim=8, cross_attn_depth=2,
                                cross_attn_heads=4, **kw)
    return PortfolioPolicyNetwork(cfg)


def _obs_bundle(B=2, N=3):
    rng = np.random.default_rng(0)
    return {
        "chart": torch.as_tensor(rng.random((B, N, 3, 32, 32)), dtype=torch.float32),
        "numeric": torch.as_tensor(rng.normal(0, 1, (B, N, 16, 32)), dtype=torch.float32),
        "context": torch.as_tensor(rng.normal(0, 1, (B, N, 10)), dtype=torch.float32),
    }


def test_legacy_bias_guards_and_works():
    p_off = _legacy(cross_attn_bias=False)  # default behaviour preserved
    p_off.eval()
    b = _obs_bundle()
    with torch.no_grad():
        o = p_off(b, torch.zeros(2, 8))
    assert o["action_logits"].shape == (2, 3, 9)
    with pytest.raises(ValueError, match="cross_attn_bias=False"):
        p_off(b, torch.zeros(2, 8), corr_bias=torch.zeros(2, 3, 3))
    # enabled: requires the bias and applies it
    p_on = _legacy(cross_attn_bias=True, cross_attn_bias_gate=5.0)
    p_on.eval()
    with pytest.raises(ValueError, match="requires corr_bias"):
        p_on(b, torch.zeros(2, 8))
    bias = torch.zeros(2, 3, 3)
    bias[:, 0, 1] = 5.0
    with torch.no_grad():
        o_bias = p_on(b, torch.zeros(2, 8), corr_bias=bias)
        o_zero = p_on(b, torch.zeros(2, 8), corr_bias=torch.zeros(2, 3, 3))
    assert o_bias["action_logits"].shape == (2, 3, 9)
    assert float((o_bias["per_instrument_embedding"] -
                  o_zero["per_instrument_embedding"]).abs().max().item()) > 1e-4


# ---------------------------------------------------------------------------
# PortfolioEnv.corr_matrix
# ---------------------------------------------------------------------------


def test_portfolio_env_corr_matrix():
    rng = np.random.default_rng(1)
    frames = {}
    for s in ("A", "B", "C"):
        idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
        frames[s] = pd.DataFrame({"open": close, "high": close * 1.001,
                                  "low": close * 0.999, "close": close,
                                  "volume": rng.uniform(1, 2, 300)}, index=idx)
    env = PortfolioEnv(frames, cfg=PortfolioConfig(
        n_instruments=3,
        env_cfg=EnvelopeConfig(window=16, image_size=32, initial_equity=1.0),
        seed=3,
    ))
    env.reset()
    # not enough history yet -> identity (causal)
    m0 = env.corr_matrix()
    assert m0.shape == (3, 3)
    assert np.allclose(np.diag(m0), 1.0)
    for _ in range(40):
        env.step(0)
    m = env.corr_matrix()
    assert m.shape == (3, 3)
    assert np.allclose(m, m.T, atol=1e-5)          # symmetric
    assert np.allclose(np.diag(m), 1.0)            # unit diagonal
    assert np.isfinite(m).all()
    # deterministic under the same seed
    env2 = PortfolioEnv(frames, cfg=PortfolioConfig(
        n_instruments=3, env_cfg=EnvelopeConfig(window=16, image_size=32),
        seed=3,
    ))
    env2.reset()
    for _ in range(40):
        env2.step(0)
    assert np.allclose(env.corr_matrix(), env2.corr_matrix(), atol=1e-6)


from zhisa.env.trading_env import EnvConfig as EnvelopeConfig