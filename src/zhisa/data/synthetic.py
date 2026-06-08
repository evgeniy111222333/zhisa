"""Synthetic OHLCV market data generator.

Generates realistic-looking candles with regime switches (trend / range /
volatility expansion), fat-tailed returns, microstructure noise, and
optional shock events. Used both as a training bootstrap and as a
benchmark / smoke-test data source.

The generator is **vectorised** for speed and supports deterministic
seeding. It produces a `pandas.DataFrame` with columns
``[open, high, low, close, volume]`` indexed by a UTC datetime range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from zhisa.utils.seeding import set_seed


_REGIME_NAMES = ("trend_up", "trend_down", "range", "vol_expansion", "crash")


@dataclass
class MarketConfig:
    """Knobs controlling the synthetic market."""

    n_bars: int = 20_000
    start: str = "2022-01-01"
    freq: str = "5min"
    initial_price: float = 30_000.0

    # Per-bar base volatility (annualised; scaled by sqrt(periods_per_year))
    base_vol: float = 0.6
    # Per-regime vol multipliers
    vol_mult_trend: float = 0.8
    vol_mult_range: float = 0.6
    vol_mult_vol_expansion: float = 2.0
    vol_mult_crash: float = 3.5

    # Mean drift (annualised) per regime
    drift_trend_up: float = 0.6
    drift_trend_down: float = -0.6
    drift_range: float = 0.0
    drift_vol_expansion: float = 0.0
    drift_crash: float = -2.0

    # Tail heaviness (df of Student-t innovations; <inf enables fat tails)
    student_t_df: float = 6.0

    # Markov transition matrix between regimes (row = from, col = to)
    transition: Optional[np.ndarray] = None

    # Volume model
    base_volume: float = 100.0
    volume_vol_sensitivity: float = 0.7
    volume_trend_bonus: float = 0.4

    # Shock events
    shock_prob: float = 0.0005
    shock_size: float = -0.05
    shock_size_std: float = 0.02

    seed: Optional[int] = 42

    periods_per_year: int = 365 * 24 * 12  # 5-min bars per year
    field_names: Tuple[str, ...] = field(
        default_factory=lambda: ("open", "high", "low", "close", "volume")
    )


def _default_transition() -> np.ndarray:
    """Reasonable default regime transition matrix.

    Rows/cols: trend_up, trend_down, range, vol_expansion, crash.
    """
    n = len(_REGIME_NAMES)
    M = np.full((n, n), 0.02, dtype=np.float64)
    np.fill_diagonal(M, 0.0)
    # Trend regimes are sticky; range is stickier; crash decays fast.
    diag = np.array([0.96, 0.96, 0.985, 0.92, 0.5])
    for i, d in enumerate(diag):
        others = (1.0 - d) / (n - 1)
        M[i, :] = others
        M[i, i] = d
    # Direct transitions into crash are slightly elevated from vol_expansion
    M[3, 4] = 0.03
    M[3, 3] -= 0.03
    # Renormalise rows
    M = M / M.sum(axis=1, keepdims=True)
    return M


def _student_t(df: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if df >= 1e6:
        return rng.standard_normal(size)
    # scipy-free Student-t via ratio of Normal / sqrt(chi2 / df)
    z = rng.standard_normal(size)
    g = rng.gamma(shape=df / 2.0, scale=2.0 / df, size=size)
    return z / np.sqrt(g)


def generate_market(cfg: Optional[MarketConfig] = None) -> pd.DataFrame:
    """Generate a synthetic OHLCV+regime DataFrame."""
    cfg = cfg or MarketConfig()
    if cfg.seed is not None:
        set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    transition = cfg.transition if cfg.transition is not None else _default_transition()
    n_regimes = transition.shape[0]

    # --- Regime path ---
    regimes = np.zeros(cfg.n_bars, dtype=np.int64)
    regimes[0] = rng.integers(0, n_regimes)
    for t in range(1, cfg.n_bars):
        regimes[t] = rng.choice(n_regimes, p=transition[regimes[t - 1]])

    drift_per_regime = np.array(
        [cfg.drift_trend_up, cfg.drift_trend_down, cfg.drift_range,
         cfg.drift_vol_expansion, cfg.drift_crash][:n_regimes]
    )
    vol_mult_per_regime = np.array(
        [cfg.vol_mult_trend, cfg.vol_mult_trend, cfg.vol_mult_range,
         cfg.vol_mult_vol_expansion, cfg.vol_mult_crash][:n_regimes]
    )
    bar_vol = cfg.base_vol / np.sqrt(cfg.periods_per_year)
    drift_bar = drift_per_regime / cfg.periods_per_year

    # --- Returns ---
    eps = _student_t(cfg.student_t_df, cfg.n_bars, rng)
    sigma = bar_vol * vol_mult_per_regime[regimes]
    mu = drift_bar[regimes]
    rets = mu + sigma * eps

    # --- Shock events ---
    if cfg.shock_prob > 0:
        shock_mask = rng.random(cfg.n_bars) < cfg.shock_prob
        shock_size = rng.normal(loc=cfg.shock_size, scale=cfg.shock_size_std,
                                size=shock_mask.sum()) if shock_mask.any() else np.array([])
        rets[shock_mask] += shock_size

    # --- Price path ---
    log_price = np.log(cfg.initial_price) + np.cumsum(rets)
    close = np.exp(log_price)
    # Open = previous close (gap = 0 in this generator; add a small random gap below)
    open_ = np.empty_like(close)
    open_[0] = cfg.initial_price
    open_[1:] = close[:-1] * (1.0 + 0.0005 * rng.standard_normal(cfg.n_bars - 1))

    # High / Low based on intra-bar volatility
    bar_range = np.abs(rng.standard_normal(cfg.n_bars)) * sigma * close
    high = np.maximum(open_, close) + 0.5 * bar_range
    low = np.minimum(open_, close) - 0.5 * bar_range
    # Ensure high >= max(o,c), low <= min(o,c)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # --- Volume ---
    base_v = cfg.base_volume * (1.0 + cfg.volume_vol_sensitivity * (np.abs(rets) / (bar_vol + 1e-12)))
    trend_mask = (regimes == 0) | (regimes == 1)
    base_v[trend_mask] *= 1.0 + cfg.volume_trend_bonus
    crash_mask = regimes == 4
    base_v[crash_mask] *= 1.5
    volume = np.maximum(1.0, base_v * (1.0 + 0.3 * rng.standard_normal(cfg.n_bars)))

    # --- Build DataFrame ---
    idx = pd.date_range(start=cfg.start, periods=cfg.n_bars, freq=cfg.freq, tz="UTC")
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "regime": regimes,
    }, index=idx)
    df.index.name = "timestamp"
    return df


def regime_name(idx: int) -> str:
    return _REGIME_NAMES[idx] if 0 <= idx < len(_REGIME_NAMES) else f"regime_{idx}"
