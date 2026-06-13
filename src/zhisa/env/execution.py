"""Realistic order execution simulator.

Models slippage (proportional to size relative to top-of-book depth),
latency (delay between signal and fill), and fees (maker / taker).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExecutionConfig:
    maker_fee_bps: float = 2.0     # 0.02%
    taker_fee_bps: float = 4.0     # 0.04%
    base_latency_bars: float = 0.0 # filled in same bar by default
    slippage_bps_per_unit: float = 1.5
    market_depth_units: float = 100.0  # bars; calibrates slippage curve
    partial_fill_prob: float = 0.0


@dataclass
class FillResult:
    requested: float
    filled: float
    price: float
    fee: float
    slippage_bps: float
    liquidity_taken: float


def execute_order(
    *,
    side: int,                  # +1 buy, -1 sell
    requested_size: float,      # in units of base asset
    ref_price: float,
    book_top_size: float,       # size at best level (for slippage model)
    cfg: Optional[ExecutionConfig] = None,
    post_only: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> FillResult:
    """Simulate the fill of a market (or marketable limit) order.

    Slippage is computed as
        ``slippage_bps = slippage_bps_per_unit * (size / book_top_size)``
    with additive Gaussian noise (mild) and clamped to a reasonable range.
    """
    cfg = cfg or ExecutionConfig()
    rng = rng or np.random.default_rng()
    if requested_size <= 0 or book_top_size <= 0 or ref_price <= 0:
        return FillResult(requested_size, 0.0, ref_price, 0.0, 0.0, 0.0)

    depth = max(cfg.market_depth_units, book_top_size)
    raw_slip_bps = cfg.slippage_bps_per_unit * (requested_size / depth)
    jitter = rng.normal(0.0, 0.5 * cfg.slippage_bps_per_unit)
    slip_bps = max(0.0, raw_slip_bps + jitter)
    slip_bps = min(slip_bps, 200.0)  # hard cap: 2% slippage
    slip_factor = slip_bps / 1e4

    if side > 0:
        fill_price = ref_price * (1.0 + slip_factor)
    else:
        fill_price = ref_price * (1.0 - slip_factor)

    # Partial fill
    filled = requested_size
    if cfg.partial_fill_prob > 0 and rng.random() < cfg.partial_fill_prob:
        filled = requested_size * rng.uniform(0.3, 0.9)

    # Fee (taker, market; if post_only and we are not marketable -> zero fill)
    if post_only:
        # If the post-only would cross the book, it gets rejected.
        filled = 0.0
        fee = 0.0
    else:
        notional = filled * fill_price
        fee = notional * (cfg.taker_fee_bps / 1e4)

    return FillResult(
        requested=requested_size,
        filled=filled,
        price=fill_price,
        fee=fee,
        slippage_bps=slip_bps,
        liquidity_taken=filled,
    )
