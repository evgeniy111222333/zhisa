"""Orderbook-derived features (depth imbalance, microprice, VWAP ratios)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def orderbook_features(
    bids: np.ndarray,
    asks: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    depth: Optional[int] = None,
) -> dict:
    """Compute a small set of orderbook microstructure features.

    Args:
        bids, asks: 1-D arrays of price levels (sorted).
        bid_sizes, ask_sizes: sizes at each level.
        depth: how many top levels to consider (None = all).

    Returns a dict with: midprice, microprice, spread (abs, bps),
    bid_size_total, ask_size_total, imbalance, weighted_mid.
    """
    if depth is not None:
        bids = bids[:depth]
        asks = asks[:depth]
        bid_sizes = bid_sizes[:depth]
        ask_sizes = ask_sizes[:depth]
    best_bid = float(bids[0])
    best_ask = float(asks[0])
    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid
    spread_bps = spread / mid * 1e4 if mid > 0 else 0.0
    bid_v = float(bid_sizes.sum())
    ask_v = float(ask_sizes.sum())
    imbalance = (bid_v - ask_v) / (bid_v + ask_v + 1e-12)
    # Microprice = (ask * bid_size + bid * ask_size) / (bid + ask)
    micro = (best_ask * bid_sizes[0] + best_bid * ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0] + 1e-12)
    # Volume-weighted midprice (size-weighted)
    bp = np.sum(bids * bid_sizes) / (bid_v + 1e-12)
    ap = np.sum(asks * ask_sizes) / (ask_v + 1e-12)
    weighted_mid = 0.5 * (bp + ap)
    return {
        "midprice": mid,
        "microprice": micro,
        "spread": float(spread),
        "spread_bps": float(spread_bps),
        "bid_size_total": bid_v,
        "ask_size_total": ask_v,
        "imbalance": float(imbalance),
        "weighted_mid": float(weighted_mid),
    }


def aggregate_orderbook_history(
    snapshots: list[dict],
) -> pd.DataFrame:
    """Stack per-snapshot orderbook feature dicts into a DataFrame."""
    return pd.DataFrame(snapshots)
