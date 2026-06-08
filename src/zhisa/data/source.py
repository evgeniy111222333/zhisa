"""Unified market data source: synthetic by default, real via adapter.

The ``MarketDataSource`` is the single point through which the rest of
the project accesses OHLCV data. It can be configured for synthetic
generation (useful for tests, smoke runs, and as a bootstrap when no
exchange connectivity is available) or for real-data fetching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from zhisa.data.crypto_loader import CCXTCryptoLoader
from zhisa.data.synthetic import MarketConfig, generate_market


@dataclass
class MarketDataSource:
    """A unified market data source.

    Args:
        kind: "synthetic" or "ccxt".
        synthetic_cfg: Config for the synthetic generator (used when kind=synthetic).
        loader: A pre-configured loader instance (used when kind=ccxt).
    """

    kind: str = "synthetic"
    synthetic_cfg: Optional[MarketConfig] = None
    loader: Optional[CCXTCryptoLoader] = None
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    max_bars: Optional[int] = None
    cache: dict = field(default_factory=dict)

    def load(self, *, force: bool = False) -> pd.DataFrame:
        key = (self.kind, self.symbol, self.timeframe, self.max_bars)
        if not force and key in self.cache:
            return self.cache[key]
        if self.kind == "synthetic":
            df = generate_market(self.synthetic_cfg or MarketConfig())
        elif self.kind == "ccxt":
            if self.loader is None:
                self.loader = CCXTCryptoLoader()
            df = self.loader.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, max_bars=self.max_bars
            )
        else:
            raise ValueError(f"Unknown data source kind: {self.kind!r}")
        self.cache[key] = df
        return df
