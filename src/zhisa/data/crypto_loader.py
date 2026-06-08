"""Real-data loaders: CCXT-based adapters for crypto exchanges.

The interface is intentionally narrow: ``load_ohlcv(symbol, timeframe, since)``
returns a DataFrame with a normalized schema. Adapters are designed to be
**swap-in**: the same DataFrame is consumed by downstream feature and
labeling code regardless of the data source.

If the optional ``ccxt`` package is not installed, importing this module
still succeeds but the loader raises a clear ``RuntimeError`` when used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

# Schema enforced across all loaders
STANDARD_COLUMNS = ("open", "high", "low", "close", "volume")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any well-formed OHLCV DF to the project standard schema."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[list(STANDARD_COLUMNS)].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="any")
    return df


@dataclass
class CCXTCryptoLoader:
    """Load OHLCV data from any CCXT-supported exchange.

    Requires the optional ``ccxt`` package. If unavailable, attempting to
    fetch raises a clear error message; the rest of the project does not
    depend on ccxt.
    """

    exchange_id: str = "binance"
    api_key: Optional[str] = None
    secret: Optional[str] = None
    timeout_ms: int = 20_000
    enable_rate_limit: bool = True

    def _make_exchange(self):
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ccxt is not installed. Install with `pip install ccxt` to use real data."
            ) from e
        cls = getattr(ccxt, self.exchange_id, None)
        if cls is None:
            raise ValueError(f"Unknown exchange: {self.exchange_id}")
        return cls({
            "apiKey": self.api_key,
            "secret": self.secret,
            "timeout": self.timeout_ms,
            "enableRateLimit": self.enable_rate_limit,
        })

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since_ms: Optional[int] = None,
        limit: int = 1000,
        max_bars: Optional[int] = None,
    ) -> pd.DataFrame:
        ex = self._make_exchange()
        rows: list[list] = []
        cursor = since_ms
        guard = 0
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            if cursor is None or last_ts == cursor:
                # Avoid infinite loop on stagnant endpoint
                break
            cursor = last_ts + 1
            if max_bars is not None and len(rows) >= max_bars:
                rows = rows[:max_bars]
                break
            guard += 1
            if guard > 50_000:
                logger.warning("CCXT fetch aborted after %d iterations", guard)
                break
        if not rows:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        arr = np.array(rows, dtype=np.float64)
        ts = pd.to_datetime(arr[:, 0], unit="ms", utc=True)
        df = pd.DataFrame(arr[:, 1:6], index=ts, columns=list(STANDARD_COLUMNS))
        return _normalize_ohlcv(df)

    @staticmethod
    def list_symbols() -> Iterable[str]:
        return ()
