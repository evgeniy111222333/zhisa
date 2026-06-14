"""Binance USD-M public WebSocket client for live shadow mode."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from zhisa.live.events import MarketEvent, symbol_to_binance, utc_from_ms


BINANCE_USDM_MARKET_WS = "wss://fstream.binance.com/market/stream"


class BinanceUSDMWebSocketClient:
    """Consume public Binance USD-M market streams.

    Only public market-data streams are used. No private listen keys and no
    order endpoints are touched here.
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        timeframe: str = "5m",
        url: str = BINANCE_USDM_MARKET_WS,
    ) -> None:
        self.symbols = [symbol_to_binance(s) for s in symbols]
        self.timeframe = timeframe
        self.url = url

    def streams(self) -> list[str]:
        out: list[str] = []
        for symbol in self.symbols:
            s = symbol.lower()
            out.extend([
                f"{s}@aggTrade",
                f"{s}@kline_{self.timeframe}",
                f"{s}@markPrice",
                f"{s}@forceOrder",
            ])
        return out

    def stream_url(self) -> str:
        return f"{self.url}?streams={'/'.join(self.streams())}"

    async def iter_events(self) -> AsyncIterator[MarketEvent]:
        async with websockets.connect(self.stream_url(), ping_interval=150, ping_timeout=540) as ws:
            async for raw in ws:
                for event in self.parse(raw):
                    yield event

    @staticmethod
    def parse(raw: str | bytes | dict[str, Any]) -> list[MarketEvent]:
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(data, dict):
            return []
        event_type = data.get("e")
        if event_type == "aggTrade":
            side = "sell" if bool(data.get("m")) else "buy"
            return [
                MarketEvent(
                    kind="trade",
                    exchange="binance_usdm",
                    symbol=str(data.get("s", "")),
                    ts=utc_from_ms(data.get("E") or data.get("T")),
                    price=float(data["p"]),
                    qty=float(data["q"]),
                    side=side,
                    raw=data,
                )
            ]
        if event_type == "kline":
            k = data.get("k", {})
            if not isinstance(k, dict):
                return []
            return [
                MarketEvent(
                    kind="kline",
                    exchange="binance_usdm",
                    symbol=str(k.get("s") or data.get("s") or ""),
                    ts=utc_from_ms(k.get("t") or data.get("E")),
                    price=float(k["c"]),
                    qty=float(k.get("v", 0.0)),
                    ohlcv={
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k.get("v", 0.0)),
                        "closed": bool(k.get("x", False)),
                    },
                    raw=data,
                )
            ]
        if event_type == "markPriceUpdate":
            return [
                MarketEvent(
                    kind="mark_price",
                    exchange="binance_usdm",
                    symbol=str(data.get("s", "")),
                    ts=utc_from_ms(data.get("E")),
                    price=float(data["p"]),
                    raw=data,
                )
            ]
        if event_type == "forceOrder":
            order = data.get("o", {})
            if not isinstance(order, dict):
                return []
            return [
                MarketEvent(
                    kind="liquidation",
                    exchange="binance_usdm",
                    symbol=str(order.get("s") or data.get("s") or ""),
                    ts=utc_from_ms(data.get("E") or order.get("T")),
                    price=float(order.get("ap") or order.get("p") or 0.0),
                    qty=float(order.get("q") or 0.0),
                    side=str(order.get("S", "")).lower(),
                    raw=data,
                )
            ]
        return []
