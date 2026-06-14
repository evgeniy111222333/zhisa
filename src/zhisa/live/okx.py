"""OKX public WebSocket client for live and demo market data."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from zhisa.live.events import MarketEvent, symbol_to_okx_swap, utc_from_ms


OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"
OKX_DEMO_PUBLIC_WS = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"


class OKXPublicWebSocketClient:
    """Consume OKX public market streams.

    Demo mode uses OKX paper-trading public WebSocket URL. This client is
    unauthenticated and never places orders.
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        timeframe: str = "5m",
        demo: bool = False,
        url: str | None = None,
    ) -> None:
        self.symbols = [symbol_to_okx_swap(s) for s in symbols]
        self.timeframe = timeframe
        self.demo = bool(demo)
        self.url = url or (OKX_DEMO_PUBLIC_WS if demo else OKX_PUBLIC_WS)

    def _candle_channel(self) -> str:
        if self.timeframe.endswith("m"):
            return f"candle{self.timeframe}"
        if self.timeframe.endswith("h"):
            return f"candle{self.timeframe.upper()}"
        return f"candle{self.timeframe}"

    def subscribe_payload(self) -> dict[str, Any]:
        args: list[dict[str, str]] = []
        candle = self._candle_channel()
        for symbol in self.symbols:
            args.extend([
                {"channel": "trades", "instId": symbol},
                {"channel": "tickers", "instId": symbol},
                {"channel": "mark-price", "instId": symbol},
                {"channel": candle, "instId": symbol},
            ])
        return {"op": "subscribe", "args": args}

    async def iter_events(self) -> AsyncIterator[MarketEvent]:
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps(self.subscribe_payload()))
            async for raw in ws:
                if raw == "pong":
                    continue
                for event in self.parse(raw):
                    yield event

    @staticmethod
    def parse(raw: str | bytes | dict[str, Any]) -> list[MarketEvent]:
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(payload, dict) or "data" not in payload:
            return []
        arg = payload.get("arg", {})
        if not isinstance(arg, dict):
            arg = {}
        channel = str(arg.get("channel", ""))
        symbol = str(arg.get("instId", ""))
        events: list[MarketEvent] = []
        for row in payload.get("data", []):
            if not isinstance(row, (dict, list)):
                continue
            if channel == "trades" and isinstance(row, dict):
                events.append(
                    MarketEvent(
                        kind="trade",
                        exchange="okx_demo" if "wspap" in str(payload.get("connId", "")) else "okx",
                        symbol=str(row.get("instId", symbol)),
                        ts=utc_from_ms(row.get("ts")),
                        price=float(row["px"]),
                        qty=float(row.get("sz", 0.0)),
                        side=str(row.get("side", "")),
                        raw=row,
                    )
                )
            elif channel == "tickers" and isinstance(row, dict):
                events.append(
                    MarketEvent(
                        kind="ticker",
                        exchange="okx",
                        symbol=str(row.get("instId", symbol)),
                        ts=utc_from_ms(row.get("ts")),
                        price=float(row["last"]),
                        raw=row,
                    )
                )
            elif channel == "mark-price" and isinstance(row, dict):
                events.append(
                    MarketEvent(
                        kind="mark_price",
                        exchange="okx",
                        symbol=str(row.get("instId", symbol)),
                        ts=utc_from_ms(row.get("ts")),
                        price=float(row["markPx"]),
                        raw=row,
                    )
                )
            elif channel.startswith("candle") and isinstance(row, list) and len(row) >= 6:
                events.append(
                    MarketEvent(
                        kind="kline",
                        exchange="okx",
                        symbol=symbol,
                        ts=utc_from_ms(row[0]),
                        price=float(row[4]),
                        qty=float(row[5]),
                        ohlcv={
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                            "closed": bool(str(row[-1]) == "1") if len(row) >= 9 else True,
                        },
                        raw=row,
                    )
                )
        return events
