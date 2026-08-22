"""OKX public and business WebSocket client for live and demo market data."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from zhisa.live.events import MarketEvent, symbol_to_okx_swap, utc_from_ms


OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"
OKX_BUSINESS_WS = "wss://ws.okx.com:8443/ws/v5/business"
OKX_DEMO_PUBLIC_WS = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
OKX_DEMO_BUSINESS_WS = "wss://wspap.okx.com:8443/ws/v5/business?brokerId=9999"


class OKXPublicWebSocketClient:
    """Consume OKX public and business market streams.

    Demo mode uses OKX paper-trading WebSocket URLs. This client is
    unauthenticated and never places orders.
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        timeframe: str = "5m",
        demo: bool = False,
        url: str | None = None,
        business_url: str | None = None,
    ) -> None:
        self.symbols = [str(s).strip() for s in symbols if str(s).strip()]
        self.timeframe = timeframe
        self.demo = bool(demo)
        self.public_url = url or (OKX_DEMO_PUBLIC_WS if demo else OKX_PUBLIC_WS)
        self.business_url = business_url or (OKX_DEMO_BUSINESS_WS if demo else OKX_BUSINESS_WS)

    def _candle_channel(self) -> str:
        if self.timeframe.endswith("m"):
            return f"candle{self.timeframe}"
        if self.timeframe.endswith("h"):
            return f"candle{self.timeframe.upper()}"
        return f"candle{self.timeframe}"

    def public_subscribe_payload(self) -> dict[str, Any]:
        args: list[dict[str, str]] = []
        for symbol in self.symbols:
            args.extend([
                {"channel": "trades", "instId": symbol},
                {"channel": "tickers", "instId": symbol},
                {"channel": "mark-price", "instId": symbol},
            ])
        return {"op": "subscribe", "args": args}

    def business_subscribe_payload(self) -> dict[str, Any]:
        args: list[dict[str, str]] = []
        candle = self._candle_channel()
        for symbol in self.symbols:
            args.append({"channel": candle, "instId": symbol})
        return {"op": "subscribe", "args": args}

    async def iter_events(self) -> AsyncIterator[MarketEvent]:
        queue: asyncio.Queue[MarketEvent | None] = asyncio.Queue(maxsize=10000)
        stop_event = asyncio.Event()

        async def _stream_ws(endpoint_url: str, payload: dict[str, Any]):
            while not stop_event.is_set():
                try:
                    async with websockets.connect(endpoint_url, ping_interval=20, ping_timeout=20) as ws:
                        await ws.send(json.dumps(payload))
                        async for raw in ws:
                            if stop_event.is_set():
                                break
                            if raw == "pong":
                                continue
                            for event in self.parse(raw):
                                await queue.put(event)
                except asyncio.CancelledError:
                    break
                except Exception:
                    if not stop_event.is_set():
                        await asyncio.sleep(1.0)

        tasks = [
            asyncio.create_task(_stream_ws(self.public_url, self.public_subscribe_payload())),
            asyncio.create_task(_stream_ws(self.business_url, self.business_subscribe_payload())),
        ]

        try:
            while not stop_event.is_set():
                event = await queue.get()
                if event is not None:
                    yield event
        finally:
            stop_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
