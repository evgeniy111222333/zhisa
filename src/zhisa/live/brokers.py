"""Broker adapters for live shadow mode.

The default broker is local-only paper trading. The OKX adapter is an
explicit demo-trading mirror and requires demo API keys; it is never used
unless selected by the caller.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from zhisa.live.events import OrderIntent


HttpPost = Callable[[str, bytes, dict[str, str], float], dict[str, Any]]


class Broker(Protocol):
    def on_price(self, symbol: str, price: float, ts: datetime) -> None: ...
    def position(self, symbol: str) -> float: ...
    def place_order(self, intent: OrderIntent) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PaperBrokerConfig:
    initial_equity: float = 1.0
    max_leverage: float = 3.0
    fee_bps: float = 4.0
    slippage_bps: float = 1.5


@dataclass
class PaperPosition:
    position: float = 0.0
    avg_entry: float = 0.0
    last_price: float = 0.0


class LocalPaperBroker:
    """A local no-money execution simulator for live market data."""

    def __init__(self, cfg: PaperBrokerConfig | None = None) -> None:
        self.cfg = cfg or PaperBrokerConfig()
        self.equity = float(self.cfg.initial_equity)
        self.positions: dict[str, PaperPosition] = {}
        self.orders: list[dict[str, Any]] = []

    def _pos(self, symbol: str) -> PaperPosition:
        return self.positions.setdefault(str(symbol), PaperPosition())

    def on_price(self, symbol: str, price: float, ts: datetime) -> None:
        if price and price > 0:
            self._pos(symbol).last_price = float(price)

    def position(self, symbol: str) -> float:
        return float(self._pos(symbol).position)

    def _unrealized(self, pos: PaperPosition) -> float:
        if pos.position == 0.0 or pos.avg_entry <= 0.0 or pos.last_price <= 0.0:
            return 0.0
        ret = pos.last_price / pos.avg_entry - 1.0
        return float(pos.position * self.cfg.max_leverage * ret)

    def snapshot(self) -> dict[str, Any]:
        unrealized = sum(self._unrealized(pos) for pos in self.positions.values())
        return {
            "equity": float(self.equity + unrealized),
            "cash_equity": float(self.equity),
            "unrealized": float(unrealized),
            "positions": {
                symbol: {
                    "position": float(pos.position),
                    "avg_entry": float(pos.avg_entry),
                    "last_price": float(pos.last_price),
                    "unrealized": self._unrealized(pos),
                }
                for symbol, pos in sorted(self.positions.items())
            },
        }

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        pos = self._pos(intent.symbol)
        mark = float(intent.price or pos.last_price)
        if mark <= 0.0:
            raise ValueError(f"No valid mark price for {intent.symbol}")

        requested_target = max(-1.0, min(1.0, float(intent.target_position)))
        current = float(pos.position)
        delta = requested_target - current
        side = 1.0 if delta > 0 else -1.0 if delta < 0 else 0.0
        if abs(delta) <= 1e-12:
            row = {
                **intent.to_dict(),
                "status": "no_order",
                "fill_price": mark,
                "filled_delta": 0.0,
                "fee": 0.0,
                "realized_pnl": 0.0,
                "equity": self.snapshot()["equity"],
            }
            self.orders.append(row)
            return row

        fill_price = mark * (1.0 + side * self.cfg.slippage_bps / 10_000.0)
        fee = abs(delta) * self.cfg.max_leverage * self.equity * self.cfg.fee_bps / 10_000.0

        realized = 0.0
        if current != 0.0 and pos.avg_entry > 0.0 and current * delta < 0.0:
            closed = min(abs(current), abs(delta))
            realized = (1.0 if current > 0 else -1.0) * closed * self.cfg.max_leverage * (
                fill_price / pos.avg_entry - 1.0
            )

        old_abs = abs(current)
        target_abs = abs(requested_target)
        if requested_target == 0.0:
            pos.avg_entry = 0.0
        elif current == 0.0 or current * requested_target < 0.0:
            pos.avg_entry = fill_price
        elif target_abs > old_abs:
            added = target_abs - old_abs
            pos.avg_entry = (old_abs * pos.avg_entry + added * fill_price) / max(target_abs, 1e-12)

        pos.position = requested_target
        pos.last_price = mark
        self.equity += realized - fee

        row = {
            **intent.to_dict(),
            "status": "filled",
            "fill_price": float(fill_price),
            "filled_delta": float(delta),
            "fee": float(fee),
            "realized_pnl": float(realized),
            "equity": float(self.snapshot()["equity"]),
        }
        self.orders.append(row)
        return row


@dataclass(frozen=True)
class OKXDemoConfig:
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = "https://www.okx.com"
    simulated_trading: bool = True
    td_mode: str = "cross"
    fixed_size: str = "1"
    timeout: float = 20.0

    @classmethod
    def from_env(cls, *, fixed_size: str, td_mode: str = "cross") -> "OKXDemoConfig":
        missing = [
            name
            for name in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                "OKX demo broker needs demo API credentials in env: " + ", ".join(missing)
            )
        return cls(
            api_key=os.environ["OKX_API_KEY"],
            api_secret=os.environ["OKX_API_SECRET"],
            passphrase=os.environ["OKX_API_PASSPHRASE"],
            base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com"),
            simulated_trading=os.getenv("OKX_SIMULATED_TRADING", "1") != "0",
            td_mode=td_mode,
            fixed_size=str(fixed_size),
        )


def _default_http_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OKX demo order failed HTTP {exc.code}: {payload}") from exc


class OKXDemoBroker:
    """Explicit OKX demo REST order adapter.

    This adapter does not hold the learning ledger. Use it through
    :class:`MirroredBroker` so local shadow accounting remains available.
    """

    def __init__(self, cfg: OKXDemoConfig, *, http_post: HttpPost = _default_http_post) -> None:
        if not cfg.simulated_trading:
            raise ValueError("OKXDemoBroker refuses to run without simulated_trading=True")
        self.cfg = cfg
        self._http_post = http_post
        self._positions: dict[str, float] = {}
        self.orders: list[dict[str, Any]] = []

    def on_price(self, symbol: str, price: float, ts: datetime) -> None:
        return None

    def position(self, symbol: str) -> float:
        return float(self._positions.get(symbol, 0.0))

    def snapshot(self) -> dict[str, Any]:
        return {"okx_demo_orders": len(self.orders)}

    def _headers(self, method: str, request_path: str, body: str) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        digest = hmac.new(
            self.cfg.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.cfg.api_key,
            "OK-ACCESS-SIGN": base64.b64encode(digest).decode("ascii"),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.cfg.passphrase,
            "x-simulated-trading": "1",
        }

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        if intent.side == "none":
            row = {**intent.to_dict(), "status": "no_order", "okx_demo": False}
            self.orders.append(row)
            return row

        request_path = "/api/v5/trade/order"
        payload = {
            "instId": intent.symbol,
            "tdMode": self.cfg.td_mode,
            "side": intent.side,
            "ordType": "market",
            "sz": self.cfg.fixed_size,
            "clOrdId": intent.client_order_id[:32] if intent.client_order_id else "",
        }
        if intent.reduce_only:
            payload["reduceOnly"] = "true"
        body = json.dumps(payload, separators=(",", ":"))
        response = self._http_post(
            f"{self.cfg.base_url}{request_path}",
            body.encode("utf-8"),
            self._headers("POST", request_path, body),
            self.cfg.timeout,
        )
        row = {
            **intent.to_dict(),
            "status": "sent_to_okx_demo",
            "okx_demo": True,
            "okx_request": payload,
            "okx_response": response,
        }
        self.orders.append(row)
        self._positions[intent.symbol] = float(intent.target_position)
        return row


class MirroredBroker:
    """Run local paper accounting and optionally mirror filled orders."""

    def __init__(self, primary: Broker, mirror: Broker) -> None:
        self.primary = primary
        self.mirror = mirror

    def on_price(self, symbol: str, price: float, ts: datetime) -> None:
        self.primary.on_price(symbol, price, ts)
        self.mirror.on_price(symbol, price, ts)

    def position(self, symbol: str) -> float:
        return self.primary.position(symbol)

    def snapshot(self) -> dict[str, Any]:
        out = self.primary.snapshot()
        out["mirror"] = self.mirror.snapshot()
        return out

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        primary = self.primary.place_order(intent)
        mirror = {"status": "not_sent"}
        if primary.get("status") == "filled" and abs(float(primary.get("filled_delta", 0.0))) > 1e-12:
            mirror = self.mirror.place_order(intent)
        return {**primary, "mirror": mirror}
