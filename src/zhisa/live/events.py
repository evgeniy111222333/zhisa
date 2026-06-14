"""Typed events and order intents for live shadow trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from zhisa.env.actions import DiscreteAction


def utc_from_ms(ms: int | float | str | None) -> datetime:
    if ms is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(int(float(ms)) / 1000.0, tz=timezone.utc)


def symbol_to_binance(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    if ":" in clean:
        clean = clean.split(":", 1)[0]
    return clean.replace("/", "").replace("-", "").replace("_", "")


def symbol_to_okx_swap(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    if "-" in clean and clean.endswith("-SWAP"):
        return clean
    if "/" in clean:
        base, quote = clean.split("/", 1)
    elif clean.endswith("USDT"):
        base, quote = clean[:-4], "USDT"
    else:
        return clean
    return f"{base}-{quote}-SWAP"


def action_target(action: int, current_position: float) -> float:
    a = DiscreteAction(int(action))
    if a == DiscreteAction.SKIP:
        return float(current_position)
    if a == DiscreteAction.CLOSE:
        return 0.0
    if a == DiscreteAction.PARTIAL_CLOSE:
        return float(current_position) * 0.5
    mapping = {
        DiscreteAction.LONG_25: 0.25,
        DiscreteAction.LONG_50: 0.50,
        DiscreteAction.LONG_100: 1.00,
        DiscreteAction.SHORT_25: -0.25,
        DiscreteAction.SHORT_50: -0.50,
        DiscreteAction.SHORT_100: -1.00,
    }
    return float(mapping.get(a, current_position))


def action_name(action: int) -> str:
    try:
        return DiscreteAction(int(action)).name
    except ValueError:
        return str(int(action))


@dataclass(frozen=True)
class MarketEvent:
    kind: str
    exchange: str
    symbol: str
    ts: datetime
    price: float | None = None
    qty: float | None = None
    side: str = ""
    ohlcv: dict[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] | list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.ts.isoformat(),
            "price": self.price,
            "qty": self.qty,
            "side": self.side,
            "ohlcv": self.ohlcv,
        }


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    action: int
    current_position: float
    target_position: float
    price: float
    ts: datetime
    reason: str = ""
    client_order_id: str = ""

    @property
    def delta(self) -> float:
        return float(self.target_position) - float(self.current_position)

    @property
    def side(self) -> str:
        if self.delta > 1e-12:
            return "buy"
        if self.delta < -1e-12:
            return "sell"
        return "none"

    @property
    def reduce_only(self) -> bool:
        current = abs(float(self.current_position))
        target = abs(float(self.target_position))
        return target < current - 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.ts.isoformat(),
            "action": int(self.action),
            "action_name": action_name(self.action),
            "current_position": float(self.current_position),
            "target_position": float(self.target_position),
            "delta": self.delta,
            "side": self.side,
            "reduce_only": self.reduce_only,
            "price": float(self.price),
            "reason": self.reason,
            "client_order_id": self.client_order_id,
        }
