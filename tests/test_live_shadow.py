from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from zhisa.env.actions import DiscreteAction
from zhisa.live.binance_usdm import BinanceUSDMWebSocketClient
from zhisa.live.brokers import LocalPaperBroker, OKXDemoBroker, OKXDemoConfig, PaperBrokerConfig
from zhisa.live.events import MarketEvent, OrderIntent
from zhisa.live.okx import OKXPublicWebSocketClient
from zhisa.live.shadow import LiveShadowEngine, ShadowConfig, run_shadow_stream


def test_binance_usdm_parser_handles_trade_kline_mark_and_liquidation():
    trade = {
        "stream": "btcusdt@aggTrade",
        "data": {"e": "aggTrade", "E": 1781466600000, "s": "BTCUSDT", "p": "100.5", "q": "0.2", "m": False},
    }
    kline = {
        "data": {
            "e": "kline",
            "E": 1781466600000,
            "k": {
                "t": 1781466600000,
                "s": "BTCUSDT",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100.8",
                "v": "10",
                "x": True,
            },
        }
    }
    mark = {"data": {"e": "markPriceUpdate", "E": 1781466600000, "s": "BTCUSDT", "p": "100.7"}}
    force = {
        "data": {
            "e": "forceOrder",
            "E": 1781466600000,
            "o": {"s": "BTCUSDT", "S": "SELL", "q": "1.5", "p": "100.1", "ap": "100.2"},
        }
    }

    events = []
    for payload in (trade, kline, mark, force):
        events.extend(BinanceUSDMWebSocketClient.parse(payload))

    assert [event.kind for event in events] == ["trade", "kline", "mark_price", "liquidation"]
    assert events[0].side == "buy"
    assert events[1].ohlcv["closed"] is True
    assert events[2].price == 100.7
    assert BinanceUSDMWebSocketClient(["BTC/USDT"], timeframe="5m").stream_url().startswith(
        "wss://fstream.binance.com/market/stream?streams="
    )


def test_okx_parser_handles_trade_ticker_mark_and_candle():
    trade = {"arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"}, "data": [{"instId": "BTC-USDT-SWAP", "px": "100", "sz": "1", "side": "buy", "ts": "1781466600000"}]}
    ticker = {"arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"}, "data": [{"instId": "BTC-USDT-SWAP", "last": "101", "ts": "1781466600000"}]}
    mark = {"arg": {"channel": "mark-price", "instId": "BTC-USDT-SWAP"}, "data": [{"instId": "BTC-USDT-SWAP", "markPx": "102", "ts": "1781466600000"}]}
    candle = {"arg": {"channel": "candle5m", "instId": "BTC-USDT-SWAP"}, "data": [["1781466600000", "100", "103", "99", "102", "12", "0", "0", "1"]]}

    events = []
    for payload in (trade, ticker, mark, candle):
        events.extend(OKXPublicWebSocketClient.parse(payload))

    assert [event.kind for event in events] == ["trade", "ticker", "mark_price", "kline"]
    assert events[-1].ohlcv["closed"] is True
    assert OKXPublicWebSocketClient(["BTC/USDT"], demo=True).url.startswith("wss://wspap.okx.com")


def test_local_paper_broker_fills_and_marks_equity():
    broker = LocalPaperBroker(PaperBrokerConfig(initial_equity=1.0, max_leverage=3.0, fee_bps=4.0, slippage_bps=0.0))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    broker.on_price("BTCUSDT", 100.0, ts)
    opened = broker.place_order(
        OrderIntent(
            symbol="BTCUSDT",
            action=int(DiscreteAction.LONG_50),
            current_position=0.0,
            target_position=0.5,
            price=100.0,
            ts=ts,
        )
    )
    broker.on_price("BTCUSDT", 110.0, ts + timedelta(minutes=5))
    closed = broker.place_order(
        OrderIntent(
            symbol="BTCUSDT",
            action=int(DiscreteAction.CLOSE),
            current_position=0.5,
            target_position=0.0,
            price=110.0,
            ts=ts + timedelta(minutes=5),
        )
    )

    assert opened["status"] == "filled"
    assert closed["realized_pnl"] > 0.0
    assert broker.snapshot()["equity"] > 1.0


async def _fake_events():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    for i in range(8):
        price += 0.5
        yield MarketEvent(
            kind="kline",
            exchange="test",
            symbol="BTCUSDT",
            ts=ts + timedelta(minutes=5 * i),
            price=price,
            ohlcv={
                "open": price - 0.2,
                "high": price + 0.3,
                "low": price - 0.4,
                "close": price,
                "volume": 100 + i,
                "closed": True,
            },
        )


def test_shadow_engine_writes_artifacts_and_experience(tmp_path: Path):
    broker = LocalPaperBroker(PaperBrokerConfig(slippage_bps=0.0))
    engine = LiveShadowEngine(
        broker,
        ShadowConfig(
            out_dir=tmp_path,
            strategy="momentum",
            min_bars=3,
            horizon_bars=2,
            seed=1,
        ),
    )

    count = asyncio.run(run_shadow_stream(_fake_events(), engine, max_events=8))
    engine.close()

    assert count == 8
    summary = json.loads((tmp_path / "live_shadow_summary.json").read_text(encoding="utf-8"))
    assert summary["event_counts"]["kline"] == 8
    assert summary["bars_by_symbol"]["BTCUSDT"] == 8
    decisions = pd.read_csv(tmp_path / "decisions.csv")
    orders = pd.read_csv(tmp_path / "orders.csv")
    experience = pd.read_csv(tmp_path / "experience.csv")
    assert len(decisions) == 8
    assert len(orders) == 8
    assert len(experience) > 0


def test_okx_demo_broker_uses_simulated_header_and_order_payload():
    captured = {}

    def fake_post(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["headers"] = headers
        return {"code": "0", "data": [{"ordId": "demo"}]}

    broker = OKXDemoBroker(
        OKXDemoConfig(
            api_key="key",
            api_secret="secret",
            passphrase="pass",
            fixed_size="2",
        ),
        http_post=fake_post,
    )
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = broker.place_order(
        OrderIntent(
            symbol="BTC-USDT-SWAP",
            action=int(DiscreteAction.LONG_25),
            current_position=0.0,
            target_position=0.25,
            price=100.0,
            ts=ts,
            client_order_id="zhisa_demo_order",
        )
    )

    assert row["status"] == "sent_to_okx_demo"
    assert captured["url"].endswith("/api/v5/trade/order")
    assert captured["headers"]["x-simulated-trading"] == "1"
    assert captured["body"]["instId"] == "BTC-USDT-SWAP"
    assert captured["body"]["side"] == "buy"
    assert captured["body"]["sz"] == "2"
