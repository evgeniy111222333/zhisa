"""Run live shadow trading on public market data with no real-money orders."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from zhisa.live.binance_usdm import BinanceUSDMWebSocketClient
from zhisa.live.brokers import (
    LocalPaperBroker,
    MirroredBroker,
    OKXDemoBroker,
    OKXDemoConfig,
    PaperBrokerConfig,
)
from zhisa.live.okx import OKXPublicWebSocketClient
from zhisa.live.shadow import LiveShadowEngine, ShadowConfig, run_shadow_stream


def _parse_symbols(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


async def _run(args: argparse.Namespace) -> dict:
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol")

    paper = LocalPaperBroker(
        PaperBrokerConfig(
            initial_equity=float(args.initial_equity),
            max_leverage=float(args.max_leverage),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
    )
    broker = paper
    safety = {
        "mode": "live_shadow",
        "real_orders_enabled": False,
        "api_keys_required": False,
        "exchange_order_api_used": False,
    }
    if args.broker == "okx_demo":
        if not args.i_understand_okx_demo_orders:
            raise RuntimeError(
                "--broker okx_demo requires --i-understand-okx-demo-orders. "
                "It sends demo orders to OKX paper trading, not real-money orders."
            )
        okx = OKXDemoBroker(
            OKXDemoConfig.from_env(
                fixed_size=str(args.okx_fixed_size),
                td_mode=str(args.okx_td_mode),
            )
        )
        broker = MirroredBroker(paper, okx)
        safety.update({
            "api_keys_required": True,
            "exchange_order_api_used": True,
            "exchange_order_api": "okx_demo_only_x_simulated_trading_1",
        })

    cfg = ShadowConfig(
        out_dir=Path(args.out),
        timeframe=args.timeframe,
        strategy=args.strategy,
        min_bars=int(args.min_bars),
        horizon_bars=int(args.horizon_bars),
        max_bars_per_symbol=int(args.max_bars_per_symbol),
        profile=args.profile,
        benchmark_symbol=args.benchmark_symbol,
        seed=int(args.seed),
    )
    engine = LiveShadowEngine(broker, cfg)
    try:
        if args.exchange == "binance_usdm":
            client = BinanceUSDMWebSocketClient(symbols, timeframe=args.timeframe)
        elif args.exchange == "okx":
            client = OKXPublicWebSocketClient(symbols, timeframe=args.timeframe, demo=False)
        elif args.exchange == "okx_demo":
            client = OKXPublicWebSocketClient(symbols, timeframe=args.timeframe, demo=True)
        else:
            raise ValueError(f"Unknown exchange: {args.exchange!r}")

        count = await run_shadow_stream(
            client.iter_events(),
            engine,
            duration_sec=float(args.duration_sec),
            max_events=int(args.max_events),
        )
    finally:
        engine.close()

    summary = engine.summary()
    summary["safety"].update(safety)
    summary["exchange"] = args.exchange
    summary["symbols"] = symbols
    summary["processed_events"] = int(count)
    summary_path = Path(args.out) / "live_shadow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live shadow-mode runner: public market data + no-money paper broker.")
    parser.add_argument("--exchange", choices=["binance_usdm", "okx", "okx_demo"], default="binance_usdm")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--out", type=str, default="artifacts/live_shadow")
    parser.add_argument("--strategy", choices=["regime", "momentum", "hold", "random"], default="regime")
    parser.add_argument("--min-bars", type=int, default=96)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--max-bars-per-symbol", type=int, default=2500)
    parser.add_argument("--profile", type=str, default="auto")
    parser.add_argument("--benchmark-symbol", type=str, default="BTC/USDT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-equity", type=float, default=1.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=1.5)
    parser.add_argument("--broker", choices=["local_paper", "okx_demo"], default="local_paper")
    parser.add_argument("--okx-fixed-size", type=str, default="1", help="OKX demo order size when --broker=okx_demo.")
    parser.add_argument("--okx-td-mode", type=str, default="cross")
    parser.add_argument("--i-understand-okx-demo-orders", action="store_true")
    args = parser.parse_args(argv)

    summary = asyncio.run(_run(args))
    compact = {
        "summary": str(Path(args.out) / "live_shadow_summary.json"),
        "processed_events": summary["processed_events"],
        "event_counts": summary["event_counts"],
        "bars_by_symbol": summary["bars_by_symbol"],
        "safety": summary["safety"],
        "artifacts": summary["artifacts"],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
