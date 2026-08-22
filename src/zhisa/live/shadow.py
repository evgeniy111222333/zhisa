"""Live shadow-mode engine: market data -> decisions -> paper fills -> experience."""
from __future__ import annotations

import asyncio
import csv
import json
import random
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhisa.env.actions import DiscreteAction
from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.live.brokers import Broker
from zhisa.live.events import MarketEvent, OrderIntent, action_name, action_target
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.regime.detector import RegimeIntelligence
from zhisa.regime.planner import plan_trade
from zhisa.regime.profiles import build_regime_profile_config, resolve_regime_profile
from zhisa.rendering.chart_renderer import render_chart


@dataclass(frozen=True)
class ShadowConfig:
    out_dir: Path
    timeframe: str = "5m"
    strategy: str = "regime"
    min_bars: int = 96
    horizon_bars: int = 12
    max_bars_per_symbol: int = 2500
    profile: str = "auto"
    benchmark_symbol: str = "BTC/USDT"
    seed: int = 0
    checkpoint: str | None = None
    device: str = "cpu"


class LiveShadowEngine:
    def __init__(self, broker: Broker, cfg: ShadowConfig) -> None:
        self.broker = broker
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(int(cfg.seed))
        self.bars: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=cfg.max_bars_per_symbol))
        self.pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.event_counts: dict[str, int] = defaultdict(int)
        self.last_event_ts: str = ""
        self.started_at = pd.Timestamp.utcnow().isoformat()
        self.device = cfg.device

        self.policy: PolicyNetwork | None = None
        self.policy_cfg: PolicyConfig | None = None
        if cfg.strategy in {"s2b", "neural"} or cfg.checkpoint:
            self._load_policy(cfg.checkpoint)

        self.events_file = (self.out_dir / "live_events.jsonl").open("a", encoding="utf-8", newline="")
        self._csv_files: list[Any] = []
        self.bars_writer = self._writer("bars.csv", [
            "symbol", "timestamp", "open", "high", "low", "close", "volume",
        ])
        self.decisions_writer = self._writer("decisions.csv", [
            "symbol", "timestamp", "strategy", "action", "action_name", "current_position",
            "target_position", "price", "reason", "primary_regime", "plan_status",
            "recommended_playbook", "tradeability_score", "transition_risk",
        ])
        self.orders_writer = self._writer("orders.csv", [
            "symbol", "timestamp", "action", "action_name", "current_position",
            "target_position", "delta", "side", "reduce_only", "price", "reason",
            "client_order_id", "status", "fill_price", "filled_delta", "fee",
            "realized_pnl", "equity", "mirror_status",
        ])
        self.equity_writer = self._writer("equity.csv", [
            "timestamp", "equity", "cash_equity", "unrealized", "positions_json",
        ])
        self.experience_writer = self._writer("experience.csv", [
            "symbol", "decision_timestamp", "resolved_timestamp", "action",
            "action_name", "entry_price", "exit_price", "raw_forward_return",
            "directional_forward_return", "horizon_bars", "reason",
        ])

    def _writer(self, name: str, columns: list[str]) -> csv.DictWriter:
        file = (self.out_dir / name).open("a", encoding="utf-8", newline="")
        self._csv_files.append(file)
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if file.tell() == 0:
            writer.writeheader()
        return writer

    def close(self) -> None:
        self.events_file.close()
        for file in self._csv_files:
            file.close()
        summary_path = self.out_dir / "live_shadow_summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def _df(self, symbol: str) -> pd.DataFrame:
        rows = list(self.bars[symbol])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def handle_event(self, event: MarketEvent) -> None:
        self.event_counts[event.kind] += 1
        self.last_event_ts = event.ts.isoformat()
        self.events_file.write(json.dumps(event.to_dict(), default=str) + "\n")
        if event.price is not None and float(event.price) > 0:
            self.broker.on_price(event.symbol, float(event.price), event.ts)
            self._write_equity(event.ts)
        if event.kind == "kline" and event.ohlcv and bool(event.ohlcv.get("closed", False)):
            self._handle_closed_bar(event)

    def _flush_files(self) -> None:
        self.events_file.flush()
        for f in self._csv_files:
            f.flush()

    def _handle_closed_bar(self, event: MarketEvent) -> None:
        row = {
            "symbol": event.symbol,
            "timestamp": event.ts.isoformat(),
            "open": float(event.ohlcv["open"]),
            "high": float(event.ohlcv["high"]),
            "low": float(event.ohlcv["low"]),
            "close": float(event.ohlcv["close"]),
            "volume": float(event.ohlcv.get("volume", 0.0)),
        }
        self.bars[event.symbol].append(row)
        self.bars_writer.writerow(row)
        self._resolve_experience(event.symbol, event.ts)
        self._decide(event.symbol, event.ts, float(row["close"]))
        self._flush_files()

    def _decide(self, symbol: str, ts: pd.Timestamp, price: float) -> None:
        action, reason, details = self._select_action(symbol)
        current = self.broker.position(symbol)
        target = action_target(action, current)
        client_order_id = f"zhisa{int(time.time() * 1000)}{abs(hash((symbol, ts.isoformat()))) % 100000}"
        intent = OrderIntent(
            symbol=symbol,
            action=int(action),
            current_position=current,
            target_position=target,
            price=price,
            ts=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
            reason=reason,
            client_order_id=client_order_id,
        )
        self.decisions_writer.writerow({
            "symbol": symbol,
            "timestamp": intent.ts.isoformat(),
            "strategy": self.cfg.strategy,
            "action": int(action),
            "action_name": action_name(action),
            "current_position": current,
            "target_position": target,
            "price": price,
            "reason": reason,
            **details,
        })
        order = self.broker.place_order(intent)
        mirror = order.get("mirror") if isinstance(order.get("mirror"), dict) else {}
        self.orders_writer.writerow({
            **order,
            "mirror_status": mirror.get("status", ""),
        })
        self.pending[symbol].append({
            "timestamp": intent.ts,
            "action": int(action),
            "entry_price": float(price),
            "reason": reason,
            "bar_index": len(self.bars[symbol]) - 1,
        })

    def _load_policy(self, checkpoint: str | None) -> None:
        import torch
        if not checkpoint:
            default_ckpt = "artifacts/s2b/s2b_mtf_champion_v2.pt"
            if Path(default_ckpt).exists():
                checkpoint = default_ckpt
            else:
                return
        try:
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model_cfg_dict = payload.get("model_config") or payload.get("config") or {}
            self.policy_cfg = PolicyConfig(**model_cfg_dict) if model_cfg_dict else PolicyConfig()
            self.policy = PolicyNetwork(self.policy_cfg).to(self.device)
            state_dict = payload.get("model") or payload.get("model_state_dict")
            if state_dict is not None:
                self.policy.load_state_dict(state_dict)
            self.policy.eval()
        except Exception as exc:
            self.policy = None

    def _select_action(self, symbol: str) -> tuple[int, str, dict[str, Any]]:
        import torch
        df = self._df(symbol)
        if len(df) < max(1, int(self.cfg.min_bars)):
            return int(DiscreteAction.SKIP), "warmup", {}
        if self.cfg.strategy == "hold":
            return int(DiscreteAction.SKIP), "hold_strategy", {}
        if self.cfg.strategy == "random":
            return int(self.rng.randrange(len(DiscreteAction))), "random_strategy", {}
        if self.cfg.strategy in {"s2b", "neural"}:
            if self.policy is None:
                return int(DiscreteAction.SKIP), "s2b_policy_not_loaded", {}
            try:
                # 1. Render ConvNeXt vision tensor (1, 3, H, W)
                chart_raw = render_chart(df.iloc[-32:], size=self.policy_cfg.image_size)
                img_tensor = chart_raw.unsqueeze(0).to(self.device)
                try:
                    from torchvision.utils import save_image
                    sym_clean = symbol.replace("/", "-")
                    save_image(chart_raw, str(self.out_dir / f"vision_chart_{sym_clean}.png"))
                    save_image(chart_raw, str(self.out_dir / "latest_vision_chart.png"))
                except Exception:
                    pass
                # 2. Compute numeric features (1, window, in_features)
                feat_df = compute_ohlcv_features(df).fillna(0.0)
                feat_arr = feat_df.values[-self.policy_cfg.window:]
                if len(feat_arr) < self.policy_cfg.window:
                    pad = np.zeros((self.policy_cfg.window - len(feat_arr), feat_arr.shape[1]))
                    feat_arr = np.vstack([pad, feat_arr])
                numeric_tensor = torch.from_numpy(feat_arr).float().unsqueeze(0).to(self.device)
                # 3. Context tensor (1, in_context_features)
                context_tensor = torch.zeros(1, self.policy_cfg.in_context_features).to(self.device)
                context_tensor[0, 0] = float(self.broker.position(symbol))
                equity_snap = self.broker.snapshot()
                context_tensor[0, 1] = float(equity_snap.get("equity", 1.0)) - 1.0
                # 4. Neural inference with precise live timing
                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    out = self.policy(chart=img_tensor, numeric=numeric_tensor, context=context_tensor)
                infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

                logits = out.get("policy_logits")
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, action].item())

                regime_id = int(torch.argmax(out.get("regime", torch.zeros(1, 4)), dim=-1).item())
                regime_names = ["bull_trend", "bear_trend", "high_volatility", "compression"]
                regime_name = regime_names[regime_id] if regime_id < len(regime_names) else "unknown"
                ret_pred = float(out.get("return_pred", torch.zeros(1)).item())
                vol_pred = float(out.get("volatility", torch.zeros(1)).item())

                return action, f"s2b_conf={confidence:.3f}", {
                    "primary_regime": regime_name,
                    "tradeability_score": confidence,
                    "transition_risk": vol_pred,
                    "s2b_pred_return": ret_pred,
                    "s2b_action_prob": confidence,
                    "inference_ms": round(infer_ms, 2),
                }
            except Exception as exc:
                return int(DiscreteAction.SKIP), f"s2b_error:{type(exc).__name__}:{str(exc)[:120]}", {}
        if self.cfg.strategy == "momentum":
            ret = float(df["close"].iloc[-1] / max(df["close"].iloc[-12], 1e-12) - 1.0) if len(df) >= 12 else 0.0
            if ret > 0.002:
                return int(DiscreteAction.LONG_25), f"momentum_ret={ret:.6f}", {}
            if ret < -0.002:
                return int(DiscreteAction.SHORT_25), f"momentum_ret={ret:.6f}", {}
            return int(DiscreteAction.SKIP), f"momentum_ret={ret:.6f}", {}
        if self.cfg.strategy == "regime":
            try:
                analyzer = self._analyzer(symbol)
                report = analyzer.analyze(df, t=len(df) - 1, symbol=symbol)
                plan = plan_trade(report)
                return int(plan.recommended_action), "regime_plan", {
                    "primary_regime": report.primary_regime,
                    "plan_status": plan.status,
                    "recommended_playbook": plan.recommended_playbook,
                    "tradeability_score": float(report.tradeability_score),
                    "transition_risk": float(report.transition_risk),
                }
            except Exception as exc:
                return int(DiscreteAction.SKIP), f"regime_error:{type(exc).__name__}:{str(exc)[:120]}", {}
        raise ValueError(f"Unknown live strategy: {self.cfg.strategy!r}")

    def _analyzer(self, symbol: str) -> RegimeIntelligence:
        if self.cfg.profile == "auto":
            cfg = resolve_regime_profile(symbol=symbol, asset_class="crypto").with_overrides(
                source_timeframe=self.cfg.timeframe,
                benchmark_symbol=self.cfg.benchmark_symbol or None,
            )
        else:
            cfg = build_regime_profile_config(
                self.cfg.profile,
                source_timeframe=self.cfg.timeframe,
                benchmark_symbol=self.cfg.benchmark_symbol or None,
            )
        return RegimeIntelligence(cfg)

    def _resolve_experience(self, symbol: str, ts: pd.Timestamp) -> None:
        rows = list(self.bars[symbol])
        still_pending: list[dict[str, Any]] = []
        for item in self.pending[symbol]:
            target_idx = int(item["bar_index"]) + int(self.cfg.horizon_bars)
            if target_idx >= len(rows):
                still_pending.append(item)
                continue
            exit_price = float(rows[target_idx]["close"])
            entry = float(item["entry_price"])
            raw_ret = exit_price / max(entry, 1e-12) - 1.0
            action = int(item["action"])
            direction = 1 if action in {1, 2, 3} else -1 if action in {4, 5, 6} else 0
            self.experience_writer.writerow({
                "symbol": symbol,
                "decision_timestamp": item["timestamp"].isoformat(),
                "resolved_timestamp": rows[target_idx]["timestamp"],
                "action": action,
                "action_name": action_name(action),
                "entry_price": entry,
                "exit_price": exit_price,
                "raw_forward_return": raw_ret,
                "directional_forward_return": raw_ret * direction,
                "horizon_bars": int(self.cfg.horizon_bars),
                "reason": item["reason"],
            })
        self.pending[symbol] = still_pending

    def _write_equity(self, ts: Any) -> None:
        snap = self.broker.snapshot()
        self.equity_writer.writerow({
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "equity": snap.get("equity", 0.0),
            "cash_equity": snap.get("cash_equity", 0.0),
            "unrealized": snap.get("unrealized", 0.0),
            "positions_json": json.dumps(snap.get("positions", {}), sort_keys=True),
        })

    def summary(self) -> dict[str, Any]:
        return {
            "safety": {
                "mode": "live_shadow",
                "real_orders_enabled": False,
                "local_paper_broker_enabled": True,
            },
            "started_at": self.started_at,
            "last_event_ts": self.last_event_ts,
            "config": {
                "timeframe": self.cfg.timeframe,
                "strategy": self.cfg.strategy,
                "min_bars": int(self.cfg.min_bars),
                "horizon_bars": int(self.cfg.horizon_bars),
                "profile": self.cfg.profile,
            },
            "event_counts": dict(sorted(self.event_counts.items())),
            "bars_by_symbol": {symbol: len(rows) for symbol, rows in sorted(self.bars.items())},
            "pending_experience": {symbol: len(rows) for symbol, rows in sorted(self.pending.items())},
            "broker": self.broker.snapshot(),
            "artifacts": {
                "events": str(self.out_dir / "live_events.jsonl"),
                "bars": str(self.out_dir / "bars.csv"),
                "decisions": str(self.out_dir / "decisions.csv"),
                "orders": str(self.out_dir / "orders.csv"),
                "equity": str(self.out_dir / "equity.csv"),
                "experience": str(self.out_dir / "experience.csv"),
            },
        }


async def run_shadow_stream(
    events: AsyncIterator[MarketEvent],
    engine: LiveShadowEngine,
    *,
    duration_sec: float = 0.0,
    max_events: int = 0,
) -> int:
    started = time.monotonic()
    count = 0
    try:
        async for event in events:
            engine.handle_event(event)
            count += 1
            if max_events > 0 and count >= max_events:
                break
            if duration_sec > 0 and time.monotonic() - started >= duration_sec:
                break
    finally:
        await asyncio.sleep(0)
    return count
