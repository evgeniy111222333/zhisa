"""Shared contracts for serious S4 training on prepared market splits."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from zhisa.data.preparation import load_prepared_split
from zhisa.env.rewards import RewardWeights
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.risk.limits import RiskLimits
from zhisa.storage.schema import Timeframe


def load_trading_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("checkpoint_meta") or {}
    if not meta.get("trading_policy_ready") or not meta.get("policy_head_trained"):
        raise ValueError(
            f"checkpoint is not a trained trading policy: stage={meta.get('stage')!r}"
        )
    if "model" not in payload:
        raise ValueError("checkpoint has no model state")
    return payload


def build_policy_from_checkpoint(payload: dict[str, Any]) -> PolicyNetwork:
    raw = payload.get("model_config") or payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint has no model_config")
    values = dict(raw)
    if isinstance(values.get("vision_channels"), list):
        values["vision_channels"] = tuple(values["vision_channels"])
    allowed = {item.name for item in fields(PolicyConfig)}
    model = PolicyNetwork(PolicyConfig(**{k: v for k, v in values.items() if k in allowed}))
    model.load_state_dict(payload["model"], strict=True)
    return model


def build_env_config(cfg, model: PolicyNetwork) -> EnvConfig:
    raw = dict((cfg.get("env_cfg", {}) if cfg else {}) or {})
    unknown = sorted(set(raw) - set(EnvConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown EnvConfig fields: {unknown}")
    if isinstance(raw.get("reward_weights"), dict):
        raw["reward_weights"] = RewardWeights(**raw["reward_weights"])
    if isinstance(raw.get("risk_limits"), dict):
        raw["risk_limits"] = RiskLimits(**raw["risk_limits"])
    raw["window"] = int(model.cfg.window)
    raw["image_size"] = int(model.cfg.image_size)
    return EnvConfig(**raw)


def load_prepared_markets(
    root: str | Path,
    split: str,
    *,
    minimum_bars: int,
    max_bars_per_symbol: int | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = load_prepared_split(root, split)
    if "symbol" not in frame.columns:
        raise ValueError("prepared RL split requires a symbol column")
    delta = pd.Timedelta(minutes=Timeframe.from_str(str(manifest["timeframe"])).minutes)
    markets: list[pd.DataFrame] = []
    names: list[str] = []
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        market = symbol_frame.drop(columns=["symbol"]).sort_index()
        if max_bars_per_symbol:
            market = market.iloc[:max_bars_per_symbol]
        segment_ids = market.index.to_series().diff().ne(delta).cumsum()
        for segment_id, segment in market.groupby(segment_ids, sort=False):
            if len(segment) < minimum_bars:
                continue
            markets.append(segment.copy())
            names.append(f"{symbol}#segment-{segment_id}")
    if not markets:
        raise ValueError(f"prepared split {split!r} has no segment with {minimum_bars} bars")
    return markets, {
        "root": str(root.resolve()),
        "split": split,
        "timeframe": manifest["timeframe"],
        "manifest_checksum": manifest["output_checksum"],
        "markets": names,
        "rows": sum(len(market) for market in markets),
    }
