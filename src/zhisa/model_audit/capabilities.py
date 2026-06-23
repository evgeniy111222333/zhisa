from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from zhisa.data.preparation import load_prepared_split


@dataclass
class DataCapabilities:
    available: set[str] = field(default_factory=set)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"available": sorted(self.available), **self.details}


def inspect_prepared_data(root: str | Path | None, split: str = "test") -> DataCapabilities:
    caps = DataCapabilities()
    if root is None:
        return caps
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        caps.details["error"] = f"manifest not found: {manifest_path}"
        return caps
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = load_prepared_split(root, split)
    caps.available.update(("ohlcv", "timestamps"))
    columns = {str(column).lower() for column in frame.columns}
    if any("funding" in column for column in columns):
        caps.available.add("funding")
    if any("liquidation" in column for column in columns):
        caps.available.add("liquidations")
    if {"bid", "ask"}.issubset(columns):
        caps.available.add("orderbook_l1")
    if any(column.startswith(("bid_", "ask_")) for column in columns):
        caps.available.add("orderbook_l2")
    timeframe = str(manifest.get("timeframe", "")).lower()
    if timeframe in {"5m", "1h"}:
        caps.available.add(f"timeframe_{timeframe}")
    start = pd.Timestamp(frame.index.min())
    end = pd.Timestamp(frame.index.max())
    start_utc = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end_utc = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    if start_utc <= pd.Timestamp("2018-01-01", tz="UTC"):
        caps.available.add("history_2018")
    if start_utc <= pd.Timestamp("2020-03-01", tz="UTC") and end_utc >= pd.Timestamp("2020-04-15", tz="UTC"):
        caps.available.add("history_2020_03")
    symbols = sorted(str(x) for x in frame["symbol"].unique()) if "symbol" in frame else []
    caps.details.update({
        "root": str(root.resolve()),
        "split": split,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": len(frame),
        "symbols": symbols,
        "columns": sorted(columns),
        "manifest_checksum": manifest.get("output_checksum"),
    })
    return caps
