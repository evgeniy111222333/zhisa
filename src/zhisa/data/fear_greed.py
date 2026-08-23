"""Fear & Greed index (alternative.me) as a deterministic global context channel.

The index is a daily sentiment series (0-100). For S1 preparation we embed it
per symbol (same value for every instrument) resampled onto the bar index with
a 1-bar shift so a bar only ever sees the value published *before* it.

Determinism: the CSV/JSON payload is cached to a local parquet snapshot; as
long as the snapshot is not refreshed, the prepared root checksum stays stable.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
DEFAULT_CACHE = Path("data/fear_greed/fear_greed.parquet")


def fetch_fear_greed_history() -> pd.DataFrame:
    """Download the full historical Fear & Greed series (daily, UTC day index)."""
    req = urllib.request.Request(FNG_URL, headers={"User-Agent": "zhisa/1.0"})
    with urllib.request.urlopen(req, timeout=20) as res:
        payload = json.loads(res.read().decode("utf-8"))
    rows = [
        {
            "fng_index": float(str(item.get("value", "nan")).strip()),
            "classification": str(item.get("value_classification", "")).strip(),
            "timestamp": pd.Timestamp(int(str(item.get("timestamp", 0))), unit="s", tz="UTC"),
        }
        for item in (payload or {}).get("data", [])
    ]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    df.index = df.index.normalize()  # day-level resolution
    df = df[~df.index.duplicated(keep="last")]
    df["fng_index"] = df["fng_index"].clip(0.0, 100.0).astype(np.float32)
    return df


def download_to_cache(path: Path | str = DEFAULT_CACHE) -> Path:
    """Fetch the index and write the snapshot parquet (deterministic cache)."""
    df = fetch_fear_greed_history()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def load_fear_greed(path: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """Read the cached snapshot; raises if absent (run the downloader first)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"fear & greed cache not found: {p}. Run zhisa-based downloads once."
        )
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def fear_greed_column(bar_index: pd.DatetimeIndex, fng: pd.DataFrame) -> pd.Series:
    """Map the daily series onto ``bar_index``: forward-fill then shift by 1 bar.

    The shift guarantees causality — a bar at ``t`` only uses F&G values known
    strictly before ``t``. Values before the earliest F&G entry stay NaN and are
    zero-filled by the downstream feature pipeline.
    """
    idx = pd.DatetimeIndex(bar_index).tz_localize(None) if getattr(bar_index, "tz", None) is None else bar_index
    vals = fng["fng_index"].reindex(idx, method="ffill")
    vals = vals.shift(1)
    return vals.astype(np.float32)


def refresh_if_stale(path: Path | str = DEFAULT_CACHE, *, force: bool = False) -> Path:
    p = Path(path)
    if force or not p.is_file():
        return download_to_cache(p)
    # Refresh if the cached snapshot is older than ~2 days (keeps training data
    # close to "now" without making each row unstable).
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    if age.days >= 2:
        return download_to_cache(p)
    return p