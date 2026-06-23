from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def ablate_observation(obs: Mapping[str, np.ndarray], modality: str) -> dict[str, np.ndarray]:
    if modality not in {"chart", "numeric", "context"}:
        raise ValueError(f"unknown modality: {modality}")
    out = {key: np.array(value, copy=True) for key, value in obs.items()}
    out[modality].fill(0)
    return out


def fake_wick(frame: pd.DataFrame, index: int, magnitude: float = 5.0) -> pd.DataFrame:
    """Inject a valid OHLC wick without creating impossible candle ordering."""
    out = frame.copy()
    atr_proxy = max(float(out["high"].iloc[index] - out["low"].iloc[index]), 1e-12)
    out.iloc[index, out.columns.get_loc("high")] += magnitude * atr_proxy
    out.iloc[index, out.columns.get_loc("low")] = max(
        1e-12, float(out["low"].iloc[index]) - magnitude * atr_proxy
    )
    return out


def spoof_volume(frame: pd.DataFrame, multiplier: float = 10.0) -> pd.DataFrame:
    out = frame.copy()
    out["volume"] = out["volume"].astype(float) * float(multiplier)
    return out


def mirror_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror returns while preserving positive prices and OHLC invariants.

    Multiplying prices by -1 is invalid for log-return features. This transform
    negates close-to-close log returns around the first close, then maps each
    candle's relative OHLC geometry into the mirrored price path.
    """
    out = frame.copy()
    close = frame["close"].to_numpy(dtype=np.float64)
    if np.any(close <= 0):
        raise ValueError("price mirroring requires strictly positive closes")
    anchor_sq = close[0] ** 2
    mirrored_open = anchor_sq / frame["open"].to_numpy(dtype=np.float64)
    mirrored_close = anchor_sq / close
    mirrored_from_high = anchor_sq / frame["high"].to_numpy(dtype=np.float64)
    mirrored_from_low = anchor_sq / frame["low"].to_numpy(dtype=np.float64)
    out["open"] = mirrored_open
    out["close"] = mirrored_close
    out["high"] = np.maximum.reduce((mirrored_open, mirrored_close, mirrored_from_low))
    out["low"] = np.minimum.reduce((mirrored_open, mirrored_close, mirrored_from_high))
    return out


def pure_noise_like(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    returns = rng.normal(0.0, 0.001, len(out))
    close = float(frame["close"].iloc[0]) * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0.001, 0.0005, len(out)))
    out["open"] = np.r_[close[0], close[:-1]]
    out["close"] = close
    out["high"] = np.maximum(out["open"], out["close"]) * (1.0 + spread)
    out["low"] = np.minimum(out["open"], out["close"]) * (1.0 - spread)
    if "volume" in out:
        out["volume"] = rng.lognormal(8.0, 0.7, len(out))
    return out


def corrupt_feed(frame: pd.DataFrame, kind: str, rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(1, int(len(frame) * rate))
    indices = np.sort(rng.choice(len(frame), size=min(n, len(frame)), replace=False))
    if kind == "missing":
        return frame.drop(frame.index[indices])
    out = frame.copy()
    if kind == "stale":
        for idx in indices:
            if idx > 0:
                out.iloc[idx] = out.iloc[idx - 1]
        return out
    if kind == "duplicate":
        return pd.concat([out, out.iloc[indices]]).sort_index(kind="stable")
    raise ValueError(f"unknown feed corruption: {kind}")
