"""Market structure, trend maturity, and liquidity/value zones."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from zhisa.features.indicators import atr, ema


@dataclass(frozen=True)
class StructureConfig:
    lookback: int = 128
    swing_window: int = 5
    value_area_pct: float = 0.70
    trend_age_window: int = 96
    late_range_quantile: float = 0.85
    exhaustion_volume_z: float = 2.0


@dataclass(frozen=True)
class LiquidityLevel:
    name: str
    side: str
    price: float
    distance_pct: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiquidityMap:
    value_area_low: float = 0.0
    value_area_high: float = 0.0
    value_area_mid: float = 0.0
    point_of_control: float = 0.0
    upper_levels: list[LiquidityLevel] = field(default_factory=list)
    lower_levels: list[LiquidityLevel] = field(default_factory=list)
    nearest_level: LiquidityLevel | None = None
    in_value_area: bool = False
    distance_to_value_mid_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["upper_levels"] = [x.to_dict() for x in self.upper_levels]
        out["lower_levels"] = [x.to_dict() for x in self.lower_levels]
        out["nearest_level"] = self.nearest_level.to_dict() if self.nearest_level else None
        return out


@dataclass(frozen=True)
class TrendState:
    phase: str = "none"
    direction: str = "flat"
    age_bars: int = 0
    maturity_score: float = 0.0
    exhaustion_score: float = 0.0
    distance_from_ema_pct: float = 0.0
    extension_atr: float = 0.0
    pullback_risk: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketStructureReport:
    trend: TrendState = field(default_factory=TrendState)
    liquidity: LiquidityMap = field(default_factory=LiquidityMap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend": self.trend.to_dict(),
            "liquidity": self.liquidity.to_dict(),
        }


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _finite(x: object, default: float = 0.0) -> float:
    try:
        out = float(x)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    if float(w.sum()) <= 1e-12:
        return float(np.quantile(v, q))
    cdf = np.cumsum(w) / float(w.sum())
    return float(np.interp(q, cdf, v))


class MarketStructureAnalyzer:
    """Causal market-structure analyzer for trend phase and liquidity zones."""

    def __init__(self, cfg: Optional[StructureConfig] = None) -> None:
        self.cfg = cfg or StructureConfig()

    def analyze(self, df: pd.DataFrame, *, t: Optional[int] = None) -> MarketStructureReport:
        if t is not None:
            if t < 0:
                raise ValueError("t must be non-negative")
            work = df.iloc[: t + 1].copy()
        else:
            work = df.copy()
        if work.empty:
            raise ValueError("df slice is empty")
        required = {"open", "high", "low", "close"}
        missing = required - set(work.columns)
        if missing:
            raise ValueError(f"df missing columns: {sorted(missing)}")
        return MarketStructureReport(
            trend=self._trend(work),
            liquidity=self._liquidity(work),
        )

    def _trend(self, df: pd.DataFrame) -> TrendState:
        cfg = self.cfg
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)
        ema21 = ema(close, 21)
        ema55 = ema(close, 55)
        atr_s = atr(df, 14).replace([np.inf, -np.inf], np.nan)
        c = float(close.iloc[-1])
        fast = float(ema21.iloc[-1])
        slow = float(ema55.iloc[-1])
        atr_v = max(_finite(atr_s.iloc[-1]), c * 1e-4)
        if fast > slow and c >= fast:
            direction = "up"
            aligned = (close >= ema21) & (ema21 >= ema55)
        elif fast < slow and c <= fast:
            direction = "down"
            aligned = (close <= ema21) & (ema21 <= ema55)
        else:
            direction = "flat"
            aligned = pd.Series(False, index=df.index)

        age = 0
        for ok in reversed(aligned.iloc[-cfg.trend_age_window:].to_list()):
            if not bool(ok):
                break
            age += 1

        lookback = min(cfg.lookback, len(close))
        recent = close.iloc[-lookback:]
        if direction == "up":
            range_pos = (c - float(recent.min())) / max(float(recent.max() - recent.min()), 1e-12)
            distance_pct = (c / max(fast, 1e-12)) - 1.0
        elif direction == "down":
            range_pos = (float(recent.max()) - c) / max(float(recent.max() - recent.min()), 1e-12)
            distance_pct = (fast / max(c, 1e-12)) - 1.0
        else:
            range_pos = 0.5
            distance_pct = abs(c / max(fast, 1e-12) - 1.0)
        extension_atr = abs(c - fast) / max(atr_v, 1e-12)

        vol_mean = volume.rolling(64, min_periods=2).mean()
        vol_std = volume.rolling(64, min_periods=2).std()
        volume_z = _finite((volume.iloc[-1] - vol_mean.iloc[-1]) / (vol_std.iloc[-1] + 1e-12))
        wick_rejection = False
        bar_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-12)
        upper_wick = float(high.iloc[-1] - max(df["open"].iloc[-1], close.iloc[-1])) / bar_range
        lower_wick = float(min(df["open"].iloc[-1], close.iloc[-1]) - low.iloc[-1]) / bar_range
        if direction == "up" and upper_wick > 0.45 and range_pos > 0.8:
            wick_rejection = True
        if direction == "down" and lower_wick > 0.45 and range_pos > 0.8:
            wick_rejection = True

        maturity = _clip01(0.45 * min(age / max(cfg.trend_age_window, 1), 1.0) + 0.35 * range_pos + 0.20 * min(extension_atr / 4.0, 1.0))
        exhaustion = _clip01(0.40 * min(extension_atr / 4.0, 1.0) + 0.25 * (volume_z > cfg.exhaustion_volume_z) + 0.25 * wick_rejection + 0.10 * (range_pos > cfg.late_range_quantile))
        pullback_risk = _clip01(0.55 * maturity + 0.45 * exhaustion)
        flags: list[str] = []
        if maturity > 0.65:
            flags.append("mature_trend")
        if maturity > 0.78 or range_pos > cfg.late_range_quantile:
            flags.append("late_trend")
        if exhaustion > 0.55:
            flags.append("exhaustion_risk")
        if extension_atr > 3.0:
            flags.append("extended_from_value")

        if direction == "flat":
            phase = "none"
        elif exhaustion > 0.6:
            phase = "exhausted"
        elif maturity > 0.78:
            phase = "late"
        elif maturity > 0.45:
            phase = "mature"
        else:
            phase = "early"

        return TrendState(
            phase=phase,
            direction=direction,
            age_bars=int(age),
            maturity_score=maturity,
            exhaustion_score=exhaustion,
            distance_from_ema_pct=float(distance_pct),
            extension_atr=float(extension_atr),
            pullback_risk=pullback_risk,
            flags=flags,
        )

    def _liquidity(self, df: pd.DataFrame) -> LiquidityMap:
        cfg = self.cfg
        work = df.iloc[-min(cfg.lookback, len(df)):].copy()
        close = work["close"].astype(float)
        high = work["high"].astype(float)
        low = work["low"].astype(float)
        volume = work.get("volume", pd.Series(1.0, index=work.index)).astype(float)
        c = float(close.iloc[-1])

        typical = ((high + low + close) / 3.0).to_numpy(dtype=np.float64)
        weights = volume.to_numpy(dtype=np.float64)
        va_low = _weighted_quantile(typical, weights, (1.0 - cfg.value_area_pct) / 2.0)
        va_high = _weighted_quantile(typical, weights, 1.0 - (1.0 - cfg.value_area_pct) / 2.0)
        poc = _weighted_quantile(typical, weights, 0.5)
        va_mid = (va_low + va_high) / 2.0

        levels: list[LiquidityLevel] = []
        win = max(1, int(cfg.swing_window))
        if len(work) >= win * 2 + 1:
            for i in range(win, len(work) - win):
                h = float(high.iloc[i])
                l = float(low.iloc[i])
                if h >= float(high.iloc[i - win : i + win + 1].max()):
                    levels.append(self._level("swing_high", "upper", h, c, volume.iloc[i], volume))
                if l <= float(low.iloc[i - win : i + win + 1].min()):
                    levels.append(self._level("swing_low", "lower", l, c, volume.iloc[i], volume))
        prior_high = float(high.iloc[:-1].max()) if len(high) > 1 else float(high.iloc[-1])
        prior_low = float(low.iloc[:-1].min()) if len(low) > 1 else float(low.iloc[-1])
        levels.append(self._level("prior_range_high", "upper", prior_high, c, float(volume.iloc[-1]), volume))
        levels.append(self._level("prior_range_low", "lower", prior_low, c, float(volume.iloc[-1]), volume))
        levels.append(self._level("value_area_high", "upper", va_high, c, float(weights.mean()), volume))
        levels.append(self._level("value_area_low", "lower", va_low, c, float(weights.mean()), volume))

        upper = sorted([x for x in levels if x.price >= c], key=lambda x: (abs(x.distance_pct), -x.strength))[:5]
        lower = sorted([x for x in levels if x.price <= c], key=lambda x: (abs(x.distance_pct), -x.strength))[:5]
        all_levels = upper + lower
        nearest = min(all_levels, key=lambda x: abs(x.distance_pct)) if all_levels else None
        return LiquidityMap(
            value_area_low=float(va_low),
            value_area_high=float(va_high),
            value_area_mid=float(va_mid),
            point_of_control=float(poc),
            upper_levels=upper,
            lower_levels=lower,
            nearest_level=nearest,
            in_value_area=bool(va_low <= c <= va_high),
            distance_to_value_mid_pct=float((c / max(va_mid, 1e-12)) - 1.0),
        )

    def _level(self, name: str, side: str, price: float, close: float, vol: float, volume: pd.Series) -> LiquidityLevel:
        vol_med = float(volume.median()) if len(volume) else 1.0
        strength = _clip01(0.5 + 0.5 * min(max(float(vol) / max(vol_med, 1e-12), 0.0), 3.0) / 3.0)
        return LiquidityLevel(
            name=name,
            side=side,
            price=float(price),
            distance_pct=float(price / max(close, 1e-12) - 1.0),
            strength=strength,
        )


__all__ = [
    "LiquidityLevel",
    "LiquidityMap",
    "MarketStructureAnalyzer",
    "MarketStructureReport",
    "StructureConfig",
    "TrendState",
]
