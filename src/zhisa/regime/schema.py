"""Structured market-regime schema.

The regime layer is intentionally richer than a flat class label.  A
regime report describes the market context, the playbooks that are
currently allowed, the risk posture, and the reasons/dangers behind the
decision so downstream policy and risk modules can use it directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MacroRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    BROAD_RANGE = "broad_range"
    HIGH_VOL_CRASH = "high_vol_crash"
    POST_CRASH_RECOVERY = "post_crash_recovery"
    LOW_LIQUIDITY_CHOP = "low_liquidity_chop"
    EVENT_DRIVEN = "event_driven"


class MesoRegime(str, Enum):
    IMPULSE = "impulse"
    PULLBACK = "pullback"
    BREAKOUT = "breakout"
    FAILED_BREAKOUT = "failed_breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    CHOP = "chop"


class MicroRegime(str, Enum):
    STOP_RUN = "stop_run"
    ABSORPTION = "absorption"
    THIN_BOOK = "thin_book"
    VOLUME_SPIKE = "volume_spike"
    WICK_REJECTION = "wick_rejection"
    VWAP_RECLAIM = "vwap_reclaim"
    ORDERFLOW_IMBALANCE = "orderflow_imbalance"
    QUIET = "quiet"
    NOISY_CHOP = "noisy_chop"


class RiskMode(str, Enum):
    OFF = "off"
    DEFENSIVE = "defensive"
    REDUCED = "reduced"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


class ExpectedDuration(str, Enum):
    VERY_SHORT = "very_short"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeFeatures:
    """Causal numeric features used by the regime classifier."""

    timeframe: str
    n_bars: int
    close: float
    ret_short: float
    ret_medium: float
    ret_long: float
    trend_score: float
    trend_efficiency: float
    realized_vol_short: float
    realized_vol_long: float
    vol_ratio: float
    atr_pct: float
    bb_width: float
    bb_width_quantile: float
    volume_z: float
    range_position: float
    drawdown: float
    rebound_from_low: float
    breakout_up: bool
    breakout_down: bool
    liquidity_sweep_high: bool
    liquidity_sweep_low: bool
    shock_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeReport:
    """Full regime decision consumed by policy, risk, and reporting."""

    primary_regime: str
    secondary_regime: str
    micro_regime: str
    confidence: float
    uncertainty: float
    expected_duration: str
    transition_risk: float
    tradeability_score: float
    allowed_playbooks: list[str]
    blocked_playbooks: list[str]
    risk_mode: str
    position_size_multiplier: float
    stop_style: str
    take_profit_style: str
    explanation: dict[str, list[str]]
    features: dict[str, Any] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ExpectedDuration",
    "MacroRegime",
    "MesoRegime",
    "MicroRegime",
    "RegimeFeatures",
    "RegimeReport",
    "RiskMode",
]
