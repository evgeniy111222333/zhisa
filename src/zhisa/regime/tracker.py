"""Temporal state tracking for Market Regime Intelligence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class RegimeTransition:
    from_primary: str
    to_primary: str
    from_secondary: str
    to_secondary: str
    timestamp: str | None
    previous_stable_for: int
    confirmed: bool
    transition_risk: float
    confidence_delta: float


@dataclass(frozen=True)
class RegimeTrackPoint:
    report: RegimeReport
    timestamp: str | None
    stable_for: int
    changed: bool


@dataclass(frozen=True)
class RegimeTrackState:
    current: RegimeReport
    previous: RegimeReport | None
    stable_for: int
    changed: bool
    transition: RegimeTransition | None
    history_size: int


@dataclass(frozen=True)
class RegimeStateTrackerConfig:
    analyzer: RegimeIntelligenceConfig = field(default_factory=RegimeIntelligenceConfig)
    history_size: int = 512
    min_persistence: int = 2


def _regime_key(report: RegimeReport) -> tuple[str, str]:
    return report.primary_regime, report.secondary_regime


def _timestamp(report: RegimeReport) -> str | None:
    value = report.features.get("timestamp") if report.features else None
    return str(value) if value is not None else None


class RegimeStateTracker:
    """Track regime persistence and transitions over sequential bars."""

    def __init__(
        self,
        cfg: RegimeStateTrackerConfig | None = None,
        *,
        analyzer: RegimeIntelligence | None = None,
    ) -> None:
        self.cfg = cfg or RegimeStateTrackerConfig()
        self.analyzer = analyzer or RegimeIntelligence(self.cfg.analyzer)
        self._history: list[RegimeTrackPoint] = []
        self._stable_for = 0

    @property
    def history(self) -> tuple[RegimeTrackPoint, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._stable_for = 0

    def update(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        extra_context: Optional[dict] = None,
    ) -> RegimeTrackState:
        report = self.analyzer.analyze(df, t=t, symbol=symbol, extra_context=extra_context)
        previous = self._history[-1].report if self._history else None
        changed = previous is not None and _regime_key(previous) != _regime_key(report)
        if changed:
            previous_stable_for = self._stable_for
            self._stable_for = 1
            transition = RegimeTransition(
                from_primary=previous.primary_regime,
                to_primary=report.primary_regime,
                from_secondary=previous.secondary_regime,
                to_secondary=report.secondary_regime,
                timestamp=_timestamp(report),
                previous_stable_for=previous_stable_for,
                confirmed=previous_stable_for >= self.cfg.min_persistence,
                transition_risk=report.transition_risk,
                confidence_delta=report.confidence - previous.confidence,
            )
        else:
            self._stable_for = self._stable_for + 1 if previous is not None else 1
            transition = None

        point = RegimeTrackPoint(
            report=report,
            timestamp=_timestamp(report),
            stable_for=self._stable_for,
            changed=changed,
        )
        self._history.append(point)
        if len(self._history) > self.cfg.history_size:
            del self._history[: len(self._history) - self.cfg.history_size]

        return RegimeTrackState(
            current=report,
            previous=previous,
            stable_for=self._stable_for,
            changed=changed,
            transition=transition,
            history_size=len(self._history),
        )

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for point in self._history:
            r = point.report
            rows.append({
                "timestamp": point.timestamp,
                "primary_regime": r.primary_regime,
                "secondary_regime": r.secondary_regime,
                "micro_regime": r.micro_regime,
                "confidence": r.confidence,
                "transition_risk": r.transition_risk,
                "tradeability_score": r.tradeability_score,
                "risk_mode": r.risk_mode,
                "stable_for": point.stable_for,
                "changed": point.changed,
            })
        return pd.DataFrame(rows)


__all__ = [
    "RegimeStateTracker",
    "RegimeStateTrackerConfig",
    "RegimeTrackPoint",
    "RegimeTrackState",
    "RegimeTransition",
]
