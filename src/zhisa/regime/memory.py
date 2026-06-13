"""Similarity memory for historical market regimes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from zhisa.regime.schema import RegimeReport
from zhisa.regime.vectorizer import RegimeFeatureVectorizer, RegimeVectorizerConfig


@dataclass(frozen=True)
class RegimeOutcome:
    forward_return: float | None = None
    realized_vol: float | None = None
    max_drawdown: float | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_return": self.forward_return,
            "realized_vol": self.realized_vol,
            "max_drawdown": self.max_drawdown,
            "label": self.label,
        }


@dataclass(frozen=True)
class RegimeMemoryItem:
    report: RegimeReport
    vector: np.ndarray
    timestamp: str | None = None
    symbol: str = ""
    outcome: RegimeOutcome | None = None


@dataclass(frozen=True)
class RegimeMemoryMatch:
    item: RegimeMemoryItem
    similarity: float


@dataclass(frozen=True)
class RegimeMemorySummary:
    n: int
    mean_forward_return: float | None
    hit_rate: float | None
    mean_realized_vol: float | None
    worst_drawdown: float | None
    regimes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_forward_return": self.mean_forward_return,
            "hit_rate": self.hit_rate,
            "mean_realized_vol": self.mean_realized_vol,
            "worst_drawdown": self.worst_drawdown,
            "regimes": self.regimes,
        }


@dataclass(frozen=True)
class RegimeMemoryConfig:
    capacity: int = 10_000
    min_similarity: float = -1.0
    vectorizer: RegimeVectorizerConfig = field(default_factory=RegimeVectorizerConfig)


def _timestamp(report: RegimeReport) -> str | None:
    value = report.features.get("timestamp") if report.features else None
    return str(value) if value is not None else None


def _symbol(report: RegimeReport) -> str:
    value = report.features.get("symbol") if report.features else ""
    return str(value or "")


def _as_outcome(outcome: RegimeOutcome | dict[str, Any] | None) -> RegimeOutcome | None:
    if outcome is None or isinstance(outcome, RegimeOutcome):
        return outcome
    return RegimeOutcome(
        forward_return=outcome.get("forward_return"),
        realized_vol=outcome.get("realized_vol"),
        max_drawdown=outcome.get("max_drawdown"),
        label=outcome.get("label"),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class RegimeMemory:
    """In-memory nearest-neighbor store for regime analogs."""

    def __init__(
        self,
        cfg: RegimeMemoryConfig | None = None,
        *,
        vectorizer: RegimeFeatureVectorizer | None = None,
    ) -> None:
        self.cfg = cfg or RegimeMemoryConfig()
        self.vectorizer = vectorizer or RegimeFeatureVectorizer(self.cfg.vectorizer)
        self._items: list[RegimeMemoryItem] = []

    @property
    def items(self) -> tuple[RegimeMemoryItem, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def add(
        self,
        report: RegimeReport,
        *,
        outcome: RegimeOutcome | dict[str, Any] | None = None,
        symbol: str | None = None,
        timestamp: str | None = None,
    ) -> RegimeMemoryItem:
        item = RegimeMemoryItem(
            report=report,
            vector=self.vectorizer.transform(report),
            timestamp=timestamp if timestamp is not None else _timestamp(report),
            symbol=symbol if symbol is not None else _symbol(report),
            outcome=_as_outcome(outcome),
        )
        self._items.append(item)
        if len(self._items) > self.cfg.capacity:
            del self._items[: len(self._items) - self.cfg.capacity]
        return item

    def query(
        self,
        report: RegimeReport,
        *,
        k: int = 5,
        same_symbol_only: bool = False,
    ) -> list[RegimeMemoryMatch]:
        if k <= 0 or not self._items:
            return []
        query_vec = self.vectorizer.transform(report)
        query_symbol = _symbol(report)
        matches: list[RegimeMemoryMatch] = []
        for item in self._items:
            if same_symbol_only and item.symbol != query_symbol:
                continue
            sim = _cosine(query_vec, item.vector)
            if sim >= self.cfg.min_similarity:
                matches.append(RegimeMemoryMatch(item=item, similarity=sim))
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:k]

    def summarize(self, matches: Sequence[RegimeMemoryMatch]) -> RegimeMemorySummary:
        regimes: dict[str, int] = {}
        rets: list[float] = []
        vols: list[float] = []
        drawdowns: list[float] = []
        for match in matches:
            regime = match.item.report.primary_regime
            regimes[regime] = regimes.get(regime, 0) + 1
            outcome = match.item.outcome
            if outcome is None:
                continue
            if outcome.forward_return is not None and np.isfinite(float(outcome.forward_return)):
                rets.append(float(outcome.forward_return))
            if outcome.realized_vol is not None and np.isfinite(float(outcome.realized_vol)):
                vols.append(float(outcome.realized_vol))
            if outcome.max_drawdown is not None and np.isfinite(float(outcome.max_drawdown)):
                drawdowns.append(float(outcome.max_drawdown))

        return RegimeMemorySummary(
            n=len(matches),
            mean_forward_return=float(np.mean(rets)) if rets else None,
            hit_rate=float(np.mean([r > 0.0 for r in rets])) if rets else None,
            mean_realized_vol=float(np.mean(vols)) if vols else None,
            worst_drawdown=float(np.min(drawdowns)) if drawdowns else None,
            regimes=regimes,
        )


__all__ = [
    "RegimeMemory",
    "RegimeMemoryConfig",
    "RegimeMemoryItem",
    "RegimeMemoryMatch",
    "RegimeMemorySummary",
    "RegimeOutcome",
]
