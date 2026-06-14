"""Registry and champion/challenger selection for learned regime models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class RegimeModelCandidate:
    name: str
    artifact_path: str
    calibration_path: str = ""
    profile: str = ""
    version: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, metric: str) -> float:
        return float(self.metrics.get(metric, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChampionChallengerReport:
    champion: dict[str, Any]
    challenger: dict[str, Any]
    metric: str
    higher_is_better: bool
    score_delta: float
    promote: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeModelRegistry:
    candidates: tuple[RegimeModelCandidate, ...] = ()
    champion_name: str = ""

    def add(self, candidate: RegimeModelCandidate) -> "RegimeModelRegistry":
        remaining = tuple(c for c in self.candidates if c.name != candidate.name)
        return RegimeModelRegistry(candidates=remaining + (candidate,), champion_name=self.champion_name)

    def get(self, name: str) -> RegimeModelCandidate:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown regime model candidate '{name}'")

    def champion(self) -> RegimeModelCandidate | None:
        if self.champion_name:
            return self.get(self.champion_name)
        return self.candidates[0] if self.candidates else None

    def best(self, metric: str, *, higher_is_better: bool = True) -> RegimeModelCandidate | None:
        if not self.candidates:
            return None
        key = lambda c: c.score(metric)
        return max(self.candidates, key=key) if higher_is_better else min(self.candidates, key=key)

    def champion_challenger_report(
        self,
        challenger: RegimeModelCandidate | str,
        *,
        metric: str = "delta_sharpe",
        higher_is_better: bool = True,
        min_improvement: float = 0.0,
    ) -> ChampionChallengerReport:
        champion = self.champion()
        if champion is None:
            raise ValueError("registry has no champion candidate")
        challenger_c = self.get(challenger) if isinstance(challenger, str) else challenger
        champ_score = champion.score(metric)
        challenger_score = challenger_c.score(metric)
        raw_delta = challenger_score - champ_score
        score_delta = raw_delta if higher_is_better else -raw_delta
        promote = score_delta >= float(min_improvement)
        reasons = [
            f"champion={champion.name} {metric}={champ_score:.6f}",
            f"challenger={challenger_c.name} {metric}={challenger_score:.6f}",
            f"required_improvement={min_improvement:.6f}",
        ]
        if promote:
            reasons.append("challenger satisfies promotion threshold")
        else:
            reasons.append("champion remains active")
        return ChampionChallengerReport(
            champion=champion.to_dict(),
            challenger=challenger_c.to_dict(),
            metric=metric,
            higher_is_better=higher_is_better,
            score_delta=float(score_delta),
            promote=bool(promote),
            reasons=reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_name": self.champion_name,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def build_regime_model_registry(
    candidates: Iterable[RegimeModelCandidate],
    *,
    champion_name: Optional[str] = None,
) -> RegimeModelRegistry:
    candidates_t = tuple(candidates)
    if champion_name is None:
        champion_name = candidates_t[0].name if candidates_t else ""
    return RegimeModelRegistry(candidates=candidates_t, champion_name=champion_name or "")


__all__ = [
    "ChampionChallengerReport",
    "RegimeModelCandidate",
    "RegimeModelRegistry",
    "build_regime_model_registry",
]
