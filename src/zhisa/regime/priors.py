"""Experience-aware playbook priors from regime memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from zhisa.regime.dataset import PLAYBOOK_DIRECTIONS
from zhisa.regime.memory import RegimeMemory, RegimeMemoryMatch
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class PlaybookPriorConfig:
    k: int = 64
    min_samples: int = 5
    min_similarity: float = -1.0
    good_return_threshold: float = 0.0
    bad_drawdown_threshold: float = -0.035
    max_boost: float = 0.22
    max_penalty: float = 0.35
    same_primary_bonus: float = 0.10
    same_secondary_bonus: float = 0.05

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {self.min_samples}")


@dataclass(frozen=True)
class PlaybookPrior:
    playbook: str
    n: int
    weighted_n: float
    hit_rate: float
    mean_signed_return: float
    mean_realized_vol: float
    worst_drawdown: float
    tail_risk: float
    reliability: float
    score_adjustment: float
    insufficient_samples: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _direction(playbook: str) -> float:
    if playbook in PLAYBOOK_DIRECTIONS:
        return float(PLAYBOOK_DIRECTIONS[playbook])
    p = playbook.lower()
    if "long" in p:
        return 1.0
    if "short" in p:
        return -1.0
    return 0.0


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    denom = float(w.sum())
    if denom <= 1e-12:
        return float(v.mean())
    return float(np.sum(v * w) / denom)


class RegimePlaybookScorer:
    """Score playbooks using nearest historical regime outcomes."""

    def __init__(
        self,
        memory: RegimeMemory,
        cfg: Optional[PlaybookPriorConfig] = None,
    ) -> None:
        self.memory = memory
        self.cfg = cfg or PlaybookPriorConfig()

    def prior(
        self,
        report: RegimeReport,
        playbook: str,
        *,
        matches: Sequence[RegimeMemoryMatch] | None = None,
    ) -> PlaybookPrior:
        cfg = self.cfg
        if matches is None:
            matches = self.memory.query(report, k=cfg.k)

        direction = _direction(playbook)
        signed_returns: list[float] = []
        vols: list[float] = []
        drawdowns: list[float] = []
        weights: list[float] = []
        hits: list[float] = []
        for match in matches:
            if match.similarity < cfg.min_similarity:
                continue
            outcome = match.item.outcome
            if outcome is None or outcome.forward_return is None:
                continue
            label = outcome.label or ""
            allowed = set(match.item.report.allowed_playbooks or [])
            if label and label != playbook and playbook not in allowed:
                continue
            if not label and playbook not in allowed:
                continue
            weight = max(float(match.similarity), 0.0)
            if match.item.report.primary_regime == report.primary_regime:
                weight += cfg.same_primary_bonus
            if match.item.report.secondary_regime == report.secondary_regime:
                weight += cfg.same_secondary_bonus
            weight = max(weight, 1e-3)
            ret = float(outcome.forward_return or 0.0)
            signed = ret * direction if direction != 0.0 else -abs(ret)
            dd = float(outcome.max_drawdown or 0.0)
            signed_returns.append(signed)
            vols.append(float(outcome.realized_vol or 0.0))
            drawdowns.append(dd)
            weights.append(weight)
            hits.append(float(signed > cfg.good_return_threshold and dd >= cfg.bad_drawdown_threshold))

        n = len(signed_returns)
        weighted_n = float(sum(weights))
        if n == 0:
            return PlaybookPrior(
                playbook=playbook,
                n=0,
                weighted_n=0.0,
                hit_rate=0.0,
                mean_signed_return=0.0,
                mean_realized_vol=0.0,
                worst_drawdown=0.0,
                tail_risk=0.0,
                reliability=0.5,
                score_adjustment=0.0,
                insufficient_samples=True,
            )

        hit_rate = _weighted_mean(hits, weights)
        mean_ret = _weighted_mean(signed_returns, weights)
        mean_vol = _weighted_mean(vols, weights)
        worst_dd = float(min(drawdowns)) if drawdowns else 0.0
        tail_risk = float(np.clip(abs(min(worst_dd, 0.0)) / max(abs(cfg.bad_drawdown_threshold), 1e-12), 0.0, 2.0))
        ret_component = float(np.clip(mean_ret / 0.03, -1.0, 1.0))
        sample_factor = float(np.clip(n / max(cfg.min_samples, 1), 0.0, 1.0))
        reliability = float(np.clip(0.62 * hit_rate + 0.26 * (0.5 + 0.5 * ret_component) + 0.12 * (1.0 - min(tail_risk, 1.0)), 0.0, 1.0))
        edge = reliability - 0.5
        adjustment = edge * (cfg.max_boost if edge >= 0.0 else cfg.max_penalty) * 2.0 * sample_factor
        if tail_risk > 1.0:
            adjustment -= min((tail_risk - 1.0) * 0.10, 0.15) * sample_factor
        if n < cfg.min_samples:
            adjustment = min(adjustment, cfg.max_boost * 0.25)
        return PlaybookPrior(
            playbook=playbook,
            n=n,
            weighted_n=weighted_n,
            hit_rate=hit_rate,
            mean_signed_return=mean_ret,
            mean_realized_vol=mean_vol,
            worst_drawdown=worst_dd,
            tail_risk=tail_risk,
            reliability=reliability,
            score_adjustment=float(np.clip(adjustment, -cfg.max_penalty, cfg.max_boost)),
            insufficient_samples=n < cfg.min_samples,
        )

    def priors(
        self,
        report: RegimeReport,
        playbooks: Sequence[str],
    ) -> dict[str, PlaybookPrior]:
        matches = self.memory.query(report, k=self.cfg.k)
        return {playbook: self.prior(report, playbook, matches=matches) for playbook in playbooks}


__all__ = [
    "PlaybookPrior",
    "PlaybookPriorConfig",
    "RegimePlaybookScorer",
]
