"""Online recalibration for regime reliability."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import numpy as np

from zhisa.regime.memory import RegimeOutcome
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class RegimeCalibrationConfig:
    window: int = 128
    min_samples: int = 8
    good_return_threshold: float = 0.0
    bad_drawdown_threshold: float = -0.03
    min_multiplier: float = 0.25
    max_multiplier: float = 1.15


@dataclass(frozen=True)
class RegimeReliability:
    key: str
    n: int
    hit_rate: float
    mean_forward_return: float
    worst_drawdown: float
    reliability_score: float
    confidence_multiplier: float
    tradeability_multiplier: float
    size_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _direction_from_playbook(playbook: str) -> float:
    p = playbook.lower()
    if "long" in p or "reversal_small" in p:
        return 1.0
    if "short" in p:
        return -1.0
    return 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return lo
    return float(np.clip(x, lo, hi))


class OnlineRegimeCalibrator:
    """Track realized outcomes and recalibrate future regime reports."""

    def __init__(self, cfg: Optional[RegimeCalibrationConfig] = None) -> None:
        self.cfg = cfg or RegimeCalibrationConfig()
        self._history: dict[str, list[tuple[RegimeOutcome, str]]] = {}

    def update(
        self,
        report: RegimeReport,
        outcome: RegimeOutcome | dict[str, Any],
        *,
        playbook: str | None = None,
    ) -> RegimeReliability:
        if isinstance(outcome, dict):
            outcome = RegimeOutcome(
                forward_return=outcome.get("forward_return"),
                realized_vol=outcome.get("realized_vol"),
                max_drawdown=outcome.get("max_drawdown"),
                label=outcome.get("label"),
            )
        pb = playbook or outcome.label or (report.allowed_playbooks[0] if report.allowed_playbooks else "unknown")
        key = self._key(report, pb)
        bucket = self._history.setdefault(key, [])
        bucket.append((outcome, pb))
        if len(bucket) > self.cfg.window:
            del bucket[: len(bucket) - self.cfg.window]
        return self.reliability(report, playbook=pb)

    def reliability(self, report: RegimeReport, *, playbook: str | None = None) -> RegimeReliability:
        pb = playbook or (report.allowed_playbooks[0] if report.allowed_playbooks else "unknown")
        key = self._key(report, pb)
        samples = self._history.get(key, [])
        if len(samples) < self.cfg.min_samples:
            return RegimeReliability(
                key=key,
                n=len(samples),
                hit_rate=1.0,
                mean_forward_return=0.0,
                worst_drawdown=0.0,
                reliability_score=1.0,
                confidence_multiplier=1.0,
                tradeability_multiplier=1.0,
                size_multiplier=1.0,
            )
        direction = _direction_from_playbook(pb)
        signed: list[float] = []
        drawdowns: list[float] = []
        hits: list[bool] = []
        for outcome, _ in samples:
            ret = float(outcome.forward_return or 0.0)
            dd = float(outcome.max_drawdown or 0.0)
            score_ret = ret * direction if direction != 0 else -abs(ret)
            signed.append(score_ret)
            drawdowns.append(dd)
            hits.append(score_ret > self.cfg.good_return_threshold and dd >= self.cfg.bad_drawdown_threshold)
        hit_rate = float(np.mean(hits)) if hits else 1.0
        mean_ret = float(np.mean(signed)) if signed else 0.0
        worst_dd = float(np.min(drawdowns)) if drawdowns else 0.0
        dd_penalty = _clip(abs(min(worst_dd, 0.0)) / max(abs(self.cfg.bad_drawdown_threshold), 1e-12), 0.0, 1.0)
        ret_bonus = _clip(mean_ret / 0.03, -1.0, 1.0)
        reliability = _clip(0.65 * hit_rate + 0.25 * (0.5 + 0.5 * ret_bonus) + 0.10 * (1.0 - dd_penalty), 0.0, 1.0)
        conf_mult = _clip(0.55 + 0.65 * reliability, self.cfg.min_multiplier, self.cfg.max_multiplier)
        trade_mult = _clip(0.40 + 0.80 * reliability, self.cfg.min_multiplier, self.cfg.max_multiplier)
        size_mult = _clip(0.30 + 0.85 * reliability, self.cfg.min_multiplier, self.cfg.max_multiplier)
        return RegimeReliability(
            key=key,
            n=len(samples),
            hit_rate=hit_rate,
            mean_forward_return=mean_ret,
            worst_drawdown=worst_dd,
            reliability_score=reliability,
            confidence_multiplier=conf_mult,
            tradeability_multiplier=trade_mult,
            size_multiplier=size_mult,
        )

    def calibrate(
        self,
        report: RegimeReport,
        *,
        playbook: str | None = None,
    ) -> RegimeReport:
        rel = self.reliability(report, playbook=playbook)
        if rel.n < self.cfg.min_samples:
            return report
        explanation = {
            "why": list(report.explanation.get("why", [])),
            "danger": list(report.explanation.get("danger", [])),
        }
        explanation["why"].append(f"online_reliability={rel.reliability_score:.2f}, n={rel.n}")
        if rel.reliability_score < 0.55:
            explanation["danger"].append("online calibration reduced regime reliability")
        features = dict(report.features)
        features["online_calibration"] = rel.to_dict()
        return replace(
            report,
            confidence=_clip(report.confidence * rel.confidence_multiplier, 0.0, 1.0),
            tradeability_score=_clip(report.tradeability_score * rel.tradeability_multiplier, 0.0, 1.0),
            position_size_multiplier=_clip(report.position_size_multiplier * rel.size_multiplier, 0.0, 2.0),
            explanation=explanation,
            features=features,
        )

    def _key(self, report: RegimeReport, playbook: str) -> str:
        return f"{report.primary_regime}|{report.secondary_regime}|{playbook}"


__all__ = [
    "OnlineRegimeCalibrator",
    "RegimeCalibrationConfig",
    "RegimeReliability",
]
