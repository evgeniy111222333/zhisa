"""Diagnostics for regime intelligence quality and playbook outcomes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from zhisa.regime.memory import RegimeOutcome
from zhisa.regime.planner import TradePlan
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class RegimeDiagnosticsConfig:
    transition_horizon: int = 6
    transition_threshold: float = 0.55
    opportunity_return_threshold: float = 0.01
    avoided_drawdown_threshold: float = 0.015

    def __post_init__(self) -> None:
        if self.transition_horizon <= 0:
            raise ValueError(f"transition_horizon must be positive, got {self.transition_horizon}")
        if not 0.0 <= self.transition_threshold <= 1.0:
            raise ValueError(f"transition_threshold must be in [0, 1], got {self.transition_threshold}")


@dataclass(frozen=True)
class OutcomeStats:
    n: int = 0
    hit_rate: float = 0.0
    mean_forward_return: float = 0.0
    median_forward_return: float = 0.0
    mean_realized_vol: float = 0.0
    worst_drawdown: float = 0.0
    mean_tradeability: float = 0.0
    mean_transition_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionDiagnostics:
    n: int = 0
    positive_rate: float = 0.0
    predicted_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    false_positive_rate: float = 0.0
    brier: float = 0.0
    auc: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoTradeDiagnostics:
    n: int = 0
    avoided_drawdown_rate: float = 0.0
    missed_opportunity_rate: float = 0.0
    mean_abs_forward_return: float = 0.0
    mean_forward_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeDiagnosticsReport:
    by_primary_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_playbook: dict[str, dict[str, Any]] = field(default_factory=dict)
    no_trade: dict[str, Any] = field(default_factory=dict)
    transition: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_outcome(value: RegimeOutcome | Mapping[str, Any]) -> RegimeOutcome:
    if isinstance(value, RegimeOutcome):
        return value
    return RegimeOutcome(
        forward_return=float(value.get("forward_return", 0.0) or 0.0),
        realized_vol=float(value.get("realized_vol", 0.0) or 0.0),
        max_drawdown=float(value.get("max_drawdown", 0.0) or 0.0),
        label=str(value.get("label", "")),
    )


def _stats(rows: list[tuple[RegimeReport, RegimeOutcome]]) -> OutcomeStats:
    if not rows:
        return OutcomeStats()
    returns = np.asarray([float(o.forward_return or 0.0) for _, o in rows], dtype=np.float64)
    vols = np.asarray([float(o.realized_vol or 0.0) for _, o in rows], dtype=np.float64)
    dds = np.asarray([float(o.max_drawdown or 0.0) for _, o in rows], dtype=np.float64)
    tradeability = np.asarray([float(r.tradeability_score) for r, _ in rows], dtype=np.float64)
    transition = np.asarray([float(r.transition_risk) for r, _ in rows], dtype=np.float64)
    return OutcomeStats(
        n=len(rows),
        hit_rate=float((returns > 0.0).mean()),
        mean_forward_return=float(returns.mean()),
        median_forward_return=float(np.median(returns)),
        mean_realized_vol=float(vols.mean()),
        worst_drawdown=float(dds.min()) if dds.size else 0.0,
        mean_tradeability=float(tradeability.mean()),
        mean_transition_risk=float(transition.mean()),
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum())
        wins += 0.5 * float((p == neg).sum())
    return float(wins / max(pos.size * neg.size, 1))


def _transition_diagnostics(reports: Sequence[RegimeReport], cfg: RegimeDiagnosticsConfig) -> TransitionDiagnostics:
    if len(reports) <= cfg.transition_horizon:
        return TransitionDiagnostics()
    scores: list[float] = []
    labels: list[int] = []
    for i in range(0, len(reports) - cfg.transition_horizon):
        current = reports[i].primary_regime
        future = reports[i + 1 : i + cfg.transition_horizon + 1]
        changed = any(r.primary_regime != current for r in future)
        labels.append(1 if changed else 0)
        scores.append(float(reports[i].transition_risk))
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    pred = score_arr >= cfg.transition_threshold
    tp = float(((pred == 1) & (label_arr == 1)).sum())
    fp = float(((pred == 1) & (label_arr == 0)).sum())
    fn = float(((pred == 0) & (label_arr == 1)).sum())
    tn = float(((pred == 0) & (label_arr == 0)).sum())
    return TransitionDiagnostics(
        n=int(label_arr.size),
        positive_rate=float(label_arr.mean()) if label_arr.size else 0.0,
        predicted_rate=float(pred.mean()) if pred.size else 0.0,
        precision=float(tp / max(tp + fp, 1.0)),
        recall=float(tp / max(tp + fn, 1.0)),
        false_positive_rate=float(fp / max(fp + tn, 1.0)),
        brier=float(np.mean((score_arr - label_arr) ** 2)) if score_arr.size else 0.0,
        auc=_auc(score_arr, label_arr),
    )


def _no_trade_diagnostics(
    reports: Sequence[RegimeReport],
    outcomes: Sequence[RegimeOutcome],
    plans: Sequence[TradePlan] | None,
    cfg: RegimeDiagnosticsConfig,
) -> NoTradeDiagnostics:
    rows: list[RegimeOutcome] = []
    for i, report in enumerate(reports[: len(outcomes)]):
        plan = plans[i] if plans is not None and i < len(plans) else None
        is_no_trade = False
        if plan is not None:
            is_no_trade = plan.status == "no_trade" or plan.recommended_playbook == "no_trade_wait"
        else:
            is_no_trade = "no_trade_wait" in report.allowed_playbooks or report.tradeability_score < 0.25
        if is_no_trade:
            rows.append(outcomes[i])
    if not rows:
        return NoTradeDiagnostics()
    returns = np.asarray([float(o.forward_return or 0.0) for o in rows], dtype=np.float64)
    dds = np.asarray([float(o.max_drawdown or 0.0) for o in rows], dtype=np.float64)
    return NoTradeDiagnostics(
        n=len(rows),
        avoided_drawdown_rate=float((dds <= -cfg.avoided_drawdown_threshold).mean()),
        missed_opportunity_rate=float((np.abs(returns) >= cfg.opportunity_return_threshold).mean()),
        mean_abs_forward_return=float(np.abs(returns).mean()),
        mean_forward_return=float(returns.mean()),
    )


def diagnose_regime_sequence(
    reports: Sequence[RegimeReport],
    outcomes: Sequence[RegimeOutcome | Mapping[str, Any]],
    *,
    plans: Optional[Sequence[TradePlan]] = None,
    cfg: Optional[RegimeDiagnosticsConfig] = None,
) -> RegimeDiagnosticsReport:
    """Evaluate regime labels, playbooks, no-trade zones, and transition risk."""
    cfg = cfg or RegimeDiagnosticsConfig()
    n = min(len(reports), len(outcomes))
    reports = list(reports[:n])
    outcomes_c = [_coerce_outcome(o) for o in outcomes[:n]]
    paired = list(zip(reports, outcomes_c))

    by_primary: dict[str, list[tuple[RegimeReport, RegimeOutcome]]] = {}
    by_playbook: dict[str, list[tuple[RegimeReport, RegimeOutcome]]] = {}
    for report, outcome in paired:
        by_primary.setdefault(report.primary_regime, []).append((report, outcome))
        playbooks = report.allowed_playbooks or ["<none>"]
        for playbook in playbooks:
            by_playbook.setdefault(playbook, []).append((report, outcome))

    return RegimeDiagnosticsReport(
        by_primary_regime={k: _stats(v).to_dict() for k, v in sorted(by_primary.items())},
        by_playbook={k: _stats(v).to_dict() for k, v in sorted(by_playbook.items())},
        no_trade=_no_trade_diagnostics(reports, outcomes_c, plans, cfg).to_dict(),
        transition=_transition_diagnostics(reports, cfg).to_dict(),
        coverage={
            "n_reports": len(reports),
            "n_outcomes": len(outcomes_c),
            "n_plans": len(plans) if plans is not None else 0,
            "primary_regime_count": len(by_primary),
            "playbook_count": len(by_playbook),
        },
    )


__all__ = [
    "NoTradeDiagnostics",
    "OutcomeStats",
    "RegimeDiagnosticsConfig",
    "RegimeDiagnosticsReport",
    "TransitionDiagnostics",
    "diagnose_regime_sequence",
]
