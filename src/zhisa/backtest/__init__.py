"""Backtest engine, walk-forward splitter, risk metrics, reports."""
from zhisa.backtest.metrics import compute_metrics, Metrics
from zhisa.backtest.regime_ab import (
    RegimeABConfig,
    RegimeABResult,
    RegimeGatedPolicy,
    RegimeVariantResult,
    run_regime_ab_backtest,
)
from zhisa.backtest.regime_walkforward import (
    RegimeWalkForwardConfig,
    RegimeWalkForwardResult,
    run_regime_walk_forward_ab,
)

__all__ = [
    "Metrics",
    "RegimeABConfig",
    "RegimeABResult",
    "RegimeGatedPolicy",
    "RegimeVariantResult",
    "RegimeWalkForwardConfig",
    "RegimeWalkForwardResult",
    "compute_metrics",
    "run_regime_ab_backtest",
    "run_regime_walk_forward_ab",
]
