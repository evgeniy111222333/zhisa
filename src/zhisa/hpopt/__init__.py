"""Hyperparameter optimisation utilities (Optuna-backed).

The :mod:`zhisa.hpopt` package wraps Optuna so that any ZHISA trainer
(BC, DAgger, PPO, CVaR-PPO, Decision Transformer, World Model, etc.)
can be tuned through a single, declarative search-space file.

Public surface:

* :class:`ParamDef`, :class:`SearchSpace`  — declarative spaces.
* :func:`get_space`                         — registry of built-in spaces.
* :class:`OptunaRunner`                     — study orchestration, pruner,
  storage, reporting.
* :func:`bc_objective`, :func:`dagger_objective`,
  :func:`ppo_objective`, :func:`s4_cvar_objective`,
  :func:`dt_objective`, :func:`world_model_objective`,
  :func:`portfolio_ppo_objective`           — ready-made objective functions
  that build and run a trainer from a base config + sampled values.

Usage example::

    from zhisa.hpopt import OptunaRunner, get_space, ppo_objective

    runner = OptunaRunner(direction="maximize", n_trials=20)
    study = runner.run(
        space=get_space("s4_ppo"),
        objective=ppo_objective,
        base_cfg={"n_bars": 1500, "n_episodes": 4, "device": "cpu"},
    )
    print(study.best_value, study.best_params)
"""
from __future__ import annotations

from zhisa.hpopt.objective import (
    ObjectiveResult,
    bc_objective,
    dagger_objective,
    dt_objective,
    ppo_objective,
    portfolio_ppo_objective,
    s4_cvar_objective,
    world_model_objective,
)
from zhisa.hpopt.runner import OptunaRunner, StudySummary
from zhisa.hpopt.search_space import ParamDef, SearchSpace, get_space, list_spaces

__all__ = [
    "ParamDef",
    "SearchSpace",
    "get_space",
    "list_spaces",
    "OptunaRunner",
    "StudySummary",
    "ObjectiveResult",
    "bc_objective",
    "dagger_objective",
    "ppo_objective",
    "s4_cvar_objective",
    "dt_objective",
    "world_model_objective",
    "portfolio_ppo_objective",
]
