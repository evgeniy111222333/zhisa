"""End-to-end tests for zhisa.hpopt.objective.

These tests run a *single* Optuna trial per objective, so they
exercise the full pipeline (model build -> train -> metric
extraction) with the smallest budget knobs that still produce a
valid result.
"""
from __future__ import annotations

import math

import optuna
import pytest

from zhisa.hpopt import (
    bc_objective,
    dagger_objective,
    dt_objective,
    get_space,
    ppo_objective,
    s4_cvar_objective,
    world_model_objective,
)
from zhisa.hpopt.objective import ObjectiveResult, _to_float


def test_to_float_handles_nan_and_inf() -> None:
    import math
    assert _to_float(float("nan")) == float("-inf")
    assert _to_float(float("inf")) == float("-inf")
    assert _to_float("-inf") == float("-inf")
    assert _to_float(1.5) == 1.5
    assert _to_float("not a number") == float("-inf")


def test_objective_result_dataclass() -> None:
    r = ObjectiveResult(value=0.1, history=[{"loss": 0.5}], params={"lr": 0.01},
                        elapsed_s=2.0)
    assert r.value == 0.1
    assert r.elapsed_s == 2.0


def test_bc_objective_runs_one_trial(device) -> None:
    space = get_space("bc")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 400, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "log_every": 1000, "expert": "triple_barrier",
    }
    res = bc_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert math.isfinite(res.value) or res.value == float("-inf")
    assert "lr" in res.params
    assert res.elapsed_s >= 0.0


def test_dagger_objective_runs_one_trial(device) -> None:
    space = get_space("dagger")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 400, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "log_every": 1000, "expert": "triple_barrier",
        "max_steps_per_episode": 50,
    }
    res = dagger_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert "n_rounds" in res.params


def test_ppo_objective_runs_one_trial(device) -> None:
    space = get_space("s4_ppo")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 500, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "n_episodes": 1, "max_steps_per_episode": 50,
    }
    res = ppo_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert "clip_ratio" in res.params


def test_s4_cvar_objective_runs_one_trial(device) -> None:
    space = get_space("s4_cvar")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 500, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "n_iterations": 1, "n_episodes": 1, "max_steps_per_episode": 50,
        "cvar_threshold": 0.1,
    }
    res = s4_cvar_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert "cvar_alpha" in res.params


def test_dt_objective_runs_one_trial(device) -> None:
    space = get_space("s6_dt")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 500, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "n_episodes": 1, "max_steps_per_episode": 30,
        "d_model": 32, "n_heads": 2, "n_layers": 1,
    }
    res = dt_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert "context_len" in res.params


def test_world_model_objective_runs_one_trial(device) -> None:
    space = get_space("s7_wm")
    trial = optuna.create_study(direction="maximize").ask()
    base = {
        "space": space, "n_bars": 500, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "n_episodes": 1, "max_steps_per_episode": 30,
    }
    res = world_model_objective(trial, base)
    assert isinstance(res, ObjectiveResult)
    assert "rollout_horizon" in res.params


def test_objective_runner_orchestrates_full_loop(device) -> None:
    """A full run with the smallest possible study."""
    from zhisa.hpopt import OptunaRunner
    space = get_space("bc")
    base = {
        "n_bars": 300, "chart_window": 16,
        "image_size": 24, "device": device, "seed": 0,
        "log_every": 1000, "expert": "triple_barrier",
    }
    runner = OptunaRunner(direction="maximize", study_name="test_loop",
                          sampler="random", seed=0)
    summary = runner.run(space, bc_objective, base_cfg=base, n_trials=2)
    assert summary.n_complete == 2
    assert "lr" in summary.best_params
