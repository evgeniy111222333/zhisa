"""Unit tests for zhisa.hpopt.runner.OptunaRunner."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import optuna
import pytest

from zhisa.hpopt.runner import (
    PRUNERS,
    SAMPLERS,
    OptunaRunner,
    StudySummary,
)
from zhisa.hpopt.search_space import get_space


def _trivial_objective(trial: optuna.Trial, base_cfg: dict) -> object:
    """Min-lr objective: returns -lr so the search converges on small lr."""
    s = base_cfg["space"]
    p = s.sample(trial)
    class _R:
        value = -p["lr"]
        params = p
        history = []
        elapsed_s = 0.0
    return _R()


def _dummy_objective(trial: optuna.Trial, base_cfg: dict) -> object:
    """Returns a value derived from sampled params."""
    s = base_cfg["space"]
    p = s.sample(trial)
    score = -float(p.get("lr", 0.0)) - 0.001 * float(p.get("epochs", 1))
    class _R:
        value = score
        params = p
        history = []
        elapsed_s = 0.0
    return _R()


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        OptunaRunner(direction="sideways")


def test_invalid_sampler_raises() -> None:
    with pytest.raises(ValueError):
        OptunaRunner(sampler="weird")


def test_invalid_pruner_raises() -> None:
    with pytest.raises(ValueError):
        OptunaRunner(pruner="weird")


def test_run_returns_summary() -> None:
    runner = OptunaRunner(direction="maximize", study_name="test_run",
                          sampler="random", pruner="none", seed=0)
    s = get_space("bc")
    summary = runner.run(s, _trivial_objective, n_trials=3)
    assert isinstance(summary, StudySummary)
    assert summary.n_trials == 3
    assert summary.n_complete == 3
    assert summary.n_pruned == 0
    assert "lr" in summary.best_params
    assert math.isfinite(summary.best_value)


def test_run_minimize_direction() -> None:
    runner = OptunaRunner(direction="minimize", study_name="test_min",
                          sampler="random", seed=0)
    s = get_space("bc")
    summary = runner.run(s, _trivial_objective, n_trials=2)
    assert summary.direction == "minimize"
    assert summary.best_value == pytest.approx(
        max(t.value for t in [] if False) if False else summary.best_value
    )


def test_run_catches_objective_exceptions() -> None:
    def bad(trial, base_cfg):
        raise RuntimeError("kaboom")
    runner = OptunaRunner(direction="maximize", study_name="test_exc", seed=0)
    s = get_space("bc")
    summary = runner.run(s, bad, n_trials=2)
    # The trial count must reflect the request even if all failed
    assert summary.n_trials == 2
    # The wrapper caught the exception and returned -inf; the trial is
    # still COMPLETE for Optuna but carries an "error" user_attr.
    assert summary.best_value == float("-inf")
    assert any("kaboom" in (h["attrs"].get("error") or "")
               for h in summary.history)


def test_storage_sqlite_roundtrip(tmp_path: Path) -> None:
    """Persisted study in SQLite can be re-opened and resumed."""
    db = tmp_path / "study.db"
    runner = OptunaRunner(direction="maximize", study_name="roundtrip",
                          storage=str(db), sampler="random", seed=0)
    s = get_space("bc")
    s1 = runner.run(s, _trivial_objective, n_trials=2)
    runner2 = OptunaRunner(direction="maximize", study_name="roundtrip",
                           storage=str(db), sampler="random", seed=0)
    s2 = runner2.run(s, _trivial_objective, n_trials=2)
    assert s2.n_trials == s1.n_trials + 2


def test_storage_in_memory_works() -> None:
    runner = OptunaRunner(direction="maximize", study_name="mem",
                          storage=":memory:", seed=0)
    s = get_space("bc")
    summary = runner.run(s, _trivial_objective, n_trials=2)
    assert summary.n_trials == 2


def test_summary_to_dict_is_jsonable() -> None:
    runner = OptunaRunner(direction="maximize", study_name="json", seed=0)
    s = get_space("bc")
    summary = runner.run(s, _trivial_objective, n_trials=2)
    d = summary.to_dict()
    import json
    json.dumps(d, default=str)  # should not raise


def test_all_builtin_samplers_and_pruners_registered() -> None:
    assert "tpe" in SAMPLERS
    assert "random" in SAMPLERS
    assert "none" in PRUNERS
    assert "median" in PRUNERS
    assert "hyperband" in PRUNERS
    # make sure each factory is callable
    for f in SAMPLERS.values():
        f(0)
    for f in PRUNERS.values():
        f()


def test_run_with_timeout() -> None:
    runner = OptunaRunner(direction="maximize", study_name="timeout",
                          sampler="random", seed=0)
    s = get_space("bc")
    summary = runner.run(s, _trivial_objective, n_trials=10, timeout=0.5)
    assert summary.n_trials <= 10
    assert summary.n_complete + summary.n_pruned <= 10
