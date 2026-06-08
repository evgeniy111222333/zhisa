"""Unit tests for zhisa.hpopt.search_space."""
from __future__ import annotations

import math

import optuna
import pytest

from zhisa.hpopt.search_space import (
    ParamDef,
    SearchSpace,
    get_space,
    list_spaces,
)


def _truncated_study() -> optuna.Study:
    """Create an Optuna study and immediately finish it (in-memory)."""
    s = optuna.create_study(direction="maximize")
    s.optimize(lambda t: 0.0, n_trials=0)
    return s


def test_paramdef_float_sample_in_range() -> None:
    pd = ParamDef(name="x", kind="float", low=0.0, high=1.0, default=0.5)
    study = _truncated_study()
    for t in range(20):
        v = pd.sample(study.ask())
        assert 0.0 <= v <= 1.0
        study.tell(study.trials[-1].number, 0.0)


def test_paramdef_int_sample_in_range() -> None:
    pd = ParamDef(name="x", kind="int", low=1, high=5, default=3)
    study = _truncated_study()
    for t in range(20):
        v = pd.sample(study.ask())
        assert isinstance(v, int)
        assert 1 <= v <= 5


def test_paramdef_loguniform() -> None:
    pd = ParamDef(name="lr", kind="loguniform", low=1e-4, high=1e-1, default=3e-4)
    study = _truncated_study()
    seen = set()
    for t in range(30):
        v = pd.sample(study.ask())
        assert v > 0
        seen.add(v)
        study.tell(study.trials[-1].number, 0.0)
    assert len(seen) > 5


def test_paramdef_categorical() -> None:
    pd = ParamDef(name="bs", kind="categorical",
                  choices=(16, 32, 64), default=32)
    study = _truncated_study()
    for t in range(10):
        v = pd.sample(study.ask())
        assert v in (16, 32, 64)
        study.tell(study.trials[-1].number, 0.0)


def test_paramdef_unknown_kind_raises() -> None:
    pd = ParamDef(name="x", kind="weird")
    study = _truncated_study()
    with pytest.raises(ValueError):
        pd.sample(study.ask())


def test_categorical_requires_choices() -> None:
    pd = ParamDef(name="x", kind="categorical", choices=None)
    study = _truncated_study()
    with pytest.raises(ValueError):
        pd.sample(study.ask())


def test_search_space_duplicate_names_raises() -> None:
    with pytest.raises(ValueError):
        SearchSpace(params=[
            ParamDef(name="x", kind="float", low=0.0, high=1.0),
            ParamDef(name="x", kind="float", low=2.0, high=3.0),
        ])


def test_search_space_sample_returns_all_params() -> None:
    space = get_space("s4_ppo")
    study = _truncated_study()
    p = space.sample(study.ask())
    assert set(p.keys()) == set(space.names())
    for v in p.values():
        assert v is not None
    study.tell(study.trials[-1].number, 0.0)


def test_search_space_iter_and_len() -> None:
    space = get_space("bc")
    assert len(space) == 5
    assert all(isinstance(p, ParamDef) for p in space)
    assert [p.name for p in space] == space.names()


def test_get_space_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_space("not_a_real_space")


def test_list_spaces_includes_all_trainers() -> None:
    names = list_spaces()
    for required in ("s4_ppo", "s4_cvar", "s6_dt", "s7_wm",
                     "s2b_bc", "s2b_dagger", "portfolio_ppo"):
        assert required in names


def test_dagger_space_includes_bc_params() -> None:
    dagger = get_space("dagger")
    bc = get_space("bc")
    for p in bc:
        assert p.name in dagger.names()
    assert "n_rounds" in dagger.names()
    assert "rollout_episodes_per_round" in dagger.names()


def test_s4_cvar_extends_ppo() -> None:
    s = get_space("s4_cvar")
    ppo = get_space("s4_ppo")
    for p in ppo:
        assert p.name in s.names()
    assert "cvar_alpha" in s.names()
    assert "cvar_lr" in s.names()
