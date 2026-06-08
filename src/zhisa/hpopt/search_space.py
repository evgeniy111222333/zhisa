"""Declarative search-space definitions backed by Optuna ``Trial``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import optuna


@dataclass(frozen=True)
class ParamDef:
    """A single hyperparameter specification.

    ``kind`` controls how the value is sampled:

    * ``"float"``  — uniform real in ``[low, high]`` (linear scale)
    * ``"loguniform"`` — log-uniform real in ``[low, high]`` (use for LRs)
    * ``"int"``   — uniform integer in ``[low, high]`` (inclusive)
    * ``"categorical"`` — one of ``choices``
    """

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    default: Any = None

    def sample(self, trial: optuna.Trial) -> Any:
        """Sample a value from this parameter using ``trial``."""
        if self.kind == "float":
            return trial.suggest_float(self.name, self.low, self.high, log=self.log)
        if self.kind == "loguniform":
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"ParamDef {self.name!r}: categorical needs 'choices'")
            return trial.suggest_categorical(self.name, list(self.choices))
        raise ValueError(f"ParamDef {self.name!r}: unknown kind {self.kind!r}")


@dataclass
class SearchSpace:
    """An ordered collection of :class:`ParamDef` objects.

    Use :meth:`sample` to pull one value per param from an Optuna
    ``Trial``, and :meth:`to_dict` for read-only inspection (e.g. for
    logging or unit tests that don't want a Trial).
    """

    params: list[ParamDef] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"SearchSpace: duplicate param name(s): {names}")

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample every parameter and return a name->value dict."""
        return {p.name: p.sample(trial) for p in self.params}

    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    def __getitem__(self, idx: int) -> ParamDef:
        return self.params[idx]


def _make_space(params: Sequence[tuple]) -> SearchSpace:
    """Build a ``SearchSpace`` from a compact tuple list.

    Each tuple is one of:

    * ``(name, "categorical", choices, default)``
    * ``(name, "loguniform" | "float" | "int", low, high, default)``
    """
    out: list[ParamDef] = []
    for spec in params:
        kind = spec[1]
        name = spec[0]
        if kind == "categorical":
            _, _, choices, default = spec
            out.append(ParamDef(name=name, kind=kind, choices=tuple(choices), default=default))
        elif kind in {"loguniform", "float", "int"}:
            _, _, low, high, default = spec
            out.append(ParamDef(name=name, kind=kind, low=low, high=high, default=default))
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return SearchSpace(params=out)


def _bc_space() -> SearchSpace:
    return _make_space([
        ("lr", "loguniform", 1e-5, 1e-2, 3e-4),
        ("weight_decay", "loguniform", 1e-6, 1e-1, 1e-2),
        ("batch_size", "categorical", (16, 32, 64, 128), 32),
        ("epochs", "int", 1, 6, 3),
        ("grad_clip", "float", 0.1, 5.0, 1.0),
    ])


def _dagger_space() -> SearchSpace:
    base = _bc_space().params
    extra = _make_space([
        ("n_rounds", "int", 1, 5, 3),
        ("rollout_episodes_per_round", "int", 1, 4, 2),
    ]).params
    return SearchSpace(params=base + extra)


def _ppo_space() -> SearchSpace:
    return _make_space([
        ("lr", "loguniform", 1e-5, 1e-2, 3e-4),
        ("weight_decay", "loguniform", 1e-6, 1e-1, 1e-2),
        ("clip_ratio", "float", 0.05, 0.4, 0.2),
        ("value_coef", "float", 0.1, 1.0, 0.5),
        ("entropy_coef", "loguniform", 1e-4, 1e-1, 1e-2),
        ("gamma", "float", 0.9, 0.999, 0.99),
        ("gae_lambda", "float", 0.8, 0.99, 0.95),
        ("minibatch_size", "categorical", (16, 32, 64), 32),
        ("n_epochs", "int", 1, 8, 4),
        ("target_kl", "float", 0.01, 0.2, 0.05),
    ])


def _s4_cvar_space() -> SearchSpace:
    base = _ppo_space().params
    extra = _make_space([
        ("cvar_alpha", "float", 0.05, 0.5, 0.1),
        ("cvar_lambda_init", "float", 0.0, 5.0, 1.0),
        ("cvar_lr", "loguniform", 1e-4, 1e-1, 5e-3),
        ("cvar_max", "float", 1.0, 20.0, 10.0),
    ]).params
    return SearchSpace(params=base + extra)


def _dt_space() -> SearchSpace:
    return _make_space([
        ("lr", "loguniform", 1e-5, 1e-2, 3e-4),
        ("weight_decay", "loguniform", 1e-6, 1e-1, 1e-2),
        ("context_len", "categorical", (8, 16, 32), 16),
        ("batch_size", "categorical", (8, 16, 32), 16),
        ("epochs", "int", 1, 5, 2),
        ("rtg_coef", "float", 0.0, 1.0, 0.1),
    ])


def _world_model_space() -> SearchSpace:
    return _make_space([
        ("lr", "loguniform", 1e-5, 1e-2, 3e-4),
        ("weight_decay", "loguniform", 1e-6, 1e-1, 1e-2),
        ("state_coef", "float", 0.5, 2.0, 1.0),
        ("reward_coef", "float", 0.5, 2.0, 1.0),
        ("done_coef", "float", 0.0, 0.5, 0.1),
        ("rollout_horizon", "categorical", (4, 8, 16), 8),
        ("epochs", "int", 1, 5, 2),
    ])


def _portfolio_ppo_space() -> SearchSpace:
    base = _ppo_space().params
    extra = _make_space([
        ("max_gross_leverage", "float", 1.0, 4.0, 1.5),
        ("n_instruments", "categorical", (2, 3, 4), 3),
    ]).params
    return SearchSpace(params=base + extra)


_SPACES: dict[str, Callable[[], SearchSpace]] = {
    "bc": _bc_space,
    "dagger": _dagger_space,
    "s2b_bc": _bc_space,
    "s2b_dagger": _dagger_space,
    "s4_ppo": _ppo_space,
    "s4_cvar": _s4_cvar_space,
    "s6_dt": _dt_space,
    "s7_wm": _world_model_space,
    "portfolio_ppo": _portfolio_ppo_space,
}


def get_space(name: str) -> SearchSpace:
    """Return a fresh copy of the named built-in search space.

    Raises ``KeyError`` for unknown names; ``list_spaces()`` shows
    the available ones.
    """
    if name not in _SPACES:
        raise KeyError(
            f"Unknown search space {name!r}. Available: {sorted(_SPACES)}"
        )
    return _SPACES[name]()


def list_spaces() -> list[str]:
    return sorted(_SPACES)
