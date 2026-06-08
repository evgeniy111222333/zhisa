"""Optuna study orchestration: pruners, storage, persistence, summary."""
from __future__ import annotations

import math
import sqlite3
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler

from zhisa.hpopt.objective import ObjectiveFn, ObjectiveResult
from zhisa.hpopt.search_space import SearchSpace


PRUNERS: dict[str, Callable[[], optuna.pruners.BasePruner]] = {
    "none": lambda: NopPruner(),
    "median": lambda: MedianPruner(n_warmup_steps=2, interval_steps=1),
    "hyperband": lambda: HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3),
}

SAMPLERS: dict[str, Callable[[int], optuna.samplers.BaseSampler]] = {
    "random": lambda seed: RandomSampler(seed=seed),
    "tpe": lambda seed: TPESampler(seed=seed, n_startup_trials=4),
}


@dataclass
class StudySummary:
    """A small, json-friendly snapshot of an Optuna study.

    Intended for CI logs and ``artifacts/hpopt/study.json`` dumps.
    """

    study_name: str
    direction: str
    n_trials: int
    n_complete: int
    n_pruned: int
    best_value: float
    best_params: dict[str, Any]
    best_trial_number: int
    storage_path: Optional[str] = None
    sampler: str = "tpe"
    pruner: str = "none"
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "n_trials": self.n_trials,
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "best_trial_number": self.best_trial_number,
            "storage_path": self.storage_path,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "history": self.history,
        }


class OptunaRunner:
    """Thin orchestrator around :class:`optuna.Study`."""

    def __init__(
        self,
        direction: str = "maximize",
        sampler: str = "tpe",
        pruner: str = "none",
        seed: int = 0,
        storage: Optional[str] = None,
        study_name: str = "zhisa_hpopt",
        show_warnings: bool = False,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        if sampler not in SAMPLERS:
            raise ValueError(f"Unknown sampler {sampler!r}. Available: {sorted(SAMPLERS)}")
        if pruner not in PRUNERS:
            raise ValueError(f"Unknown pruner {pruner!r}. Available: {sorted(PRUNERS)}")
        self.direction = direction
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.seed = int(seed)
        self.study_name = study_name
        self.storage_path = self._resolve_storage(storage)
        self.show_warnings = show_warnings

    @staticmethod
    def _resolve_storage(storage: Optional[str]) -> Optional[str]:
        if storage is None:
            return None
        if storage == ":memory:":
            return "sqlite:///:memory:"
        if storage.startswith("sqlite://") or storage.startswith("mysql://") or storage.startswith("postgresql://"):
            return storage
        p = Path(storage)
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p.as_posix()}"

    def _make_study(self) -> optuna.Study:
        sampler = SAMPLERS[self.sampler_name](self.seed)
        pruner = PRUNERS[self.pruner_name]()
        if not self.show_warnings:
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        kw: dict[str, Any] = {
            "direction": self.direction,
            "sampler": sampler,
            "pruner": pruner,
            "study_name": self.study_name,
        }
        if self.storage_path is not None:
            kw["storage"] = self.storage_path
            kw["load_if_exists"] = True
        return optuna.create_study(**kw)

    def _run_one_trial(self, study: optuna.Study, objective: ObjectiveFn,
                        base_cfg: dict[str, Any]) -> ObjectiveResult | None:
        def _wrapped(trial: optuna.Trial) -> float:
            try:
                res = objective(trial, base_cfg)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as exc:
                # Treat uncaught exceptions as a worst-case trial so the
                # pruner / sampler can learn to avoid that region.
                trial.set_user_attr("error", repr(exc)[:500])
                return float("-inf") if self.direction == "maximize" else float("inf")
            for k, v in (res.params or {}).items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("elapsed_s", float(res.elapsed_s))
            return float(res.value)
        study.optimize(_wrapped, n_trials=1, catch=(Exception,))
        if not study.trials:
            return None
        return ObjectiveResult(value=study.trials[-1].value or float("nan"),
                               history=[], params=study.trials[-1].params,
                               elapsed_s=study.trials[-1].user_attrs.get("elapsed_s", 0.0))

    def run(self, space: SearchSpace, objective: ObjectiveFn,
            base_cfg: Optional[dict[str, Any]] = None,
            n_trials: int = 10, timeout: Optional[float] = None) -> StudySummary:
        """Run ``n_trials`` evaluations of ``objective`` and return a summary.

        The ``base_cfg`` dict is forwarded to the objective verbatim; the
        convention is that it always contains a ``"space"`` key (the
        :class:`SearchSpace` used by the objective).
        """
        bc = dict(base_cfg or {})
        bc["space"] = space
        study = self._make_study()
        study.optimize(
            lambda t: self._objective_wrapper(t, objective, bc),
            n_trials=int(n_trials),
            timeout=timeout,
            catch=(Exception,),
        )
        return self._summarise(study)

    @staticmethod
    def _objective_wrapper(trial: optuna.Trial, objective: ObjectiveFn,
                           base_cfg: dict[str, Any]) -> float:
        try:
            res = objective(trial, base_cfg)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", repr(exc)[:500])
            return float("-inf")
        for k, v in (res.params or {}).items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("elapsed_s", float(res.elapsed_s))
        return float(res.value)

    def _summarise(self, study: optuna.Study) -> StudySummary:
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        best = study.best_trial if complete else None
        history: list[dict[str, Any]] = []
        for t in study.trials:
            history.append({
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "params": t.params,
                "attrs": {k: v for k, v in t.user_attrs.items()
                          if isinstance(v, (int, float, str, bool))},
            })
        return StudySummary(
            study_name=self.study_name,
            direction=self.direction,
            n_trials=len(study.trials),
            n_complete=len(complete),
            n_pruned=len(pruned),
            best_value=float(best.value) if best else float("nan"),
            best_params=dict(best.params) if best else {},
            best_trial_number=int(best.number) if best else -1,
            storage_path=self.storage_path,
            sampler=self.sampler_name,
            pruner=self.pruner_name,
            history=history,
        )
