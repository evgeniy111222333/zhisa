"""Run a hyperparameter search with Optuna over a ZHISA trainer.

Usage::

    python -m zhisa.scripts.hpopt --trainer s4_ppo --n-trials 20 \\
        --storage artifacts/hpopt/s4_ppo.db \\
        --out artifacts/hpopt/s4_ppo_summary.json

    python -m zhisa.scripts.hpopt --trainer s2b_bc --n-trials 5 \\
        --seed 0 --device cpu

The ``--trainer`` flag picks a built-in (objective, search-space)
pair. The full list of available trainers is printed by ``--list``.

The base config is taken from the matching YAML in ``configs/`` (if
present) and may be overridden with ``--n-bars``, ``--n-episodes``,
``--max-steps``, ``--device`` flags so that you can shrink the
budget for a smoke run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

from zhisa.config import load_config
from zhisa.hpopt import (
    OptunaRunner,
    bc_objective,
    dagger_objective,
    dt_objective,
    get_space,
    list_spaces,
    ppo_objective,
    portfolio_ppo_objective,
    s4_cvar_objective,
    world_model_objective,
)
from zhisa.utils.logging import get_logger


_TRAINER_REGISTRY: dict[str, Callable] = {
    "s2b_bc": bc_objective,
    "s2b_dagger": dagger_objective,
    "s4_ppo": ppo_objective,
    "s4_cvar": s4_cvar_objective,
    "s6_dt": dt_objective,
    "s7_wm": world_model_objective,
    "portfolio_ppo": portfolio_ppo_objective,
}

_DEFAULT_CONFIGS: dict[str, str] = {
    "s2b_bc": "configs/s2b_imitation.yaml",
    "s2b_dagger": "configs/s2b_imitation.yaml",
    "s4_ppo": "configs/s4_rl.yaml",
    "s4_cvar": "configs/s4_cvar_ppo.yaml",
    "s6_dt": "configs/s6_dt.yaml",
    "s7_wm": "configs/s7_world_model.yaml",
    "portfolio_ppo": "configs/portfolio_rl.yaml",
}

_LOG = get_logger(__name__)


def _default_device() -> str:
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Optuna hyperparameter search.")
    p.add_argument("--trainer", type=str, choices=sorted(_TRAINER_REGISTRY),
                   default="s4_ppo", help="Trainer / objective to optimise.")
    p.add_argument("--space", type=str, default=None,
                   help="Override the search-space name (default: same as --trainer).")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--timeout", type=float, default=None,
                   help="Wall-clock timeout in seconds (optional).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--storage", type=str, default=None,
                   help="SQLite path (omit for in-memory).")
    p.add_argument("--sampler", type=str, choices=("tpe", "random"), default="tpe")
    p.add_argument("--pruner", type=str, choices=("none", "median", "hyperband"),
                   default="none")
    p.add_argument("--out", type=str, default="artifacts/hpopt/summary.json",
                   help="Where to write the JSON summary.")
    p.add_argument("--config", type=str, default=None,
                   help="Override base YAML config (default: per-trainer).")
    p.add_argument("--n-bars", type=int, default=None)
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--list", action="store_true",
                   help="Print registered trainers and exit.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress Optuna's per-trial logging.")
    return p


def _build_base_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = Path(args.config or _DEFAULT_CONFIGS.get(args.trainer, ""))
    base: dict[str, Any] = {}
    if cfg_path.exists():
        cfg_obj = load_config(cfg_path)
        if hasattr(cfg_obj, "to_dict"):
            base = cfg_obj.to_dict()
        elif isinstance(cfg_obj, dict):
            base = dict(cfg_obj)
        else:
            base = {}
    if args.n_bars is not None:
        base["n_bars"] = int(args.n_bars)
    if args.n_episodes is not None:
        base["n_episodes"] = int(args.n_episodes)
    if args.max_steps is not None:
        base["max_steps_per_episode"] = int(args.max_steps)
    base["device"] = args.device or _default_device()
    base.setdefault("seed", int(args.seed))
    return base


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.list:
        for name in sorted(_TRAINER_REGISTRY):
            print(f"{name:>20s}   space={name}")
        print(f"available spaces: {list_spaces()}")
        return 0

    objective = _TRAINER_REGISTRY[args.trainer]
    space_name = args.space or args.trainer
    space = get_space(space_name)
    base_cfg = _build_base_cfg(args)

    runner = OptunaRunner(
        direction="maximize",
        sampler=args.sampler,
        pruner=args.pruner,
        seed=int(args.seed),
        storage=args.storage,
        study_name=f"zhisa_{args.trainer}",
        show_warnings=False,
    )
    if args.quiet:
        optuna_log = logging.getLogger("optuna")
        optuna_log.setLevel(logging.WARNING)

    _LOG.info("starting hpopt trainer=%s n_trials=%d space=%s",
              args.trainer, int(args.n_trials), space_name)
    summary = runner.run(space, objective, base_cfg=base_cfg,
                         n_trials=int(args.n_trials), timeout=args.timeout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=float)
    _LOG.info("done best_value=%.6f best_params=%s",
              summary.best_value, summary.best_params)
    _LOG.info("summary written to: %s", str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
