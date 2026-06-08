"""Configuration loader: YAML files with deep-merge and dotted overrides.

The loader is intentionally minimal (no Hydra/OmegaConf dependency).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import yaml

from zhisa.utils.containers import Config


PathLike = Union[str, os.PathLike]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`; override wins on scalars."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_dotted_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Turn ['a.b.c=1', 'd=2'] into {'a': {'b': {'c': 1}}, 'd': 2}."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        path, _, raw = item.partition("=")
        path = path.strip()
        if not path:
            raise ValueError(f"Empty key in override: {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cur = out
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(f"Override path conflict at {p!r} in {item!r}")
        cur[parts[-1]] = value
    return out


def load_config(
    paths: Union[PathLike, Iterable[PathLike]],
    overrides: Optional[Iterable[str]] = None,
) -> Config:
    """Load and merge a list of YAML config files, then apply dotted overrides.

    Later files in `paths` override earlier ones. Overrides are dot-notation
    keys with `=value` (YAML-parsed).
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    merged: Dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, _read_yaml(Path(p)))
    if overrides:
        override_dict = _parse_dotted_overrides(overrides)
        merged = deep_merge(merged, override_dict)
    return Config(data=merged)
