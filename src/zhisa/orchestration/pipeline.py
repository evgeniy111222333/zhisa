"""Pipeline / DAG model for end-to-end orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class Stage:
    """A single training stage.

    Attributes:
        name: unique identifier within the pipeline.
        entry: console-script entry point (e.g. ``"zhisa-train-s1"``)
            or a module path (e.g. ``"zhisa.scripts.train_s1"``).
        config: path to the YAML config (relative to the pipeline
            file or absolute).
        args: extra CLI flags forwarded as ``--key value`` pairs.
            ``True`` / ``False`` becomes a flag with no value.
            String values can contain ``{stage_name}`` placeholders
            that resolve to the dependent stage's checkpoint path.
        output_checkpoint: file name under the pipeline's
            ``artifacts_dir``; defaults to ``f"{name}.pt"``.
        depends_on: names of stages that must complete first.
        timeout_s: per-stage timeout in seconds (None = no limit).
    """

    name: str
    entry: str
    config: str | None = None
    args: Mapping[str, Any] = field(default_factory=dict)
    output_checkpoint: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    timeout_s: float | None = None

    def checkpoint_filename(self) -> str:
        return self.output_checkpoint or f"{self.name}.pt"


@dataclass
class Pipeline:
    """A list of :class:`Stage` objects with shared global config."""

    stages: list[Stage]
    seed: int = 0
    device: str = "cpu"
    artifacts_dir: str = "artifacts/pipeline"
    python: str = "python"
    source: str | None = None  # path the pipeline was loaded from

    def stage_by_name(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"no stage named {name!r}")

    def checkpoint_path(self, stage: Stage) -> Path:
        return Path(self.artifacts_dir) / stage.checkpoint_filename()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "artifacts_dir": self.artifacts_dir,
            "python": self.python,
            "source": self.source,
            "stages": [
                {
                    "name": s.name, "entry": s.entry, "config": s.config,
                    "args": dict(s.args), "output_checkpoint": s.output_checkpoint,
                    "depends_on": list(s.depends_on), "timeout_s": s.timeout_s,
                }
                for s in self.stages
            ],
        }


def validate_dag(stages: Sequence[Stage]) -> None:
    """Raise :class:`ValueError` on cycles, unknown deps, or duplicate names."""
    names = [s.name for s in stages]
    if len(names) != len(set(names)):
        dups = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate stage names: {dups}")
    name_set = set(names)
    for s in stages:
        for dep in s.depends_on:
            if dep not in name_set:
                raise ValueError(
                    f"stage {s.name!r} depends on unknown stage {dep!r}"
                )
    for s in stages:
        if s.name in s.depends_on:
            raise ValueError(f"stage {s.name!r} depends on itself")
    # Cycle detection via DFS colouring.
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in names}
    succ = {s.name: list(s.depends_on) for s in stages}

    def dfs(u: str) -> None:
        if color[u] == GREY:
            raise ValueError(f"cycle detected involving stage {u!r}")
        if color[u] == BLACK:
            return
        color[u] = GREY
        for v in succ[u]:
            dfs(v)
        color[u] = BLACK

    for n in names:
        if color[n] == WHITE:
            dfs(n)


def toposort(stages: Sequence[Stage]) -> list[Stage]:
    """Return a topological order over ``stages``.

    Raises :class:`ValueError` on cycles. Order is deterministic
    (Kahn's algorithm, ties broken by declaration order).
    """
    validate_dag(stages)
    name_to_stage = {s.name: s for s in stages}
    in_deg: dict[str, int] = {s.name: 0 for s in stages}
    succ: dict[str, list[str]] = {s.name: [] for s in stages}
    for s in stages:
        for d in s.depends_on:
            succ[d].append(s.name)
            in_deg[s.name] += 1
    ready = [n for n, d in in_deg.items() if d == 0]
    out: list[Stage] = []
    while ready:
        ready.sort()  # deterministic
        u = ready.pop(0)
        out.append(name_to_stage[u])
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                ready.append(v)
    if len(out) != len(stages):
        raise ValueError("toposort failed (likely a cycle)")
    return out


def _coerce_arg_value(v: Any) -> list[str]:
    """Convert a YAML value to a list of CLI tokens.

    * ``True``  -> ``[]`` (boolean flag, only key emitted later)
    * ``False`` -> filtered out (skip the flag)
    * list/tuple -> repeated ``--key item1 --key item2``
    * scalar  -> ``[str(v)]``
    """
    if v is False:
        return ["__SKIP__"]
    if v is True:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def render_stage_cli(stage: Stage, *, config_arg: str | None = None,
                     checkpoint_arg: str | None = None) -> list[str]:
    """Render a stage into a list of CLI tokens (no executable prefix).

    The render uses only stable string conversions so the result is
    suitable for unit testing.

    Special args key ``__pos__`` is rendered as bare positional tokens
    (no leading ``--``), useful for command entry points such as
    ``python script.py --flag value`` where the script path is
    positional.
    """
    tokens: list[str] = []
    if config_arg is not None and stage.config:
        tokens += [config_arg, stage.config]
    if checkpoint_arg is not None and stage.output_checkpoint:
        tokens += [checkpoint_arg, stage.checkpoint_filename()]
    args = dict(stage.args)
    pos = args.pop("__pos__", None)
    if pos is not None:
        if isinstance(pos, (list, tuple)):
            tokens.extend(str(x) for x in pos)
        else:
            tokens.append(str(pos))
    for k, v in args.items():
        rendered = _coerce_arg_value(v)
        if rendered == ["__SKIP__"]:
            continue
        if not rendered:
            tokens.append(f"--{k}")
        else:
            tokens.append(f"--{k}")
            tokens.extend(rendered)
    return tokens


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a :class:`Pipeline` from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pipeline file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level pipeline YAML must be a mapping, got {type(data).__name__}")
    stages_raw = data.get("stages", []) or []
    if not isinstance(stages_raw, list):
        raise ValueError("'stages' must be a list")
    stages: list[Stage] = []
    for s in stages_raw:
        if not isinstance(s, dict):
            raise ValueError(f"each stage must be a mapping, got {type(s).__name__}")
        stages.append(Stage(
            name=str(s["name"]),
            entry=str(s["entry"]),
            config=s.get("config"),
            args=dict(s.get("args", {}) or {}),
            output_checkpoint=s.get("output_checkpoint"),
            depends_on=tuple(s.get("depends_on", []) or ()),
            timeout_s=s.get("timeout_s"),
        ))
    return Pipeline(
        stages=stages,
        seed=int(data.get("seed", 0)),
        device=str(data.get("device", "cpu")),
        artifacts_dir=str(data.get("artifacts_dir", "artifacts/pipeline")),
        python=str(data.get("python", "python")),
        source=str(p),
    )
