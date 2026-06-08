"""End-to-end pipeline orchestration for ZHISA.

The :mod:`zhisa.orchestration` package lets you wire multiple
trainers (S1 → S2 → S2b → S4 → S6 → S7 → S-portfolio) into a
single declarative YAML and execute them as a directed acyclic
graph. Each stage runs in a subprocess, its output checkpoint
becomes available to dependent stages, and a JSON manifest records
the full run.

Public surface:

* :class:`Stage`     — a single training stage (entry, config, args).
* :class:`Pipeline`  — a list of stages with dependency edges.
* :func:`toposort`   — Kahn's algorithm topological order.
* :class:`PipelineRunner` — executes the pipeline in subprocesses.
* :func:`load_pipeline`  — load a :class:`Pipeline` from a YAML file.
"""
from __future__ import annotations

from zhisa.orchestration.pipeline import (
    Pipeline,
    Stage,
    load_pipeline,
    toposort,
    validate_dag,
)
from zhisa.orchestration.runner import (
    PipelineRunner,
    StageResult,
    RunManifest,
)

__all__ = [
    "Stage",
    "Pipeline",
    "PipelineRunner",
    "StageResult",
    "RunManifest",
    "load_pipeline",
    "toposort",
    "validate_dag",
]
