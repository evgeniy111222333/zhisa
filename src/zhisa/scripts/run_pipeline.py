"""End-to-end pipeline orchestrator CLI.

Usage::

    python -m zhisa.scripts.run_pipeline --config configs/pipeline_default.yaml
    python -m zhisa.scripts.run_pipeline --config my_pipeline.yaml \\
        --out artifacts/pipeline/manifest.json --no-fail-fast

The pipeline YAML declares a list of stages and their dependencies.
Stages run as subprocesses in topological order, each in its own
working directory (the repo root), and the resulting manifest is
written to ``--out`` (or to ``artifacts/pipeline/manifest.json`` by
default).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from zhisa.orchestration import PipelineRunner, load_pipeline
from zhisa.utils.logging import get_logger


_LOG = get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a multi-stage ZHISA training pipeline.",
    )
    p.add_argument("--config", type=str, required=True,
                   help="YAML pipeline file.")
    p.add_argument("--out", type=str, default=None,
                   help="Manifest output path (default: <artifacts_dir>/manifest.json).")
    p.add_argument("--no-fail-fast", action="store_true",
                   help="Continue to the next stage even when a stage fails.")
    p.add_argument("--repo-root", type=str, default=None,
                   help="Override the working directory for stage subprocesses.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pipeline = load_pipeline(args.config)
    if not pipeline.stages:
        _LOG.warning("pipeline has no stages; exiting")
        return 0
    out: Optional[Path] = None
    if args.out:
        out = Path(args.out)
    elif pipeline.artifacts_dir:
        out = Path(pipeline.artifacts_dir) / "manifest.json"
    repo_root = Path(args.repo_root) if args.repo_root else None
    runner = PipelineRunner(
        pipeline, repo_root=repo_root, fail_fast=not args.no_fail_fast,
    )
    manifest = runner.run_and_persist(out or Path("manifest.json"))
    _LOG.info("pipeline complete: %d/%d stages succeeded, total=%.2fs",
              manifest.n_succeeded, manifest.n_stages, manifest.total_elapsed_s)
    if out is not None:
        _LOG.info("manifest written to: %s", str(out))
    return 0 if manifest.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
