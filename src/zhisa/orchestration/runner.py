"""End-to-end pipeline runner: executes stages in subprocesses.

The runner is intentionally thin — its only job is to translate a
:class:`Pipeline` into a sequence of subprocess invocations, in
topological order, capture their output, and assemble a JSON
manifest.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from zhisa.orchestration.pipeline import (
    Pipeline,
    Stage,
    render_stage_cli,
    toposort,
)
from zhisa.utils.logging import get_logger


_LOG = get_logger(__name__)


@dataclass
class StageResult:
    """The captured result of running a single :class:`Stage`."""

    name: str
    entry: str
    returncode: int
    elapsed_s: float
    cmd: list[str]
    stdout_tail: str = ""
    stderr_tail: str = ""
    checkpoint: str | None = None
    checkpoint_exists: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """A persisted record of a pipeline run."""

    pipeline_source: str | None
    seed: int
    device: str
    artifacts_dir: str
    n_stages: int
    n_succeeded: int
    n_failed: int
    total_elapsed_s: float
    started_at: float
    finished_at: float
    stages: list[StageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d


class PipelineRunner:
    """Execute a :class:`Pipeline` stage by stage."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        repo_root: Optional[Path] = None,
        extra_env: Optional[Mapping[str, str]] = None,
        tail_lines: int = 20,
        fail_fast: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.repo_root = Path(repo_root or Path(pipeline.source).parent if pipeline.source else Path.cwd())
        self.extra_env = dict(extra_env or {})
        self.tail_lines = int(tail_lines)
        self.fail_fast = bool(fail_fast)
        self._outputs: dict[str, Path] = {}

    def _entry_to_module(self, entry: str) -> tuple[str, str]:
        """Map a console-script entry to a ``(executable, module)`` pair.

        Special prefixes:

        * ``zhisa-<name>``  -> ``(python, zhisa.scripts.<name>)``
        * ``python:<path>`` -> ``(python, <path>)`` (run a script directly)
        * ``<dotted>``      -> ``(python, <dotted>)`` (assume ``-m``)
        * anything else     -> ``(python, <entry>)``
        """
        if entry.startswith("zhisa-"):
            return (sys.executable, "zhisa.scripts." + entry[len("zhisa-"):].replace("-", "_"))
        if entry.startswith("python:"):
            return (sys.executable, entry[len("python:"):])
        return (sys.executable, entry)

    def _resolve_arg(self, v: Any, outputs: dict[str, Path]) -> Any:
        """Substitute ``{stage_name}`` placeholders with resolved paths."""
        if isinstance(v, str) and "{" in v:
            for name, path in outputs.items():
                v = v.replace("{" + name + "}", str(path))
        return v

    def _build_cmd(self, stage: Stage) -> list[str]:
        """Build the subprocess command list for one stage."""
        executable, mod = self._entry_to_module(stage.entry)
        # Resolve {stage_name} placeholders against already-completed stages.
        resolved_args = {
            k: self._resolve_arg(v, self._outputs) for k, v in stage.args.items()
        }
        # Render into a positional CLI.
        stage_resolved = Stage(
            name=stage.name, entry=stage.entry, config=stage.config,
            args=resolved_args, output_checkpoint=stage.output_checkpoint,
            depends_on=stage.depends_on, timeout_s=stage.timeout_s,
        )
        cli = render_stage_cli(
            stage_resolved, config_arg="--config",
            checkpoint_arg="--checkpoint",
        )
        # If the entry is a real .py file, run it directly. Otherwise
        # treat it as a ``-m`` module.
        if mod.endswith(".py") or ("\\" in mod) or ("/" in mod):
            return [executable, mod, *cli]
        return [executable, "-m", mod, *cli]

    def _run_one(self, cmd: list[str], stage: Stage,
                 cwd: Path, env: dict[str, str]) -> StageResult:
        _LOG.info("running stage=%s cmd=%s", stage.name, " ".join(shlex.quote(c) for c in cmd))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env,
                capture_output=True, text=True,
                timeout=stage.timeout_s,
            )
            elapsed = time.perf_counter() - t0
            stdout_tail = "\n".join(proc.stdout.splitlines()[-self.tail_lines:])
            stderr_tail = "\n".join(proc.stderr.splitlines()[-self.tail_lines:])
            checkpoint_path = self.pipeline.checkpoint_path(stage)
            res = StageResult(
                name=stage.name, entry=stage.entry,
                returncode=proc.returncode, elapsed_s=elapsed,
                cmd=cmd, stdout_tail=stdout_tail, stderr_tail=stderr_tail,
                checkpoint=str(checkpoint_path),
                checkpoint_exists=checkpoint_path.exists(),
                error=None if proc.returncode == 0 else f"exit={proc.returncode}",
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            res = StageResult(
                name=stage.name, entry=stage.entry,
                returncode=-1, elapsed_s=elapsed,
                cmd=cmd,
                stdout_tail=(exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                stderr_tail=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                checkpoint=str(self.pipeline.checkpoint_path(stage)),
                checkpoint_exists=False,
                error=f"timeout after {exc.timeout}s",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            res = StageResult(
                name=stage.name, entry=stage.entry,
                returncode=-2, elapsed_s=elapsed,
                cmd=cmd, error=f"runner exception: {exc!r}",
            )
        _LOG.info("stage=%s done returncode=%s elapsed=%.2fs",
                  stage.name, res.returncode, res.elapsed_s)
        return res

    def _setup_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("ZHISA_TEST_DEVICE", str(self.pipeline.device))
        env.setdefault("ZHISA_FAST_RENDER", "1")
        env.setdefault("PYTHONHASHSEED", str(self.pipeline.seed))
        env.update({k: str(v) for k, v in self.extra_env.items()})
        return env

    def run(self) -> RunManifest:
        """Execute all stages in topological order. Returns a manifest."""
        Path(self.pipeline.artifacts_dir).mkdir(parents=True, exist_ok=True)
        env = self._setup_env()
        cwd = self.repo_root
        order = toposort(self.pipeline.stages)
        started = time.time()
        results: list[StageResult] = []
        n_succeeded = 0
        n_failed = 0
        for stage in order:
            cmd = self._build_cmd(stage)
            res = self._run_one(cmd, stage, cwd, env)
            results.append(res)
            if res.returncode == 0:
                n_succeeded += 1
                self._outputs[stage.name] = self.pipeline.checkpoint_path(stage)
            else:
                n_failed += 1
                if self.fail_fast:
                    _LOG.error("stage=%s failed; stopping pipeline (fail_fast=True)",
                               stage.name)
                    break
        finished = time.time()
        manifest = RunManifest(
            pipeline_source=self.pipeline.source,
            seed=self.pipeline.seed, device=self.pipeline.device,
            artifacts_dir=self.pipeline.artifacts_dir,
            n_stages=len(order), n_succeeded=n_succeeded, n_failed=n_failed,
            total_elapsed_s=finished - started,
            started_at=started, finished_at=finished,
            stages=results,
        )
        return manifest

    def run_and_persist(self, manifest_path: str | Path) -> RunManifest:
        """Run the pipeline and write a JSON manifest to disk."""
        manifest = self.run()
        out = Path(manifest_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        return manifest
