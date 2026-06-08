"""Unit tests for zhisa.orchestration.runner.PipelineRunner."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.orchestration import Pipeline, PipelineRunner, Stage
from zhisa.orchestration.runner import RunManifest, StageResult


def _echo_args_stage(name: str, marker: str, output: str) -> Stage:
    """A stage that runs ``python -c '...print(marker); write file'``."""
    return Stage(
        name=name, entry=marker,
        args={"marker": marker, "output": output},
        output_checkpoint=Path(output).name,
    )


def test_runner_class_basic_construction(tmp_path: Path) -> None:
    pipe = Pipeline(stages=[
        Stage(name="a", entry="zhisa-train-s1"),
    ], artifacts_dir=str(tmp_path))
    r = PipelineRunner(pipe, repo_root=tmp_path)
    assert r.pipeline is pipe
    assert r.fail_fast is True


def test_entry_to_module_maps_dashes() -> None:
    pipe = Pipeline(stages=[])
    r = PipelineRunner(pipe)
    assert r._entry_to_module("zhisa-train-s1") == (sys.executable, "zhisa.scripts.train_s1")
    assert r._entry_to_module("zhisa.scripts.train_s1") == (sys.executable, "zhisa.scripts.train_s1")


def test_entry_to_module_python_prefix() -> None:
    pipe = Pipeline(stages=[])
    r = PipelineRunner(pipe)
    assert r._entry_to_module("python:/abs/s.py") == (sys.executable, "/abs/s.py")


def test_resolve_arg_substitutes_placeholder() -> None:
    pipe = Pipeline(stages=[])
    r = PipelineRunner(pipe)
    s1_path = str(Path("artifacts") / "s1.pt")
    out = {"s1": Path(s1_path)}
    assert r._resolve_arg("{s1}", out) == s1_path
    assert r._resolve_arg("prefix/{s1}/suffix", out) == f"prefix/{s1_path}/suffix"
    assert r._resolve_arg("plain", out) == "plain"
    assert r._resolve_arg(42, out) == 42


def test_run_one_captures_success(tmp_path: Path) -> None:
    pipe = Pipeline(stages=[
        Stage(name="ok", entry="zhisa-train-s1", output_checkpoint="ok.pt"),
    ], artifacts_dir=str(tmp_path))
    r = PipelineRunner(pipe, repo_root=tmp_path)

    # Synthesise a successful subprocess call by using python -c
    cmd = [sys.executable, "-c", "import sys; print('hello from stage')"]
    stage = pipe.stages[0]
    res = r._run_one(cmd, stage, cwd=tmp_path, env=None)
    assert res.returncode == 0
    assert "hello" in res.stdout_tail
    assert res.elapsed_s >= 0.0
    assert res.error is None


def test_run_one_captures_failure(tmp_path: Path) -> None:
    pipe = Pipeline(stages=[
        Stage(name="bad", entry="zhisa-train-s1", output_checkpoint="bad.pt"),
    ], artifacts_dir=str(tmp_path))
    r = PipelineRunner(pipe, repo_root=tmp_path)
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    stage = pipe.stages[0]
    res = r._run_one(cmd, stage, cwd=tmp_path, env=None)
    assert res.returncode == 3
    assert "boom" in res.stderr_tail
    assert res.error == "exit=3"


def test_run_one_captures_timeout(tmp_path: Path) -> None:
    pipe = Pipeline(stages=[
        Stage(name="slow", entry="zhisa-train-s1", output_checkpoint="s.pt",
              timeout_s=0.5),
    ], artifacts_dir=str(tmp_path))
    r = PipelineRunner(pipe, repo_root=tmp_path)
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    res = r._run_one(cmd, pipe.stages[0], cwd=tmp_path, env=None)
    assert res.returncode == -1
    assert "timeout" in (res.error or "")


def test_run_end_to_end_with_echo_stages(tmp_path: Path) -> None:
    """A real subprocess-based end-to-end run with two echo stages.

    We monkey-patch ``_build_cmd`` to a simple ``python -c '...'`` command
    so the test does not depend on the actual zhisa-train-* entry points.
    """
    s1 = Stage(
        name="s1",
        entry="zhisa-train-s1",
        args={"--marker": "M1", "--out": str(tmp_path / "s1.txt")},
        output_checkpoint="s1.txt",
    )
    s2 = Stage(
        name="s2",
        entry="zhisa-train-s2",
        args={"--out": str(tmp_path / "s2.txt")},
        output_checkpoint="s2.txt",
        depends_on=("s1",),
    )
    pipe = Pipeline(
        stages=[s1, s2],
        artifacts_dir=str(tmp_path / "art"),
    )
    r = PipelineRunner(pipe, repo_root=tmp_path)

    def _fake_cmd(stage: Stage) -> list[str]:
        if stage.name == "s1":
            return [sys.executable, "-c",
                    f"open(r'{tmp_path / 's1.txt'}', 'w').write('M1'); print('s1 done')"]
        return [sys.executable, "-c",
                f"open(r'{tmp_path / 's2.txt'}', 'w').write("
                f"open(r'{tmp_path / 's1.txt'}').read() + '+s2')"]

    r._build_cmd = _fake_cmd  # type: ignore[assignment]
    manifest = r.run()
    assert manifest.n_stages == 2
    assert manifest.n_succeeded == 2
    assert manifest.n_failed == 0
    assert (tmp_path / "s1.txt").read_text() == "M1"
    assert (tmp_path / "s2.txt").read_text() == "M1+s2"
    # Outputs dict must contain resolved paths.
    assert "s1" in r._outputs
    assert str(r._outputs["s1"]).endswith("s1.txt")


def test_run_fail_fast_stops_pipeline(tmp_path: Path) -> None:
    """When a stage fails and fail_fast=True, subsequent stages don't run."""
    s1 = Stage(
        name="bad", entry="zhisa-train-s1",
        args={}, output_checkpoint="bad.txt",
    )
    s2 = Stage(
        name="ok", entry="zhisa-train-s1",
        args={}, output_checkpoint="ok.txt",
        depends_on=("bad",),
    )
    pipe = Pipeline(stages=[s1, s2], artifacts_dir=str(tmp_path / "art"))
    r = PipelineRunner(pipe, repo_root=tmp_path, fail_fast=True)

    def _fake_cmd(stage: Stage) -> list[str]:
        if stage.name == "bad":
            return [sys.executable, "-c", "import sys; sys.exit(7)"]
        return [sys.executable, "-c", "print('should not run')"]

    r._build_cmd = _fake_cmd  # type: ignore[assignment]
    manifest = r.run()
    assert manifest.n_succeeded == 0
    assert manifest.n_failed == 1
    assert len(manifest.stages) == 1  # s2 was skipped


def test_run_no_fail_fast_continues(tmp_path: Path) -> None:
    s1 = Stage(
        name="bad", entry="zhisa-train-s1",
        args={}, output_checkpoint="bad.txt",
    )
    s2 = Stage(
        name="ok", entry="zhisa-train-s1",
        args={}, output_checkpoint="ok.txt",
        depends_on=("bad",),
    )
    pipe = Pipeline(stages=[s1, s2], artifacts_dir=str(tmp_path / "art"))
    r = PipelineRunner(pipe, repo_root=tmp_path, fail_fast=False)

    def _fake_cmd(stage: Stage) -> list[str]:
        if stage.name == "bad":
            return [sys.executable, "-c", "import sys; sys.exit(7)"]
        return [sys.executable, "-c", "print('hello')"]

    r._build_cmd = _fake_cmd  # type: ignore[assignment]
    manifest = r.run()
    assert manifest.n_failed == 1
    assert manifest.n_succeeded == 1
    assert len(manifest.stages) == 2


def test_run_and_persist_writes_json(tmp_path: Path) -> None:
    s1 = Stage(
        name="ok", entry="zhisa-train-s1",
        args={}, output_checkpoint="ok.txt",
    )
    pipe = Pipeline(stages=[s1], artifacts_dir=str(tmp_path / "art"))
    r = PipelineRunner(pipe, repo_root=tmp_path)
    r._build_cmd = lambda stage: [sys.executable, "-c", "print('hello')"]  # type: ignore[assignment]
    out = tmp_path / "manifest.json"
    manifest = r.run_and_persist(out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_stages"] == 1
    assert payload["n_succeeded"] == 1
    assert "stages" in payload


def test_manifest_to_dict_is_jsonable(tmp_path: Path) -> None:
    s1 = Stage(name="ok", entry="zhisa-train-s1",
               args={}, output_checkpoint="ok.txt")
    pipe = Pipeline(stages=[s1], artifacts_dir=str(tmp_path / "art"))
    r = PipelineRunner(pipe, repo_root=tmp_path)
    r._build_cmd = lambda stage: [sys.executable, "-c", "print('hi')"]  # type: ignore[assignment]
    m = r.run()
    json.dumps(m.to_dict(), default=str)  # no raise
