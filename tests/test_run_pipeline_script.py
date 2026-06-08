"""Smoke tests for the orchestrator CLI script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run_cli(args: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "zhisa.scripts.run_pipeline", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )


def test_cli_help(tmp_path: Path) -> None:
    proc = _run_cli(["--help"], cwd=tmp_path, timeout=20)
    assert proc.returncode == 0
    assert "pipeline" in proc.stdout.lower() or "stage" in proc.stdout.lower()


def test_cli_missing_config(tmp_path: Path) -> None:
    proc = _run_cli(["--config", "no_such.yaml"], cwd=tmp_path, timeout=20)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "not found" in combined.lower() or "no such" in combined.lower()


def test_cli_runs_minimal_pipeline(tmp_path: Path) -> None:
    """Run a real two-stage pipeline via the CLI: each stage invokes
    a tiny Python helper that writes its output file."""
    art = tmp_path / "art"
    art.mkdir()
    cfg = tmp_path / "pipe.yaml"
    s1_out = str(art / "s1.pt")
    s2_out = str(art / "s2.pt")
    helper = REPO_ROOT / "tests" / "_support" / "echo_helper.py"
    cfg.write_text(
        f"seed: 1\n"
        f"device: cpu\n"
        f"artifacts_dir: {art.as_posix()}\n"
        f"stages:\n"
        f"  - name: s1\n"
        f"    entry: 'python:{helper.as_posix()}'\n"
        f"    config: null\n"
        f"    args:\n"
        f"      out: {s1_out}\n"
        f"      text: s1-data\n"
        f"    output_checkpoint: s1.pt\n"
        f"  - name: s2\n"
        f"    entry: 'python:{helper.as_posix()}'\n"
        f"    config: null\n"
        f"    args:\n"
        f"      out: {s2_out}\n"
        f"      text: s2-data\n"
        f"    output_checkpoint: s2.pt\n"
        f"    depends_on: [s1]\n"
    )
    manifest_path = tmp_path / "manifest.json"
    proc = _run_cli(
        ["--config", str(cfg), "--out", str(manifest_path)],
        cwd=tmp_path, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["n_stages"] == 2
    assert payload["n_succeeded"] == 2
    assert payload["n_failed"] == 0
    assert (art / "s1.pt").exists()
    assert (art / "s2.pt").exists()


def test_cli_fail_fast_returns_nonzero(tmp_path: Path) -> None:
    art = tmp_path / "art"
    art.mkdir()
    helper = tmp_path / "fail.py"
    helper.write_text("import sys; sys.stderr.write('boom'); sys.exit(7)\n")
    cfg = tmp_path / "pipe.yaml"
    cfg.write_text(
        f"seed: 1\n"
        f"device: cpu\n"
        f"artifacts_dir: {art.as_posix()}\n"
        f"stages:\n"
        f"  - name: bad\n"
        f"    entry: 'python:{helper.as_posix()}'\n"
        f"    config: null\n"
        f"    args: {{}}\n"
        f"    output_checkpoint: bad.pt\n"
    )
    proc = _run_cli(
        ["--config", str(cfg), "--out", str(tmp_path / "m.json")],
        cwd=tmp_path, timeout=20,
    )
    assert proc.returncode != 0
    # Manifest still gets written even on failure.
    assert (tmp_path / "m.json").exists()
