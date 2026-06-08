"""Subprocess smoke tests for zhisa.scripts.hpopt."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args: list[str], env: dict | None = None, timeout: float = 120.0) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    full_env.setdefault("ZHISA_TEST_DEVICE", "cpu")
    full_env.setdefault("ZHISA_FAST_RENDER", "1")
    if env:
        full_env.update(env)
    return subprocess.run(
        [PYTHON, "-m", "zhisa.scripts.hpopt", *args],
        cwd=str(REPO), env=full_env,
        capture_output=True, text=True, timeout=timeout,
    )


def test_hpopt_list_smoke() -> None:
    p = _run(["--list"], timeout=20.0)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "s4_ppo" in out
    assert "portfolio_ppo" in out


def test_hpopt_unknown_trainer_falls_back() -> None:
    """Invalid trainer surfaces argparse error, exits non-zero."""
    p = _run(["--trainer", "not_a_real_trainer", "--n-trials", "1"], timeout=10.0)
    assert p.returncode != 0


def test_hpopt_s2b_bc_smoke(tmp_path: Path) -> None:
    """End-to-end: 2 trials, BC objective, summary.json on disk."""
    out = tmp_path / "summary.json"
    p = _run([
        "--trainer", "s2b_bc",
        "--n-trials", "2",
        "--n-bars", "200",
        "--sampler", "random",
        "--pruner", "none",
        "--out", str(out),
        "--quiet",
    ], timeout=120.0)
    assert p.returncode == 0, f"stdout={p.stdout!r}\nstderr={p.stderr!r}"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 2
    assert "lr" in payload["best_params"]


def test_hpopt_s4_ppo_smoke(tmp_path: Path) -> None:
    """End-to-end: 2 trials, S4 PPO objective."""
    out = tmp_path / "s4_summary.json"
    p = _run([
        "--trainer", "s4_ppo",
        "--n-trials", "2",
        "--n-bars", "300",
        "--n-episodes", "1",
        "--max-steps", "30",
        "--sampler", "random",
        "--out", str(out),
        "--quiet",
    ], timeout=120.0)
    assert p.returncode == 0, f"stdout={p.stdout!r}\nstderr={p.stderr!r}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 2
    assert "clip_ratio" in payload["best_params"]


def test_hpopt_resume_via_storage(tmp_path: Path) -> None:
    """Two runs share the same SQLite storage and accumulate trials."""
    db = tmp_path / "hp.db"
    base = [
        "--trainer", "s2b_bc",
        "--n-bars", "200",
        "--sampler", "random",
        "--storage", str(db),
        "--quiet",
    ]
    p1 = _run([*base, "--n-trials", "1", "--out", str(tmp_path / "s1.json")],
              timeout=120.0)
    assert p1.returncode == 0, p1.stderr
    p2 = _run([*base, "--n-trials", "1", "--out", str(tmp_path / "s2.json")],
              timeout=120.0)
    assert p2.returncode == 0, p2.stderr
    s1 = json.loads((tmp_path / "s1.json").read_text())
    s2 = json.loads((tmp_path / "s2.json").read_text())
    assert s2["n_trials"] == s1["n_trials"] + 1
