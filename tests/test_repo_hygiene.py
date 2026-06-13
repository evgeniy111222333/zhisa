"""Repository hygiene checks for source files that must be versioned."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def test_source_env_package_is_not_gitignored() -> None:
    if not (REPO / ".git").exists():
        pytest.skip("not running inside a git checkout")

    result = subprocess.run(
        ["git", "check-ignore", "src/zhisa/env/trading_env.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout + result.stderr
