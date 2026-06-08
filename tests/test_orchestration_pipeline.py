"""Unit tests for zhisa.orchestration.pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from zhisa.orchestration.pipeline import (
    Pipeline,
    Stage,
    load_pipeline,
    render_stage_cli,
    toposort,
    validate_dag,
)


def _stage(name: str, *, deps: tuple[str, ...] = (),
           args: dict | None = None,
           output: str | None = None,
           config: str | None = None) -> Stage:
    return Stage(
        name=name, entry=f"zhisa-train-{name}",
        config=config if config is not None else f"configs/{name}.yaml",
        args=args or {},
        output_checkpoint=output or f"{name}.pt",
        depends_on=deps,
    )


def test_stage_default_checkpoint_filename() -> None:
    s = Stage(name="s1", entry="zhisa-train-s1")
    assert s.checkpoint_filename() == "s1.pt"


def test_stage_custom_checkpoint_filename() -> None:
    s = Stage(name="s1", entry="zhisa-train-s1", output_checkpoint="custom.pt")
    assert s.checkpoint_filename() == "custom.pt"


def test_validate_dag_accepts_linear() -> None:
    stages = [_stage("a"), _stage("b", deps=("a",)), _stage("c", deps=("b",))]
    validate_dag(stages)  # no raise


def test_validate_dag_accepts_diamond() -> None:
    stages = [
        _stage("a"),
        _stage("b", deps=("a",)),
        _stage("c", deps=("a",)),
        _stage("d", deps=("b", "c")),
    ]
    validate_dag(stages)


def test_validate_dag_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_dag([_stage("a"), _stage("a")])


def test_validate_dag_rejects_self_dependency() -> None:
    with pytest.raises(ValueError, match="depends on itself"):
        validate_dag([_stage("a", deps=("a",))])


def test_validate_dag_rejects_unknown_dep() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        validate_dag([_stage("a", deps=("zz",))])


def test_validate_dag_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        validate_dag([_stage("a", deps=("b",)), _stage("b", deps=("a",))])


def test_toposort_linear() -> None:
    stages = [_stage("a"), _stage("b", deps=("a",)), _stage("c", deps=("b",))]
    order = toposort(stages)
    assert [s.name for s in order] == ["a", "b", "c"]


def test_toposort_diamond() -> None:
    stages = [
        _stage("a"),
        _stage("b", deps=("a",)),
        _stage("c", deps=("a",)),
        _stage("d", deps=("b", "c")),
    ]
    order = toposort(stages)
    names = [s.name for s in order]
    assert names[0] == "a"
    assert names[-1] == "d"
    assert names.index("b") < names.index("d")
    assert names.index("c") < names.index("d")


def test_toposort_independent() -> None:
    stages = [_stage("a"), _stage("b"), _stage("c")]
    order = toposort(stages)
    assert {s.name for s in order} == {"a", "b", "c"}


def test_toposort_deterministic_with_ties() -> None:
    stages = [_stage("z"), _stage("a"), _stage("m")]
    order = toposort(stages)
    assert [s.name for s in order] == ["a", "m", "z"]


def test_render_stage_cli_basic() -> None:
    s = _stage("s1", args={"bars": 2000, "epochs": 3})
    cli = render_stage_cli(s, config_arg="--config", checkpoint_arg="--checkpoint")
    assert "--config" in cli
    assert "configs/s1.yaml" in cli
    assert "--checkpoint" in cli
    assert "s1.pt" in cli
    assert "--bars" in cli
    assert "2000" in cli
    assert "--epochs" in cli
    assert "3" in cli


def test_render_stage_cli_boolean_flags() -> None:
    s_true = _stage("s1", args={"flag": True})
    s_false = _stage("s1", args={"flag": False})
    cli_true = render_stage_cli(s_true)
    cli_false = render_stage_cli(s_false)
    assert "--flag" in cli_true
    assert "--flag" not in cli_false


def test_render_stage_cli_list_value() -> None:
    s = _stage("s1", args={"a_list": [1, 2, 3]})
    cli = render_stage_cli(s)
    assert cli.count("--a_list") == 1
    assert "1" in cli and "2" in cli and "3" in cli


def test_render_stage_cli_no_config() -> None:
    s = Stage(name="s1", entry="zhisa-train-s1", config=None,
              output_checkpoint="s1.pt", args={})
    cli = render_stage_cli(s, config_arg="--config")
    assert "--config" not in cli


def test_pipeline_checkpoint_path() -> None:
    p = Pipeline(stages=[_stage("a")], artifacts_dir="artifacts/x")
    s = p.stage_by_name("a")
    assert p.checkpoint_path(s) == Path("artifacts/x/a.pt")


def test_pipeline_to_dict() -> None:
    p = Pipeline(stages=[_stage("a"), _stage("b", deps=("a",))])
    d = p.to_dict()
    assert d["seed"] == 0
    assert len(d["stages"]) == 2
    assert d["stages"][1]["depends_on"] == ["a"]


def test_load_pipeline_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        "seed: 7\n"
        "device: cuda\n"
        "artifacts_dir: out\n"
        "stages:\n"
        "  - name: a\n"
        "    entry: zhisa-train-s1\n"
        "    config: c1.yaml\n"
        "    args: {bars: 100}\n"
        "    output_checkpoint: a.pt\n"
        "  - name: b\n"
        "    entry: zhisa-train-s2\n"
        "    depends_on: [a]\n"
        "    output_checkpoint: b.pt\n"
    )
    pipe = load_pipeline(p)
    assert pipe.seed == 7
    assert pipe.device == "cuda"
    assert pipe.artifacts_dir == "out"
    assert [s.name for s in pipe.stages] == ["a", "b"]
    assert pipe.stages[1].depends_on == ("a",)
    assert pipe.stages[0].args == {"bars": 100}


def test_load_pipeline_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_pipeline("/no/such/file.yaml")


def test_load_pipeline_bad_top_level() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("- not\n- a\n- dict\n")
        name = f.name
    try:
        with pytest.raises(ValueError, match="mapping"):
            load_pipeline(name)
    finally:
        Path(name).unlink()


def test_load_pipeline_bad_stage_type() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("stages: [not_a_dict]\n")
        name = f.name
    try:
        with pytest.raises(ValueError, match="mapping"):
            load_pipeline(name)
    finally:
        Path(name).unlink()
