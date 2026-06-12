"""Tests for the S3 synthetic curriculum trainer.

The tests cover, in order of complexity:
    - config helpers (default_curriculum, iter_curriculum, to_market_config)
    - :class:`StageMetrics` and :class:`CurriculumResult` serialisation
    - internal helpers (regime distribution, mix_markets)
    - input-shape validation
    - the trainer's per-stage and full-curriculum behaviour
    - end-to-end runs with both the S1 and S2 inner trainers
    - checkpoint persistence
    - determinism under a fixed base seed
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.s1_ssl import SSLPretrainer, SSLConfig
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.s3_curriculum import (
    CurriculumResult,
    CurriculumStage,
    CurriculumTrainer,
    StageMetrics,
    _mix_markets,
    _regime_label_distribution,
    default_curriculum,
    iter_curriculum,
)
from zhisa.training.optim import OptimConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_spec() -> SampleSpec:
    return SampleSpec(
        chart_window=16, image_size=16,
        horizons=(4, 8), n_regime_states=3,
    )


@pytest.fixture
def probe_model(small_spec):
    """A model built with the correct feature dim for the small spec."""
    df = generate_market(MarketConfig(n_bars=200, seed=0))
    ds = MarketDataset(df, spec=small_spec)
    return build_default_policy(
        in_numeric_features=ds._features.shape[1],
        in_context_features=ds._time_features.shape[1],
        window=small_spec.chart_window, image_size=small_spec.image_size,
        n_actions=9, n_regime_classes=small_spec.n_regime_states,
    )


def _ssl_factory():
    """Return a factory that produces a fresh S1 trainer for the model."""
    def factory(model):
        return SSLPretrainer(model, SSLConfig(
            epochs=1, batch_size=8, lr=1e-3, device="cpu", log_every=1000,
        ))
    return factory


def _s2_factory():
    """Return a factory that produces a fresh S2 trainer for the model."""
    def factory(model):
        loss = MultiTaskLoss(LossWeights())
        return SupervisedTrainer(
            model, loss, TrainConfig(
                epochs=1, batch_size=8, device="cpu", log_every=1000,
                optim=OptimConfig(lr=1e-3),
            ),
        )
    return factory


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def test_default_curriculum_has_three_stages():
    cs = default_curriculum()
    assert len(cs) == 3
    assert [s.name for s in cs] == ["clean", "mixed", "stressed"]


def test_default_curriculum_progresses_in_difficulty():
    """Each stage must be strictly harder than the previous one in vol/shocks/tails."""
    cs = default_curriculum()
    for prev, cur in zip(cs, cs[1:]):
        assert cur.base_vol > prev.base_vol
        assert cur.shock_prob > prev.shock_prob
        assert cur.student_t_df < prev.student_t_df


def test_curriculum_stage_to_market_config():
    stage = CurriculumStage("custom", n_bars=500, base_vol=0.7,
                            shock_prob=0.001, student_t_df=10.0)
    cfg = stage.to_market_config(seed=123)
    assert isinstance(cfg, MarketConfig)
    assert cfg.n_bars == 500
    assert cfg.base_vol == 0.7
    assert cfg.shock_prob == 0.001
    assert cfg.student_t_df == 10.0
    assert cfg.seed == 123


def test_iter_curriculum_yields_name_config_pairs():
    pairs = iter_curriculum()
    assert len(pairs) == 3
    for name, cfg in pairs:
        assert isinstance(name, str)
        assert isinstance(cfg, MarketConfig)


# ---------------------------------------------------------------------------
# StageMetrics / CurriculumResult
# ---------------------------------------------------------------------------


def test_stage_metrics_as_dict():
    m = StageMetrics(
        stage="clean", n_bars=1000, epochs=1,
        final_loss=0.5, best_loss=0.4, elapsed_s=12.3,
        extra={"regime_dist": {"regime_0": 200}},
    )
    d = m.as_dict()
    assert d["stage"] == "clean"
    assert d["final_loss"] == 0.5
    assert d["regime_dist"] == {"regime_0": 200}


def test_curriculum_result_as_frame():
    stages = [
        StageMetrics("a", 100, 1, 0.5, 0.4, 1.0),
        StageMetrics("b", 100, 1, 0.3, 0.2, 1.0),
    ]
    res = CurriculumResult(stages=stages, final_loss=0.3)
    df = res.as_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["stage"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_regime_label_distribution_with_regime_col():
    df = pd.DataFrame({"regime": [0, 1, 1, 2, 2, 2]})
    dist = _regime_label_distribution(df)
    assert dist == {"regime_0": 1, "regime_1": 2, "regime_2": 3}


def test_regime_label_distribution_without_regime_col_returns_empty():
    df = pd.DataFrame({"close": [1, 2, 3]})
    assert _regime_label_distribution(df) == {}


def test_mix_markets_zero_fraction_returns_primary():
    primary = generate_market(MarketConfig(n_bars=200, seed=1))
    secondary = generate_market(MarketConfig(n_bars=100, seed=2))
    mixed = _mix_markets(primary, secondary, fraction=0.0)
    assert len(mixed) == len(primary)
    assert list(mixed.columns) == list(primary.columns)


def test_mix_markets_partial_fraction_concatenates():
    primary = generate_market(MarketConfig(n_bars=200, seed=1))
    secondary = generate_market(MarketConfig(n_bars=100, seed=2))
    mixed = _mix_markets(primary, secondary, fraction=0.5)
    # primary unchanged; secondary contributes ~50 of its 100 rows.
    assert len(mixed) >= len(primary)
    assert len(mixed) <= len(primary) + 60  # rough upper bound
    # Sorted by index.
    assert mixed.index.is_monotonic_increasing


def test_mix_markets_secondary_offset_avoids_overlap():
    """The secondary block should start *after* primary ends, not overlap."""
    primary = generate_market(MarketConfig(n_bars=100, seed=1))
    secondary = generate_market(MarketConfig(n_bars=50, seed=2))
    mixed = _mix_markets(primary, secondary, fraction=1.0)
    # No overlapping timestamps between the two blocks.
    assert len(mixed) == len(primary) + len(secondary)
    assert mixed.index.is_monotonic_increasing
    # First secondary timestamp is strictly after last primary timestamp.
    assert mixed.index[len(primary)] > primary.index[-1]


def test_mix_markets_fraction_clamped_to_unit_interval():
    primary = generate_market(MarketConfig(n_bars=100, seed=1))
    secondary = generate_market(MarketConfig(n_bars=50, seed=2))
    mixed = _mix_markets(primary, secondary, fraction=5.0)
    # All 50 secondary rows included (fraction=5.0 clamps to 1.0).
    assert len(mixed) == len(primary) + len(secondary)


def test_mix_markets_empty_secondary_returns_primary():
    primary = generate_market(MarketConfig(n_bars=100, seed=1))
    empty = primary.iloc[:0]
    mixed = _mix_markets(primary, empty, fraction=0.5)
    assert len(mixed) == len(primary)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_trainer_raises_on_wrong_n_features(small_spec):
    bad = build_default_policy(
        in_numeric_features=99,  # wrong
        in_context_features=6,
        window=16, image_size=16, n_actions=9, n_regime_classes=3,
    )
    with pytest.raises(ValueError, match="in_numeric_features"):
        CurriculumTrainer(
            _ssl_factory(), bad, stages=[
                CurriculumStage("clean", n_bars=200, epochs=1),
            ],
            sample_spec=small_spec, base_seed=0,
        )


def test_trainer_raises_on_wrong_n_context(small_spec):
    bad = build_default_policy(
        in_numeric_features=probe_model_features(small_spec),
        in_context_features=99,  # wrong
        window=16, image_size=16, n_actions=9, n_regime_classes=3,
    )
    with pytest.raises(ValueError, match="in_context_features"):
        CurriculumTrainer(
            _ssl_factory(), bad, stages=[
                CurriculumStage("clean", n_bars=200, epochs=1),
            ],
            sample_spec=small_spec, base_seed=0,
        )


def probe_model_features(spec):
    df = generate_market(MarketConfig(n_bars=200, seed=0))
    ds = MarketDataset(df, spec=spec)
    return ds._features.shape[1]


# ---------------------------------------------------------------------------
# Trainer behaviour
# ---------------------------------------------------------------------------


def test_trainer_single_stage_runs(probe_model, small_spec):
    ct = CurriculumTrainer(
        _ssl_factory(), probe_model,
        stages=[CurriculumStage("only", n_bars=200, epochs=1)],
        sample_spec=small_spec, base_seed=0,
    )
    result = ct.fit()
    assert len(result.stages) == 1
    assert result.stages[0].stage == "only"
    assert result.stages[0].elapsed_s > 0


def test_trainer_multi_stage_records_per_stage_metrics(probe_model, small_spec):
    stages = [
        CurriculumStage("a", n_bars=200, base_vol=0.4, shock_prob=0.0, student_t_df=20.0, epochs=1),
        CurriculumStage("b", n_bars=200, base_vol=0.6, shock_prob=0.001, student_t_df=8.0, epochs=1, mix_with_previous=0.0),
        CurriculumStage("c", n_bars=200, base_vol=0.9, shock_prob=0.002, student_t_df=4.0, epochs=1, mix_with_previous=0.0),
    ]
    ct = CurriculumTrainer(_ssl_factory(), probe_model, stages=stages,
                            sample_spec=small_spec, base_seed=0)
    result = ct.fit()
    assert [s.stage for s in result.stages] == ["a", "b", "c"]
    assert all(s.elapsed_s > 0 for s in result.stages)
    assert result.final_loss == result.stages[-1].final_loss


def test_trainer_with_mix_increases_n_bars(probe_model, small_spec):
    """With mix_with_previous > 0, the stage's n_bars should exceed the base."""
    stages = [
        CurriculumStage("a", n_bars=300, base_vol=0.4, epochs=1),
        CurriculumStage("b", n_bars=300, base_vol=0.6, epochs=1, mix_with_previous=0.3),
    ]
    ct = CurriculumTrainer(_ssl_factory(), probe_model, stages=stages,
                            sample_spec=small_spec, base_seed=0)
    result = ct.fit()
    assert result.stages[0].n_bars == 300
    assert result.stages[1].n_bars >= 300  # at least the primary


def test_trainer_zero_epochs_skips_inner_trainer(probe_model, small_spec):
    """A stage with epochs=0 must skip the inner trainer entirely."""
    stages = [
        CurriculumStage("noop", n_bars=100, base_vol=0.4, epochs=0),
    ]
    ct = CurriculumTrainer(_ssl_factory(), probe_model, stages=stages,
                            sample_spec=small_spec, base_seed=0)
    result = ct.fit()
    assert result.stages[0].final_loss == 0.0  # empty loss curve default
    assert result.stages[0].best_loss == 0.0


def test_trainer_writes_checkpoint(probe_model, small_spec, tmp_path):
    ckpt = tmp_path / "curriculum.pt"
    stages = [CurriculumStage("only", n_bars=200, epochs=1)]
    ct = CurriculumTrainer(_ssl_factory(), probe_model, stages=stages,
                            sample_spec=small_spec, base_seed=0,
                            checkpoint=str(ckpt))
    ct.fit()
    assert ckpt.exists()
    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert "model" in payload
    assert "stages" in payload
    assert len(payload["stages"]) == 1


# ---------------------------------------------------------------------------
# Inner-trainer interop
# ---------------------------------------------------------------------------


def test_curriculum_with_s2_supervised_trainer(probe_model, small_spec):
    """The curriculum must work with the S2 supervised trainer, not just S1."""
    stages = [
        CurriculumStage("clean", n_bars=200, base_vol=0.4, epochs=1),
        CurriculumStage("mixed", n_bars=200, base_vol=0.6, epochs=1),
    ]
    ct = CurriculumTrainer(_s2_factory(), probe_model, stages=stages,
                            sample_spec=small_spec, base_seed=0)
    result = ct.fit()
    assert all(s.final_loss == s.final_loss for s in result.stages)  # all finite


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_curriculum_is_deterministic_under_seed(small_spec):
    """Same seed + same init + same trainer state → same per-stage losses."""
    stages = [
        CurriculumStage("a", n_bars=200, base_vol=0.4, epochs=1),
        CurriculumStage("b", n_bars=200, base_vol=0.6, epochs=1, mix_with_previous=0.0),
    ]

    def run_once():
        # Build a fresh model with the correct feature dims.
        df = generate_market(MarketConfig(n_bars=200, seed=0))
        ds = MarketDataset(df, spec=small_spec)
        m = build_default_policy(
            in_numeric_features=ds._features.shape[1],
            in_context_features=ds._time_features.shape[1],
            window=small_spec.chart_window, image_size=small_spec.image_size,
            n_actions=9, n_regime_classes=small_spec.n_regime_states,
        )
        torch.manual_seed(42)
        ct = CurriculumTrainer(_ssl_factory(), m, stages=stages,
                                sample_spec=small_spec, base_seed=42)
        return [s.final_loss for s in ct.fit().stages]

    a = run_once()
    b = run_once()
    assert a == b
