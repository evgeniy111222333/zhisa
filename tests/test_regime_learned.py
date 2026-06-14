"""Tests for outcome-learned regime intelligence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from zhisa.backtest.splitter import SplitSpec
from zhisa.regime import (
    ChampionChallengerReport,
    LearnedRegimeLoss,
    LearnedRegimeModel,
    LearnedRegimeModelConfig,
    RegimeDecisionAdapter,
    RegimeFeatureEngine,
    RegimeFeatureEngineConfig,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeModelCandidate,
    RegimeModelTrainConfig,
    RegimeModelTrainer,
    RegimeModelWalkForwardConfig,
    RegimeOutcomeDataset,
    RegimeOutcomeDatasetConfig,
    RuleRegimePrior,
    build_regime_model_registry,
    calibrate_learned_regime_model,
    predict_learned_regime,
    regime_outcome_collate,
    run_regime_model_walk_forward,
)
from zhisa.training.optim import OptimConfig


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.full(close.size, float(volume)) if np.isscalar(volume) else np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def _mixed_df(n: int = 180) -> pd.DataFrame:
    up = np.linspace(100.0, 122.0, n // 2)
    down = np.linspace(122.0, 92.0, n - len(up))
    close = np.r_[up, down] + 0.25 * np.sin(np.arange(n) / 3)
    volume = np.r_[np.full(len(up), 100.0), np.full(len(down), 320.0)]
    return _ohlcv_from_close(close, volume=volume)


def _dataset_cfg() -> RegimeOutcomeDatasetConfig:
    return RegimeOutcomeDatasetConfig(
        horizons=(3, 6),
        stride=8,
        min_history=48,
        symbol="BTC/USDT",
        analyzer=RegimeIntelligenceConfig(timeframes=("5m", "15m")),
    )


def test_feature_engine_is_causal_and_has_no_decision_fields() -> None:
    df = _mixed_df()
    engine = RegimeFeatureEngine(RegimeFeatureEngineConfig(timeframes=("5m", "15m")))

    full_at_t = engine.snapshot(df, t=80, symbol="BTC/USDT")
    truncated = engine.snapshot(df.iloc[:81], symbol="BTC/USDT")

    assert full_at_t.to_dict() == truncated.to_dict()
    assert "primary_regime" not in full_at_t.to_dict()
    assert "aggregate" in full_at_t.to_dict()


def test_rule_regime_prior_marks_weak_prior_metadata() -> None:
    prior = RuleRegimePrior()
    report = prior.analyze(_mixed_df(), symbol="BTC/USDT")

    assert report.features["inference_source"] == "rule_prior"
    assert report.features["weak_prior"] is True


def test_regime_outcome_dataset_uses_causal_features_and_future_labels() -> None:
    df = _mixed_df()
    ds = RegimeOutcomeDataset(df, _dataset_cfg())
    item = ds[0]
    t = int(item.meta["t"])
    report = RegimeIntelligence(_dataset_cfg().analyzer).analyze(df.iloc[: t + 1], symbol="BTC/USDT")

    assert len(ds) > 0
    assert item.rule_report.features["aggregate"] == report.features["aggregate"]
    assert item.return_targets.shape == (2,)
    assert item.volatility_targets.shape == (2,)
    assert item.drawdown_targets.shape == (2,)
    assert item.mfe_targets.shape == (2,)
    assert item.playbook_utility.ndim == 1
    assert item.meta["horizons"] == (3, 6)
    assert torch.isfinite(item.transition_event)


def test_learned_regime_model_outputs_expected_heads_and_loss_backpropagates() -> None:
    ds = RegimeOutcomeDataset(_mixed_df(), _dataset_cfg())
    batch = regime_outcome_collate([ds[0], ds[1], ds[2], ds[3]])
    model = LearnedRegimeModel(LearnedRegimeModelConfig(hidden_dim=32, latent_dim=10, n_horizons=2, dropout=0.0))
    loss_fn = LearnedRegimeLoss(quantiles=model.cfg.return_quantiles)

    out = model(batch.x)
    losses = loss_fn(out, batch)
    losses["total"].backward()

    assert out["latent_regime_embedding"].shape == (4, 10)
    assert out["return_quantiles"].shape == (4, 2, len(model.cfg.return_quantiles))
    assert out["volatility_forecast"].shape == (4, 2)
    assert out["drawdown_quantiles"].shape == (4, 2)
    assert out["playbook_utility"].shape[0] == 4
    assert torch.all((out["transition_hazard"] >= 0.0) & (out["transition_hazard"] <= 1.0))
    assert losses["total"].item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_regime_model_trainer_reduces_tiny_training_loss_and_calibrates() -> None:
    ds = RegimeOutcomeDataset(_mixed_df(210), _dataset_cfg())
    model = LearnedRegimeModel(LearnedRegimeModelConfig(hidden_dim=48, latent_dim=12, n_horizons=2, dropout=0.0))
    trainer = RegimeModelTrainer(
        model,
        RegimeModelTrainConfig(
            epochs=3,
            batch_size=6,
            device="cpu",
            optim=OptimConfig(lr=3e-3, scheduler="none", warmup_steps=0),
        ),
    )

    result = trainer.fit(ds, val_ds=ds)
    calibration, metrics = calibrate_learned_regime_model(model, ds)

    assert result["history"][-1]["total"] <= result["history"][0]["total"]
    assert metrics["transition_brier_after"] <= metrics["transition_brier_before"] + 1e-9
    assert metrics["tradeability_brier_after"] <= metrics["tradeability_brier_before"] + 1e-9
    assert calibration.version == "intercept_v1"


def test_decision_adapter_and_regime_intelligence_hybrid_mode() -> None:
    df = _mixed_df()
    rule = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(df)
    model = LearnedRegimeModel(LearnedRegimeModelConfig(hidden_dim=32, latent_dim=8, n_horizons=2, dropout=0.0))
    prediction = predict_learned_regime(model, rule)
    adapted = RegimeDecisionAdapter().adapt(rule, prediction)
    hybrid = RegimeIntelligence(
        RegimeIntelligenceConfig(timeframes=("5m", "15m"), inference_mode="hybrid"),
        learned_model=model,
    ).analyze(df)

    assert adapted.features["inference_source"] == "hybrid"
    assert "model_outputs" in adapted.features
    assert "rule_prior" in adapted.features
    assert "guardrails" in adapted.explanation
    assert hybrid.features["inference_source"] == "hybrid"
    assert 0.0 <= hybrid.tradeability_score <= 1.0
    assert 0.0 <= hybrid.transition_risk <= 1.0


def test_rule_only_fallback_adds_inference_metadata() -> None:
    report = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"), inference_mode="hybrid")).analyze(_mixed_df())

    assert report.features["inference_source"] == "rule_only"
    assert "rule_prior" in report.features
    assert report.features["guardrail_overrides"] == []


def test_registry_champion_challenger_report_is_deterministic() -> None:
    champion = RegimeModelCandidate("champion", "a.pt", metrics={"delta_sharpe": 0.10})
    challenger = RegimeModelCandidate("challenger", "b.pt", metrics={"delta_sharpe": 0.16})
    registry = build_regime_model_registry((champion, challenger), champion_name="champion")
    report = registry.champion_challenger_report("challenger", metric="delta_sharpe", min_improvement=0.03)

    assert isinstance(report, ChampionChallengerReport)
    assert report.promote is True
    assert report.score_delta == 0.06
    assert report.champion["name"] == "champion"
    assert report.challenger["name"] == "challenger"


def test_regime_model_walk_forward_trains_locked_folds() -> None:
    result = run_regime_model_walk_forward(
        _mixed_df(170),
        cfg=RegimeModelWalkForwardConfig(
            split=SplitSpec(train_size=95, test_size=50, step=50, n_splits=1),
            dataset=RegimeOutcomeDatasetConfig(
                horizons=(2,),
                stride=12,
                min_history=24,
                analyzer=RegimeIntelligenceConfig(timeframes=("5m",)),
            ),
            model=LearnedRegimeModelConfig(hidden_dim=24, latent_dim=8, n_horizons=1, dropout=0.0),
            train=RegimeModelTrainConfig(
                epochs=1,
                batch_size=4,
                device="cpu",
                optim=OptimConfig(lr=2e-3, scheduler="none", warmup_steps=0),
            ),
        ),
        seed=0,
    )

    assert result.summary["n_folds"] == 1
    assert result.best_candidate
    assert result.registry.champion_name == result.best_candidate
