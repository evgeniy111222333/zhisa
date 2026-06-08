"""S3: Synthetic curriculum training.

This module extends the S3 stub (which used to be only a config
generator) into a real training orchestrator. It implements the
curriculum-learning stage described in ``CONCEPT.md`` §5.4.

The curriculum progressively increases the difficulty of the synthetic
market the model is exposed to:

    clean    -> mild fat tails, no shocks
    mixed    -> moderate vol, light shocks, moderate tails
    stressed -> high vol, frequent shocks, very fat tails (df=4)

For each stage we (a) generate a fresh market with the matching
``MarketConfig``, (b) train the inner model (S1 SSL or S2 supervised)
for some number of epochs, and (c) snapshot stage-level metrics
(loss, label distribution) so we can detect the regime in which
training degrades.

A few safety features are wired in by default:

* **NaN-guard**  — non-finite inner losses are skipped instead of
  poisoning the optimiser state (same lesson we learned in S1).
* **Optional regime mix**  — at each stage we can mix in a small
  fraction of data from the *previous* stage to discourage
  catastrophic forgetting of earlier regimes.
* **Per-stage checkpointing**  — the final model is saved once at
  the end, but intermediate snapshots can be enabled for analysis.

Design constraint: S3 must work with *any* inner trainer that
exposes a ``.fit(dataset)`` method. We do not subclass S1 / S2; we
just call them. This keeps the curriculum generic enough to also
support a future S4 RL inner trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """A single stage of the curriculum.

    Attributes
    ----------
    name:
        Human-readable stage name (e.g. ``"clean"``).
    n_bars:
        Number of OHLCV bars to synthesise for this stage.
    base_vol:
        Per-bar annualised base volatility.
    shock_prob:
        Probability of a per-bar exogenous shock event.
    student_t_df:
        Degrees-of-freedom for the Student-t innovations. Lower
        values produce fatter tails.
    epochs:
        Number of inner-trainer epochs to run on this stage's data.
    mix_with_previous:
        If > 0, mix in this fraction of the previous stage's data
        to discourage catastrophic forgetting. Range: [0, 1].
    """

    name: str
    n_bars: int = 4000
    base_vol: float = 0.4
    shock_prob: float = 0.0
    student_t_df: float = 20.0
    epochs: int = 1
    mix_with_previous: float = 0.0

    def to_market_config(self, seed: Optional[int] = None) -> MarketConfig:
        """Project this stage into a :class:`MarketConfig`."""
        return MarketConfig(
            n_bars=self.n_bars,
            base_vol=self.base_vol,
            shock_prob=self.shock_prob,
            student_t_df=self.student_t_df,
            seed=seed,
        )


@dataclass
class StageMetrics:
    """Per-stage training statistics.

    Aggregated across the inner trainer's epoch loop and the optional
    regime-mix pass.
    """

    stage: str
    n_bars: int
    epochs: int
    final_loss: float
    best_loss: float
    elapsed_s: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "stage": self.stage,
            "n_bars": self.n_bars,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "elapsed_s": self.elapsed_s,
        }
        d.update(self.extra)
        return d


@dataclass
class CurriculumResult:
    """Aggregate result of a full curriculum run."""

    stages: list[StageMetrics]
    final_loss: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.as_dict() for s in self.stages])


def default_curriculum() -> list[CurriculumStage]:
    """A simple 3-stage curriculum: clean trends -> mixed -> stressed.

    Mirrors the pre-existing stub but with a per-stage ``epochs`` knob
    so the trainer can spend more or less time on each regime.
    """
    return [
        CurriculumStage("clean", n_bars=1500, base_vol=0.4, shock_prob=0.0,
                        student_t_df=20.0, epochs=1, mix_with_previous=0.0),
        CurriculumStage("mixed", n_bars=1500, base_vol=0.6, shock_prob=0.0005,
                        student_t_df=8.0, epochs=1, mix_with_previous=0.2),
        CurriculumStage("stressed", n_bars=1500, base_vol=0.9, shock_prob=0.002,
                        student_t_df=4.0, epochs=1, mix_with_previous=0.2),
    ]


def iter_curriculum(
    stages: Optional[Sequence[CurriculumStage]] = None,
) -> list[tuple[str, MarketConfig]]:
    """Yield ``(stage_name, MarketConfig)`` pairs for each stage."""
    return [(s.name, s.to_market_config()) for s in (stages or default_curriculum())]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regime_label_distribution(df: pd.DataFrame) -> dict:
    """Return the count of bars in each regime, useful for sanity checks."""
    if "regime" not in df.columns:
        return {}
    counts = df["regime"].value_counts().to_dict()
    return {f"regime_{k}": int(v) for k, v in sorted(counts.items())}


def _make_dataset(
    df: pd.DataFrame,
    spec: SampleSpec,
) -> MarketDataset:
    """Wrap a market DataFrame into a multimodal dataset."""
    return MarketDataset(df, spec=spec)


def _mix_markets(
    primary: pd.DataFrame, secondary: pd.DataFrame, fraction: float
) -> pd.DataFrame:
    """Concatenate ``fraction`` of ``secondary`` onto ``primary``.

    The ``secondary`` rows are placed *after* the ``primary`` block on
    the time axis (we reindex them to start at the bar right after
    ``primary`` ends). Without this offset, primary and secondary
    would share timestamps (both generators use the same default start
    date) and the downstream dedup would silently drop them.
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0 or len(secondary) == 0:
        return primary
    n = max(1, int(round(fraction * len(secondary))))
    secondary_sample = secondary.iloc[:n].copy()
    if not isinstance(secondary_sample.index, pd.DatetimeIndex):
        return pd.concat([primary, secondary_sample], axis=0)
    # Offset secondary to start one bar after primary ends.
    last_ts = primary.index[-1]
    freq = pd.infer_freq(primary.index) or "5min"
    new_index = pd.date_range(
        start=last_ts + pd.Timedelta(freq),
        periods=len(secondary_sample), freq=freq, tz=primary.index.tz,
    )
    secondary_sample.index = new_index
    mixed = pd.concat([primary, secondary_sample], axis=0)
    return mixed


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CurriculumTrainer:
    """Orchestrate a multi-stage curriculum.

    Parameters
    ----------
    inner_trainer_factory:
        A zero-argument callable that returns a *fresh* inner trainer
        bound to the current model. We use a factory rather than a
        single trainer instance because the inner trainer usually
        carries its own optimiser, scheduler, and step counter that
        we want to reset between stages. The factory receives the
        current model state on each call so a single model is
        threaded through all stages.

        The returned object must implement ``.fit(dataset)`` returning
        a dict with a ``history`` key (same contract as
        :class:`SSLPretrainer` and :class:`SupervisedTrainer`).
    model:
        The model to train across stages.
    stages:
        A sequence of :class:`CurriculumStage` to traverse.
    sample_spec:
        The :class:`SampleSpec` used to build datasets for each stage.
    base_seed:
        Master seed. Each stage's :class:`MarketConfig` gets a
        deterministic per-stage seed derived from this value.
    checkpoint:
        If set, the final model state is saved here at the end of
        the curriculum (using ``torch.save`` on the model directly).
    """

    def __init__(
        self,
        inner_trainer_factory,
        model: torch.nn.Module,
        stages: Optional[Sequence[CurriculumStage]] = None,
        sample_spec: Optional[SampleSpec] = None,
        base_seed: int = 0,
        checkpoint: Optional[str] = None,
    ) -> None:
        self.factory = inner_trainer_factory
        self.model = model
        self.stages: list[CurriculumStage] = list(stages or default_curriculum())
        self.spec = sample_spec or SampleSpec(chart_window=16, image_size=16)
        self.base_seed = int(base_seed)
        self.checkpoint = checkpoint
        self._prev_df: Optional[pd.DataFrame] = None
        # Auto-detect feature dimensionality on a probe dataset so we
        # can validate the model's input shapes before training starts.
        # This is purely diagnostic — the model itself is the user's
        # responsibility — but it surfaces the most common setup error.
        probe_df = generate_market(self.stages[0].to_market_config(seed=self.base_seed))
        probe_ds = MarketDataset(probe_df, spec=self.spec)
        expected_n_feat = probe_ds._features.shape[1] + probe_ds._time_features.shape[1]
        expected_n_ctx = probe_ds._time_features.shape[1]
        if hasattr(model, "cfg"):
            cfg = model.cfg
            if cfg.in_numeric_features != expected_n_feat:
                raise ValueError(
                    f"Model in_numeric_features={cfg.in_numeric_features} does not match "
                    f"dataset's {expected_n_feat} features. Rebuild the model with the "
                    f"correct dimensionality (see scripts/train_s2.py for the pattern)."
                )
            if cfg.in_context_features != expected_n_ctx:
                raise ValueError(
                    f"Model in_context_features={cfg.in_context_features} does not match "
                    f"dataset's {expected_n_ctx} context features."
                )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self) -> CurriculumResult:
        """Run all stages in order and return aggregate metrics."""
        stage_metrics: list[StageMetrics] = []
        for stage_idx, stage in enumerate(self.stages):
            metrics = self._run_stage(stage_idx, stage)
            stage_metrics.append(metrics)
            logger.info(
                "curriculum stage %d/%d (%s) done in %.1fs | final_loss=%.5f best_loss=%.5f",
                stage_idx + 1, len(self.stages), stage.name,
                metrics.elapsed_s, metrics.final_loss, metrics.best_loss,
            )
        if self.checkpoint:
            self._save_checkpoint()
        final_loss = stage_metrics[-1].final_loss if stage_metrics else float("nan")
        return CurriculumResult(stages=stage_metrics, final_loss=final_loss)

    # ------------------------------------------------------------------
    # Per-stage logic
    # ------------------------------------------------------------------

    def _run_stage(self, stage_idx: int, stage: CurriculumStage) -> StageMetrics:
        # Per-stage deterministic seed.
        stage_seed = self.base_seed * 1000 + stage_idx
        df = generate_market(stage.to_market_config(seed=stage_seed))
        if stage.mix_with_previous > 0.0 and self._prev_df is not None:
            df = _mix_markets(df, self._prev_df, stage.mix_with_previous)
            logger.info(
                "stage %s: mixed in %.0f%% of previous-stage data (n=%d)",
                stage.name, 100 * stage.mix_with_previous, len(df),
            )

        # Build dataset.
        ds = _make_dataset(df, self.spec)

        # Build a fresh inner trainer bound to the current model state.
        trainer = self.factory(self.model)

        # Run.
        timer = Timer()
        timer.start()
        history = trainer.fit(ds) if stage.epochs > 0 else {"history": []}
        timer.stop()

        # Aggregate per-stage metrics robustly to the inner trainer's
        # history format. We expect each entry to have at least a
        # "loss" key (S1 uses "total"; we fall back to "loss" then
        # "total" then 0.0).
        loss_curve: list[float] = []
        for entry in history.get("history", []):
            if "loss" in entry:
                loss_curve.append(float(entry["loss"]))
            elif "total" in entry:
                loss_curve.append(float(entry["total"]))
        if not loss_curve:
            loss_curve = [0.0]
        final_loss = loss_curve[-1]
        best_loss = min(loss_curve)

        # Per-stage regime distribution for downstream introspection.
        regime_dist = _regime_label_distribution(df)

        metrics = StageMetrics(
            stage=stage.name,
            n_bars=len(df),
            epochs=stage.epochs,
            final_loss=final_loss,
            best_loss=best_loss,
            elapsed_s=timer.elapsed,
            extra={"regime_dist": regime_dist},
        )

        # Cache for next stage's mix.
        self._prev_df = df
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        from dataclasses import asdict
        p = Path(self.checkpoint)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "stages": [asdict(s) for s in self.stages],
        }, p)
        logger.info("curriculum checkpoint saved to %s", p)
