"""Outcome-trained regime model, dataset, calibration, and hybrid adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from zhisa.backtest.splitter import SplitSpec, walk_forward_splits
from zhisa.regime.dataset import (
    MACRO_TO_ID,
    MESO_TO_ID,
    PLAYBOOK_NAMES,
    PLAYBOOK_TO_ID,
    RISK_MODE_TO_ID,
    _playbook_scores,
)
from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.memory import RegimeOutcome
from zhisa.regime.schema import MacroRegime, MesoRegime, RegimeReport, RiskMode
from zhisa.regime.vectorizer import RegimeFeatureVectorizer, RegimeVectorizerConfig
from zhisa.regime.registry import RegimeModelCandidate, build_regime_model_registry


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _default_optim_config() -> Any:
    from zhisa.training.optim import OptimConfig

    return OptimConfig()


@dataclass(frozen=True)
class RegimeOutcomeDatasetConfig:
    horizons: tuple[int, ...] = (6, 12, 24, 48)
    stride: int = 1
    min_history: int = 128
    symbol: str = ""
    cache_items: bool = True
    transition_return_threshold: float = 0.015
    transition_vol_multiplier: float = 1.35
    crash_drawdown_threshold: float = -0.04
    no_trade_abs_return_threshold: float = 0.004
    no_trade_drawdown_threshold: float = -0.015
    execution_vol_norm: float = 0.03
    execution_drawdown_norm: float = 0.05
    analyzer: RegimeIntelligenceConfig = field(default_factory=RegimeIntelligenceConfig)
    vectorizer: RegimeVectorizerConfig = field(default_factory=RegimeVectorizerConfig)

    def __post_init__(self) -> None:
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_history < 2:
            raise ValueError("min_history must be >= 2")


@dataclass(frozen=True)
class RegimeOutcomeItem:
    x: torch.Tensor
    return_targets: torch.Tensor
    volatility_targets: torch.Tensor
    drawdown_targets: torch.Tensor
    mfe_targets: torch.Tensor
    transition_event: torch.Tensor
    crash_event: torch.Tensor
    no_trade_correct: torch.Tensor
    execution_risk: torch.Tensor
    playbook_utility: torch.Tensor
    weak_macro: torch.Tensor
    weak_meso: torch.Tensor
    weak_risk_mode: torch.Tensor
    rule_report: RegimeReport
    outcomes: list[RegimeOutcome]
    meta: dict[str, Any]


@dataclass(frozen=True)
class RegimeOutcomeBatch:
    x: torch.Tensor
    return_targets: torch.Tensor
    volatility_targets: torch.Tensor
    drawdown_targets: torch.Tensor
    mfe_targets: torch.Tensor
    transition_event: torch.Tensor
    crash_event: torch.Tensor
    no_trade_correct: torch.Tensor
    execution_risk: torch.Tensor
    playbook_utility: torch.Tensor
    weak_macro: torch.Tensor
    weak_meso: torch.Tensor
    weak_risk_mode: torch.Tensor
    rule_reports: list[RegimeReport]
    outcomes: list[list[RegimeOutcome]]
    meta: list[dict[str, Any]]


def _future_path(close: pd.Series, t: int, horizon: int) -> pd.Series:
    c0 = float(close.iloc[t])
    future = close.iloc[t + 1 : t + horizon + 1].astype(float)
    if c0 <= 0 or not np.isfinite(c0) or future.empty:
        return pd.Series([c0], index=[close.index[t]], dtype=float)
    return pd.concat([pd.Series([c0], index=[close.index[t]]), future])


def _outcome(close: pd.Series, t: int, horizon: int) -> tuple[RegimeOutcome, float, float]:
    path = _future_path(close, t, horizon)
    c0 = float(path.iloc[0])
    future = path.iloc[1:]
    if future.empty or c0 <= 0:
        return RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0), 0.0, 0.0
    rel = future / c0 - 1.0
    logret = np.log(path.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    outcome = RegimeOutcome(
        forward_return=float(rel.iloc[-1]) if np.isfinite(float(rel.iloc[-1])) else 0.0,
        realized_vol=float(logret.std(ddof=0) if not logret.empty else 0.0),
        max_drawdown=min(0.0, float(rel.min())),
    )
    mfe = max(0.0, float(rel.max()))
    abs_path_return = float(np.abs(rel).max()) if not rel.empty else 0.0
    return outcome, mfe, abs_path_return


class RegimeOutcomeDataset(Dataset):
    """Causal features at ``t`` with outcome-first future labels over several horizons."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Optional[RegimeOutcomeDatasetConfig] = None,
        *,
        analyzer: RegimeIntelligence | None = None,
        vectorizer: RegimeFeatureVectorizer | None = None,
    ) -> None:
        self.df = df
        self.cfg = cfg or RegimeOutcomeDatasetConfig()
        self.analyzer = analyzer or RegimeIntelligence(self.cfg.analyzer)
        self.vectorizer = vectorizer or RegimeFeatureVectorizer(self.cfg.vectorizer)
        if "close" not in df.columns:
            raise ValueError("df must contain a close column")
        max_h = max(self.cfg.horizons)
        if len(df) <= self.cfg.min_history + max_h:
            raise ValueError(
                "df is too short for regime outcome learning: "
                f"len={len(df)}, min_history={self.cfg.min_history}, max_horizon={max_h}"
            )
        start = self.cfg.min_history - 1
        stop = len(df) - max_h - 1
        self.indices = list(range(start, stop + 1, self.cfg.stride))
        self._cache: dict[int, RegimeOutcomeItem] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> RegimeOutcomeItem:
        if self.cfg.cache_items and idx in self._cache:
            return self._cache[idx]
        t = self.indices[idx]
        report = self.analyzer.analyze(self.df, t=t, symbol=self.cfg.symbol)
        x = torch.from_numpy(self.vectorizer.transform(report)).float()
        close = self.df["close"].astype(float)
        outcomes: list[RegimeOutcome] = []
        returns: list[float] = []
        vols: list[float] = []
        drawdowns: list[float] = []
        mfes: list[float] = []
        max_abs_return = 0.0
        for horizon in self.cfg.horizons:
            out, mfe, abs_ret = _outcome(close, t, horizon)
            outcomes.append(out)
            returns.append(float(out.forward_return or 0.0))
            vols.append(float(out.realized_vol or 0.0))
            drawdowns.append(float(out.max_drawdown or 0.0))
            mfes.append(float(mfe))
            max_abs_return = max(max_abs_return, abs_ret)
        primary_outcome = outcomes[0]
        playbook_scores, _, best_playbook = _playbook_scores(report, primary_outcome)
        vol_baseline = max(float(report.features.get("aggregate", {}).get("atr_pct", 0.0)), 1e-6)
        transition = float(
            max_abs_return >= self.cfg.transition_return_threshold
            or max(vols or [0.0]) >= vol_baseline * self.cfg.transition_vol_multiplier
        )
        crash = float(min(drawdowns or [0.0]) <= self.cfg.crash_drawdown_threshold)
        no_trade_correct = float(
            abs(float(primary_outcome.forward_return or 0.0)) <= self.cfg.no_trade_abs_return_threshold
            or float(primary_outcome.max_drawdown or 0.0) <= self.cfg.no_trade_drawdown_threshold
        )
        execution_risk = _clip01(
            max(vols or [0.0]) / max(self.cfg.execution_vol_norm, 1e-12)
            + abs(min(drawdowns or [0.0])) / max(self.cfg.execution_drawdown_norm, 1e-12)
        )
        item = RegimeOutcomeItem(
            x=x,
            return_targets=torch.as_tensor(returns, dtype=torch.float32),
            volatility_targets=torch.as_tensor(vols, dtype=torch.float32),
            drawdown_targets=torch.as_tensor(drawdowns, dtype=torch.float32),
            mfe_targets=torch.as_tensor(mfes, dtype=torch.float32),
            transition_event=torch.tensor(transition, dtype=torch.float32),
            crash_event=torch.tensor(crash, dtype=torch.float32),
            no_trade_correct=torch.tensor(no_trade_correct, dtype=torch.float32),
            execution_risk=torch.tensor(execution_risk, dtype=torch.float32),
            playbook_utility=torch.from_numpy(playbook_scores).float(),
            weak_macro=torch.tensor(MACRO_TO_ID[report.primary_regime], dtype=torch.long),
            weak_meso=torch.tensor(MESO_TO_ID[report.secondary_regime], dtype=torch.long),
            weak_risk_mode=torch.tensor(RISK_MODE_TO_ID[report.risk_mode], dtype=torch.long),
            rule_report=report,
            outcomes=outcomes,
            meta={
                "t": t,
                "timestamp": report.features.get("timestamp") if report.features else None,
                "symbol": self.cfg.symbol,
                "horizons": self.cfg.horizons,
                "best_playbook": best_playbook,
            },
        )
        if self.cfg.cache_items:
            self._cache[idx] = item
        return item


def regime_outcome_collate(items: list[RegimeOutcomeItem]) -> RegimeOutcomeBatch:
    return RegimeOutcomeBatch(
        x=torch.stack([it.x for it in items]),
        return_targets=torch.stack([it.return_targets for it in items]),
        volatility_targets=torch.stack([it.volatility_targets for it in items]),
        drawdown_targets=torch.stack([it.drawdown_targets for it in items]),
        mfe_targets=torch.stack([it.mfe_targets for it in items]),
        transition_event=torch.stack([it.transition_event for it in items]),
        crash_event=torch.stack([it.crash_event for it in items]),
        no_trade_correct=torch.stack([it.no_trade_correct for it in items]),
        execution_risk=torch.stack([it.execution_risk for it in items]),
        playbook_utility=torch.stack([it.playbook_utility for it in items]),
        weak_macro=torch.stack([it.weak_macro for it in items]),
        weak_meso=torch.stack([it.weak_meso for it in items]),
        weak_risk_mode=torch.stack([it.weak_risk_mode for it in items]),
        rule_reports=[it.rule_report for it in items],
        outcomes=[it.outcomes for it in items],
        meta=[it.meta for it in items],
    )


@dataclass(frozen=True)
class LearnedRegimeModelConfig:
    input_dim: int | None = None
    hidden_dim: int = 128
    latent_dim: int = 32
    dropout: float = 0.1
    n_horizons: int = 4
    n_playbooks: int = len(PLAYBOOK_NAMES)
    return_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    vectorizer: RegimeVectorizerConfig = field(default_factory=RegimeVectorizerConfig)


class LearnedRegimeModel(nn.Module):
    """Outcome-trained regime model; rule labels are only auxiliary weak priors."""

    def __init__(
        self,
        cfg: Optional[LearnedRegimeModelConfig] = None,
        *,
        vectorizer: RegimeFeatureVectorizer | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LearnedRegimeModelConfig()
        self.vectorizer = vectorizer or RegimeFeatureVectorizer(self.cfg.vectorizer)
        input_dim = self.cfg.input_dim or self.vectorizer.dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
        )
        self.latent = nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim)
        self.macro = nn.Linear(self.cfg.hidden_dim, len(tuple(MacroRegime)))
        self.meso = nn.Linear(self.cfg.hidden_dim, len(tuple(MesoRegime)))
        self.risk_mode = nn.Linear(self.cfg.hidden_dim, len(tuple(RiskMode)))
        q_dim = self.cfg.n_horizons * len(self.cfg.return_quantiles)
        self.return_quantiles = nn.Linear(self.cfg.hidden_dim, q_dim)
        self.volatility = nn.Linear(self.cfg.hidden_dim, self.cfg.n_horizons)
        self.drawdown = nn.Linear(self.cfg.hidden_dim, self.cfg.n_horizons)
        self.transition = nn.Linear(self.cfg.hidden_dim, 1)
        self.crash = nn.Linear(self.cfg.hidden_dim, 1)
        self.no_trade = nn.Linear(self.cfg.hidden_dim, 1)
        self.execution = nn.Linear(self.cfg.hidden_dim, 1)
        self.tradeability = nn.Linear(self.cfg.hidden_dim, 1)
        self.risk_budget = nn.Linear(self.cfg.hidden_dim, 1)
        self.playbook_utility = nn.Linear(self.cfg.hidden_dim, self.cfg.n_playbooks)
        self.uncertainty = nn.Linear(self.cfg.hidden_dim, 1)
        self.ood = nn.Linear(self.cfg.hidden_dim, 1)

    @property
    def input_dim(self) -> int:
        return int(self.cfg.input_dim or self.vectorizer.dim)

    def vectorize_reports(self, reports: Sequence[RegimeReport], *, device: torch.device | str | None = None) -> torch.Tensor:
        arr = self.vectorizer.transform_many(reports)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor | np.ndarray | Sequence[RegimeReport]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, torch.Tensor):
            x_t = x.to(device=device, dtype=torch.float32)
        else:
            x_t = self.vectorize_reports(x, device=device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        h = self.backbone(x_t)
        z = torch.tanh(self.latent(h))
        n_q = len(self.cfg.return_quantiles)
        return {
            "latent_regime_embedding": z,
            "macro_logits": self.macro(h),
            "meso_logits": self.meso(h),
            "risk_logits": self.risk_mode(h),
            "return_quantiles": self.return_quantiles(h).view(-1, self.cfg.n_horizons, n_q),
            "volatility_forecast": F.softplus(self.volatility(h)),
            "drawdown_quantiles": -F.softplus(self.drawdown(h)),
            "transition_hazard": torch.sigmoid(self.transition(h)).squeeze(-1),
            "crash_risk": torch.sigmoid(self.crash(h)).squeeze(-1),
            "no_trade_probability": torch.sigmoid(self.no_trade(h)).squeeze(-1),
            "liquidity_execution_risk": torch.sigmoid(self.execution(h)).squeeze(-1),
            "tradeability": torch.sigmoid(self.tradeability(h)).squeeze(-1),
            "risk_budget": torch.sigmoid(self.risk_budget(h)).squeeze(-1),
            "playbook_utility": self.playbook_utility(h),
            "uncertainty": torch.sigmoid(self.uncertainty(h)).squeeze(-1),
            "ood_score": torch.sigmoid(self.ood(h)).squeeze(-1),
        }


@dataclass(frozen=True)
class LearnedRegimeLossWeights:
    return_quantiles: float = 1.0
    volatility: float = 0.5
    drawdown: float = 0.75
    transition: float = 0.8
    crash: float = 0.4
    no_trade: float = 0.4
    execution: float = 0.35
    playbook_utility: float = 0.7
    tradeability: float = 0.35
    risk_budget: float = 0.35
    weak_macro: float = 0.08
    weak_meso: float = 0.06
    weak_risk_mode: float = 0.05
    monotonic: float = 0.15


class LearnedRegimeLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[LearnedRegimeLossWeights] = None,
        *,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.weights = weights or LearnedRegimeLossWeights()
        self.quantiles = quantiles

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_q = target.unsqueeze(-1).expand_as(pred)
        err = target_q - pred
        qs = torch.as_tensor(self.quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
        return torch.maximum(qs * err, (qs - 1.0) * err).mean()

    def forward(self, outputs: dict[str, torch.Tensor], batch: RegimeOutcomeBatch) -> dict[str, torch.Tensor]:
        device = outputs["latent_regime_embedding"].device
        ret = batch.return_targets.to(device=device)
        vol = batch.volatility_targets.to(device=device)
        dd = batch.drawdown_targets.to(device=device)
        transition = batch.transition_event.to(device=device)
        crash = batch.crash_event.to(device=device)
        no_trade = batch.no_trade_correct.to(device=device)
        execution = batch.execution_risk.to(device=device)
        playbook = batch.playbook_utility.to(device=device)
        losses = {
            "return_quantiles": self._quantile_loss(outputs["return_quantiles"], ret),
            "volatility": F.smooth_l1_loss(outputs["volatility_forecast"], vol),
            "drawdown": F.smooth_l1_loss(outputs["drawdown_quantiles"], dd),
            "transition": F.binary_cross_entropy(outputs["transition_hazard"].clamp(1e-6, 1.0 - 1e-6), transition),
            "crash": F.binary_cross_entropy(outputs["crash_risk"].clamp(1e-6, 1.0 - 1e-6), crash),
            "no_trade": F.binary_cross_entropy(outputs["no_trade_probability"].clamp(1e-6, 1.0 - 1e-6), no_trade),
            "execution": F.smooth_l1_loss(outputs["liquidity_execution_risk"], execution),
            "playbook_utility": F.smooth_l1_loss(outputs["playbook_utility"], playbook),
            "tradeability": F.smooth_l1_loss(outputs["tradeability"], (1.0 - execution).clamp(0.0, 1.0)),
            "risk_budget": F.smooth_l1_loss(outputs["risk_budget"], (1.0 - transition).clamp(0.0, 1.0)),
            "weak_macro": F.cross_entropy(outputs["macro_logits"], batch.weak_macro.to(device=device)),
            "weak_meso": F.cross_entropy(outputs["meso_logits"], batch.weak_meso.to(device=device)),
            "weak_risk_mode": F.cross_entropy(outputs["risk_logits"], batch.weak_risk_mode.to(device=device)),
        }
        bad_risk = outputs["liquidity_execution_risk"].detach() + outputs["transition_hazard"].detach()
        monotonic_target = (1.0 - 0.5 * bad_risk).clamp(0.0, 1.0)
        losses["monotonic"] = F.relu(outputs["tradeability"] - monotonic_target).mean()
        total = torch.zeros((), device=device)
        for name, loss in losses.items():
            total = total + float(getattr(self.weights, name)) * loss
        losses["total"] = total
        return losses


@dataclass
class RegimeModelTrainConfig:
    epochs: int = 5
    batch_size: int = 64
    grad_clip: float = 1.0
    num_workers: int = 0
    checkpoint: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    optim: Any = field(default_factory=_default_optim_config)
    loss_weights: LearnedRegimeLossWeights = field(default_factory=LearnedRegimeLossWeights)


def _move_batch(batch: RegimeOutcomeBatch, device: torch.device) -> RegimeOutcomeBatch:
    return RegimeOutcomeBatch(
        x=batch.x.to(device),
        return_targets=batch.return_targets.to(device),
        volatility_targets=batch.volatility_targets.to(device),
        drawdown_targets=batch.drawdown_targets.to(device),
        mfe_targets=batch.mfe_targets.to(device),
        transition_event=batch.transition_event.to(device),
        crash_event=batch.crash_event.to(device),
        no_trade_correct=batch.no_trade_correct.to(device),
        execution_risk=batch.execution_risk.to(device),
        playbook_utility=batch.playbook_utility.to(device),
        weak_macro=batch.weak_macro.to(device),
        weak_meso=batch.weak_meso.to(device),
        weak_risk_mode=batch.weak_risk_mode.to(device),
        rule_reports=batch.rule_reports,
        outcomes=batch.outcomes,
        meta=batch.meta,
    )


class RegimeModelTrainer:
    def __init__(
        self,
        model: LearnedRegimeModel,
        cfg: Optional[RegimeModelTrainConfig] = None,
        *,
        loss: LearnedRegimeLoss | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or RegimeModelTrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.loss = loss or LearnedRegimeLoss(self.cfg.loss_weights, quantiles=self.model.cfg.return_quantiles)
        from zhisa.training.optim import build_optimizer, build_scheduler

        self.opt = build_optimizer(self.model, self.cfg.optim)
        self.sched = build_scheduler(self.opt, self.cfg.optim)
        self._step = 0

    def fit(self, train_ds: RegimeOutcomeDataset, val_ds: RegimeOutcomeDataset | None = None) -> dict[str, Any]:
        torch.manual_seed(int(self.cfg.seed))
        loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=regime_outcome_collate,
            drop_last=len(train_ds) >= self.cfg.batch_size,
        )
        history: list[dict[str, Any]] = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            totals: dict[str, float] = {}
            n = 0
            for batch in loader:
                batch_d = _move_batch(batch, self.device)
                out = self.model(batch_d.x)
                losses = self.loss(out, batch_d)
                self.opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                self._step += 1
                bs = batch_d.x.size(0)
                n += bs
                for key, val in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(val.item()) * bs
            row = {"epoch": epoch, "step": self._step}
            row.update({k: v / max(1, n) for k, v in totals.items()})
            if val_ds is not None:
                row["val"] = self.evaluate(val_ds)
            history.append(row)
        if self.cfg.checkpoint:
            self.save(self.cfg.checkpoint)
        return {"history": history, "final_step": self._step}

    @torch.no_grad()
    def evaluate(self, ds: RegimeOutcomeDataset) -> dict[str, float]:
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=regime_outcome_collate)
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch_d = _move_batch(batch, self.device)
            out = self.model(batch_d.x)
            losses = self.loss(out, batch_d)
            bs = batch_d.x.size(0)
            n += bs
            for key, val in losses.items():
                totals[key] = totals.get(key, 0.0) + float(val.item()) * bs
            pred = (out["transition_hazard"] >= 0.5).float()
            totals["transition_acc"] = totals.get("transition_acc", 0.0) + float((pred == batch_d.transition_event).float().mean().item()) * bs
        return {k: v / max(1, n) for k, v in totals.items()}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "model_config": asdict(self.model.cfg),
            "train_config": asdict(self.cfg),
            "step": self._step,
        }, p)


@dataclass(frozen=True)
class RegimeModelWalkForwardConfig:
    split: SplitSpec
    dataset: RegimeOutcomeDatasetConfig = field(default_factory=RegimeOutcomeDatasetConfig)
    model: LearnedRegimeModelConfig = field(default_factory=LearnedRegimeModelConfig)
    train: RegimeModelTrainConfig = field(default_factory=RegimeModelTrainConfig)
    validation_fraction: float = 0.25
    artifact_dir: str = ""
    selection_metric: str = "test_total"
    higher_is_better: bool = False


@dataclass(frozen=True)
class RegimeModelWalkForwardResult:
    fold_metrics: list[dict[str, Any]]
    candidates: list[RegimeModelCandidate]
    registry: Any
    best_candidate: str
    summary: dict[str, Any]


def _slice_for_dataset(
    df: pd.DataFrame,
    start: int,
    end: int,
    *,
    min_history: int,
) -> pd.DataFrame:
    left = max(0, start - max(min_history, 1))
    return df.iloc[left:end].copy()


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def run_regime_model_walk_forward(
    df: pd.DataFrame,
    *,
    cfg: RegimeModelWalkForwardConfig,
    seed: int = 0,
) -> RegimeModelWalkForwardResult:
    """Train, calibrate, and evaluate learned regime models on locked walk-forward folds."""
    folds = walk_forward_splits(len(df), cfg.split)
    fold_metrics: list[dict[str, Any]] = []
    candidates: list[RegimeModelCandidate] = []
    artifact_dir = Path(cfg.artifact_dir) if cfg.artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        train_start, train_end = fold.train
        test_start, test_end = fold.test
        train_size = train_end - train_start
        val_size = max(int(train_size * cfg.validation_fraction), cfg.dataset.min_history + max(cfg.dataset.horizons) + 2)
        val_start = max(train_start, train_end - val_size)
        fit_df = _slice_for_dataset(df, train_start, val_start, min_history=cfg.dataset.min_history)
        val_df = _slice_for_dataset(df, val_start, train_end, min_history=cfg.dataset.min_history)
        test_df = _slice_for_dataset(df, test_start, test_end, min_history=cfg.dataset.min_history)
        try:
            train_ds = RegimeOutcomeDataset(fit_df, cfg.dataset)
            val_ds = RegimeOutcomeDataset(val_df, replace(cfg.dataset, cache_items=False))
            test_ds = RegimeOutcomeDataset(test_df, replace(cfg.dataset, cache_items=False))
        except ValueError:
            continue
        model_cfg = replace(cfg.model, n_horizons=len(cfg.dataset.horizons))
        model = LearnedRegimeModel(model_cfg)
        ckpt = str(artifact_dir / f"regime_model_fold_{i}.pt") if artifact_dir is not None else None
        train_cfg = replace(cfg.train, checkpoint=ckpt, seed=seed + i)
        trainer = RegimeModelTrainer(model, train_cfg)
        train_result = trainer.fit(train_ds, val_ds=val_ds)
        val_metrics = trainer.evaluate(val_ds)
        calibration, calibration_metrics = calibrate_learned_regime_model(model, val_ds, device=train_cfg.device)
        test_metrics = trainer.evaluate(test_ds)
        row = {
            "fold": i,
            "train": fold.train,
            "validation": (val_start, train_end),
            "test": fold.test,
            "train_final_step": int(train_result["final_step"]),
            "val_total": float(val_metrics.get("total", 0.0)),
            "test_total": float(test_metrics.get("total", 0.0)),
            "test_transition_acc": float(test_metrics.get("transition_acc", 0.0)),
            **{f"calibration_{k}": float(v) for k, v in calibration_metrics.items() if isinstance(v, (int, float))},
        }
        fold_metrics.append(row)
        candidates.append(
            RegimeModelCandidate(
                name=f"fold_{i}",
                artifact_path=ckpt or "",
                calibration_path="",
                version=f"wf_fold_{i}",
                metrics={
                    "test_total": row["test_total"],
                    "val_total": row["val_total"],
                    "test_transition_acc": row["test_transition_acc"],
                    "transition_brier_after": row.get("calibration_transition_brier_after", 0.0),
                },
                metadata={
                    "fold": i,
                    "calibration": calibration.to_dict(),
                    "train": fold.train,
                    "validation": (val_start, train_end),
                    "test": fold.test,
                },
            )
        )
    if candidates:
        key = lambda c: c.score(cfg.selection_metric)
        best = max(candidates, key=key) if cfg.higher_is_better else min(candidates, key=key)
        registry = build_regime_model_registry(candidates, champion_name=best.name)
        best_name = best.name
    else:
        registry = build_regime_model_registry(())
        best_name = ""
    summary = {
        "n_folds": len(fold_metrics),
        "best_candidate": best_name,
        "selection_metric": cfg.selection_metric,
        "higher_is_better": cfg.higher_is_better,
        "mean_test_total": _mean_metric(fold_metrics, "test_total"),
        "mean_val_total": _mean_metric(fold_metrics, "val_total"),
        "mean_test_transition_acc": _mean_metric(fold_metrics, "test_transition_acc"),
    }
    return RegimeModelWalkForwardResult(
        fold_metrics=fold_metrics,
        candidates=candidates,
        registry=registry,
        best_candidate=best_name,
        summary=summary,
    )


@dataclass(frozen=True)
class RegimeProbabilityCalibration:
    transition_scale: float = 1.0
    transition_bias: float = 0.0
    tradeability_scale: float = 1.0
    tradeability_bias: float = 0.0
    risk_budget_scale: float = 1.0
    risk_budget_bias: float = 0.0
    version: str = "uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_probability(self, value: float, *, scale: float, bias: float) -> float:
        value = _clip01(value)
        logit = np.log(max(value, 1e-6) / max(1.0 - value, 1e-6))
        return _clip01(1.0 / (1.0 + np.exp(-(scale * logit + bias))))

    def transition(self, value: float) -> float:
        return self.apply_probability(value, scale=self.transition_scale, bias=self.transition_bias)

    def tradeability(self, value: float) -> float:
        return self.apply_probability(value, scale=self.tradeability_scale, bias=self.tradeability_bias)

    def risk_budget(self, value: float) -> float:
        return self.apply_probability(value, scale=self.risk_budget_scale, bias=self.risk_budget_bias)


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean((scores - labels) ** 2))


def _ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if scores.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        out += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(out)


@torch.no_grad()
def calibrate_learned_regime_model(
    model: LearnedRegimeModel,
    ds: RegimeOutcomeDataset,
    *,
    device: str | torch.device = "cpu",
) -> tuple[RegimeProbabilityCalibration, dict[str, Any]]:
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=regime_outcome_collate)
    model.to(device)
    model.eval()
    transitions: list[float] = []
    transition_labels: list[float] = []
    tradeability: list[float] = []
    trade_labels: list[float] = []
    risk_budget: list[float] = []
    risk_labels: list[float] = []
    for batch in loader:
        batch_d = _move_batch(batch, torch.device(device))
        out = model(batch_d.x)
        transitions.extend(out["transition_hazard"].detach().cpu().numpy().tolist())
        transition_labels.extend(batch.transition_event.numpy().tolist())
        tradeability.extend(out["tradeability"].detach().cpu().numpy().tolist())
        trade_labels.extend((1.0 - batch.execution_risk).clamp(0.0, 1.0).numpy().tolist())
        risk_budget.extend(out["risk_budget"].detach().cpu().numpy().tolist())
        risk_labels.extend((1.0 - batch.transition_event).clamp(0.0, 1.0).numpy().tolist())
    t = np.asarray(transitions, dtype=np.float64)
    tl = np.asarray(transition_labels, dtype=np.float64)
    tr = np.asarray(tradeability, dtype=np.float64)
    trl = np.asarray(trade_labels, dtype=np.float64)
    rb = np.asarray(risk_budget, dtype=np.float64)
    rbl = np.asarray(risk_labels, dtype=np.float64)
    # Conservative intercept-only calibration keeps ranking intact and fixes base-rate bias.
    def bias(pred: np.ndarray, label: np.ndarray) -> float:
        if pred.size == 0:
            return 0.0
        p = _clip01(float(pred.mean()))
        y = _clip01(float(label.mean()))
        return float(np.log(max(y, 1e-6) / max(1.0 - y, 1e-6)) - np.log(max(p, 1e-6) / max(1.0 - p, 1e-6)))

    t_bias = bias(t, tl)
    tr_bias = bias(tr, trl)
    rb_bias = bias(rb, rbl)
    candidate = RegimeProbabilityCalibration(
        transition_bias=t_bias,
        tradeability_bias=tr_bias,
        risk_budget_bias=rb_bias,
        version="intercept_v1",
    )
    if _brier(np.asarray([candidate.transition(x) for x in t], dtype=np.float64), tl) > _brier(t, tl):
        t_bias = 0.0
    if _brier(np.asarray([candidate.tradeability(x) for x in tr], dtype=np.float64), trl) > _brier(tr, trl):
        tr_bias = 0.0
    if _brier(np.asarray([candidate.risk_budget(x) for x in rb], dtype=np.float64), rbl) > _brier(rb, rbl):
        rb_bias = 0.0
    cal = RegimeProbabilityCalibration(
        transition_bias=t_bias,
        tradeability_bias=tr_bias,
        risk_budget_bias=rb_bias,
        version="intercept_v1",
    )
    t_cal = np.asarray([cal.transition(x) for x in t], dtype=np.float64)
    tr_cal = np.asarray([cal.tradeability(x) for x in tr], dtype=np.float64)
    rb_cal = np.asarray([cal.risk_budget(x) for x in rb], dtype=np.float64)
    metrics = {
        "transition_brier_before": _brier(t, tl),
        "transition_brier_after": _brier(t_cal, tl),
        "transition_ece_before": _ece(t, tl),
        "transition_ece_after": _ece(t_cal, tl),
        "tradeability_brier_before": _brier(tr, trl),
        "tradeability_brier_after": _brier(tr_cal, trl),
        "risk_budget_brier_before": _brier(rb, rbl),
        "risk_budget_brier_after": _brier(rb_cal, rbl),
        "n": int(t.size),
    }
    return cal, metrics


@dataclass(frozen=True)
class RegimeModelPrediction:
    outputs: dict[str, Any]
    macro: str
    meso: str
    risk_mode: str
    transition_risk: float
    tradeability: float
    risk_budget: float
    uncertainty: float
    ood_score: float
    playbook_utility: dict[str, float]


def _macro_name(idx: int) -> str:
    return tuple(MacroRegime)[int(idx)].value


def _meso_name(idx: int) -> str:
    return tuple(MesoRegime)[int(idx)].value


def _risk_name(idx: int) -> str:
    return tuple(RiskMode)[int(idx)].value


@torch.no_grad()
def predict_learned_regime(
    model: LearnedRegimeModel,
    report: RegimeReport,
    *,
    calibration: RegimeProbabilityCalibration | None = None,
    device: torch.device | str = "cpu",
) -> RegimeModelPrediction:
    calibration = calibration or RegimeProbabilityCalibration()
    model.to(device)
    model.eval()
    out_t = model([report])
    out = {k: v.detach().cpu() for k, v in out_t.items()}
    macro_probs = torch.softmax(out["macro_logits"][0], dim=-1).numpy()
    meso_probs = torch.softmax(out["meso_logits"][0], dim=-1).numpy()
    risk_probs = torch.softmax(out["risk_logits"][0], dim=-1).numpy()
    playbook = out["playbook_utility"][0].numpy()
    transition = calibration.transition(float(out["transition_hazard"][0]))
    trade = calibration.tradeability(float(out["tradeability"][0]))
    risk_budget = calibration.risk_budget(float(out["risk_budget"][0]))
    outputs = {
        "macro_probabilities": {_macro_name(i): float(v) for i, v in enumerate(macro_probs)},
        "meso_probabilities": {_meso_name(i): float(v) for i, v in enumerate(meso_probs)},
        "risk_mode_probabilities": {_risk_name(i): float(v) for i, v in enumerate(risk_probs)},
        "return_quantiles": out["return_quantiles"][0].numpy().tolist(),
        "volatility_forecast": out["volatility_forecast"][0].numpy().tolist(),
        "drawdown_quantiles": out["drawdown_quantiles"][0].numpy().tolist(),
        "transition_hazard": transition,
        "tradeability": trade,
        "risk_budget": risk_budget,
        "liquidity_execution_risk": float(out["liquidity_execution_risk"][0]),
        "uncertainty": float(out["uncertainty"][0]),
        "ood_score": float(out["ood_score"][0]),
    }
    return RegimeModelPrediction(
        outputs=outputs,
        macro=_macro_name(int(np.argmax(macro_probs))),
        meso=_meso_name(int(np.argmax(meso_probs))),
        risk_mode=_risk_name(int(np.argmax(risk_probs))),
        transition_risk=transition,
        tradeability=trade,
        risk_budget=risk_budget,
        uncertainty=float(out["uncertainty"][0]),
        ood_score=float(out["ood_score"][0]),
        playbook_utility={PLAYBOOK_NAMES[i]: float(v) for i, v in enumerate(playbook[: len(PLAYBOOK_NAMES)])},
    )


@dataclass(frozen=True)
class RegimeDecisionAdapterConfig:
    mode: str = "hybrid"
    model_version: str = "in_memory"
    calibration_version: str = "uncalibrated"
    uncertainty_guardrail: float = 0.75
    ood_guardrail: float = 0.75
    transition_guardrail: float = 0.70
    min_tradeability_on_guardrail: float = 0.25


class RegimeDecisionAdapter:
    """Convert learned predictions plus rule prior into the existing RegimeReport API."""

    def __init__(self, cfg: RegimeDecisionAdapterConfig | None = None) -> None:
        self.cfg = cfg or RegimeDecisionAdapterConfig()

    def adapt(self, rule_report: RegimeReport, prediction: RegimeModelPrediction | None) -> RegimeReport:
        features = dict(rule_report.features)
        explanation = {
            "why": list(rule_report.explanation.get("why", [])),
            "danger": list(rule_report.explanation.get("danger", [])),
            "guardrails": list(rule_report.explanation.get("guardrails", [])),
        }
        features["rule_prior"] = rule_report.to_dict()
        if prediction is None:
            features["inference_source"] = "rule_only"
            features["guardrail_overrides"] = []
            return replace(rule_report, features=features, explanation=explanation)
        overrides: list[str] = []
        source = "learned" if self.cfg.mode == "learned" else "hybrid"
        transition = prediction.transition_risk
        tradeability = prediction.tradeability
        size = prediction.risk_budget
        if self.cfg.mode == "hybrid":
            transition = max(float(rule_report.transition_risk), transition)
            tradeability = min(float(rule_report.tradeability_score), tradeability)
            size = min(float(rule_report.position_size_multiplier), size)
        if prediction.uncertainty >= self.cfg.uncertainty_guardrail:
            tradeability = min(tradeability, self.cfg.min_tradeability_on_guardrail)
            size = min(size, self.cfg.min_tradeability_on_guardrail)
            overrides.append("model_uncertainty")
        if prediction.ood_score >= self.cfg.ood_guardrail:
            tradeability = min(tradeability, self.cfg.min_tradeability_on_guardrail)
            size = min(size, self.cfg.min_tradeability_on_guardrail)
            overrides.append("model_ood")
        if transition >= self.cfg.transition_guardrail:
            size = min(size, 0.35)
            overrides.append("high_transition_hazard")
        for reason in overrides:
            explanation["guardrails"].append(reason)
        explanation["why"].append(
            f"learned_regime={prediction.macro}/{prediction.meso}, transition={transition:.2f}, tradeability={tradeability:.2f}"
        )
        features["inference_source"] = source
        features["model_version"] = self.cfg.model_version
        features["calibration_version"] = self.cfg.calibration_version
        features["model_outputs"] = prediction.outputs
        features["playbook_utility"] = prediction.playbook_utility
        features["guardrail_overrides"] = overrides
        learned_probs = prediction.outputs.get("macro_probabilities", {})
        allowed = list(rule_report.allowed_playbooks)
        blocked = list(rule_report.blocked_playbooks)
        if transition >= self.cfg.transition_guardrail and "transition_wait" not in allowed:
            allowed.append("transition_wait")
            if "regime_transition_chase" not in blocked:
                blocked.append("regime_transition_chase")
        return replace(
            rule_report,
            primary_regime=prediction.macro if self.cfg.mode == "learned" else rule_report.primary_regime,
            secondary_regime=prediction.meso if self.cfg.mode == "learned" else rule_report.secondary_regime,
            risk_mode=prediction.risk_mode if self.cfg.mode == "learned" else rule_report.risk_mode,
            transition_risk=_clip01(transition),
            tradeability_score=_clip01(tradeability),
            position_size_multiplier=max(0.0, float(size)),
            uncertainty=_clip01(max(rule_report.uncertainty, prediction.uncertainty, prediction.ood_score)),
            allowed_playbooks=sorted(set(allowed)),
            blocked_playbooks=sorted(set(blocked)),
            explanation=explanation,
            features=features,
            probabilities={**rule_report.probabilities, **{str(k): float(v) for k, v in learned_probs.items()}},
        )


def load_learned_regime_model(path: str | Path, *, map_location: str | torch.device = "cpu") -> LearnedRegimeModel:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg_raw = payload.get("model_config", {})
    cfg = LearnedRegimeModelConfig(**{k: v for k, v in cfg_raw.items() if k in LearnedRegimeModelConfig.__dataclass_fields__})
    model = LearnedRegimeModel(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


__all__ = [
    "LearnedRegimeLoss",
    "LearnedRegimeLossWeights",
    "LearnedRegimeModel",
    "LearnedRegimeModelConfig",
    "RegimeDecisionAdapter",
    "RegimeDecisionAdapterConfig",
    "RegimeModelPrediction",
    "RegimeModelTrainConfig",
    "RegimeModelTrainer",
    "RegimeModelWalkForwardConfig",
    "RegimeModelWalkForwardResult",
    "RegimeOutcomeBatch",
    "RegimeOutcomeDataset",
    "RegimeOutcomeDatasetConfig",
    "RegimeOutcomeItem",
    "RegimeProbabilityCalibration",
    "calibrate_learned_regime_model",
    "load_learned_regime_model",
    "predict_learned_regime",
    "regime_outcome_collate",
    "run_regime_model_walk_forward",
]
