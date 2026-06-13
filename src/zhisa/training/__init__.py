"""Training scripts for the 5 learning phases (+ S2b imitation, S6 DT, S7 WM, S4-CVaR, S-portfolio)."""

from zhisa.training.cvar_ppo import CVaRPPOConfig, CVaRPPOTrainer
from zhisa.training.decision_transformer import (
    DTConfig,
    DTTrainResult,
    DecisionTransformer,
    DecisionTransformerConfig,
    DecisionTransformerTrainer,
    build_default_dt,
    embed_trajectories,
)
from zhisa.training.dyna_ppo import (
    DynaPPOConfig,
    DynaPPOResult,
    DynaPPOTrainer,
    ImaginedBatch,
)
from zhisa.training.portfolio_ppo import (
    PortfolioPPOConfig,
    PortfolioPPOTrainer,
    PortfolioRolloutBuffer,
    PortfolioTransition,
)
from zhisa.training.regime_supervised import (
    RegimeEncoderLoss,
    RegimeEncoderTrainer,
    RegimeLossWeights,
    RegimeTrainConfig,
)
from zhisa.training.s2b_imitation import (
    BCConfig,
    BehavioralCloningTrainer,
    DAggerConfig,
    DAggerResult,
    DAggerRoundMetrics,
    DAggerTrainer,
)
from zhisa.training.world_model_trainer import (
    WorldModelDataset,
    WorldModelTrainResult,
    WorldModelTrainer,
    WorldModelTrainerConfig,
)

__all__ = [
    "BCConfig",
    "BehavioralCloningTrainer",
    "CVaRPPOConfig",
    "CVaRPPOTrainer",
    "DAggerConfig",
    "DAggerResult",
    "DAggerRoundMetrics",
    "DAggerTrainer",
    "DTConfig",
    "DTTrainResult",
    "DecisionTransformer",
    "DecisionTransformerConfig",
    "DecisionTransformerTrainer",
    "DynaPPOConfig",
    "DynaPPOResult",
    "DynaPPOTrainer",
    "ImaginedBatch",
    "PortfolioPPOConfig",
    "PortfolioPPOTrainer",
    "PortfolioRolloutBuffer",
    "PortfolioTransition",
    "RegimeEncoderLoss",
    "RegimeEncoderTrainer",
    "RegimeLossWeights",
    "RegimeTrainConfig",
    "WorldModelDataset",
    "WorldModelTrainer",
    "WorldModelTrainerConfig",
    "WorldModelTrainResult",
    "build_default_dt",
    "embed_trajectories",
]
