"""Risk management: position sizing, hard limits, stops, kill-switch."""
from zhisa.risk.cvar import cvar_constraint_violation, cvar_numpy, cvar_torch

__all__ = [
    "cvar_constraint_violation",
    "cvar_numpy",
    "cvar_torch",
]

