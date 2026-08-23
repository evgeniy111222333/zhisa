"""Cross-instrument attention for multi-instrument portfolio policies.

Stage 2 of the portfolio policy: instrument tokens attend to each
other so the per-instrument policy/value heads receive context that
includes information about *all* instruments in the portfolio, not
just the focal one. The shared encoder in Stage 1 already aggregates
information through the portfolio summary vector; cross-attention
adds a direct, content-based pathway.

The module is intentionally minimal:

* :class:`CrossInstrumentAttention` is a stack of
  ``nn.TransformerEncoderLayer`` blocks operating on the
  ``(B, N, embed_dim)`` instrument-token tensor. It is
  permutation-equivariant (an instrument's output is invariant to
  the order in which the others are presented).
* An optional *instrument-id embedding* (instrument index -> learned
  vector) is added to the tokens before the first attention block,
  so the model can break the permutation symmetry when needed.
* An optional *portfolio summary bias* (B, embed_dim) is added to
  every token after the attention stack, so the portfolio state can
  bias the policy heads even without going through the MLP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CrossInstrumentConfig:
    """Configuration for :class:`CrossInstrumentAttention`."""

    embed_dim: int = 64
    depth: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    use_instrument_id: bool = True
    n_instruments_max: int = 8
    feedforward_mult: int = 4
    norm_first: bool = True
    field_overrides: dict = field(default_factory=dict)
    # Additive attention bias (e.g. a cross-asset correlation matrix).
    # When enabled the module's forward accepts ``bias`` of shape
    # (B, N, N) that is added to the scaled scores BEFORE softmax —
    # the direct "graph" pathway connecting cross_asset statistics to
    # cross-instrument attention (#4 proposal). Default disabled.
    use_attention_bias: bool = False
    bias_gate: float = 1.0  # lambda scaling of the additive bias

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.n_heads <= 0 or self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must divide embed_dim={self.embed_dim}"
            )


class CrossInstrumentAttention(nn.Module):
    """Stacked bidirectional self-attention over instrument tokens.

    Input shape: ``(B, N, embed_dim)`` — instrument embeddings.
    Output shape: ``(B, N, embed_dim)`` — cross-instrument context.

    With ``depth=0`` the module is the identity (apart from the
    optional instrument-id embedding and portfolio bias). This
    makes it cheap to default-enable in the policy config and let
    Stage 1 callers explicitly set ``depth=0`` to disable.
    """

    def __init__(self, cfg: Optional[CrossInstrumentConfig] = None) -> None:
        super().__init__()
        cfg = cfg or CrossInstrumentConfig()
        self.cfg = cfg
        if cfg.use_instrument_id:
            self.instrument_id = nn.Embedding(cfg.n_instruments_max, cfg.embed_dim)
        else:
            self.instrument_id = None
        if cfg.depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.embed_dim * cfg.feedforward_mult,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=cfg.norm_first,
            )
            self.layers = nn.TransformerEncoder(
                layer, num_layers=cfg.depth, enable_nested_tensor=False,
            )
        else:
            self.layers = None
        self.norm: Optional[nn.LayerNorm] = None
        self.portfolio_proj: Optional[nn.Linear] = None

    def set_portfolio_dim(self, portfolio_dim: int) -> None:
        """Enable the optional portfolio bias projection.

        Call this once after construction if you want the model to
        use the portfolio summary vector as an additive bias on
        every instrument token.
        """
        if self.portfolio_proj is None and portfolio_dim > 0:
            self.portfolio_proj = nn.Linear(portfolio_dim, self.cfg.embed_dim)

    def _add_instrument_id(self, x: torch.Tensor) -> torch.Tensor:
        if self.instrument_id is None:
            return x
        B, N, D = x.shape
        if N > self.cfg.n_instruments_max:
            raise ValueError(
                f"N={N} > n_instruments_max={self.cfg.n_instruments_max}; "
                "raise n_instruments_max in the config or pre-pad instruments."
            )
        ids = torch.arange(N, device=x.device)
        return x + self.instrument_id(ids).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,                       # (B, N, D)
        portfolio: Optional[torch.Tensor] = None,  # (B, portfolio_dim)
        bias: Optional[torch.Tensor] = None,   # (B, N, N) additive score bias
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D) input, got {tuple(x.shape)}")
        if x.size(-1) != self.cfg.embed_dim:
            raise ValueError(
                f"input embed_dim={x.size(-1)} != config embed_dim={self.cfg.embed_dim}"
            )
        y = self._add_instrument_id(x)
        if self.layers is not None:
            if self.cfg.use_attention_bias:
                if bias is None:
                    raise ValueError("use_attention_bias=True requires a bias tensor")
                y = self._biased_stack(y, bias)
            else:
                y = self.layers(y)
        if portfolio is not None and self.portfolio_proj is not None:
            y = y + self.portfolio_proj(portfolio).unsqueeze(1)
        return y

    def _biased_stack(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Full encoder-depth forward with additive attention bias.

        Replicates each TransformerEncoderLayer exactly (attention + FFN +
        norms, norm_first ordering) except that every attention block receives
        the additive ``bias`` in its scores. With ``bias_gate=0`` the output
        is bit-identical to the unbiased :attr:`layers` path.
        """
        if bias.shape[:2] != x.shape[:2] or bias.shape[2] != x.shape[1]:
            raise ValueError(
                f"bias must be (B, N, N), got {tuple(bias.shape)} vs x {tuple(x.shape)}"
            )
        gate = self.cfg.bias_gate
        for layer in self.layers.layers:
            if layer.norm_first:
                _x = layer.norm1(x)
                attn_out = self._mha(layer, _x, _x, _x, bias, gate)
                x = x + self._pw_drop(layer.dropout1, attn_out)
                _x2 = layer.norm2(x)
                ff_out = layer.linear2(layer.activation(layer.linear1(_x2)))
                x = x + self._pw_drop(layer.dropout2, ff_out)
            else:
                attn_out = self._mha(layer, x, x, x, bias, gate)
                x = layer.norm1(x + self._pw_drop(layer.dropout1, attn_out))
                ff_out = layer.linear2(layer.activation(layer.linear1(x)))
                x = layer.norm2(x + self._pw_drop(layer.dropout2, ff_out))
        return x

    @staticmethod
    def _pw_drop(dropout, t: torch.Tensor) -> torch.Tensor:
        if dropout is None or dropout.p == 0.0 or not dropout.training:
            return t
        return dropout(t)

    def _mha(self, layer, q, k, v, bias: torch.Tensor, gate: float) -> torch.Tensor:
        """Replicate nn.MultiheadAttention forward with an additive bias."""
        mha = layer.self_attn
        W, bq = mha.in_proj_weight, mha.in_proj_bias
        D = q.size(-1)
        H = mha.num_heads
        hd = D // H
        B, N, _ = q.shape
        qf = q.reshape(B * N, D)
        qq = (qf @ W[:D].T + bq[:D]).reshape(B, N, H, hd).permute(0, 2, 1, 3)
        kk = (qf @ W[D:2 * D].T + bq[D:2 * D]).reshape(B, N, H, hd).permute(0, 2, 1, 3)
        vv = (qf @ W[2 * D:].T + bq[2 * D:]).reshape(B, N, H, hd).permute(0, 2, 1, 3)
        scores = (qq @ kk.transpose(-1, -2)) / (hd ** 0.5)
        if gate != 0.0:
            scores = scores + gate * bias.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        if mha.training and getattr(mha, "dropout", 0.0) > 0.0:
            attn = torch.nn.functional.dropout(attn, p=float(mha.dropout), training=True)
        out = (attn @ vv).permute(0, 2, 1, 3).reshape(B, N, D)
        return mha.out_proj(out)


__all__ = ["CrossInstrumentAttention", "CrossInstrumentConfig"]
