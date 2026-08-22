"""End-to-end policy network: encoders + fusion + memory + heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from zhisa.models.encoders.context import ContextEncoder, ContextEncoderConfig
from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.models.encoders.vision_columnformer import ColumnFormerVision, VisionColumnFormerConfig
from zhisa.models.fusion import (
    POS_CONTEXT,
    POS_FREQ,
    CrossModalFusion,
    FusionConfig,
    TokenCrossFusion,
    TokenFusionConfig,
)
from zhisa.models.heads import HeadsConfig, MultiTaskHeads
from zhisa.models.memory import MemoryConfig, WorkingMemory


@dataclass
class PolicyConfig:
    image_size: int = 64
    in_numeric_features: int = 32
    in_macro_features: int = 32
    window: int = 32
    macro_window: int = 64
    in_context_features: int = 10
    embed_dim: int = 128
    n_actions: int = 9
    n_regime_classes: int = 4
    n_instruments: int = 1
    market_horizons: tuple = ()
    use_macro_context: bool = False
    use_memory: bool = True
    memory_max_len: int = 64
    fusion_layers: int = 2
    memory_layers: int = 2
    numeric_layers: int = 2
    numeric_causal: bool = False
    encoder_ff_mult: float = 2.0
    vision_mode: str = "cnn"              # "cnn" | "columnformer"
    vision_n_layers: int = 4
    vision_n_heads: int = 8
    vision_causal: bool = True
    vision_reader: str = "attention_pool" # "attention_pool" | "cls"
    token_fusion: bool = False
    fusion_token_layers: int = 3
    fusion_token_heads: int = 8
    freq_branch: bool = False
    freq_k: int = 8
    two_tokens_per_bar: bool = False
    vision_volume: bool = True
    vision_channels: tuple = (32, 64, 128, 192)
    dropout: float = 0.1
    field_overrides: dict = field(default_factory=dict)


class PolicyNetwork(nn.Module):
    """The end-to-end multimodal policy / feature extractor."""

    def __init__(self, cfg: Optional[PolicyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        if cfg.vision_mode == "columnformer":
            self.vision = ColumnFormerVision(VisionColumnFormerConfig(
                image_size=cfg.image_size, n_bars=cfg.window,
                d_model=cfg.embed_dim,
                n_heads=cfg.vision_n_heads, n_layers=cfg.vision_n_layers,
                dim_ff=int(cfg.embed_dim * cfg.encoder_ff_mult),
                dropout=cfg.dropout, out_dim=cfg.embed_dim,
                causal=cfg.vision_causal, reader=cfg.vision_reader,
                freq_branch=cfg.freq_branch, freq_k=cfg.freq_k,
                two_tokens_per_bar=cfg.two_tokens_per_bar,
                include_volume=cfg.vision_volume,
            ))
        else:
            self.vision = VisionEncoder(VisionEncoderConfig(
                image_size=cfg.image_size, out_dim=cfg.embed_dim,
                channels=cfg.vision_channels, dropout=cfg.dropout,
            ))
        self.numeric = NumericEncoder(NumericEncoderConfig(
            in_features=cfg.in_numeric_features, window=cfg.window,
            d_model=cfg.embed_dim, out_dim=cfg.embed_dim,
            n_layers=cfg.numeric_layers, dropout=cfg.dropout,
            dim_ff=int(cfg.embed_dim * cfg.encoder_ff_mult),
            causal=cfg.numeric_causal,
            summary_position="end" if cfg.numeric_causal else "front",
        ))
        self.context = ContextEncoder(ContextEncoderConfig(
            in_dim=cfg.in_context_features, out_dim=cfg.embed_dim,
            n_instruments=cfg.n_instruments, dropout=cfg.dropout,
        ))
        if cfg.use_macro_context:
            self.macro_numeric = NumericEncoder(NumericEncoderConfig(
                in_features=cfg.in_macro_features, window=cfg.macro_window,
                d_model=cfg.embed_dim, out_dim=cfg.embed_dim,
                n_layers=2, dropout=cfg.dropout,
            ))
            self.timeframe_embed = nn.Embedding(2, cfg.embed_dim)
            self.macro_gate = nn.Sequential(
                nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.Sigmoid(),
            )
            self.macro_proj = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            self.macro_norm = nn.LayerNorm(cfg.embed_dim)
        else:
            self.macro_numeric = None
            self.timeframe_embed = None
            self.macro_gate = None
            self.macro_proj = None
            self.macro_norm = None
        if cfg.token_fusion:
            self.fusion = TokenCrossFusion(TokenFusionConfig(
                d_model=cfg.embed_dim, n_heads=cfg.fusion_token_heads,
                n_layers=cfg.fusion_token_layers,
                dim_ff=int(cfg.embed_dim * cfg.encoder_ff_mult),
                dropout=cfg.dropout, out_dim=cfg.embed_dim,
            ))
        else:
            self.fusion = CrossModalFusion(FusionConfig(
                embed_dim=cfg.embed_dim, n_layers=cfg.fusion_layers,
                dropout=cfg.dropout, dim_ff=int(cfg.embed_dim * cfg.encoder_ff_mult),
            ))
        self.memory = (
            WorkingMemory(MemoryConfig(
                embed_dim=cfg.embed_dim, n_layers=cfg.memory_layers,
                max_len=cfg.memory_max_len, dropout=cfg.dropout,
                dim_ff=int(cfg.embed_dim * cfg.encoder_ff_mult),
            )) if cfg.use_memory else None
        )
        self.heads = MultiTaskHeads(HeadsConfig(
            embed_dim=cfg.embed_dim, n_actions=cfg.n_actions,
            n_regime_classes=cfg.n_regime_classes,
            n_market_horizons=len(cfg.market_horizons),
            dropout=cfg.dropout,
        ))

    def plain_vision(self, chart: torch.Tensor) -> torch.Tensor:
        """Pooled vision embedding regardless of encoder mode (CNN vs ColumnFormer)."""
        if self.cfg.vision_mode == "columnformer":
            v, _vt, freq = self.vision(chart)
            if freq is not None:
                v = v + freq.squeeze(1)
            return v
        return self.vision(chart)

    def encode(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        macro_numeric: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.context(context, instrument_id=instrument_id)
        if self.cfg.token_fusion:
            vv, vtok, vfreq = self.vision(chart)
            n_cls, n_tokens = self.numeric(numeric)
            dev = chart.device
            B = chart.size(0)
            # shared time axis for aligned temporal PE:
            # vision bar token k -> time index k (2-token-per-bar halves share k)
            if self.cfg.two_tokens_per_bar:
                pos_v = torch.arange(vtok.size(1), device=dev) // 2
            else:
                pos_v = torch.arange(vtok.size(1), device=dev)
            pos_v = pos_v.view(1, -1).expand(B, -1).to(torch.long)
            # numeric patches aligned by their start bar; summary at window end
            ps = int(self.numeric.cfg.patch_size)
            np_ = int(self.numeric.n_patches)
            W = int(self.cfg.window)
            if self.numeric.cfg.summary_position == "end":
                pos_n = torch.cat(
                    [torch.arange(0, np_, device=dev) * ps,
                     torch.full((1,), W, device=dev)], dim=0)
            else:
                pos_n = torch.cat(
                    [torch.full((1,), W, device=dev),
                     torch.arange(0, np_, device=dev) * ps], dim=0)
            pos_n = pos_n.view(1, -1).expand(B, -1).to(torch.long)
            streams = [(vtok, 0, pos_v)]
            if vfreq is not None:
                streams.append((vfreq, 3, torch.full((B, 1), POS_FREQ, dtype=torch.long, device=dev)))
            streams.append((n_tokens, 1, pos_n))
            streams.append((c.unsqueeze(1), 2, torch.full((B, 1), POS_CONTEXT, dtype=torch.long, device=dev)))
            z = self.fusion(streams)
        else:
            if self.cfg.vision_mode == "columnformer":
                vv, _vt, vfreq = self.vision(chart)
                if vfreq is not None:
                    vv = vv + vfreq.squeeze(1)
                v = vv
            else:
                v = self.vision(chart)
            n, _ = self.numeric(numeric)
            if self.cfg.use_macro_context:
                if self.timeframe_embed is None:
                    raise RuntimeError("macro context is enabled but timeframe embeddings are missing")
                n = n + self.timeframe_embed.weight[0].view(1, -1)
            z = self.fusion(v, n, c)
        if self.cfg.use_macro_context:
            if macro_numeric is None:
                macro_numeric = torch.zeros(
                    numeric.size(0),
                    int(self.cfg.macro_window),
                    int(self.cfg.in_macro_features),
                    device=numeric.device,
                    dtype=numeric.dtype,
                )
            if self.macro_numeric is None or self.macro_gate is None or self.macro_proj is None or self.macro_norm is None:
                raise RuntimeError("macro context modules are not initialised")
            m, _ = self.macro_numeric(macro_numeric)
            m = m + self.timeframe_embed.weight[1].view(1, -1)
            gate = self.macro_gate(torch.cat([z, m], dim=-1))
            z = self.macro_norm(z + gate * self.macro_proj(m))
        return z

    def forward(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        macro_numeric: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            chart:    (B, 3, H, W)
            numeric:  (B, T, F)
            context:  (B, C)
            history:  optional (B, S, D) rolling memory; updated internally.
            instrument_id: optional (B,) long.
        """
        z = self.encode(
            chart,
            numeric,
            context,
            macro_numeric=macro_numeric,
            instrument_id=instrument_id,
        )
        if self.memory is not None:
            max_hist_len = self.memory.cfg.max_len - 1
            if history is None:
                history = torch.zeros(
                    z.size(0), max_hist_len, z.size(-1),
                    device=z.device, dtype=z.dtype
                )
            z_with_hist = torch.cat([history, z.unsqueeze(1)], dim=1)
            out_seq = self.memory(z_with_hist)
            z = out_seq[:, -1]
            next_history = z_with_hist[:, -max_hist_len:].detach()
        else:
            next_history = None
        out = self.heads(z)
        out["embedding"] = z
        out["next_history"] = next_history
        return out


def build_default_policy(
    in_numeric_features: int = 32,
    in_context_features: int = 10,
    n_actions: int = 9,
    n_regime_classes: int = 4,
    **kwargs,
) -> PolicyNetwork:
    cfg = PolicyConfig(
        in_numeric_features=in_numeric_features,
        in_context_features=in_context_features,
        n_actions=n_actions,
        n_regime_classes=n_regime_classes,
        **kwargs,
    )
    return PolicyNetwork(cfg)
