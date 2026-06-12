"""Smoke test: do model components actually do a forward pass?"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, r"D:\zhisa\src")

from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
from zhisa.models.encoders.context import ContextEncoder, ContextEncoderConfig
from zhisa.models.fusion import CrossModalFusion, FusionConfig
from zhisa.models.policy import PolicyNetwork, PolicyConfig, build_default_policy

print("=" * 60)
print("FORWARD-PASS SMOKE TEST")
print("=" * 60)

device = "cpu"
torch.manual_seed(0)

# Realistic shapes (matching env obs structure)
B = 4
chart = torch.randn(B, 3, 64, 64)
numeric = torch.randn(B, 32)
context = torch.randn(B, 16)

# 1. Vision encoder
t0 = time.perf_counter()
ve = VisionEncoder(VisionEncoderConfig(image_size=64, in_channels=3, out_dim=128))
v_out = ve(chart)
print(f"[1] VisionEncoder:    in={tuple(chart.shape)}   out={tuple(v_out.shape)}  ({1000*(time.perf_counter()-t0):.1f}ms)")

# 2. Numeric encoder
t0 = time.perf_counter()
ne = NumericEncoder(NumericEncoderConfig(in_dim=32, out_dim=64))
n_out = ne(numeric)
print(f"[2] NumericEncoder:   in={tuple(numeric.shape)} out={tuple(n_out.shape)}  ({1000*(time.perf_counter()-t0):.1f}ms)")

# 3. Context encoder
t0 = time.perf_counter()
ce = ContextEncoder(ContextEncoderConfig(in_dim=16, out_dim=32))
c_out = ce(context)
print(f"[3] ContextEncoder:   in={tuple(context.shape)} out={tuple(c_out.shape)}  ({1000*(time.perf_counter()-t0):.1f}ms)")

# 4. Cross-modal fusion
t0 = time.perf_counter()
fs = CrossModalFusion(FusionConfig(in_dim=128 + 64 + 32, hidden_dim=128, out_dim=128, n_layers=2))
fusion_in = torch.cat([v_out, n_out, c_out], dim=-1)
f_out = fs(fusion_in)
print(f"[4] CrossModalFusion: in={tuple(fusion_in.shape)} out={tuple(f_out.shape)}  ({1000*(time.perf_counter()-t0):.1f}ms)")

# 5. Policy (actor + critic)
t0 = time.perf_counter()
pol = PolicyNetwork(PolicyConfig(in_dim=128, n_actions=9, hidden_dim=128))
logits, value = pol(f_out)
print(f"[5] PolicyNetwork:    logits={tuple(logits.shape)} value={tuple(value.shape)}  ({1000*(time.perf_counter()-t0):.1f}ms)")

# 6. Action sampling
probs = torch.softmax(logits, dim=-1)
print(f"[6] Action probs sample: {probs[0].detach().numpy().round(3)}")
print(f"    Sum-to-1 check: {[round(x, 4) for x in probs.sum(-1).tolist()]}")

# 7. Value range
print(f"[7] Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

# 8. Are outputs finite?
all_finite = all(torch.isfinite(x).all().item() for x in [v_out, n_out, c_out, f_out, logits, value])
print(f"[8] All outputs finite: {all_finite}")

# 9. Full build_default_policy (the production way)
print()
print("[9] build_default_policy end-to-end:")
t0 = time.perf_counter()
pol2, info = build_default_policy(
    chart_shape=(3, 64, 64), numeric_dim=32, context_dim=16, n_actions=9, hidden_dim=128
)
print(f"    policy built in {1000*(time.perf_counter()-t0):.1f}ms")
print(f"    info: {info}")

# forward with real obs dict
obs = {
    "chart": torch.randn(2, 3, 64, 64),
    "numeric": torch.randn(2, 32),
    "context": torch.randn(2, 16),
}
t0 = time.perf_counter()
logits2, value2 = pol2(obs)
print(f"    forward(obs_dict): logits={tuple(logits2.shape)} value={tuple(value2.shape)} ({1000*(time.perf_counter()-t0):.1f}ms)")

print()
all_finite2 = torch.isfinite(logits2).all().item() and torch.isfinite(value2).all().item()
print("RESULT:", "PASS" if (all_finite and all_finite2) else "FAIL")
