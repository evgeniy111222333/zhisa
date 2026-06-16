import os, time
os.environ['ZHISA_FAST_RENDER'] = '1'
import numpy as np, torch
from zhisa.data.trajectory import Trajectory, TrajectoryWindowDataset

# Make a few short fake trajectories
trajs = []
for i in range(5):
    t = Trajectory()
    T = 200 + i*20
    t.actions = np.random.randint(0, 9, size=T).astype(np.int64)
    t.rewards = np.random.randn(T).astype(np.float32) * 0.01
    t.dones = np.zeros(T, dtype=bool); t.dones[-1] = True
    t.obs = [{'numeric': np.random.randn(2, 16).astype(np.float32)} for _ in range(T)]
    trajs.append(t)

ds = TrajectoryWindowDataset(trajs, context_length=20, n_actions=9)
n = len(ds)
print(f'len(ds)={n}, __fast_getitem__={ds.__fast_getitem__}')

# Cold pass
t0 = time.perf_counter()
for i in range(n): _ = ds[i]
cold = time.perf_counter() - t0

# Warm pass
t0 = time.perf_counter()
for i in range(n): _ = ds[i]
warm = time.perf_counter() - t0

print(f'cold: {cold:.4f}s | {cold/n*1e6:.1f} us/sample | {n/cold:.0f}/s')
print(f'warm: {warm:.4f}s | {warm/n*1e6:.1f} us/sample | {n/warm:.0f}/s')

b = ds[0]
shapes = {k: v.shape if hasattr(v, 'shape') else v for k,v in b.items()}
print('shapes:', shapes)
