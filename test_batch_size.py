import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\zhisa\src")
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import MultiTaskLoss, LossWeights

print("="*60)
print("BATCH SIZE DIAGNOSTICS (Empirical Testing)")
print("="*60)

import os
os.environ["ZHISA_FAST_RENDER"] = "1"

class FakeArgs:
    data_source = "tsdb"
    tsdb_root = "data/tsdb"
    symbol = "BTC/USDT"
    timeframe = "5m"
    with_futures_context = True
    futures_context_root = "data/futures_context/binance_usdm"

args = FakeArgs()
def main():
    print("1. Loading mock dataset for benchmarking (5000 bars)...")
    df = load_market_dataframe(args, default_bars=5000)
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32)
    ds = MarketDataset(df, spec=spec, cache_charts=True)

    n_feat = ds._features.shape[1]
    n_ctx = ds._time_features.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    ).to(device)

    loss_fn = MultiTaskLoss(LossWeights()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch_sizes = [32, 64, 128, 256, 512]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=multimodal_collate, drop_last=True)
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        model.train()
        
        # Warmup
        loader_iter = iter(loader)
        try:
            warmup_batch = next(loader_iter)
        except StopIteration:
            print("Dataset too small for this batch size.")
            continue
            
        # Benchmark loop
        steps = 10
        start_time = time.perf_counter()
        
        grad_norms = []
        
        for i in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
                
            # Move to device
            batch_chart = batch.chart.to(device)
            batch_numeric = batch.numeric.to(device)
            batch_context = batch.context.to(device)
            
            optimizer.zero_grad()
            out = model(
                chart=batch_chart,
                numeric=batch_numeric,
                context=batch_context,
                instrument_id=None
            )
            
            # Fake backward pass to measure memory and grad norm
            dummy_loss = sum([v.sum() for v in out.values() if isinstance(v, torch.Tensor)])
            dummy_loss.backward()
            
            # Measure gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            optimizer.step()
            
        elapsed = time.perf_counter() - start_time
        samples_processed = steps * bs
        throughput = samples_processed / elapsed
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0
        avg_grad = sum(grad_norms)/len(grad_norms) if grad_norms else 0
        
        print(f"Throughput:  {throughput:.1f} samples/sec")
        print(f"Peak VRAM:   {peak_mem_mb:.1f} MB")
        print(f"Avg Grad L2: {avg_grad:.4f}")
        
        results.append({
            "batch_size": bs,
            "throughput": throughput,
            "vram": peak_mem_mb,
            "grad_norm": avg_grad
        })

    print("\n" + "="*60)
    print("SUMMARY DIAGNOSTICS")
    print("="*60)
    for r in results:
        print(f"BS: {r['batch_size']:<4} | VRAM: {r['vram']:>6.1f} MB | Speed: {r['throughput']:>6.1f} samp/s | GradNorm: {r['grad_norm']:.4f}")
    print("============================================================")

if __name__ == "__main__":
    main()
