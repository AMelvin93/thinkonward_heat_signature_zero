#!/usr/bin/env python
"""
Quick test of JAX Hybrid optimizer.

Run from WSL:
    uv run python scripts/test_jax_hybrid.py
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
from src.jax_hybrid_optimizer import JAXHybridOptimizer
from src.jax_simulator import check_gpu

# Check GPU
print("=" * 60)
print("JAX Device Info:")
gpu_info = check_gpu()
print(f"  Backend: {gpu_info.get('default_backend', 'unknown')}")
print(f"  Devices: {gpu_info.get('devices', [])}")
print(f"  GPU Available: {gpu_info.get('gpu_available', False)}")
print("=" * 60)

# Load data
with open(project_root / 'data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples'][:5]  # Test on 5 samples
meta = data['meta']

print(f"\nTesting on {len(samples)} samples...")

# Create optimizer
optimizer = JAXHybridOptimizer(
    n_smart_inits=2,
    n_random_inits=4,
    n_max_candidates=3,
)

# Process samples
total_time = 0
rmses = []

for i, sample in enumerate(samples):
    sample_id = sample['sample_id']
    n_sources = sample['n_sources']

    print(f"\n[{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

    start = time.time()
    estimates, rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=(0.5, 2.0),
        max_iter=50,
        verbose=True,
    )
    elapsed = time.time() - start
    total_time += elapsed
    rmses.append(rmse)

    print(f"  Result: RMSE={rmse:.4f}, {len(candidates)} candidates, {elapsed:.1f}s")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Samples processed: {len(samples)}")
print(f"Total time: {total_time:.1f}s")
print(f"Avg time per sample: {total_time/len(samples):.1f}s")
print(f"Avg RMSE: {sum(rmses)/len(rmses):.4f}")
print(f"\nProjected time for 400 samples: {(total_time/len(samples) * 400) / 60:.1f} min")
print("=" * 60)
