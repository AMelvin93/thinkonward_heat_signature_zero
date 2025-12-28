#!/usr/bin/env python
"""
Test Coarse-to-Fine Optimizer.

Strategy:
- Coarse: 50x25 grid, 1/4 timesteps → ~16x faster per forward pass
- Fine: 100x50 grid, all timesteps → accurate final result

Run: uv run python scripts/test_coarse_to_fine.py
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_coarse_to_fine_optimizer import CoarseToFineOptimizer, check_gpu
import numpy as np


def main():
    print("=" * 60)
    print("COARSE-TO-FINE OPTIMIZER TEST")
    print("=" * 60)

    device_info = check_gpu()
    print(f"\nDevice: {device_info}")

    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'heat-signature-zero-test-data.pkl'
    )

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    print(f"Loaded {len(samples)} samples")

    optimizer = CoarseToFineOptimizer(time_subsample=4)

    n_test = 10
    test_samples = samples[:n_test]

    print(f"\nTesting on {n_test} samples...")
    print("-" * 60)

    total_time = 0
    rmses = []
    times_list = []

    for i, sample in enumerate(test_samples):
        n_sources = sample['n_sources']
        nt = sample['sample_metadata']['nt']

        start = time.time()
        sources, rmse, _ = optimizer.estimate_sources(
            sample, meta,
            coarse_iters=15,
            fine_iters=10,
            verbose=True
        )
        elapsed = time.time() - start

        total_time += elapsed
        rmses.append(rmse)
        times_list.append(elapsed)

        print(f"Sample {i:2d}: nt={nt}, {n_sources} src, RMSE={rmse:.4f}, time={elapsed:.1f}s\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples: {n_test}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time/sample: {total_time/n_test:.1f}s")
    print(f"Min/Max: {min(times_list):.1f}s / {max(times_list):.1f}s")
    print(f"Avg RMSE: {np.mean(rmses):.4f}")

    projected_400 = total_time / n_test * 400 / 60
    print(f"\nProjected for 400 samples: {projected_400:.1f} min")

    target = 60
    if projected_400 < target:
        print(f"\n[OK] Under {target} min target!")
    else:
        print(f"\n[WARNING] Need {projected_400/target:.1f}x more speedup")

    per_sample = total_time / n_test
    if per_sample < 9:
        print(f"[OK] {per_sample:.1f}s per sample (target: 9s)")
    else:
        print(f"[WARNING] {per_sample:.1f}s per sample (target: 9s)")


if __name__ == '__main__':
    main()
