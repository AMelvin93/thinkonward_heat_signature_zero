#!/usr/bin/env python
"""
Test Pure JAX Optimizer.

Key difference: No scipy, no Python callbacks.
Entire optimization runs in JAX on GPU.

Run: uv run python scripts/test_pure_jax.py
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_pure_optimizer import PureJAXOptimizer, check_gpu
import numpy as np


def main():
    print("=" * 60)
    print("PURE JAX OPTIMIZER TEST (No scipy)")
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

    optimizer = PureJAXOptimizer()

    n_test = 10
    test_samples = samples[:n_test]

    print(f"\nTesting {n_test} samples...")
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
            n_steps=50,
            lr=0.05,
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
    print(f"Total: {total_time:.1f}s")
    print(f"Avg: {total_time/n_test:.1f}s/sample")
    print(f"Min/Max: {min(times_list):.1f}s / {max(times_list):.1f}s")
    print(f"Avg RMSE: {np.mean(rmses):.4f}")

    projected_400 = total_time / n_test * 400 / 60
    print(f"\nProjected 400 samples: {projected_400:.1f} min")

    if projected_400 < 60:
        print("\n[OK] Under 60 min!")
    else:
        print(f"\n[WARNING] Need {projected_400/60:.1f}x speedup")

    if total_time/n_test < 9:
        print(f"[OK] {total_time/n_test:.1f}s/sample (target: 9s)")
    else:
        print(f"[WARNING] {total_time/n_test:.1f}s/sample (target: 9s)")


if __name__ == '__main__':
    main()
