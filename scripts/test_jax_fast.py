#!/usr/bin/env python
"""
Test script for JAX Fast Optimizer (autodiff + L-BFGS-B).

Run from WSL: uv run python scripts/test_jax_fast.py
"""

import sys
import os
import time
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_fast_optimizer import JAXFastOptimizer, check_gpu
import numpy as np


def main():
    print("=" * 60)
    print("JAX FAST OPTIMIZER TEST (Autodiff + L-BFGS-B)")
    print("=" * 60)

    # Check GPU
    device_info = check_gpu()
    print(f"\nDevice: {device_info}")

    # Load test data
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'heat-signature-zero-test-data.pkl'
    )
    print(f"\nLoading data from: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    print(f"Loaded {len(samples)} samples")

    # Initialize optimizer
    optimizer = JAXFastOptimizer(
        n_smart_inits=1,
        n_random_inits=2,
        min_candidate_distance=0.15,
        n_max_candidates=3,
    )

    # Test on first N samples
    n_test = 5
    test_samples = samples[:n_test]

    print(f"\nTesting on {n_test} samples...")
    print("-" * 60)

    total_time = 0
    rmses = []

    for i, sample in enumerate(test_samples):
        n_sources = sample['n_sources']
        print(f"\nSample {i}: {n_sources} source(s)")

        start = time.time()
        sources, rmse, candidates = optimizer.estimate_sources(
            sample, meta, max_iter=30, verbose=True
        )
        elapsed = time.time() - start

        total_time += elapsed
        rmses.append(rmse)

        print(f"  Time: {elapsed:.1f}s, RMSE: {rmse:.4f}")
        for j, (x, y, q) in enumerate(sources):
            print(f"    Source {j}: x={x:.3f}, y={y:.3f}, q={q:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples processed: {n_test}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per sample: {total_time/n_test:.1f}s")
    print(f"Avg RMSE: {np.mean(rmses):.4f}")
    print(f"\nProjected time for 400 samples: {total_time/n_test * 400 / 60:.1f} min")

    # Target check
    target_time = 60  # minutes
    projected = total_time / n_test * 400 / 60
    if projected < target_time:
        print(f"\n[OK] Under {target_time} min target!")
    else:
        print(f"\n[WARNING] Over {target_time} min target. Need {projected/target_time:.1f}x speedup.")


if __name__ == '__main__':
    main()
