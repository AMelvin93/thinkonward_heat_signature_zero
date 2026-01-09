#!/usr/bin/env python
"""
Test script for Ultra-Fast JAX Optimizer.

Target: 400 samples in <60 min = 9s per sample

Run from WSL: uv run python scripts/test_ultrafast.py
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_ultrafast_optimizer import JAXUltraFastOptimizer, precompile_for_dataset, check_gpu
import numpy as np


def main():
    print("=" * 60)
    print("ULTRA-FAST JAX OPTIMIZER TEST")
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

    # Pre-compile for all nt values (one-time cost)
    print("\n" + "-" * 60)
    print("PRE-COMPILATION PHASE")
    print("-" * 60)
    precompile_start = time.time()
    precompile_for_dataset(samples)
    precompile_time = time.time() - precompile_start
    print(f"Pre-compilation time: {precompile_time:.1f}s")

    # Initialize optimizer
    optimizer = JAXUltraFastOptimizer(
        n_smart_inits=1,
        n_random_inits=1,
        n_max_candidates=3,
    )

    # Test on first N samples
    n_test = 10
    test_samples = samples[:n_test]

    print("\n" + "-" * 60)
    print(f"TESTING ON {n_test} SAMPLES")
    print("-" * 60)

    total_time = 0
    rmses = []
    times_per_sample = []

    for i, sample in enumerate(test_samples):
        n_sources = sample['n_sources']
        nt = sample['sample_metadata']['nt']

        start = time.time()
        sources, rmse, candidates = optimizer.estimate_sources(
            sample, meta, max_iter=20, verbose=False
        )
        elapsed = time.time() - start

        total_time += elapsed
        rmses.append(rmse)
        times_per_sample.append(elapsed)

        print(f"Sample {i:2d}: nt={nt}, {n_sources} src, RMSE={rmse:.4f}, time={elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples processed: {n_test}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per sample: {total_time/n_test:.1f}s")
    print(f"Min/Max time: {min(times_per_sample):.1f}s / {max(times_per_sample):.1f}s")
    print(f"Avg RMSE: {np.mean(rmses):.4f}")

    # Projections
    projected_80 = total_time / n_test * 80 / 60
    projected_400 = total_time / n_test * 400 / 60

    print(f"\nProjected time for 80 samples: {projected_80:.1f} min")
    print(f"Projected time for 400 samples: {projected_400:.1f} min")

    # Target check
    target_time = 60  # minutes for 400 samples
    if projected_400 < target_time:
        print(f"\n[OK] Under {target_time} min target for 400 samples!")
    else:
        speedup_needed = projected_400 / target_time
        print(f"\n[WARNING] Over {target_time} min target. Need {speedup_needed:.1f}x more speedup.")

    # Per-sample target
    per_sample_target = 9.0  # seconds
    avg_per_sample = total_time / n_test
    if avg_per_sample < per_sample_target:
        print(f"[OK] Under {per_sample_target}s per sample target!")
    else:
        print(f"[WARNING] {avg_per_sample:.1f}s per sample, target is {per_sample_target}s")


if __name__ == '__main__':
    main()
