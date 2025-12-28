#!/usr/bin/env python
"""
Test fast NumPy optimizer with minimal iterations.

Target: 60 min for 400 samples with 7-way parallelism
= 63s per sample
= ~70 forward passes (at 0.9s each)
= ~6 L-BFGS-B iterations with numerical gradients

Run: uv run python scripts/test_fast_numpy.py
"""

import sys
import os
import time
import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import numpy as np
from joblib import Parallel, delayed
from src.hybrid_optimizer import HybridOptimizer


def process_sample(sample, meta, max_iter=5):
    """Process single sample with minimal iterations."""
    optimizer = HybridOptimizer(
        Lx=2.0, Ly=1.0, nx=100, ny=50,
        n_smart_inits=1,      # Only 1 smart init
        n_random_inits=0,     # No random inits
        min_candidate_distance=0.15,
        n_max_candidates=1,
    )

    start = time.time()
    estimates, rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=(0.5, 2.0),
        max_iter=max_iter,
        parallel=False,
        verbose=False,
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'rmse': rmse,
        'time': elapsed,
    }


def main():
    print("=" * 60)
    print("FAST NUMPY OPTIMIZER TEST")
    print("=" * 60)

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    print(f"Loaded {len(samples)} samples")

    n_cpus = os.cpu_count()
    n_workers = max(1, n_cpus - 1)
    print(f"CPUs: {n_cpus}, Workers: {n_workers}")

    # Test different iteration counts
    for max_iter in [5, 8, 10]:
        print(f"\n{'='*60}")
        print(f"Testing max_iter={max_iter}")
        print("=" * 60)

        n_test = 10
        test_samples = samples[:n_test]

        # Sequential test first (to measure per-sample time)
        print(f"\nSequential test ({n_test} samples)...")
        seq_times = []
        seq_rmses = []

        for i, sample in enumerate(test_samples):
            result = process_sample(sample, meta, max_iter=max_iter)
            seq_times.append(result['time'])
            seq_rmses.append(result['rmse'])
            print(f"  Sample {i}: {result['time']:.1f}s, RMSE={result['rmse']:.4f}")

        avg_time = np.mean(seq_times)
        avg_rmse = np.mean(seq_rmses)

        print(f"\nSequential results:")
        print(f"  Avg time: {avg_time:.1f}s")
        print(f"  Avg RMSE: {avg_rmse:.4f}")

        # Project for 400 samples
        projected_seq = avg_time * 400 / 60
        projected_par = avg_time * 400 / n_workers / 60

        print(f"\nProjections for 400 samples:")
        print(f"  Sequential: {projected_seq:.1f} min")
        print(f"  Parallel ({n_workers} workers): {projected_par:.1f} min")

        if projected_par < 60:
            print(f"  [OK] Under 60 min target!")
        else:
            print(f"  [WARNING] Need {projected_par/60:.1f}x more speedup")

    # Full parallel test with best setting
    print(f"\n{'='*60}")
    print("PARALLEL TEST (max_iter=5, all samples)")
    print("=" * 60)

    start = time.time()
    results = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(process_sample)(sample, meta, max_iter=5)
        for sample in samples
    )
    total_time = time.time() - start

    rmses = [r['rmse'] for r in results]
    times = [r['time'] for r in results]

    print(f"\nFull dataset results ({len(samples)} samples):")
    print(f"  Total time: {total_time:.1f}s = {total_time/60:.1f} min")
    print(f"  Avg time/sample: {np.mean(times):.1f}s")
    print(f"  Avg RMSE: {np.mean(rmses):.4f}")
    print(f"  RMSE std: {np.std(rmses):.4f}")

    # Project for 400 samples
    projected_400 = total_time * (400 / len(samples)) / 60
    print(f"\nProjected for 400 samples: {projected_400:.1f} min")

    if projected_400 < 60:
        print("[OK] Under 60 min target!")
    else:
        print(f"[WARNING] Need {projected_400/60:.1f}x more speedup")


if __name__ == '__main__':
    main()
