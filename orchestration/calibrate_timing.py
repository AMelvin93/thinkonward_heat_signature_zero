#!/usr/bin/env python3
"""
Timing Calibration Script

Run this ONCE in a single container to verify timing matches G4dn.2xlarge.

Expected baseline: ~57 min for 80 samples (projects to ~57 min for 400 samples)

Usage:
    # Inside a worker container:
    uv run python /workspace/orchestration/calibrate_timing.py

This runs the exact baseline config and compares against known timing.
"""

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _project_root)

# Import the production optimizer
from src.robust_fallback_optimizer import RobustFallbackOptimizer


def process_single_sample(args):
    """Process one sample with baseline config."""
    idx, sample, meta = args

    # EXACT baseline config from best result
    optimizer = RobustFallbackOptimizer(
        max_fevals_1src=20,
        max_fevals_2src=36,
        rmse_threshold_1src=0.35,
        rmse_threshold_2src=0.45,
        fallback_fevals=18,
        sigma0_1src=0.15,
        sigma0_2src=0.20,
    )

    start = time.time()
    try:
        candidates, best_rmse, _, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        return {
            'idx': idx,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'elapsed': elapsed,
            'success': True,
        }
    except Exception as e:
        return {
            'idx': idx,
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'elapsed': time.time() - start,
            'success': False,
            'error': str(e),
        }


def calculate_score(results):
    """Calculate submission score."""
    def sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    scores = [sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    return np.mean(scores)


def main():
    print("\n" + "=" * 70)
    print("        TIMING CALIBRATION - G4dn.2xlarge Equivalent")
    print("=" * 70)

    # Load data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    # Use 7 workers (G4dn.2xlarge has 8 vCPUs, use 7 for workers)
    n_workers = 7

    print(f"\nConfig: Baseline (0.35/0.45 thresholds, 18 fallback fevals)")
    print(f"Samples: {n_samples}")
    print(f"Workers: {n_workers}")
    print(f"\nExpected: ~57 min for 80 samples (based on G4dn.2xlarge)")
    print("=" * 70)
    print("\nRunning calibration...\n")

    # Prepare work items
    np.random.seed(42)  # Same seed as baseline
    work_items = [(i, samples[i], meta) for i in range(n_samples)]

    # Run
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (len(results) % 10 == 0) or len(results) == n_samples:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed * 60  # samples per minute
                eta = (n_samples - len(results)) / (len(results) / elapsed) if len(results) > 0 else 0
                print(f"[{len(results):3d}/{n_samples}] {elapsed/60:.1f} min elapsed, "
                      f"{rate:.1f} samples/min, ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time
    total_min = total_time / 60
    projected_400 = (total_time / n_samples) * 400 / 60

    # Calculate score
    score = calculate_score(results)

    # Results
    print("\n" + "=" * 70)
    print("        CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nTiming:")
    print(f"  This run:     {total_min:.1f} min for {n_samples} samples")
    print(f"  Projected:    {projected_400:.1f} min for 400 samples")
    print(f"  Expected:     57.2 min (G4dn.2xlarge baseline)")

    ratio = total_min / 57.2 * (80 / n_samples)  # Normalize to 80 samples
    print(f"\n  Timing ratio: {ratio:.2f}x vs G4dn.2xlarge")

    if ratio > 1.1:
        print(f"  WARNING: Your system is {(ratio-1)*100:.0f}% SLOWER than G4dn.2xlarge")
        print(f"           Apply {ratio:.2f}x correction to timing estimates")
    elif ratio < 0.9:
        print(f"  NOTE: Your system is {(1-ratio)*100:.0f}% FASTER than G4dn.2xlarge")
        print(f"        Apply {ratio:.2f}x correction to timing estimates")
    else:
        print(f"  GOOD: Timing is within 10% of G4dn.2xlarge")

    print(f"\nScore:")
    print(f"  This run:     {score:.4f}")
    print(f"  Expected:     1.1247 (baseline)")
    print(f"  Difference:   {score - 1.1247:+.4f}")

    # Save calibration result
    calibration = {
        'timing_ratio': ratio,
        'score': score,
        'total_min': total_min,
        'projected_400_min': projected_400,
        'n_samples': n_samples,
        'n_workers': n_workers,
    }

    calibration_path = os.path.join(_project_root, 'orchestration', 'shared', 'calibration.json')
    import json
    with open(calibration_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: orchestration/shared/calibration.json")
    print(f"Workers can use timing_ratio={ratio:.2f} to correct estimates")
    print("=" * 70 + "\n")

    return ratio, score


if __name__ == '__main__':
    main()
