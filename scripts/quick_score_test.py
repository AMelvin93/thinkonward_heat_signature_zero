#!/usr/bin/env python
"""
Quick score test with time-constrained configurations.

Target: 60 min for 400 samples on G4dn.2xlarge (7 workers)
= ~63 seconds per sample

This tests configurations that can fit within this constraint.
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
from src.hybrid_optimizer import HybridOptimizer

# Scoring parameters
LAMBDA = 0.3
TAU = 0.2
N_MAX = 3


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX):
    """Calculate score for a single sample."""
    n_valid = len(rmses)
    if n_valid == 0:
        return 0.0
    accuracy_term = sum(1.0 / (1.0 + rmse) for rmse in rmses) / n_valid
    diversity_term = lambda_ * (n_valid / n_max)
    return accuracy_term + diversity_term


def process_sample(sample, meta, config):
    """Process a single sample with given configuration."""
    optimizer = HybridOptimizer(
        Lx=2.0, Ly=1.0, nx=100, ny=50,
        n_smart_inits=config['n_smart_inits'],
        n_random_inits=config['n_random_inits'],
        min_candidate_distance=TAU,
        n_max_candidates=config['n_max_candidates'],
    )

    start = time.time()
    estimates, best_rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=(0.5, 2.0),
        max_iter=config['max_iter'],
        parallel=False,  # Sequential within sample
        verbose=False,
    )
    elapsed = time.time() - start

    rmses = [c.rmse for c in candidates]
    score = calculate_sample_score(rmses)

    return {
        'rmses': rmses,
        'n_candidates': len(candidates),
        'best_rmse': best_rmse,
        'time': elapsed,
        'score': score,
    }


def main():
    print("=" * 60)
    print("QUICK SCORE TEST (Per-sample timing)")
    print("Target: <63s per sample for G4dn.2xlarge")
    print("=" * 60)

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    print(f"Loaded {len(samples)} samples")

    # Test a subset
    n_test = 5
    test_samples = samples[:n_test]

    # Configurations to test (ordered by expected speed)
    configs = [
        # Fast: 1 candidate, minimal optimization
        {'n_smart_inits': 1, 'n_random_inits': 0, 'max_iter': 5,
         'n_max_candidates': 1, 'name': '1 init, 5 iter, 1 cand'},

        # Medium: 1 candidate, more iterations
        {'n_smart_inits': 1, 'n_random_inits': 0, 'max_iter': 10,
         'n_max_candidates': 1, 'name': '1 init, 10 iter, 1 cand'},

        # Diversity: 2 candidates
        {'n_smart_inits': 2, 'n_random_inits': 0, 'max_iter': 5,
         'n_max_candidates': 2, 'name': '2 init, 5 iter, 2 cand'},

        # Diversity: 3 candidates (target for max diversity bonus)
        {'n_smart_inits': 3, 'n_random_inits': 0, 'max_iter': 5,
         'n_max_candidates': 3, 'name': '3 init, 5 iter, 3 cand'},

        # Balance: 2 inits, more iterations
        {'n_smart_inits': 2, 'n_random_inits': 0, 'max_iter': 8,
         'n_max_candidates': 2, 'name': '2 init, 8 iter, 2 cand'},
    ]

    results_summary = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print("=" * 60)

        times = []
        scores = []
        rmses_all = []
        n_cands = []

        for i, sample in enumerate(test_samples):
            result = process_sample(sample, meta, config)
            times.append(result['time'])
            scores.append(result['score'])
            rmses_all.extend(result['rmses'])
            n_cands.append(result['n_candidates'])
            print(f"  Sample {i}: {result['time']:.1f}s, "
                  f"RMSE={result['best_rmse']:.4f}, "
                  f"cands={result['n_candidates']}, "
                  f"score={result['score']:.4f}")

        avg_time = np.mean(times)
        avg_score = np.mean(scores)
        avg_rmse = np.mean(rmses_all)
        avg_cands = np.mean(n_cands)

        # Project for 400 samples with 7 workers
        projected_400 = avg_time * 400 / 7 / 60  # minutes

        print(f"\nSummary:")
        print(f"  Avg time/sample: {avg_time:.1f}s")
        print(f"  Avg score: {avg_score:.4f}")
        print(f"  Avg RMSE: {avg_rmse:.4f}")
        print(f"  Avg candidates: {avg_cands:.1f}")
        print(f"  Projected 400 samples (7 workers): {projected_400:.1f} min")

        if projected_400 < 60:
            print(f"  [OK] Under 60 min target!")
            results_summary.append({
                'config': config['name'],
                'time': avg_time,
                'score': avg_score,
                'rmse': avg_rmse,
                'projected': projected_400,
                'feasible': True,
            })
        else:
            print(f"  [WARNING] Over 60 min target - need {projected_400/60:.1f}x speedup")
            results_summary.append({
                'config': config['name'],
                'time': avg_time,
                'score': avg_score,
                'rmse': avg_rmse,
                'projected': projected_400,
                'feasible': False,
            })

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - FEASIBLE CONFIGURATIONS")
    print("=" * 60)
    print(f"{'Config':<30} {'Time':>8} {'Score':>8} {'RMSE':>8} {'Proj':>8}")
    print("-" * 60)

    feasible = [r for r in results_summary if r['feasible']]
    for r in feasible:
        print(f"{r['config']:<30} {r['time']:>7.1f}s {r['score']:>8.4f} "
              f"{r['rmse']:>8.4f} {r['projected']:>7.1f}m")

    if feasible:
        best = max(feasible, key=lambda x: x['score'])
        print(f"\nBest score within time: {best['config']}")
        print(f"  Score: {best['score']:.4f}")
        print(f"  Projected time: {best['projected']:.1f} min")

    # Theoretical max score analysis
    print("\n" + "=" * 60)
    print("SCORE ANALYSIS")
    print("=" * 60)
    print("Score = (1/N_valid) * sum(1/(1+RMSE_i)) + 0.3 * (N_valid/3)")
    print("\nTheoretical bounds:")
    print("  Max score (RMSE=0, 3 cands): 1 + 0.3 = 1.30")
    print("  Good score (RMSE=0.1, 3 cands): 0.91 + 0.3 = 1.21")
    print("  Decent score (RMSE=0.3, 3 cands): 0.77 + 0.3 = 1.07")
    print("  1 cand, RMSE=0: 1 + 0.1 = 1.10")
    print("  1 cand, RMSE=0.1: 0.91 + 0.1 = 1.01")


if __name__ == '__main__':
    main()
