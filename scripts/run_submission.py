#!/usr/bin/env python
"""
Run final competition submission.

This script runs the optimized NumPy-based hybrid optimizer on all test samples
and calculates the competition score.

Target: Complete 400 samples in <60 min on G4dn.2xlarge (7 workers)

Usage:
    uv run python scripts/run_submission.py
    uv run python scripts/run_submission.py --n-workers 7  # For G4dn
"""

import sys
import os
import time
import pickle
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import numpy as np
from joblib import Parallel, delayed
from src.hybrid_optimizer import HybridOptimizer


# Scoring parameters (from competition rules)
LAMBDA = 0.3
TAU = 0.2
N_MAX = 3
MAX_RMSE_THRESHOLD = 1.0


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX, max_rmse=MAX_RMSE_THRESHOLD):
    """
    Calculate competition score for a single sample.

    Score = (1/N_valid) * sum(1/(1+RMSE_i)) + lambda * (N_valid/N_max)

    Candidates with RMSE > max_rmse are excluded.
    """
    good_rmses = [r for r in rmses if r <= max_rmse]
    n_valid = len(good_rmses)
    if n_valid == 0:
        return 0.0

    accuracy_term = sum(1.0 / (1.0 + rmse) for rmse in good_rmses) / n_valid
    diversity_term = lambda_ * (n_valid / n_max)

    return accuracy_term + diversity_term


def process_sample(sample, meta, config):
    """Process a single sample and return results."""
    optimizer = HybridOptimizer(
        Lx=config['Lx'],
        Ly=config['Ly'],
        nx=config['nx'],
        ny=config['ny'],
        n_smart_inits=config['n_smart_inits'],
        n_random_inits=config['n_random_inits'],
        min_candidate_distance=config['min_candidate_distance'],
        n_max_candidates=config['n_max_candidates'],
    )

    start = time.time()
    estimates, best_rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=tuple(config['q_range']),
        max_iter=config['max_iter'],
        parallel=config['parallel'],
        verbose=False,
    )
    elapsed = time.time() - start

    rmses = [c.rmse for c in candidates]
    score = calculate_sample_score(rmses)

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': estimates,
        'rmses': rmses,
        'n_candidates': len(candidates),
        'best_rmse': best_rmse,
        'score': score,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Run competition submission')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPUs - 1)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--max-iter', type=int, default=2,
                        help='L-BFGS-B iterations (default: 2, try 3 if G4dn is fast)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample results')
    args = parser.parse_args()

    print("=" * 60)
    print("COMPETITION SUBMISSION")
    print("=" * 60)

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    if args.n_samples:
        samples = samples[:args.n_samples]

    print(f"Loaded {len(samples)} samples")

    # Configuration (optimized for time/score tradeoff)
    # Tested on Windows 32-core with 7 workers:
    # - max_iter=2: 54.7 min projected, score 0.52 [SAFE]
    # - max_iter=3: 68.7 min projected, score 0.57 [slightly over]
    # G4dn.2xlarge (Linux) may be 10-20% faster
    config = {
        'Lx': 2.0,
        'Ly': 1.0,
        'nx': 100,
        'ny': 50,
        'n_smart_inits': 1,
        'n_random_inits': 0,
        'min_candidate_distance': TAU,
        'n_max_candidates': 1,
        'max_iter': args.max_iter,  # Default 2; try 3 if G4dn is fast
        'parallel': False,
        'q_range': [0.5, 2.0],
    }

    n_workers = args.n_workers if args.n_workers else max(1, os.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers")
    print(f"Config: {config['n_smart_inits']} smart inits, {config['max_iter']} iterations")

    # Run optimization
    print(f"\nProcessing {len(samples)} samples...")
    start_total = time.time()

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_sample)(sample, meta, config)
        for sample in samples
    )

    total_time = time.time() - start_total

    # Aggregate results
    scores = [r['score'] for r in results]
    times = [r['time'] for r in results]
    rmses = []
    for r in results:
        rmses.extend([x for x in r['rmses'] if x <= MAX_RMSE_THRESHOLD])

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if args.verbose:
        print("\nPer-sample results:")
        for r in results:
            print(f"  Sample {r['sample_id']}: RMSE={r['best_rmse']:.4f}, "
                  f"score={r['score']:.4f}, time={r['time']:.1f}s")

    print(f"\nSummary ({len(samples)} samples):")
    print(f"  Total time: {total_time:.1f}s = {total_time/60:.1f} min")
    print(f"  Avg time/sample: {np.mean(times):.1f}s")
    print(f"  Avg RMSE: {np.mean(rmses):.4f}" if rmses else "  No valid candidates!")
    print(f"  Avg score: {np.mean(scores):.4f}")
    print(f"  Min/Max score: {np.min(scores):.4f} / {np.max(scores):.4f}")

    # Calculate final competition score
    final_score = np.mean(scores)
    print(f"\n  FINAL COMPETITION SCORE: {final_score:.4f}")

    # Projection for 400 samples
    if len(samples) < 400:
        projected_400 = total_time * (400 / len(samples)) / 60
        print(f"\n  Projected for 400 samples: {projected_400:.1f} min")
        if projected_400 < 60:
            print(f"  [OK] Under 60 min target!")
        else:
            print(f"  [WARNING] Over 60 min target by {projected_400 - 60:.1f} min")

    # Save results
    output_path = project_root / "results" / "submission_results.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'config': config,
            'final_score': final_score,
            'total_time': total_time,
        }, f)
    print(f"\nResults saved to: {output_path}")

    return final_score


if __name__ == '__main__':
    main()
