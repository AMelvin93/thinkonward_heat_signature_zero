#!/usr/bin/env python
"""
Calculate the competition score for our optimizer.

Score = (1/N_valid) * sum(1/(1+L_i)) + lambda * (N_valid/N_max)

Where:
- L_i = RMSE for candidate i
- lambda = 0.3
- N_max = 3
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


# Scoring parameters (from starter notebook)
LAMBDA = 0.3
TAU = 0.2
N_MAX = 3
LX, LY = 2.0, 1.0
Q_RANGE = (0.5, 2.0)
SCALE_FACTORS = (LX, LY, Q_RANGE[1])


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX):
    """
    Calculate score for a single sample given RMSE values for each candidate.

    Score = (1/N_valid) * sum(1/(1+L_i)) + lambda * (N_valid/N_max)
    """
    n_valid = len(rmses)
    if n_valid == 0:
        return 0.0

    accuracy_term = sum(1.0 / (1.0 + rmse) for rmse in rmses) / n_valid
    diversity_term = lambda_ * (n_valid / n_max)

    return accuracy_term + diversity_term


def process_sample(sample, meta, n_smart_inits=2, n_random_inits=1, max_iter=10):
    """
    Process single sample and return candidates with RMSEs.
    """
    optimizer = HybridOptimizer(
        Lx=2.0, Ly=1.0, nx=100, ny=50,
        n_smart_inits=n_smart_inits,
        n_random_inits=n_random_inits,
        min_candidate_distance=TAU,  # Use tau for filtering
        n_max_candidates=N_MAX,
    )

    start = time.time()
    estimates, best_rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=Q_RANGE,
        max_iter=max_iter,
        parallel=True,
        verbose=False,
    )
    elapsed = time.time() - start

    # Get RMSEs for all candidates
    rmses = [c.rmse for c in candidates]

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'rmses': rmses,
        'n_candidates': len(candidates),
        'best_rmse': best_rmse,
        'time': elapsed,
    }


def main():
    print("=" * 60)
    print("COMPETITION SCORE CALCULATION")
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
    print(f"Using {n_workers} parallel workers")

    # Test different configurations
    configs = [
        {'n_smart_inits': 1, 'n_random_inits': 0, 'max_iter': 5, 'name': '1 smart, 5 iter'},
        {'n_smart_inits': 2, 'n_random_inits': 1, 'max_iter': 8, 'name': '3 inits, 8 iter'},
        {'n_smart_inits': 3, 'n_random_inits': 2, 'max_iter': 10, 'name': '5 inits, 10 iter'},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print("=" * 60)

        start = time.time()
        results = Parallel(n_jobs=n_workers, verbose=5)(
            delayed(process_sample)(
                sample, meta,
                n_smart_inits=config['n_smart_inits'],
                n_random_inits=config['n_random_inits'],
                max_iter=config['max_iter']
            )
            for sample in samples
        )
        total_time = time.time() - start

        # Calculate scores
        sample_scores = []
        all_rmses = []
        n_candidates_list = []

        for r in results:
            score = calculate_sample_score(r['rmses'])
            sample_scores.append(score)
            all_rmses.extend(r['rmses'])
            n_candidates_list.append(r['n_candidates'])

        final_score = np.mean(sample_scores)

        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Avg time/sample: {np.mean([r['time'] for r in results]):.1f}s")
        print(f"  Avg candidates/sample: {np.mean(n_candidates_list):.1f}")
        print(f"  Avg RMSE (all candidates): {np.mean(all_rmses):.4f}")
        print(f"  Avg best RMSE: {np.mean([r['best_rmse'] for r in results]):.4f}")
        print(f"\n  FINAL SCORE: {final_score:.4f}")

        # Theoretical analysis
        print(f"\n  Score breakdown:")
        avg_rmse = np.mean(all_rmses)
        avg_n_valid = np.mean(n_candidates_list)
        accuracy = (1.0 / (1.0 + avg_rmse))
        diversity = LAMBDA * (avg_n_valid / N_MAX)
        print(f"    Accuracy component (1/(1+RMSE)): {accuracy:.4f}")
        print(f"    Diversity component (0.3 * N/3): {diversity:.4f}")

        # Projection for 400 samples
        projected_400 = total_time * (400 / len(samples)) / 60
        print(f"\n  Projected for 400 samples: {projected_400:.1f} min")
        if projected_400 < 60:
            print(f"  [OK] Under 60 min target!")
        else:
            print(f"  [WARNING] Over 60 min target")


if __name__ == '__main__':
    main()
