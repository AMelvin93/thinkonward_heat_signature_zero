#!/usr/bin/env python3
"""
Run Sparse L1 Optimization experiment.

Based on compressed sensing principles for heat source identification.

Usage:
    uv run python experiments/sparse_l1_optimization/run.py --workers 7 --shuffle
    uv run python experiments/sparse_l1_optimization/run.py --workers 7 --max-samples 10  # Quick test
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import mlflow
import numpy as np

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Add simulator path
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))

from optimizer import SparseL1Optimizer, N_MAX


LAMBDA = 0.3


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX, max_rmse=1.0):
    """Calculate sample score from candidate RMSEs."""
    valid_rmses = [r for r in rmses if r <= max_rmse]
    n_valid = len(valid_rmses)
    if n_valid == 0:
        return 0.0
    accuracy_sum = sum(1.0 / (1.0 + r) for r in valid_rmses)
    return accuracy_sum / n_valid + lambda_ * (n_valid / n_max)


def load_test_data():
    """Load test dataset."""
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def process_sample(args):
    """Process a single sample."""
    idx, sample, meta, optimizer_config = args

    optimizer = SparseL1Optimizer(**optimizer_config)

    start_time = time.time()
    try:
        candidates, best_rmse, results, n_evals = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )

        elapsed = time.time() - start_time

        # Calculate sample score from candidate RMSEs
        candidate_rmses = [r.rmse for r in results] if results else [best_rmse]
        sample_score = calculate_sample_score(candidate_rmses)

        return {
            'idx': idx,
            'candidates': candidates,
            'candidate_rmses': candidate_rmses,
            'best_rmse': best_rmse,
            'sample_score': sample_score,
            'n_sources': sample['n_sources'],
            'n_evals': n_evals,
            'elapsed': elapsed,
            'init_types': [r.init_type for r in results] if results else [],
            'error': None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'idx': idx,
            'candidates': [],
            'candidate_rmses': [],
            'best_rmse': float('inf'),
            'sample_score': 0.0,
            'n_sources': sample['n_sources'],
            'n_evals': 0,
            'elapsed': elapsed,
            'init_types': [],
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--grid-size', type=str, default='20,10',
                        help='Grid size as "nx,ny" (default: 20,10)')
    parser.add_argument('--lambda-l1', type=float, default=0.1,
                        help='L1 regularization strength')
    parser.add_argument('--refine-fevals-1src', type=int, default=10)
    parser.add_argument('--refine-fevals-2src', type=int, default=18)
    parser.add_argument('--no-mlflow', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Sparse L1 Optimization Experiment")
    print("=" * 70)

    # Parse grid size
    grid_nx, grid_ny = map(int, args.grid_size.split(','))

    # Load data
    data = load_test_data()
    samples = data['samples']
    meta = data['meta']

    n_samples = len(samples)
    if args.max_samples:
        n_samples = min(n_samples, args.max_samples)

    print(f"\nConfig:")
    print(f"  Workers: {args.workers}")
    print(f"  Samples: {n_samples}")
    print(f"  Grid size: {grid_nx}x{grid_ny} = {grid_nx * grid_ny} points")
    print(f"  Lambda L1: {args.lambda_l1}")
    print(f"  Refine fevals: {args.refine_fevals_1src}/{args.refine_fevals_2src}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  Seed: {args.seed}")
    print()

    # Prepare samples
    indices = list(range(n_samples))
    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    optimizer_config = {
        'grid_size': (grid_nx, grid_ny),
        'lambda_l1': args.lambda_l1,
        'max_fevals_refine_1src': args.refine_fevals_1src,
        'max_fevals_refine_2src': args.refine_fevals_2src,
    }

    sample_args = [
        (idx, samples[idx], meta, optimizer_config)
        for idx in indices
    ]

    # Run optimization
    results_by_idx = {}
    pred_dataset = [None] * len(samples)

    total_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, arg): arg[0] for arg in sample_args}

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result['idx']
            results_by_idx[idx] = result

            pred_dataset[idx] = result['candidates'] if result['candidates'] else []

            completed += 1
            if completed % 10 == 0 or completed == n_samples:
                elapsed = time.time() - total_start
                avg_time = elapsed / completed
                remaining = avg_time * (n_samples - completed)
                print(f"  [{completed}/{n_samples}] Elapsed: {elapsed:.1f}s, "
                      f"Remaining: {remaining:.1f}s, "
                      f"Last RMSE: {result['best_rmse']:.4f}")

    total_time = time.time() - total_start
    projected_400 = (total_time / n_samples) * 400 / 60

    # Analyze results
    rmse_1src = []
    rmse_2src = []
    sample_scores = []
    init_type_counts = defaultdict(int)
    n_errors = 0

    for idx, result in results_by_idx.items():
        if result['error']:
            n_errors += 1
            print(f"  ERROR in sample {idx}: {result['error']}")
            continue

        if result['n_sources'] == 1:
            rmse_1src.append(result['best_rmse'])
        else:
            rmse_2src.append(result['best_rmse'])

        sample_scores.append(result['sample_score'])

        for itype in result['init_types']:
            init_type_counts[itype] += 1

    # Calculate overall score (mean of sample scores)
    score = np.mean(sample_scores) if sample_scores else 0.0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_rmse = rmse_1src + rmse_2src
    mean_rmse = np.mean(all_rmse) if all_rmse else 0
    std_rmse = np.std(all_rmse) if all_rmse else 0

    print(f"RMSE:             {mean_rmse:.6f} +/- {std_rmse:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    print(f"  1-source: RMSE={np.mean(rmse_1src):.4f} (n={len(rmse_1src)})")
    print(f"  2-source: RMSE={np.mean(rmse_2src):.4f} (n={len(rmse_2src)})")
    print()

    if init_type_counts:
        total_inits = sum(init_type_counts.values())
        print("Init types:")
        for itype, count in sorted(init_type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_inits
            print(f"  {itype}: {count} ({pct:.1f}%)")
        print()

    if n_errors > 0:
        print(f"Errors: {n_errors}")
        print()

    # Compare to baseline
    print("-" * 70)
    print(f"Baseline: 1.0329 @ 57.0 min (A3b Adaptive Budget)")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")

    if score > 1.0329 and projected_400 < 60:
        print("[IMPROVED!]")
    elif projected_400 >= 60:
        print("[OVER BUDGET]")
    else:
        print("[NO IMPROVEMENT]")
    print("=" * 70)

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_experiment("sparse_l1_optimization")
        with mlflow.start_run():
            mlflow.log_params({
                'optimizer': 'SparseL1Optimizer',
                'grid_nx': grid_nx,
                'grid_ny': grid_ny,
                'lambda_l1': args.lambda_l1,
                'refine_fevals_1src': args.refine_fevals_1src,
                'refine_fevals_2src': args.refine_fevals_2src,
                'workers': args.workers,
                'n_samples': n_samples,
                'shuffle': args.shuffle,
                'seed': args.seed,
            })
            mlflow.log_metrics({
                'submission_score': score,
                'rmse_mean': mean_rmse,
                'rmse_std': std_rmse,
                'rmse_1src': np.mean(rmse_1src) if rmse_1src else 0,
                'rmse_2src': np.mean(rmse_2src) if rmse_2src else 0,
                'total_time_sec': total_time,
                'projected_400_samples_min': projected_400,
                'n_errors': n_errors,
            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(
        _project_root, 'results',
        f'sparse_l1_{grid_nx}x{grid_ny}_{timestamp}_results.pkl'
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'results': results_by_idx,
            'pred_dataset': pred_dataset,
            'score': score,
            'rmse_mean': mean_rmse,
            'projected_400_min': projected_400,
        }, f)

    print(f"\nResults saved to: {results_path}")

    return score, projected_400


if __name__ == '__main__':
    main()
