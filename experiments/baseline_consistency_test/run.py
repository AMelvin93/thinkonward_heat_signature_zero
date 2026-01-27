"""
Baseline Consistency Test

Run the baseline (early_timestep_filtering) multiple times with different seeds
to measure variance and establish confidence intervals.

Usage:
    python run.py [--workers N] [--seeds S1 S2 S3]
"""

import os
import sys
import pickle
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import baseline optimizer
sys.path.insert(0, str(project_root / 'experiments' / 'early_timestep_filtering'))
from optimizer import TemporalFidelityOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = TemporalFidelityOptimizer(
        max_fevals_1src=config.get('max_fevals_1src', 20),
        max_fevals_2src=config.get('max_fevals_2src', 36),
        timestep_fraction=config.get('timestep_fraction', 0.25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def run_single_test(samples, meta, workers, seed, config):
    """Run a single consistency test with given seed."""
    np.random.seed(seed)

    n_samples = len(samples)
    indices = np.arange(n_samples)

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    # Calculate score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (total_time / n_samples) * 400 / 60

    return {
        'seed': seed,
        'score': score,
        'projected_400_min': projected_400,
        'total_time_sec': total_time,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else 0,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds to test')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = len(samples) - n_1src

    print(f"\n{'='*60}")
    print(f"BASELINE CONSISTENCY TEST")
    print(f"{'='*60}")
    print(f"Samples: {len(samples)} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Seeds to test: {args.seeds}")
    print(f"{'='*60}")

    config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': 0.25,
        'refine_maxiter': 3,
        'refine_top_n': 2,
    }

    all_results = []

    for seed in args.seeds:
        print(f"\n--- Running with seed {seed} ---")
        result = run_single_test(samples, meta, args.workers, seed, config)
        all_results.append(result)
        print(f"Seed {seed}: Score={result['score']:.4f}, Time={result['projected_400_min']:.1f} min")

    # Calculate statistics
    scores = [r['score'] for r in all_results]
    times = [r['projected_400_min'] for r in all_results]

    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_min = min(scores)
    score_max = max(scores)

    time_mean = np.mean(times)
    time_std = np.std(times)

    print(f"\n{'='*70}")
    print("CONSISTENCY TEST RESULTS")
    print(f"{'='*70}")
    print(f"\nScore Statistics:")
    print(f"  Mean:  {score_mean:.4f}")
    print(f"  Std:   {score_std:.4f}")
    print(f"  Min:   {score_min:.4f}")
    print(f"  Max:   {score_max:.4f}")
    print(f"  Range: {score_max - score_min:.4f}")
    print(f"\nTime Statistics:")
    print(f"  Mean:  {time_mean:.1f} min")
    print(f"  Std:   {time_std:.1f} min")
    print(f"\n95% Confidence Interval for Score:")
    ci_low = score_mean - 1.96 * score_std
    ci_high = score_mean + 1.96 * score_std
    print(f"  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"\nTo beat baseline with 95% confidence, new score must be > {ci_high:.4f}")
    print(f"{'='*70}\n")

    # Summary
    summary = {
        'seeds': args.seeds,
        'n_runs': len(all_results),
        'individual_results': all_results,
        'score_mean': score_mean,
        'score_std': score_std,
        'score_min': score_min,
        'score_max': score_max,
        'score_range': score_max - score_min,
        'time_mean': time_mean,
        'time_std': time_std,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'min_score_to_beat': ci_high,
    }

    # Log to MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(str(project_root / 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"baseline_consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_name', 'baseline_consistency_test')
            mlflow.log_param('experiment_id', 'EXP_BASELINE_CONSISTENCY_TEST_001')
            mlflow.log_param('worker', 'W1')
            mlflow.log_param('n_seeds', len(args.seeds))
            mlflow.log_param('seeds', str(args.seeds))
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('score_mean', score_mean)
            mlflow.log_metric('score_std', score_std)
            mlflow.log_metric('score_min', score_min)
            mlflow.log_metric('score_max', score_max)
            mlflow.log_metric('score_range', score_max - score_min)
            mlflow.log_metric('time_mean', time_mean)
            mlflow.log_metric('time_std', time_std)
            mlflow.log_metric('ci_95_low', ci_low)
            mlflow.log_metric('ci_95_high', ci_high)

            mlflow_run_id = run.info.run_id
            print(f"MLflow run ID: {mlflow_run_id}")
            summary['mlflow_run_id'] = mlflow_run_id

    # Update STATE.json
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)

        state['tuning_runs'].append({
            'run': len(state['tuning_runs']) + 1,
            'config': {
                'seeds': args.seeds,
                'n_workers': args.workers,
            },
            'results': summary
        })

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    return summary


if __name__ == '__main__':
    main()
