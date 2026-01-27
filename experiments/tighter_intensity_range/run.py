"""
Tighter Intensity Range Experiment

Test whether reducing q_range from [0.5, 2.0] to [0.6, 1.8] improves CMA-ES convergence.

Usage:
    python run.py [--workers N] [--q-min Q_MIN] [--q-max Q_MAX]
"""

import os
import sys
import pickle
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
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

    q_range = (config.get('q_min', 0.5), config.get('q_max', 2.0))

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=q_range, verbose=False
        )
        elapsed = time.time() - start

        # Check if any candidate hit the bounds
        bound_hits = []
        for sources in candidates:
            for src in sources:
                x, y, q = src
                if abs(q - q_range[0]) < 0.01 or abs(q - q_range[1]) < 0.01:
                    bound_hits.append(q)

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'success': True,
            'bound_hits': bound_hits,
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
            'bound_hits': [],
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    """Calculate sample score using simplified formula."""
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--q-min', type=float, default=0.6, help='Lower bound for intensity')
    parser.add_argument('--q-max', type=float, default=1.8, help='Upper bound for intensity')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"TIGHTER INTENSITY RANGE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Q range: [{args.q_min}, {args.q_max}] (baseline: [0.5, 2.0])")
    print(f"{'='*60}")

    config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': 0.25,
        'refine_maxiter': 3,
        'refine_top_n': 2,
        'q_min': args.q_min,
        'q_max': args.q_max,
    }

    start_time = time.time()
    results = []

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['success']:
                print(f"  Sample {result['idx']:3d}: RMSE={result['best_rmse']:.4f}, "
                      f"n_cand={result['n_candidates']}, "
                      f"bounds_hit={len(result['bound_hits'])}, "
                      f"time={result['elapsed']:.1f}s")
            else:
                print(f"  Sample {result['idx']:3d}: FAILED - {result.get('error', 'unknown')}")

    total_time = time.time() - start_time

    # Calculate metrics
    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    all_bound_hits = [hit for r in results for hit in r.get('bound_hits', [])]
    n_bound_hits = len(all_bound_hits)
    n_samples_with_bound_hits = sum(1 for r in results if r.get('bound_hits', []))

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Score: {score:.4f}")
    print(f"RMSE mean: {np.mean(rmses):.4f}")
    print(f"RMSE mean 1-src: {np.mean(rmses_1src):.4f}")
    print(f"RMSE mean 2-src: {np.mean(rmses_2src):.4f}")
    print(f"Projected 400-sample time: {projected_400:.1f} min")
    print(f"\nBound Hits Analysis:")
    print(f"  Total bound hits: {n_bound_hits}")
    print(f"  Samples with bound hits: {n_samples_with_bound_hits}/{n_samples}")
    if all_bound_hits:
        print(f"  Bound hit values: {sorted(set(all_bound_hits))}")
    print(f"{'='*70}\n")

    # Summary
    summary = {
        'q_range': [args.q_min, args.q_max],
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src),
        'rmse_mean_2src': np.mean(rmses_2src),
        'n_bound_hits': n_bound_hits,
        'n_samples_with_bound_hits': n_samples_with_bound_hits,
    }

    # Log to MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(str(project_root / 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"tighter_intensity_range_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_name', 'tighter_intensity_range')
            mlflow.log_param('experiment_id', 'EXP_TIGHTER_INTENSITY_RANGE_001')
            mlflow.log_param('worker', 'W1')
            mlflow.log_param('q_min', args.q_min)
            mlflow.log_param('q_max', args.q_max)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('score', score)
            mlflow.log_metric('rmse_mean', np.mean(rmses))
            mlflow.log_metric('rmse_mean_1src', np.mean(rmses_1src))
            mlflow.log_metric('rmse_mean_2src', np.mean(rmses_2src))
            mlflow.log_metric('projected_400_min', projected_400)
            mlflow.log_metric('n_bound_hits', n_bound_hits)
            mlflow.log_metric('n_samples_with_bound_hits', n_samples_with_bound_hits)

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
                'q_min': args.q_min,
                'q_max': args.q_max,
                'n_workers': args.workers,
            },
            'results': summary
        })

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    return summary


if __name__ == '__main__':
    main()
