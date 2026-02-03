"""
TEMPLATE: Run script with proper seeding for reproducibility.

This demonstrates the correct pattern for seeding in parallel experiments.
Copy this template when creating new experiments.

Key changes from original pattern:
1. SeedManager generates deterministic per-sample seeds
2. Seed is passed to each worker and initialized at start
3. CMA-ES receives explicit seed in options
4. Seed is stored in STATE.json for reproducibility
5. Seeds are logged to MLflow

Usage:
    python run.py --seed 42 --workers 7
    python run.py --seed 12345  # Different seed for variant runs
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

# Import seed manager - this is the key addition
from src.seed_manager import SeedManager


def process_single_sample(args):
    """
    Process a single sample with proper seeding.

    CRITICAL: The sample_seed must be used to initialize RNG at the start
    of this function since ProcessPoolExecutor creates new processes with
    independent random states.
    """
    idx, sample, meta, config, sample_seed = args

    # CRITICAL: Seed the worker process at the very start
    # This ensures all np.random calls in this worker are reproducible
    np.random.seed(sample_seed)

    # Import optimizer here to avoid issues with multiprocessing
    # The import path should be adjusted for your actual optimizer
    from optimizer import YourOptimizer  # Replace with actual import

    optimizer = YourOptimizer(
        # Pass config parameters
        **config['optimizer_params']
    )

    start = time.time()
    try:
        # Run optimization
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
            'sample_seed': sample_seed,  # Store seed used for this sample
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
            'sample_seed': sample_seed,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master seed for reproducibility')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Skip MLflow logging')
    # Add your experiment-specific arguments here
    args = parser.parse_args()

    # Initialize seed manager with master seed
    seed_manager = SeedManager(master_seed=args.seed)

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    # Seed the main process for any pre-processing randomness
    np.random.seed(args.seed)

    indices = np.arange(len(samples))
    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    n_1src = sum(1 for s in samples_to_process if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"SEEDED EXPERIMENT TEMPLATE")
    print(f"{'='*60}")
    print(f"Master seed: {args.seed}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}")

    # Configuration dict - adjust for your experiment
    config = {
        'optimizer_params': {
            # Add your optimizer parameters here
        },
        'seed': args.seed,  # Store seed in config for logging
    }

    start_time = time.time()
    results = []

    # Create work items WITH per-sample seeds
    # This is the key change: each sample gets a deterministic seed
    work_items = [
        (
            indices[i],
            samples_to_process[i],
            meta,
            config,
            seed_manager.get_sample_seed(indices[i])  # Per-sample seed
        )
        for i in range(n_samples)
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_sample, item): item[0]
            for item in work_items
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s "
                  f"[{status}] seed={result['sample_seed']}")

    total_time = time.time() - start_time

    # Calculate score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [
        calculate_sample_score(r['best_rmse'], r['n_candidates'])
        for r in results
    ]
    score = np.mean(sample_scores)

    # Statistics
    rmses = [r['best_rmse'] for r in results if r['success']]
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Master seed:      {args.seed}")
    print(f"Score:            {score:.4f}")
    print(f"RMSE mean:        {np.mean(rmses):.4f}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print(f"{'='*70}\n")

    results_summary = {
        'score': score,
        'total_time_sec': total_time,
        'projected_400_min': projected_400,
        'rmse_mean': float(np.mean(rmses)),
        'seed': args.seed,  # IMPORTANT: Store seed
        'sample_seeds': {r['idx']: r['sample_seed'] for r in results},  # Per-sample seeds
    }

    # Save STATE.json with seed information
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment': 'seeded_template',
            'experiment_id': 'EXP_SEEDED_TEMPLATE_001',
            'tuning_runs': [],
            'best_in_budget': None,
        }

    # Add tuning run with seed info
    state['tuning_runs'].append({
        'run': len(state['tuning_runs']) + 1,
        'config': config,
        'results': results_summary,
        'seed_info': seed_manager.to_dict(),  # Store full seed info
        'timestamp': datetime.now().isoformat(),
    })

    if projected_400 <= 60:
        if state['best_in_budget'] is None or score > state['best_in_budget'].get('score', 0):
            state['best_in_budget'] = {
                'run': len(state['tuning_runs']),
                'score': score,
                'time_min': projected_400,
                'config': config,
                'seed': args.seed,  # Store the seed of best run
            }

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    # Log to MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(str(project_root / 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"seeded_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_name', 'seeded_template')
            mlflow.log_param('seed', args.seed)  # Log seed
            mlflow.log_param('n_samples', n_samples)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('submission_score', score)
            mlflow.log_metric('projected_400_samples_min', projected_400)
            mlflow.log_metric('rmse_mean', float(np.mean(rmses)))

            print(f"MLflow run ID: {run.info.run_id}")

    return results_summary


if __name__ == '__main__':
    main()
