#!/usr/bin/env python3
"""
Run script for Early Timestep Optimization experiment.

Key Innovation: Focus position optimization on early timesteps (containing
onset/timing information) which should be more discriminative for source positions.

Usage:
    uv run python experiments/early_timestep_opt/run.py --workers 7 --shuffle
    uv run python experiments/early_timestep_opt/run.py --workers 7 --max-samples 20 --early-fraction 0.3
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# Add project root
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Local import
from optimizer import EarlyTimestepOptimizer

# MLflow (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = EarlyTimestepOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        early_fraction=config['early_fraction'],
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )

        elapsed = time.time() - start
        init_types = [r.init_type for r in results]

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'init_types': init_types,
            'success': True,
        }
    except Exception as e:
        return {
            'idx': idx,
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'init_types': [],
            'success': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Early Timestep optimizer')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of parallel workers (default: 7 for G4dn)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (default: all)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle samples for balanced batches')
    parser.add_argument('--max-fevals-1src', type=int, default=15,
                        help='Max fevals for 1-source (default: 15)')
    parser.add_argument('--max-fevals-2src', type=int, default=20,
                        help='Max fevals for 2-source (default: 20)')
    parser.add_argument('--early-fraction', type=float, default=0.3,
                        help='Fraction of early timesteps to use (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Load data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    # Shuffle if requested
    np.random.seed(args.seed)
    if args.shuffle:
        indices = np.random.permutation(len(samples))
    else:
        indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    print(f"\nEarly Timestep Optimizer")
    print(f"=" * 60)
    print(f"Samples: {n_samples}")
    print(f"Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Early fraction: {args.early_fraction:.0%}")
    print(f"Shuffle: {args.shuffle}")
    print(f"Seed: {args.seed}")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'early_fraction': args.early_fraction,
    }

    # Process samples
    start_time = time.time()
    results = []

    work_items = [
        (indices[i], samples_to_process[i], meta, config)
        for i in range(n_samples)
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0]
                   for item in work_items}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            n_src = result['n_sources']
            rmse = result['best_rmse']
            ncand = result['n_candidates']
            elapsed = result['elapsed']
            success = "OK" if result['success'] else "FAIL"

            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{n_src}-src RMSE={rmse:.4f} cands={ncand} "
                  f"sims={result['n_sims']:3d} time={elapsed:.1f}s [{success}]")

    total_time = time.time() - start_time

    # Calculate score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        accuracy_term = 1.0 / (1.0 + rmse)
        diversity_term = lambda_ * (n_candidates / n_max)
        return accuracy_term + diversity_term

    sample_scores = []
    for r in results:
        s = calculate_sample_score(r['best_rmse'], r['n_candidates'])
        sample_scores.append(s)

    score = np.mean(sample_scores)

    # Statistics
    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)

    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    n_candidates = [r['n_candidates'] for r in results if r['success']]
    avg_candidates = np.mean(n_candidates)

    total_sims = sum(r['n_sims'] for r in results)
    avg_sims = total_sims / n_samples

    projected_400 = (total_time / n_samples) * 400 / 60

    # Init type stats
    all_init_types = []
    for r in results:
        all_init_types.extend(r['init_types'])
    init_counts = {}
    for t in all_init_types:
        init_counts[t] = init_counts.get(t, 0) + 1

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"RMSE:             {rmse_mean:.6f} +/- {rmse_std:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} +/- {np.std(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} +/- {np.std(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"  Avg candidates: {avg_candidates:.1f}")
    print(f"  Avg sims/sample: {avg_sims:.1f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print("Init type distribution:")
    for t, count in sorted(init_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(init_counts.values())
        print(f"  {t}: {count} ({pct:.1f}%)")
    print()

    # Comparison to baseline
    baseline_score = 0.9951
    baseline_time = 55.6
    diff_score = score - baseline_score
    diff_time = projected_400 - baseline_time

    print(f"Baseline: {baseline_score:.4f} @ {baseline_time:.1f} min")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:    {diff_score:+.4f} score, {diff_time:+.1f} min")

    if projected_400 > 60:
        print("❌ OVER BUDGET")
    elif score > baseline_score:
        print("✅ IMPROVED!")
    else:
        print("❌ NO IMPROVEMENT")

    print(f"{'='*60}\n")

    # MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("early_timestep_opt")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"early_{args.early_fraction:.0%}_{args.max_fevals_1src}_{args.max_fevals_2src}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'max_fevals_1src': args.max_fevals_1src,
                'max_fevals_2src': args.max_fevals_2src,
                'early_fraction': args.early_fraction,
                'workers': args.workers,
                'n_samples': n_samples,
                'shuffle': args.shuffle,
                'seed': args.seed,
            })
            mlflow.log_metrics({
                'submission_score': score,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'rmse_1src': np.mean(rmses_1src) if rmses_1src else 0,
                'rmse_2src': np.mean(rmses_2src) if rmses_2src else 0,
                'avg_candidates': avg_candidates,
                'avg_sims_per_sample': avg_sims,
                'total_time_sec': total_time,
                'projected_400_samples_min': projected_400,
            })
            print(f"Logged to MLflow: {run_name}")

    return score, projected_400


if __name__ == '__main__':
    main()
