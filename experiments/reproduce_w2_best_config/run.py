#!/usr/bin/env python3
"""
Run script for Reproducing W2 Best Config.

Verifies the W2 1.1688 score with 3 different seeds to estimate variance.

W2 Best Config:
- timestep_fraction=0.40 (40%)
- sigma0_1src=0.18, sigma0_2src=0.22
- final_polish_maxiter=8
- max_fevals_1src=20, max_fevals_2src=36
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import TemporalFidelityWithPolishOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = TemporalFidelityWithPolishOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def run_single_config(config, samples, meta, indices, n_workers, seed_label):
    """Run the experiment with a single configuration."""
    n_samples = len(samples)

    print(f"\n{'='*70}")
    print(f"Running {seed_label}")
    print(f"{'='*70}")
    print(f"Samples: {n_samples}, Workers: {n_workers}")
    print(f"Fevals: {config['max_fevals_1src']}/{config['max_fevals_2src']}")
    print(f"Sigma: {config['sigma0_1src']}/{config['sigma0_2src']}")
    print(f"Timestep fraction: {config['timestep_fraction']*100:.0f}%")
    print(f"Final polish: {config['final_polish_maxiter']} iterations")
    print(f"{'='*70}")

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]")

    total_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - {seed_label}")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"W2 Baseline: 1.1688 @ 58.4 min")
    print(f"This run:    {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:       {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print(f"{'='*70}\n")

    return {
        'seed_label': seed_label,
        'score': score,
        'rmse_mean': rmse_mean,
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else None,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
        'total_time_sec': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Reproduce W2 Best Config')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    # Use W2 BEST config as defaults
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--final-polish-maxiter', type=int, default=8)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    # Run 3 seeds by default
    parser.add_argument('--seeds', type=str, default='42,123,999')
    # MLflow
    parser.add_argument('--no-mlflow', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    indices = np.arange(len(samples))
    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    seeds = [int(s) for s in args.seeds.split(',')]

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'timestep_fraction': args.timestep_fraction,
        'final_polish_maxiter': args.final_polish_maxiter,
        'candidate_pool_size': 10,
        'nx_coarse': 50,
        'ny_coarse': 25,
        'refine_maxiter': 3,
        'refine_top_n': 2,
        'rmse_threshold_1src': 0.4,
        'rmse_threshold_2src': 0.5,
    }

    print("\n" + "="*70)
    print("REPRODUCE W2 BEST CONFIG - Verification Experiment")
    print("="*70)
    print(f"W2 Best Config:")
    print(f"  sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"  timestep_fraction: {args.timestep_fraction}")
    print(f"  final_polish_maxiter: {args.final_polish_maxiter}")
    print(f"  fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"\nRunning with seeds: {seeds}")
    print("="*70)

    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        seed_label = f"seed_{seed}"
        result = run_single_config(config, samples_to_process, meta, indices, args.workers, seed_label)
        all_results.append(result)

    # Summary statistics
    print("\n" + "="*70)
    print("FINAL SUMMARY - All Seeds")
    print("="*70)

    scores = [r['score'] for r in all_results]
    times = [r['projected_400_min'] for r in all_results]

    print(f"\nScores across seeds: {[f'{s:.4f}' for s in scores]}")
    print(f"Times across seeds:  {[f'{t:.1f}' for t in times]} min")
    print()
    print(f"Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Mean Time:  {np.mean(times):.1f} ± {np.std(times):.1f} min")
    print()
    print(f"W2 Baseline: 1.1688 @ 58.4 min")
    print(f"Difference:  {np.mean(scores) - 1.1688:+.4f}")
    print()

    in_budget = np.mean(times) <= 60
    score_matches = abs(np.mean(scores) - 1.1688) < 0.02  # Within 2%

    if in_budget and score_matches:
        print("✓ VERIFIED: W2 config is reproducible")
    elif in_budget:
        print(f"⚠ PARTIAL: In budget but score differs by {abs(np.mean(scores) - 1.1688):.4f}")
    else:
        print("✗ FAILED: Over budget")

    print("="*70 + "\n")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment('reproduce_w2_best_config')

        run_name = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('sigma0_1src', args.sigma0_1src)
            mlflow.log_param('sigma0_2src', args.sigma0_2src)
            mlflow.log_param('timestep_fraction', args.timestep_fraction)
            mlflow.log_param('final_polish_maxiter', args.final_polish_maxiter)
            mlflow.log_param('seeds', args.seeds)
            mlflow.log_param('n_samples', len(samples_to_process))
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('mean_score', np.mean(scores))
            mlflow.log_metric('std_score', np.std(scores))
            mlflow.log_metric('mean_time_min', np.mean(times))
            mlflow.log_metric('std_time_min', np.std(times))

            for i, r in enumerate(all_results):
                mlflow.log_metric(f'score_seed{seeds[i]}', r['score'])
                mlflow.log_metric(f'time_seed{seeds[i]}', r['projected_400_min'])

            print(f"Logged to MLflow: {run_name}")

    return {
        'results': all_results,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_time': np.mean(times),
        'verified': in_budget and score_matches,
    }


if __name__ == '__main__':
    main()
