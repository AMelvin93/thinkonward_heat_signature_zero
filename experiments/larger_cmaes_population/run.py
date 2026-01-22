#!/usr/bin/env python3
"""
Run script for Larger CMA-ES Population experiment.

Tests the hypothesis that larger population sizes improve convergence quality
in inverse problems.
"""

import argparse
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

from optimizer import LargerPopulationOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = LargerPopulationOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        popsize_1src=config.get('popsize_1src'),
        popsize_2src=config.get('popsize_2src'),
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


def main():
    parser = argparse.ArgumentParser(description='Run Larger CMA-ES Population experiment')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    # Fevals
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    # Sigma
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    # Temporal fidelity
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    # Polish
    parser.add_argument('--final-polish-maxiter', type=int, default=8)
    # NEW: Population size
    parser.add_argument('--popsize-1src', type=int, default=None, help='CMA-ES population for 1-src (default: auto)')
    parser.add_argument('--popsize-2src', type=int, default=None, help='CMA-ES population for 2-src (default: auto)')
    # Other
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    # MLflow
    parser.add_argument('--no-mlflow', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    np.random.seed(args.seed)
    if args.shuffle:
        indices = np.random.permutation(len(samples))
    else:
        indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    pop_1src_str = str(args.popsize_1src) if args.popsize_1src else "auto"
    pop_2src_str = str(args.popsize_2src) if args.popsize_2src else "auto"

    print(f"\nLarger CMA-ES Population Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Timestep fraction: {args.timestep_fraction*100:.0f}%")
    print(f"Final polish: {args.final_polish_maxiter} iterations")
    print(f"Population size (1-src): {pop_1src_str}")
    print(f"Population size (2-src): {pop_2src_str}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'timestep_fraction': args.timestep_fraction,
        'final_polish_maxiter': args.final_polish_maxiter,
        'popsize_1src': args.popsize_1src,
        'popsize_2src': args.popsize_2src,
        'candidate_pool_size': args.candidate_pool_size,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    init_type_counts = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"

            for it in result['init_types']:
                init_type_counts[it] = init_type_counts.get(it, 0) + 1

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
    print(f"RESULTS")
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
    print(f"Init type distribution: {init_type_counts}")
    print()
    print(f"W2 Baseline (40% temporal + 8 polish): 1.1688 @ 58.4 min")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs W2: {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print()
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score >= 1.17:
        print("SUCCESS - Beat W2 baseline!")
    elif score >= 1.14:
        print("PARTIAL SUCCESS")
    else:
        print("NO IMPROVEMENT")
    print(f"{'='*70}\n")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment('larger_cmaes_population')

        run_name = args.run_name or f"popsize_{pop_1src_str}_{pop_2src_str}_{datetime.now().strftime('%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('max_fevals_1src', args.max_fevals_1src)
            mlflow.log_param('max_fevals_2src', args.max_fevals_2src)
            mlflow.log_param('sigma0_1src', args.sigma0_1src)
            mlflow.log_param('sigma0_2src', args.sigma0_2src)
            mlflow.log_param('timestep_fraction', args.timestep_fraction)
            mlflow.log_param('final_polish_maxiter', args.final_polish_maxiter)
            mlflow.log_param('popsize_1src', args.popsize_1src or 'auto')
            mlflow.log_param('popsize_2src', args.popsize_2src or 'auto')
            mlflow.log_param('n_samples', n_samples)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('submission_score', score)
            mlflow.log_metric('rmse_mean', rmse_mean)
            if rmses_1src:
                mlflow.log_metric('rmse_mean_1src', np.mean(rmses_1src))
            if rmses_2src:
                mlflow.log_metric('rmse_mean_2src', np.mean(rmses_2src))
            mlflow.log_metric('projected_400_samples_min', projected_400)
            mlflow.log_metric('total_time_sec', total_time)

            print(f"\nLogged to MLflow: {run_name}")

    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else None,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
        'init_type_counts': init_type_counts,
    }


if __name__ == '__main__':
    main()
