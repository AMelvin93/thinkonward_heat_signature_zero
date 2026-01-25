#!/usr/bin/env python3
"""
Run script for Boundary Handling Methods experiment.

Tests different CMA-ES boundary constraint handling methods:
1. BoundTransform (default) - smooth transformation into feasible domain
2. BoundPenalty - penalty-based constraint handling
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

from optimizer import BoundaryHandlingOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = BoundaryHandlingOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        boundary_handler=config['boundary_handler'],
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


def run_single_config(samples_to_process, indices, meta, config, args):
    """Run a single configuration and return results."""
    n_samples = len(samples_to_process)

    print(f"\nBoundary Handler: {config['boundary_handler']}")
    print(f"=" * 60)

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            if (len(results) % 10 == 0) or len(results) == n_samples:
                print(f"[{len(results):3d}/{n_samples}] {result['n_sources']}-src "
                      f"RMSE={result['best_rmse']:.4f} cands={result['n_candidates']} [{status}]")

    total_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    return {
        'score': score,
        'time_min': projected_400,
        'rmse_mean': rmse_mean,
        'rmse_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_2src': np.mean(rmses_2src) if rmses_2src else None,
        'n_1src': len(rmses_1src),
        'n_2src': len(rmses_2src),
    }


def main():
    parser = argparse.ArgumentParser(description='Boundary Handling Methods Experiment')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--final-polish-maxiter', type=int, default=8)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--boundary-handler', type=str, default=None,
                        help='Specific handler to test (BoundTransform or BoundPenalty)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    np.random.seed(args.seed)
    indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    print(f"\nBoundary Handling Methods Experiment")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"CMA-ES fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Timestep: {args.timestep_fraction:.0%}, NM polish: {args.final_polish_maxiter} iters")
    print(f"=" * 60)

    # Set up MLflow
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    # Handlers to test
    if args.boundary_handler:
        handlers = [args.boundary_handler]
    else:
        handlers = ['BoundTransform', 'BoundPenalty']

    all_results = {}

    for handler in handlers:
        config = {
            'max_fevals_1src': args.max_fevals_1src,
            'max_fevals_2src': args.max_fevals_2src,
            'timestep_fraction': args.timestep_fraction,
            'final_polish_maxiter': args.final_polish_maxiter,
            'sigma0_1src': args.sigma0_1src,
            'sigma0_2src': args.sigma0_2src,
            'boundary_handler': handler,
        }

        run_name = f"boundary_{handler.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("experiment_id", "EXP_BOUNDARY_HANDLING_001")
            mlflow.log_param("worker", "W2")
            mlflow.log_param("boundary_handler", handler)
            mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
            mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
            mlflow.log_param("final_polish_maxiter", args.final_polish_maxiter)
            mlflow.log_param("timestep_fraction", args.timestep_fraction)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("platform", "wsl")

            result = run_single_config(samples_to_process, indices, meta, config, args)
            all_results[handler] = result

            mlflow.log_metric("submission_score", result['score'])
            mlflow.log_metric("projected_400_samples_min", result['time_min'])
            mlflow.log_metric("rmse_mean", result['rmse_mean'])
            if result['rmse_1src']:
                mlflow.log_metric("rmse_1src", result['rmse_1src'])
            if result['rmse_2src']:
                mlflow.log_metric("rmse_2src", result['rmse_2src'])

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY - Boundary Handling Methods")
    print(f"{'='*70}")
    print(f"{'Handler':<20} {'Score':>10} {'Time':>12} {'RMSE':>10} {'1-src':>10} {'2-src':>10}")
    print(f"{'-'*70}")

    for handler, result in all_results.items():
        in_budget = result['time_min'] <= 60
        status = "OK" if in_budget else "OVER"
        print(f"{handler:<20} {result['score']:>10.4f} {result['time_min']:>8.1f} min {result['rmse_mean']:>10.4f} "
              f"{result['rmse_1src'] or 0:>10.4f} {result['rmse_2src'] or 0:>10.4f} [{status}]")

    print(f"{'-'*70}")
    print(f"Baseline: BoundTransform @ 1.1688 / 58.4 min")
    print(f"{'='*70}\n")

    # Find best in-budget
    best_handler = None
    best_score = 0
    for handler, result in all_results.items():
        if result['time_min'] <= 60 and result['score'] > best_score:
            best_score = result['score']
            best_handler = handler

    if best_handler:
        print(f"Best in-budget: {best_handler} @ {best_score:.4f}")
        if best_score > 1.1688:
            print("SUCCESS: Improvement over baseline!")
        else:
            print(f"Delta vs baseline: {best_score - 1.1688:+.4f}")
    else:
        print("All runs OVER BUDGET")

    return all_results


if __name__ == '__main__':
    main()
