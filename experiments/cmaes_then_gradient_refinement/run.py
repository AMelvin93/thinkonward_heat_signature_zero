#!/usr/bin/env python3
"""
Run script for CMA-ES + L-BFGS-B hybrid optimizer.

Tests the hypothesis that L-BFGS-B polish converges faster than NM polish
due to gradient information (even via finite differences).
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

from optimizer import HybridCMAESLBFGSBOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = HybridCMAESLBFGSBOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        polish_maxiter=config['polish_maxiter'],
        polish_top_n=config['polish_top_n'],
        polish_method=config['polish_method'],
        timestep_fraction=config['timestep_fraction'],
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


def main():
    parser = argparse.ArgumentParser(description='Run CMA-ES + L-BFGS-B hybrid optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    # Polish configuration
    parser.add_argument('--polish-maxiter', type=int, default=3,
                        help='Max iterations for polish (L-BFGS-B: 3-5, NM: 8)')
    parser.add_argument('--polish-top', type=int, default=2,
                        help='Number of top candidates to polish')
    parser.add_argument('--polish-method', type=str, default='L-BFGS-B',
                        choices=['L-BFGS-B', 'Nelder-Mead'],
                        help='Method for polishing')
    # Temporal fidelity
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--threshold-1src', type=float, default=0.4)
    parser.add_argument('--threshold-2src', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
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

    print(f"\nCMA-ES + {args.polish_method} Hybrid Optimizer")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Polish: {args.polish_method} x {args.polish_maxiter} iters on top-{args.polish_top}")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'polish_maxiter': args.polish_maxiter,
        'polish_top_n': args.polish_top,
        'polish_method': args.polish_method,
        'timestep_fraction': args.timestep_fraction,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            flag = " [FB]" if result['best_rmse'] > 0.4 and result['n_sources'] == 1 else ""
            flag = " [FB]" if result['best_rmse'] > 0.5 and result['n_sources'] == 2 else flag
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s "
                  f"sims={result['n_sims']} [{status}]{flag}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - CMA-ES + {args.polish_method}")
    print(f"{'='*70}")
    print(f"RMSE mean:        {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Baseline (NM polish x8):  1.1362 @ 39 min")
    print(f"This run:                 {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                    {score - 1.1362:+.4f} score, {projected_400 - 39:+.1f} min")
    print()

    if projected_400 > 60:
        status = "OVER BUDGET"
    elif score >= 1.17 and projected_400 <= 58:
        status = "SUCCESS! Meets experiment criteria"
    elif score >= 1.1362 and projected_400 <= 60:
        status = "COMPARABLE to baseline"
    elif score >= 1.12:
        status = "PARTIAL: Check accuracy vs time tradeoff"
    else:
        status = "NEEDS IMPROVEMENT"

    print(f"Status: {status}")
    print(f"{'='*70}\n")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"hybrid_{args.polish_method.lower().replace('-', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("experiment_id", "EXP_HYBRID_CMAES_LBFGSB_001")
            mlflow.log_param("worker", "W1")
            mlflow.log_param("optimizer", f"cmaes_{args.polish_method.lower()}")
            mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
            mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
            mlflow.log_param("sigma0_1src", args.sigma0_1src)
            mlflow.log_param("sigma0_2src", args.sigma0_2src)
            mlflow.log_param("polish_method", args.polish_method)
            mlflow.log_param("polish_maxiter", args.polish_maxiter)
            mlflow.log_param("timestep_fraction", args.timestep_fraction)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("workers", args.workers)

            mlflow.log_metric("submission_score", score)
            mlflow.log_metric("rmse_mean", rmse_mean)
            mlflow.log_metric("projected_400_samples_min", projected_400)
            mlflow.log_metric("total_time_sec", total_time)

            print(f"MLflow run ID: {run.info.run_id}")

    # Print outliers
    high_rmse = [r for r in results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:5]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f}")

    return score, projected_400


if __name__ == '__main__':
    main()
