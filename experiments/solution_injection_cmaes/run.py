#!/usr/bin/env python3
"""
Run script for Solution Injection CMA-ES optimizer.

Tests the hypothesis that injecting best solutions from early CMA-ES inits
into later inits can improve convergence (GloMPO-inspired approach).
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import SolutionInjectionOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = SolutionInjectionOptimizer(
        fevals_1src=config['fevals_1src'],
        fevals_2src=config['fevals_2src'],
        sigma_1src=config['sigma_1src'],
        sigma_2src=config['sigma_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 8),
        refine_top_n=config.get('refine_top_n', 3),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
        timestep_fraction=config['timestep_fraction'],
        inject_best_n=config['inject_best_n'],
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
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start,
            'init_types': [], 'success': False, 'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Solution Injection CMA-ES optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    # Core parameters
    parser.add_argument('--fevals-1src', type=int, default=20)
    parser.add_argument('--fevals-2src', type=int, default=36)
    parser.add_argument('--sigma-1src', type=float, default=0.18)
    parser.add_argument('--sigma-2src', type=float, default=0.22)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    # Injection parameters
    parser.add_argument('--inject-best-n', type=int, default=1)
    # Shared parameters
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=8)
    parser.add_argument('--refine-top', type=int, default=3)
    parser.add_argument('--threshold-1src', type=float, default=0.4)
    parser.add_argument('--threshold-2src', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mlflow-run-name', type=str, default=None)
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

    print(f"\nSolution Injection CMA-ES Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Inject best N: {args.inject_best_n}")
    print(f"fevals: {args.fevals_1src}/{args.fevals_2src}, sigma: {args.sigma_1src}/{args.sigma_2src}")
    print(f"CMA-ES timesteps: {args.timestep_fraction:.0%}")
    print(f"NM Polish: {args.refine_maxiter} iters on top-{args.refine_top}")
    print(f"=" * 70)

    config = {
        'fevals_1src': args.fevals_1src,
        'fevals_2src': args.fevals_2src,
        'sigma_1src': args.sigma_1src,
        'sigma_2src': args.sigma_2src,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
        'timestep_fraction': args.timestep_fraction,
        'inject_best_n': args.inject_best_n,
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
                  f"cands={result['n_candidates']} sims={result['n_sims']} "
                  f"time={result['elapsed']:.1f}s [{status}]{flag}")

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
    print(f"RESULTS - Solution Injection CMA-ES (inject_best_n={args.inject_best_n})")
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
    print(f"Baseline (W2):  1.1688 @ 58.4 min")
    print(f"This run:       {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:          {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print()

    in_budget = projected_400 <= 60
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score >= 1.18 and projected_400 <= 55:
        print("SUCCESS! Score improved with good timing")
    elif score >= 1.17:
        print("PARTIAL: Good score, checking time")
    else:
        print("NEEDS TUNING")
    print(f"{'='*70}\n")

    # MLflow logging
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    run_name = args.mlflow_run_name or f"solution_injection_n{args.inject_best_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_SOLUTION_INJECTION_001")
        mlflow.log_param("worker", "W1")
        mlflow.log_param("algorithm", "solution_injection_cmaes")
        mlflow.log_param("platform", "wsl")
        mlflow.log_param("inject_best_n", args.inject_best_n)
        mlflow.log_param("fevals_1src", args.fevals_1src)
        mlflow.log_param("fevals_2src", args.fevals_2src)
        mlflow.log_param("sigma_1src", args.sigma_1src)
        mlflow.log_param("sigma_2src", args.sigma_2src)
        mlflow.log_param("timestep_fraction", args.timestep_fraction)
        mlflow.log_param("refine_maxiter", args.refine_maxiter)
        mlflow.log_param("n_workers", args.workers)
        mlflow.log_param("n_samples", n_samples)

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("rmse_1src_mean", np.mean(rmses_1src) if rmses_1src else 0)
        mlflow.log_metric("rmse_2src_mean", np.mean(rmses_2src) if rmses_2src else 0)
        mlflow.log_metric("total_time_sec", total_time)

        mlflow_run_id = run.info.run_id
        print(f"MLflow run ID: {mlflow_run_id}")

    return {
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': rmse_mean,
        'in_budget': in_budget,
        'mlflow_run_id': mlflow_run_id,
        'config': config,
    }


if __name__ == '__main__':
    main()
