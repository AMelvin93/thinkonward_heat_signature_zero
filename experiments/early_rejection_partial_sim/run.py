#!/usr/bin/env python3
"""
Run script for Early Rejection optimizer.

Tests the hypothesis that using 10% timesteps to reject obviously bad
candidates can save compute while maintaining accuracy.
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

from optimizer import EarlyRejectionOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = EarlyRejectionOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 8),
        refine_top_n=config.get('refine_top_n', 3),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
        # Early rejection parameters
        filter_timestep_fraction=config['filter_timestep_fraction'],
        filter_threshold_multiplier=config['filter_threshold_multiplier'],
        cmaes_timestep_fraction=config['cmaes_timestep_fraction'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, n_rejected = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'n_rejected': n_rejected, 'elapsed': elapsed,
            'init_types': init_types, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'n_rejected': 0, 'elapsed': time.time() - start,
            'init_types': [], 'success': False, 'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Early Rejection optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=8)
    parser.add_argument('--refine-top', type=int, default=3)
    parser.add_argument('--threshold-1src', type=float, default=0.4)
    parser.add_argument('--threshold-2src', type=float, default=0.5)
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    parser.add_argument('--seed', type=int, default=42)
    # Early rejection parameters
    parser.add_argument('--filter-timestep', type=float, default=0.10,
                        help='Fraction of timesteps for quick filter (default 0.10)')
    parser.add_argument('--filter-threshold', type=float, default=2.0,
                        help='Reject if quick RMSE > threshold * best (default 2.0)')
    parser.add_argument('--cmaes-timestep', type=float, default=0.40,
                        help='Fraction of timesteps for CMA-ES evaluation (default 0.40)')
    parser.add_argument('--mlflow-run-name', type=str, default=None,
                        help='MLflow run name (auto-generated if not specified)')
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

    print(f"\nEarly Rejection Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Filter: {args.filter_timestep:.0%} timesteps, reject if > {args.filter_threshold}x best")
    print(f"CMA-ES: {args.cmaes_timestep:.0%} timesteps")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"NM Polish: {args.refine_maxiter} iters on top-{args.refine_top}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'filter_timestep_fraction': args.filter_timestep,
        'filter_threshold_multiplier': args.filter_threshold,
        'cmaes_timestep_fraction': args.cmaes_timestep,
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
            reject_info = f"rej={result['n_rejected']}" if result['n_rejected'] > 0 else ""
            flag = " [FB]" if result['best_rmse'] > 0.4 and result['n_sources'] == 1 else ""
            flag = " [FB]" if result['best_rmse'] > 0.5 and result['n_sources'] == 2 else flag
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} sims={result['n_sims']} {reject_info} "
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

    # Rejection statistics
    total_rejected = sum(r['n_rejected'] for r in results)
    total_sims = sum(r['n_sims'] for r in results)
    rejection_rate = total_rejected / (total_rejected + total_sims) if (total_rejected + total_sims) > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS - Early Rejection ({args.filter_timestep:.0%} filter, {args.cmaes_timestep:.0%} CMA-ES)")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    print(f"Rejection Stats:")
    print(f"  Total rejected:  {total_rejected}")
    print(f"  Total simulated: {total_sims}")
    print(f"  Rejection rate:  {rejection_rate:.1%}")
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

    # Analyze success criteria
    in_budget = projected_400 <= 60
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score >= 1.17 and projected_400 <= 55:
        print("SUCCESS! Score improved with time savings")
    elif score >= 1.16:
        print("PARTIAL: Good score, checking time improvement")
    else:
        print("NEEDS TUNING")
    print(f"{'='*70}\n")

    # MLflow logging
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    run_name = args.mlflow_run_name or f"early_rejection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_EARLY_REJECTION_001")
        mlflow.log_param("worker", "W1")
        mlflow.log_param("algorithm", "early_rejection_cmaes")
        mlflow.log_param("platform", "wsl")
        mlflow.log_param("filter_timestep_fraction", args.filter_timestep)
        mlflow.log_param("filter_threshold_multiplier", args.filter_threshold)
        mlflow.log_param("cmaes_timestep_fraction", args.cmaes_timestep)
        mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
        mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
        mlflow.log_param("refine_maxiter", args.refine_maxiter)
        mlflow.log_param("sigma0_1src", args.sigma0_1src)
        mlflow.log_param("sigma0_2src", args.sigma0_2src)
        mlflow.log_param("n_workers", args.workers)
        mlflow.log_param("n_samples", n_samples)

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("rmse_1src_mean", np.mean(rmses_1src) if rmses_1src else 0)
        mlflow.log_metric("rmse_2src_mean", np.mean(rmses_2src) if rmses_2src else 0)
        mlflow.log_metric("total_rejected", total_rejected)
        mlflow.log_metric("rejection_rate", rejection_rate)
        mlflow.log_metric("total_time_sec", total_time)

        mlflow_run_id = run.info.run_id
        print(f"MLflow run ID: {mlflow_run_id}")

    return {
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': rmse_mean,
        'rejection_rate': rejection_rate,
        'in_budget': in_budget,
        'mlflow_run_id': mlflow_run_id,
        'config': config,
    }


if __name__ == '__main__':
    main()
