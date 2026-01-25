#!/usr/bin/env python3
"""
Run script for Asymmetric Polish Budget experiment.

Tests hypothesis: 1-source problems converge faster and need fewer polish iterations,
while 2-source problems are harder and benefit from more polish iterations.
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

from optimizer import AsymmetricPolishOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = AsymmetricPolishOptimizer(
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
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        final_polish_maxiter_1src=config['final_polish_maxiter_1src'],
        final_polish_maxiter_2src=config['final_polish_maxiter_2src'],
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
    parser = argparse.ArgumentParser(description='Run Asymmetric Polish Budget')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    # Asymmetric polish parameters
    parser.add_argument('--polish-1src', type=int, default=6, help='NM polish iters for 1-source')
    parser.add_argument('--polish-2src', type=int, default=10, help='NM polish iters for 2-source')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tuning-run', type=int, default=1)
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

    print(f"\nAsymmetric Polish Budget Experiment")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Sigma: 1-src={args.sigma0_1src}, 2-src={args.sigma0_2src}")
    print(f"Polish: 1-src={args.polish_1src} iters, 2-src={args.polish_2src} iters")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'final_polish_maxiter_1src': args.polish_1src,
        'final_polish_maxiter_2src': args.polish_2src,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            polished = " [P]" if 'polished' in result.get('init_types', []) else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{polished}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    times_1src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 1]
    times_2src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    # MLflow logging
    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"asymmetric_polish_run{args.tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_ASYMMETRIC_POLISH_001")
        mlflow.log_param("worker", "W1")
        mlflow.log_param("tuning_run", args.tuning_run)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_workers", args.workers)
        mlflow.log_param("timestep_fraction", args.timestep_fraction)
        mlflow.log_param("sigma0_1src", args.sigma0_1src)
        mlflow.log_param("sigma0_2src", args.sigma0_2src)
        mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
        mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
        mlflow.log_param("polish_1src", args.polish_1src)
        mlflow.log_param("polish_2src", args.polish_2src)

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("total_time_sec", total_time)
        mlflow.log_metric("rmse_mean", np.mean(rmses))
        if rmses_1src:
            mlflow.log_metric("rmse_1src_mean", np.mean(rmses_1src))
            mlflow.log_metric("time_1src_mean", np.mean(times_1src))
        if rmses_2src:
            mlflow.log_metric("rmse_2src_mean", np.mean(rmses_2src))
            mlflow.log_metric("time_2src_mean", np.mean(times_2src))

        mlflow_run_id = run.info.run_id

    print(f"\n{'='*70}")
    print(f"RESULTS - Asymmetric Polish: 1-src={args.polish_1src}, 2-src={args.polish_2src}")
    print(f"{'='*70}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print(f"In Budget:        {'YES' if projected_400 <= 60 else 'NO'}")
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f}, time={np.mean(times_1src):.1f}s (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f}, time={np.mean(times_2src):.1f}s (n={len(rmses_2src)})")
    print()
    print(f"Baseline (8 NM both): 1.1688 @ 58.4 min")
    print(f"This run:             {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print(f"MLflow run ID:        {mlflow_run_id}")
    print(f"{'='*70}\n")

    return {
        'score': score,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
        'rmse_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_2src': np.mean(rmses_2src) if rmses_2src else None,
        'time_1src': np.mean(times_1src) if times_1src else None,
        'time_2src': np.mean(times_2src) if times_2src else None,
        'mlflow_run_id': mlflow_run_id,
        'config': config,
    }


if __name__ == '__main__':
    main()
