#!/usr/bin/env python3
"""Run script for Adaptive NM Iterations experiment with MLflow logging."""

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

from optimizer import AdaptiveNMPolishOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = AdaptiveNMPolishOptimizer(
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
        # Adaptive NM parameters
        min_polish_iters=config['min_polish_iters'],
        max_polish_iters=config['max_polish_iters'],
        polish_batch_size=config['polish_batch_size'],
        convergence_threshold=config['convergence_threshold'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, polish_iters = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'polish_iters': polish_iters, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'polish_iters': 0, 'success': False, 'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive NM Iterations Experiment')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    # Adaptive NM parameters
    parser.add_argument('--min-polish-iters', type=int, default=4)
    parser.add_argument('--max-polish-iters', type=int, default=12)
    parser.add_argument('--polish-batch-size', type=int, default=2)
    parser.add_argument('--convergence-threshold', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    parser.add_argument('--tuning-run', type=int, default=1, help='Tuning run number')
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

    print(f"\nAdaptive NM Iterations Experiment")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Adaptive NM polish: {args.min_polish_iters}-{args.max_polish_iters} iters, "
          f"batch={args.polish_batch_size}, threshold={args.convergence_threshold}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'min_polish_iters': args.min_polish_iters,
        'max_polish_iters': args.max_polish_iters,
        'polish_batch_size': args.polish_batch_size,
        'convergence_threshold': args.convergence_threshold,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    # Track polish iterations distribution
    polish_iters_1src = []
    polish_iters_2src = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            polished = " [P]" if 'polished' in result.get('init_types', []) else ""
            iters = result.get('polish_iters', 0)
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} polish={iters} time={result['elapsed']:.1f}s [{status}]{polished}")

            # Track polish iterations by source count
            if result['success']:
                if result['n_sources'] == 1:
                    polish_iters_1src.append(iters)
                else:
                    polish_iters_2src.append(iters)

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
    projected_400 = (total_time / n_samples) * 400 / 60

    # Calculate polish iteration stats
    avg_polish_1src = np.mean(polish_iters_1src) if polish_iters_1src else 0
    avg_polish_2src = np.mean(polish_iters_2src) if polish_iters_2src else 0
    avg_polish_all = np.mean(polish_iters_1src + polish_iters_2src) if (polish_iters_1src + polish_iters_2src) else 0

    print(f"\n{'='*70}")
    print(f"RESULTS - Adaptive NM Iterations")
    print(f"{'='*70}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print(f"Total time (80):  {total_time:.1f} s")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)}), avg polish iters={avg_polish_1src:.1f}")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)}), avg polish iters={avg_polish_2src:.1f}")
    print()
    print(f"Polish iterations distribution:")
    print(f"  1-source: min={min(polish_iters_1src) if polish_iters_1src else 0}, "
          f"max={max(polish_iters_1src) if polish_iters_1src else 0}, "
          f"avg={avg_polish_1src:.1f}")
    print(f"  2-source: min={min(polish_iters_2src) if polish_iters_2src else 0}, "
          f"max={max(polish_iters_2src) if polish_iters_2src else 0}, "
          f"avg={avg_polish_2src:.1f}")
    print()
    print(f"Baseline (fixed 8 NM): 1.1688 @ 58.4 min")
    print(f"This run:              {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                 {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print(f"{'='*70}\n")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"adaptive_nm_iter_run{args.tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("experiment_id", "EXP_ADAPTIVE_NM_POLISH_001")
            mlflow.log_param("worker", "W2")
            mlflow.log_param("tuning_run", args.tuning_run)
            mlflow.log_param("platform", "wsl")
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("timestep_fraction", args.timestep_fraction)
            mlflow.log_param("min_polish_iters", args.min_polish_iters)
            mlflow.log_param("max_polish_iters", args.max_polish_iters)
            mlflow.log_param("polish_batch_size", args.polish_batch_size)
            mlflow.log_param("convergence_threshold", args.convergence_threshold)

            # Log metrics
            mlflow.log_metric("submission_score", score)
            mlflow.log_metric("projected_400_samples_min", projected_400)
            mlflow.log_metric("total_time_sec", total_time)
            mlflow.log_metric("rmse_mean", np.mean(rmses))
            if rmses_1src:
                mlflow.log_metric("rmse_1src_mean", np.mean(rmses_1src))
            if rmses_2src:
                mlflow.log_metric("rmse_2src_mean", np.mean(rmses_2src))
            mlflow.log_metric("avg_polish_iters", avg_polish_all)
            mlflow.log_metric("avg_polish_iters_1src", avg_polish_1src)
            mlflow.log_metric("avg_polish_iters_2src", avg_polish_2src)

            print(f"MLflow run ID: {run.info.run_id}")

            # Return run ID for STATE.json
            return {
                'score': score,
                'projected_400': projected_400,
                'mlflow_run_id': run.info.run_id,
                'avg_polish_iters': avg_polish_all,
                'avg_polish_1src': avg_polish_1src,
                'avg_polish_2src': avg_polish_2src,
                'rmse_mean': np.mean(rmses),
                'rmse_1src': np.mean(rmses_1src) if rmses_1src else None,
                'rmse_2src': np.mean(rmses_2src) if rmses_2src else None,
            }

    return {
        'score': score,
        'projected_400': projected_400,
        'mlflow_run_id': None,
        'avg_polish_iters': avg_polish_all,
    }


if __name__ == '__main__':
    main()
