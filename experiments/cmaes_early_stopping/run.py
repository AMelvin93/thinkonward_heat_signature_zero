#!/usr/bin/env python3
"""Run script for CMA-ES Early Stopping experiment with MLflow logging."""

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

from optimizer import EarlyStopCMAESOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = EarlyStopCMAESOptimizer(
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
        base_polish_iters=config['base_polish_iters'],
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
        stagnation_threshold=config['stagnation_threshold'],
        stagnation_generations=config['stagnation_generations'],
        extra_polish_per_saved=config['extra_polish_per_saved'],
        max_polish_iters=config['max_polish_iters'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, saved_fevals, polish_iters = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'saved_fevals': saved_fevals, 'polish_iters': polish_iters,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'saved_fevals': 0, 'polish_iters': 0,
            'success': False, 'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run CMA-ES Early Stopping Experiment')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--base-polish-iters', type=int, default=8)
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    parser.add_argument('--stagnation-threshold', type=float, default=0.01)
    parser.add_argument('--stagnation-generations', type=int, default=3)
    parser.add_argument('--extra-polish-per-saved', type=float, default=0.5)
    parser.add_argument('--max-polish-iters', type=int, default=12)
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

    print(f"\nCMA-ES Early Stopping Experiment")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Stagnation: {args.stagnation_threshold*100:.0f}% for {args.stagnation_generations} gens")
    print(f"Base polish: {args.base_polish_iters}, max: {args.max_polish_iters}")
    print(f"Extra polish per saved feval: {args.extra_polish_per_saved}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'base_polish_iters': args.base_polish_iters,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'stagnation_threshold': args.stagnation_threshold,
        'stagnation_generations': args.stagnation_generations,
        'extra_polish_per_saved': args.extra_polish_per_saved,
        'max_polish_iters': args.max_polish_iters,
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
            polish_info = f"polish={result['polish_iters']}" if result['saved_fevals'] > 0 else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} saved={result['saved_fevals']} {polish_info} "
                  f"time={result['elapsed']:.1f}s [{status}]")

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
    saved_fevals_total = sum(r['saved_fevals'] for r in results if r['success'])
    avg_polish_iters = np.mean([r['polish_iters'] for r in results if r['success']])
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - CMA-ES Early Stopping")
    print(f"{'='*70}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print(f"Total time (80):  {total_time:.1f} s")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Early stopping stats:")
    print(f"  Total saved fevals: {saved_fevals_total}")
    print(f"  Avg polish iters: {avg_polish_iters:.1f} (base: {args.base_polish_iters})")
    print()
    print(f"Baseline (no early stop): 1.1688 @ 58.4 min")
    print(f"This run (early stop): {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print(f"{'='*70}\n")

    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"early_stop_run{args.tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("experiment_id", "EXP_EARLY_STOP_CMA_001")
            mlflow.log_param("worker", "W2")
            mlflow.log_param("tuning_run", args.tuning_run)
            mlflow.log_param("platform", "wsl")
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("stagnation_threshold", args.stagnation_threshold)
            mlflow.log_param("stagnation_generations", args.stagnation_generations)
            mlflow.log_param("base_polish_iters", args.base_polish_iters)
            mlflow.log_param("max_polish_iters", args.max_polish_iters)

            mlflow.log_metric("submission_score", score)
            mlflow.log_metric("projected_400_samples_min", projected_400)
            mlflow.log_metric("total_time_sec", total_time)
            mlflow.log_metric("rmse_mean", np.mean(rmses))
            mlflow.log_metric("saved_fevals_total", saved_fevals_total)
            mlflow.log_metric("avg_polish_iters", avg_polish_iters)
            if rmses_1src:
                mlflow.log_metric("rmse_1src_mean", np.mean(rmses_1src))
            if rmses_2src:
                mlflow.log_metric("rmse_2src_mean", np.mean(rmses_2src))

            print(f"MLflow run ID: {run.info.run_id}")

            return {
                'score': score,
                'projected_400': projected_400,
                'mlflow_run_id': run.info.run_id,
            }

    return {
        'score': score,
        'projected_400': projected_400,
        'mlflow_run_id': None,
    }


if __name__ == '__main__':
    main()
