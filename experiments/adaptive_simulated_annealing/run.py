#!/usr/bin/env python3
"""
Run script for Adaptive Simulated Annealing experiment.
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import DualAnnealingOptimizer


def process_single_sample(args):
    """Process a single sample - for parallel execution."""
    idx, sample, meta, config = args

    optimizer = DualAnnealingOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        n_restarts=config['n_restarts'],
        refine_maxiter=config['refine_maxiter'],
        refine_top_n=config['refine_top_n'],
        timestep_fraction=config['timestep_fraction'],
        initial_temp=config.get('initial_temp', 5000),
        visit=config.get('visit', 2.62),
        accept=config.get('accept', -5.0),
        no_local_search=config.get('no_local_search', False),
    )

    q_range = meta['q_range']
    start_time = time.time()

    try:
        sources, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=q_range, verbose=False
        )
        success = True
    except Exception as e:
        n_sources = sample['n_sources']
        if n_sources == 1:
            sources = [(1.0, 0.5, 1.0)]
        else:
            sources = [(0.7, 0.5, 1.0), (1.3, 0.5, 1.0)]
        best_rmse = float('inf')
        n_sims = 0
        success = False

    elapsed = time.time() - start_time

    return {
        'idx': idx,
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'sources': sources,
        'best_rmse': best_rmse,
        'n_candidates': len(sources) if sources else 0,
        'n_sims': n_sims,
        'elapsed': elapsed,
        'success': success,
    }


def run_experiment(
    max_fevals_1src=100,
    max_fevals_2src=200,
    n_restarts=3,
    refine_maxiter=8,
    refine_top_n=2,
    timestep_fraction=0.4,
    initial_temp=5000,
    visit=2.62,
    accept=-5.0,
    no_local_search=False,
    n_workers=None,
    shuffle=True,
    verbose=True,
    tuning_run=1,
):
    """Run dual_annealing optimization experiment."""
    if n_workers is None:
        n_workers = os.cpu_count()

    # Load test data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    if shuffle:
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        samples = [samples[i] for i in indices]
    else:
        indices = np.arange(len(samples))

    n_samples = len(samples)

    if verbose:
        print(f"\nDual Annealing Optimizer")
        print(f"=" * 60)
        print(f"Samples: {n_samples}, Workers: {n_workers}")
        print(f"Fevals: {max_fevals_1src}/{max_fevals_2src}")
        print(f"Restarts: {n_restarts}")
        print(f"Refine: {refine_maxiter} iters on top-{refine_top_n}")
        print(f"Initial temp: {initial_temp}, Visit: {visit}, Accept: {accept}")
        print(f"No local search: {no_local_search}")
        print(f"=" * 60)

    config = {
        'max_fevals_1src': max_fevals_1src,
        'max_fevals_2src': max_fevals_2src,
        'n_restarts': n_restarts,
        'refine_maxiter': refine_maxiter,
        'refine_top_n': refine_top_n,
        'timestep_fraction': timestep_fraction,
        'initial_temp': initial_temp,
        'visit': visit,
        'accept': accept,
        'no_local_search': no_local_search,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            if verbose and (len(results) % 10 == 0 or len(results) == n_samples):
                elapsed = time.time() - start_time
                avg_time = elapsed / len(results)
                projected_400 = avg_time * 400 / 60
                rmses = [r['best_rmse'] for r in results if r['success']]
                avg_rmse = np.mean(rmses) if rmses else float('inf')
                print(f"[{len(results):3d}/{n_samples}] Elapsed: {elapsed/60:.1f} min, "
                      f"Avg RMSE: {avg_rmse:.4f}, Projected 400: {projected_400:.1f} min")

    total_time = time.time() - start_time

    # Calculate score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)
    total_sims = sum(r['n_sims'] for r in results)

    projected_400_min = (total_time / n_samples) * 400 / 60

    # Log to MLflow
    run_name = f"dual_annealing_run{tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_ADAPTIVE_SA_001")
        mlflow.log_param("worker", "W2")
        mlflow.log_param("tuning_run", tuning_run)
        mlflow.log_param("max_fevals_1src", max_fevals_1src)
        mlflow.log_param("max_fevals_2src", max_fevals_2src)
        mlflow.log_param("n_restarts", n_restarts)
        mlflow.log_param("refine_maxiter", refine_maxiter)
        mlflow.log_param("timestep_fraction", timestep_fraction)
        mlflow.log_param("initial_temp", initial_temp)
        mlflow.log_param("visit", visit)
        mlflow.log_param("accept", accept)
        mlflow.log_param("no_local_search", no_local_search)
        mlflow.log_param("n_workers", n_workers)
        mlflow.log_param("platform", "wsl")

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400_min)
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("total_simulations", total_sims)
        mlflow.log_metric("total_time_min", total_time / 60)

        mlflow_run_id = run.info.run_id

    results_dict = {
        'score': score,
        'total_time_sec': total_time,
        'total_time_min': total_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'total_simulations': total_sims,
        'n_samples': n_samples,
        'mlflow_run_id': mlflow_run_id,
    }

    if verbose:
        print(f"\n=== Results ===")
        print(f"Score: {score:.4f}")
        print(f"Time: {total_time/60:.1f} min ({n_samples} samples)")
        print(f"Projected 400 samples: {projected_400_min:.1f} min")
        print(f"Mean RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
        print(f"Total simulations: {total_sims}")
        print(f"MLflow run ID: {mlflow_run_id}")

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-fevals-1src', type=int, default=100)
    parser.add_argument('--max-fevals-2src', type=int, default=200)
    parser.add_argument('--n-restarts', type=int, default=3)
    parser.add_argument('--refine-maxiter', type=int, default=8)
    parser.add_argument('--refine-top-n', type=int, default=2)
    parser.add_argument('--timestep-fraction', type=float, default=0.4)
    parser.add_argument('--initial-temp', type=float, default=5000)
    parser.add_argument('--visit', type=float, default=2.62)
    parser.add_argument('--accept', type=float, default=-5.0)
    parser.add_argument('--no-local-search', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--tuning-run', type=int, default=1)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_experiment(
        max_fevals_1src=args.max_fevals_1src,
        max_fevals_2src=args.max_fevals_2src,
        n_restarts=args.n_restarts,
        refine_maxiter=args.refine_maxiter,
        refine_top_n=args.refine_top_n,
        timestep_fraction=args.timestep_fraction,
        initial_temp=args.initial_temp,
        visit=args.visit,
        accept=args.accept,
        no_local_search=args.no_local_search,
        n_workers=args.workers,
        tuning_run=args.tuning_run,
        shuffle=not args.no_shuffle,
        verbose=not args.quiet,
    )
