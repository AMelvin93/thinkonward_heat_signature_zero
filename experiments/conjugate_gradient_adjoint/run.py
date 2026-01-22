#!/usr/bin/env python3
"""
Run script for Conjugate Gradient with Adjoint Gradients experiment.
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import AdjointOptimizer
from utils import score_submission


def run_experiment(
    n_restarts_1src=3,
    n_restarts_2src=5,
    max_iter=20,
    timestep_fraction=0.4,
    n_workers=None,
    shuffle=True,
    verbose=True,
    tuning_run=1,
):
    """Run adjoint gradient optimization experiment."""
    if n_workers is None:
        n_workers = os.cpu_count()

    # Load test data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    q_range = (meta['q_min'], meta['q_max'])

    if shuffle:
        indices = np.random.permutation(len(samples))
        samples = [samples[i] for i in indices]

    # Create optimizer
    optimizer = AdjointOptimizer(
        n_restarts_1src=n_restarts_1src,
        n_restarts_2src=n_restarts_2src,
        max_iter_per_restart=max_iter,
        timestep_fraction=timestep_fraction,
    )

    # Run optimization (sequential for now - adjoint solver has state)
    predictions = []
    rmses = []
    times_per_sample = []
    total_sims = 0

    start_time = time.time()

    for i, sample in enumerate(samples):
        sample_start = time.time()

        try:
            sources, best_rmse, results, n_sims = optimizer.estimate_sources(
                sample, meta, q_range=q_range, verbose=verbose
            )
            total_sims += n_sims
        except Exception as e:
            if verbose:
                print(f"Sample {i}: ERROR - {e}")
            # Fallback to center of domain
            n_sources = sample['n_sources']
            if n_sources == 1:
                sources = [(1.0, 0.5, 1.0)]
            else:
                sources = [(0.7, 0.5, 1.0), (1.3, 0.5, 1.0)]
            best_rmse = float('inf')

        sample_time = time.time() - sample_start
        times_per_sample.append(sample_time)
        rmses.append(best_rmse)
        predictions.append(sources)

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            projected_total = avg_time * len(samples)
            projected_400 = avg_time * 400 / 60
            print(f"  [{i+1}/{len(samples)}] Elapsed: {elapsed/60:.1f} min, "
                  f"Avg RMSE: {np.mean(rmses):.4f}, "
                  f"Projected 400: {projected_400:.1f} min")

    total_time = time.time() - start_time

    # Calculate score
    pred_dataset = []
    for i, sample in enumerate(samples):
        pred_dataset.append({
            'sample_id': sample['sample_id'],
            'sources': [{'x': s[0], 'y': s[1], 'q': s[2]} for s in predictions[i]]
        })

    score = score_submission(
        data, pred_dataset, N_max=3, lambda_=0.3, tau=0.2,
        scale_factors=(2.0, 1.0, 2.0), forward_loss="rmse",
        solver_kwargs={"Lx": 2.0, "Ly": 1.0, "nx": 100, "ny": 50}
    )

    # Calculate metrics
    n_samples = len(samples)
    projected_400_min = (total_time / n_samples) * 400 / 60

    results_dict = {
        'score': score,
        'total_time_sec': total_time,
        'total_time_min': total_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
        'total_simulations': total_sims,
        'avg_sims_per_sample': total_sims / n_samples,
        'n_samples': n_samples,
    }

    # Log to MLflow
    run_name = f"adjoint_gradient_run{tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_CONJUGATE_GRADIENT_001")
        mlflow.log_param("worker", "W2")
        mlflow.log_param("tuning_run", tuning_run)
        mlflow.log_param("n_restarts_1src", n_restarts_1src)
        mlflow.log_param("n_restarts_2src", n_restarts_2src)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("timestep_fraction", timestep_fraction)
        mlflow.log_param("n_workers", n_workers)
        mlflow.log_param("platform", "wsl")

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400_min)
        mlflow.log_metric("rmse_mean", np.mean(rmses))
        mlflow.log_metric("total_simulations", total_sims)
        mlflow.log_metric("total_time_min", total_time / 60)

        mlflow_run_id = run.info.run_id

    results_dict['mlflow_run_id'] = mlflow_run_id

    if verbose:
        print(f"\n=== Results ===")
        print(f"Score: {score:.4f}")
        print(f"Time: {total_time/60:.1f} min ({n_samples} samples)")
        print(f"Projected 400 samples: {projected_400_min:.1f} min")
        print(f"Mean RMSE: {np.mean(rmses):.4f}")
        print(f"Total simulations: {total_sims}")
        print(f"MLflow run ID: {mlflow_run_id}")

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-restarts-1src', type=int, default=3)
    parser.add_argument('--n-restarts-2src', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--timestep-fraction', type=float, default=0.4)
    parser.add_argument('--tuning-run', type=int, default=1)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_experiment(
        n_restarts_1src=args.n_restarts_1src,
        n_restarts_2src=args.n_restarts_2src,
        max_iter=args.max_iter,
        timestep_fraction=args.timestep_fraction,
        tuning_run=args.tuning_run,
        shuffle=not args.no_shuffle,
        verbose=not args.quiet,
    )
