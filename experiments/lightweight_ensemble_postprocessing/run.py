"""
Run script for lightweight ensemble postprocessing experiment.

Based on production baseline with minimal ensemble addition.
"""

import os
import sys
import time
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import LightweightEnsembleOptimizer


def process_sample(args):
    """Process a single sample."""
    idx, sample, meta, optimizer_kwargs = args
    optimizer = LightweightEnsembleOptimizer(**optimizer_kwargs)
    candidate_sources, best_rmse, results, n_sims, ensemble_win = optimizer.estimate_sources(
        sample, meta, q_range=(0.5, 2.0)
    )
    return {
        'idx': idx,
        'candidates': candidate_sources,
        'best_rmse': best_rmse,
        'n_candidates': len(candidate_sources),
        'n_sources': sample['n_sources'],
        'n_sims': n_sims,
        'ensemble_win': ensemble_win,
        'init_types': [r.init_type for r in results],
    }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    """Calculate score for a single sample."""
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def run_experiment(
    n_samples=80,
    n_workers=None,
    ensemble_top_n=5,
):
    """Run the lightweight ensemble experiment."""
    if n_workers is None:
        n_workers = os.cpu_count()

    # Load data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples'][:n_samples]
    meta = data['meta']

    optimizer_kwargs = {
        'ensemble_top_n': ensemble_top_n,
        # All other params use production baseline defaults
    }

    print(f"[Ensemble] Running with ensemble_top_n={ensemble_top_n}")
    print(f"[Ensemble] Processing {n_samples} samples with {n_workers} workers")

    start_time = time.time()

    results = []
    ensemble_wins = 0
    total_sims = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_sample, (i, sample, meta, optimizer_kwargs)): i
            for i, sample in enumerate(samples)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            total_sims += result['n_sims']
            if result['ensemble_win']:
                ensemble_wins += 1
            if len(results) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {len(results)}/{n_samples} samples ({elapsed:.1f}s)")

    total_time = time.time() - start_time

    # Calculate scores
    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    # Stats
    rmses = [r['best_rmse'] for r in results]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['n_sources'] == 2]

    projected_400_min = (total_time / n_samples) * 400 / 60
    in_budget = projected_400_min < 60

    print(f"\n[Ensemble] Results:")
    print(f"  Score: {score:.4f}")
    print(f"  RMSE mean: {rmse_mean:.4f}")
    if rmses_1src:
        print(f"    1-source RMSE: {np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"    2-source RMSE: {np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Projected 400 samples: {projected_400_min:.1f} min")
    print(f"  In budget (<60 min): {in_budget}")
    print(f"  Ensemble wins: {ensemble_wins}/{n_samples} ({100*ensemble_wins/n_samples:.1f}%)")

    # Log to MLflow
    mlflow.set_experiment("lightweight_ensemble_postprocessing")
    with mlflow.start_run():
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_workers", n_workers)
        mlflow.log_param("ensemble_top_n", ensemble_top_n)
        mlflow.log_param("platform", "wsl")

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("total_time_sec", total_time)
        mlflow.log_metric("projected_400_samples_min", projected_400_min)
        mlflow.log_metric("in_budget", 1 if in_budget else 0)
        mlflow.log_metric("ensemble_wins", ensemble_wins)
        mlflow.log_metric("ensemble_win_pct", 100 * ensemble_wins / n_samples)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[Ensemble] MLflow run ID: {run_id}")

    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'total_time': total_time,
        'projected_400_min': projected_400_min,
        'in_budget': in_budget,
        'ensemble_wins': ensemble_wins,
        'ensemble_win_pct': 100 * ensemble_wins / n_samples,
        'mlflow_id': run_id,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=80)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--ensemble_top_n', type=int, default=5)
    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        n_workers=args.n_workers,
        ensemble_top_n=args.ensemble_top_n,
    )
