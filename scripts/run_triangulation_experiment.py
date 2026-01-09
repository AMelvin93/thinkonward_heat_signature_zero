#!/usr/bin/env python
"""
Run full experiment with triangulation-enabled HybridOptimizer and log to MLflow.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import pickle
import time
import numpy as np
import mlflow
from datetime import datetime
from joblib import Parallel, delayed

from src.hybrid_optimizer import HybridOptimizer

# Default workers
DEFAULT_N_WORKERS = max(1, os.cpu_count() - 1)


def process_sample(sample, meta, q_range, max_iter):
    """Process a single sample (for parallel execution)."""
    optimizer = HybridOptimizer(
        n_smart_inits=1,
        n_random_inits=0,
        use_triangulation=True,  # Uses triangulation for 1-source, hottest for 2-source
    )

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, max_iter=max_iter, verbose=False
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'n_candidates': len(candidates),
        'time': elapsed,
    }


def main():
    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    q_range = tuple(meta['q_range'])

    n_samples = len(samples)
    max_iter = 3  # Production config
    n_workers = DEFAULT_N_WORKERS

    print(f"Running triangulation experiment on {n_samples} samples...")
    print(f"Config: max_iter={max_iter}, n_smart_inits=1, use_triangulation=True (1-src only)")
    print(f"Workers: {n_workers}")
    print("=" * 70)

    # Set up MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"triangulation_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("optimizer", "HybridOptimizer")
        mlflow.log_param("use_triangulation", "1-source only")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_smart_inits", 1)
        mlflow.log_param("n_random_inits", 0)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_workers", n_workers)

        # Process samples in parallel
        start_total = time.time()

        print(f"Processing {n_samples} samples with {n_workers} workers...")
        results = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(process_sample)(sample, meta, q_range, max_iter)
            for sample in samples
        )

        total_time = time.time() - start_total

        # Aggregate results
        all_rmses = [r['rmse'] for r in results]
        sample_times = [r['time'] for r in results]
        all_predictions = results

        rmse_by_nsources = {}
        for r in results:
            n_src = r['n_sources']
            if n_src not in rmse_by_nsources:
                rmse_by_nsources[n_src] = []
            rmse_by_nsources[n_src].append(r['rmse'])

        total_time = time.time() - start_total

        # Calculate metrics
        rmse_mean = np.mean(all_rmses)
        rmse_std = np.std(all_rmses)
        avg_time = np.mean(sample_times)

        # Calculate competition score
        # P = (1/N) * sum(1/(1+RMSE)) + 0.3 * (N_valid/3)
        scores = [1.0 / (1.0 + r) for r in all_rmses]
        base_score = np.mean(scores)
        n_valid = len([r for r in all_rmses if r < float('inf')])
        competition_score = base_score + 0.3 * (n_valid / 3)

        # Log metrics
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("rmse_std", rmse_std)
        mlflow.log_metric("rmse_median", np.median(all_rmses))
        mlflow.log_metric("avg_sample_time_sec", avg_time)
        mlflow.log_metric("total_time_sec", total_time)
        mlflow.log_metric("total_time_min", total_time / 60)
        mlflow.log_metric("score", competition_score)
        mlflow.log_metric("base_score", base_score)
        mlflow.log_metric("n_valid", n_valid)

        # Log per-source metrics
        for n_src, rmses in rmse_by_nsources.items():
            mlflow.log_metric(f"rmse_{n_src}src_mean", np.mean(rmses))
            mlflow.log_metric(f"rmse_{n_src}src_count", len(rmses))

        # Save predictions artifact
        predictions_path = project_root / "outputs" / f"{run_name}_predictions.pkl"
        predictions_path.parent.mkdir(exist_ok=True)
        with open(predictions_path, 'wb') as f:
            pickle.dump(all_predictions, f)
        mlflow.log_artifact(str(predictions_path))

        # Print summary
        print()
        print("=" * 70)
        print(f"RESULTS: {run_name}")
        print("=" * 70)
        print(f"RMSE: {rmse_mean:.6f} +/- {rmse_std:.6f}")
        print(f"Competition Score: {competition_score:.4f}")
        print(f"Total time: {total_time/60:.1f} min")
        print(f"Avg time per sample: {avg_time:.1f}s")
        print()
        for n_src in sorted(rmse_by_nsources.keys()):
            rmses = rmse_by_nsources[n_src]
            print(f"  {n_src}-source: {np.mean(rmses):.6f} +/- {np.std(rmses):.6f} (n={len(rmses)})")
        print("=" * 70)

        return competition_score, rmse_mean


if __name__ == "__main__":
    main()
