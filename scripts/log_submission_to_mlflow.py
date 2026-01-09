#!/usr/bin/env python
"""
Log existing submission results to MLflow.

Usage:
    uv run python scripts/log_submission_to_mlflow.py
    uv run python scripts/log_submission_to_mlflow.py --results results/submission_results.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tracking import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Log submission results to MLflow")
    parser.add_argument(
        "--results", "-r",
        default="results/submission_results.pkl",
        help="Path to submission results pickle file"
    )
    parser.add_argument(
        "--experiment-name", "-e",
        default="jax_hybrid_submission",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", "-n",
        default=None,
        help="Custom run name"
    )
    args = parser.parse_args()

    # Load results
    results_path = project_root / args.results
    print(f"Loading results from: {results_path}")

    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    config = data['config']
    final_score = float(data['final_score'])
    total_time = data['total_time']

    print(f"\n{'='*60}")
    print(f"SUBMISSION RESULTS")
    print(f"{'='*60}")
    print(f"Final Score: {final_score:.6f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Samples: {len(results)}")
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Calculate metrics
    rmse_values = []
    n_candidates_values = []
    rmse_by_nsources = {}

    for res in results:
        # Handle both 'rmse' and 'best_rmse' keys
        rmse = res.get('best_rmse') or res.get('rmse')
        if rmse is not None:
            rmse_values.append(float(rmse))
        n_cand = res.get('n_candidates', 1)
        n_candidates_values.append(n_cand)

        # Track by n_sources if available
        n_src = res.get('n_sources', 1)
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
        if rmse is not None:
            rmse_by_nsources[n_src].append(float(rmse))

    rmse_mean = np.mean(rmse_values) if rmse_values else 0
    rmse_std = np.std(rmse_values) if rmse_values else 0
    avg_candidates = np.mean(n_candidates_values)

    print(f"\nMetrics:")
    print(f"  RMSE Mean: {rmse_mean:.6f}")
    print(f"  RMSE Std: {rmse_std:.6f}")
    print(f"  Avg Candidates: {avg_candidates:.2f}")

    # Log to MLflow
    run_name = args.run_name or f"submission_score_{final_score:.4f}"

    with ExperimentTracker(
        experiment_name=args.experiment_name,
        run_name=run_name,
        tags={"source": "submission_results.pkl", "type": "historical"}
    ) as tracker:
        # Log config as params
        tracker.log_params(config)

        # Log main metrics
        tracker.log_metric("submission_score", final_score)
        tracker.log_metric("rmse_mean", rmse_mean)
        tracker.log_metric("rmse_std", rmse_std)
        tracker.log_metric("avg_candidates_per_sample", avg_candidates)
        tracker.log_metric("total_time_sec", total_time)
        tracker.log_metric("n_samples", len(results))
        tracker.log_metric("avg_sample_time_sec", total_time / len(results))
        tracker.log_metric("projected_400_samples_min", (total_time / len(results) * 400) / 60)

        # Log by n_sources
        for n_src, rmse_list in rmse_by_nsources.items():
            if rmse_list:
                tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
                tracker.log_metric(f"rmse_{n_src}src_std", np.std(rmse_list))
                tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

        # Log competition score components
        accuracy_component = 1 / (1 + rmse_mean)
        diversity_component = 0.3 * (min(avg_candidates, 3) / 3)
        tracker.log_metric("score_accuracy_component", accuracy_component)
        tracker.log_metric("score_diversity_component", diversity_component)

        # Log the results file as artifact
        tracker.log_artifact(str(results_path))

        print(f"\n{'='*60}")
        print(f"Logged to MLflow!")
        print(f"Run ID: {tracker.run_id}")
        print(f"Experiment: {args.experiment_name}")
        print(f"{'='*60}")
        print(f"\nView results: uv run mlflow ui --port 5000")


if __name__ == "__main__":
    main()
