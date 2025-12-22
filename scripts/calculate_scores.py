#!/usr/bin/env python
"""
Calculate and display competition scores for all MLflow runs.

This script:
1. Fetches all completed runs from MLflow
2. Calculates the competition score for each
3. Updates MLflow with the score (if --update flag used)
4. Displays a comparison table

Usage:
    python scripts/calculate_scores.py           # Display scores
    python scripts/calculate_scores.py --update  # Update MLflow with scores
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
from mlflow.tracking import MlflowClient
from src.scoring import score_from_rmse_and_candidates, DEFAULT_LAMBDA, DEFAULT_N_MAX


def get_all_runs(experiment_name: str = "heat-signature-zero"):
    """Get all completed runs from an experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
    )

    return runs


def calculate_score_for_run(run) -> dict:
    """Calculate competition score for a single run."""
    metrics = run.data.metrics
    params = run.data.params

    rmse_mean = metrics.get('rmse_mean', None)
    avg_candidates = metrics.get('avg_candidates_per_sample', 1.0)

    # Try to get from params if not in metrics
    if avg_candidates == 1.0:
        avg_candidates = float(metrics.get('avg_candidates', 1.0))

    if rmse_mean is None:
        return {
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'error': 'No rmse_mean metric found',
        }

    # Calculate score
    score = score_from_rmse_and_candidates(rmse_mean, avg_candidates, DEFAULT_LAMBDA, DEFAULT_N_MAX)
    accuracy_component = 1 / (1 + rmse_mean)
    diversity_component = DEFAULT_LAMBDA * (min(avg_candidates, DEFAULT_N_MAX) / DEFAULT_N_MAX)

    return {
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'experiment_type': run.data.tags.get('experiment_type', 'unknown'),
        'rmse_mean': rmse_mean,
        'avg_candidates': avg_candidates,
        'score': score,
        'accuracy_component': accuracy_component,
        'diversity_component': diversity_component,
    }


def update_run_with_score(run_id: str, score_data: dict):
    """Update MLflow run with competition score."""
    client = MlflowClient()

    if 'error' in score_data:
        return False

    client.log_metric(run_id, "competition_score", score_data['score'])
    client.log_metric(run_id, "score_accuracy_component", score_data['accuracy_component'])
    client.log_metric(run_id, "score_diversity_component", score_data['diversity_component'])

    return True


def main():
    parser = argparse.ArgumentParser(description="Calculate competition scores for MLflow runs")
    parser.add_argument("--update", action="store_true", help="Update MLflow with calculated scores")
    parser.add_argument("--experiment", default="heat-signature-zero", help="Experiment name")
    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("mlruns")

    print(f"Fetching runs from experiment: {args.experiment}")
    runs = get_all_runs(args.experiment)

    if not runs:
        print("No completed runs found.")
        return

    print(f"Found {len(runs)} completed runs.\n")

    # Calculate scores
    results = []
    for run in runs:
        score_data = calculate_score_for_run(run)
        results.append(score_data)

        if args.update and 'error' not in score_data:
            update_run_with_score(run.info.run_id, score_data)

    # Sort by score (highest first)
    results_with_scores = [r for r in results if 'score' in r]
    results_with_scores.sort(key=lambda x: x['score'], reverse=True)

    # Display table
    print("=" * 100)
    print(f"{'Rank':<5} {'Experiment':<25} {'RMSE':<12} {'Candidates':<12} {'Score':<10} {'Run Name':<30}")
    print("=" * 100)

    for i, r in enumerate(results_with_scores, 1):
        print(f"{i:<5} {r['experiment_type']:<25} {r['rmse_mean']:<12.6f} {r['avg_candidates']:<12.1f} {r['score']:<10.4f} {r['run_name']:<30}")

    print("=" * 100)

    # Show errors
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\nRuns with errors ({len(errors)}):")
        for r in errors:
            print(f"  - {r['run_name']}: {r['error']}")

    if args.update:
        print(f"\n[OK] Updated {len(results_with_scores)} runs with competition scores.")

    # Summary
    if results_with_scores:
        best = results_with_scores[0]
        print(f"\n*** BEST MODEL: {best['experiment_type']} ***")
        print(f"   Score: {best['score']:.4f} (RMSE: {best['rmse_mean']:.6f}, Candidates: {best['avg_candidates']:.1f})")


if __name__ == "__main__":
    main()
