#!/usr/bin/env python
"""
Run an experiment with MLflow tracking.

Usage:
    python scripts/run_experiment.py --experiment baseline_lbfgs
    python scripts/run_experiment.py --experiment baseline_lbfgs --config configs/custom.yaml
    python scripts/run_experiment.py --experiment jax_gradient --n-samples 10

This script:
1. Loads the experiment module from experiments/<name>/
2. Runs the experiment with MLflow tracking
3. Logs metrics, parameters, and artifacts
"""

import argparse
import importlib.util
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tracking import ExperimentTracker, load_config, merge_configs


def load_experiment_module(experiment_name: str):
    """Dynamically load an experiment module."""
    experiment_path = project_root / "experiments" / experiment_name / "train.py"

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment not found: {experiment_path}\n"
            f"Create it with: python scripts/create_experiment.py {experiment_name}"
        )

    spec = importlib.util.spec_from_file_location("experiment", experiment_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run an experiment with MLflow tracking")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment name (folder in experiments/)")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--n-samples", "-n", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--run-name", "-r", default=None, help="Custom run name")
    parser.add_argument("--tags", "-t", nargs="*", default=[], help="Tags in key=value format")
    args = parser.parse_args()

    # Parse tags
    tags = {}
    for tag in args.tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key] = value

    # Load configs
    base_config = load_config(args.config)
    experiment_config_path = project_root / "experiments" / args.experiment / "config.yaml"

    if experiment_config_path.exists():
        experiment_config = load_config(str(experiment_config_path))
        config = merge_configs(base_config, experiment_config)
    else:
        config = base_config

    # Override with CLI args
    if args.n_samples:
        config["n_samples"] = args.n_samples

    # Load experiment module
    print(f"Loading experiment: {args.experiment}")
    experiment = load_experiment_module(args.experiment)

    # Run with tracking
    print(f"Starting MLflow run...")
    with ExperimentTracker(
        experiment_name=args.experiment,
        run_name=args.run_name,
        config_path=args.config,
        tags=tags,
    ) as tracker:
        # Log configuration
        tracker.log_params(config)

        # Run the experiment
        results = experiment.run(config, tracker)

        # Log summary metrics
        if results:
            if "rmse_mean" in results:
                tracker.log_metric("rmse_mean", results["rmse_mean"])
            if "rmse_std" in results:
                tracker.log_metric("rmse_std", results["rmse_std"])

            # Calculate and log competition score
            rmse_mean = results.get("rmse_mean", 0)
            avg_candidates = results.get("avg_candidates", 1.0)
            score = tracker.log_final_score(rmse_mean, avg_candidates)
            print(f"\n{'='*50}")
            print(f"COMPETITION SCORE: {score:.4f}")
            print(f"  Accuracy (1/(1+L)): {1/(1+rmse_mean):.4f}")
            print(f"  Diversity bonus:    {0.3 * min(avg_candidates, 3) / 3:.4f}")
            print(f"{'='*50}")

        print(f"\nRun completed! Run ID: {tracker.run_id}")
        print(f"View results: mlflow ui --port 5000")

    return results


if __name__ == "__main__":
    main()
