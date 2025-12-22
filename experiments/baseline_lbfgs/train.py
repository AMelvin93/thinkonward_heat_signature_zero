#!/usr/bin/env python
"""
Baseline L-BFGS-B optimizer experiment.

This is the simplest approach using scipy's L-BFGS-B optimizer
with finite difference gradients.

Run with:
    python scripts/run_experiment.py --experiment baseline_lbfgs
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from src.optimizer import HeatSourceOptimizer
from simulator import Heat2D


def run(config: dict, tracker) -> dict:
    """
    Run the baseline L-BFGS-B experiment.

    Args:
        config: Configuration dictionary
        tracker: ExperimentTracker instance for logging

    Returns:
        Dictionary with summary metrics
    """
    # Load test data
    data_path = project_root / config["data"]["test_path"]
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    # Limit samples if specified
    n_samples = config.get("n_samples", len(samples))
    samples = samples[:n_samples]

    print(f"Processing {len(samples)} samples with L-BFGS-B optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Create optimizer
    optimizer = HeatSourceOptimizer(Lx, Ly, nx, ny)

    # Process each sample
    all_predictions = []
    all_rmse = []

    # Track by number of sources for detailed metrics
    rmse_by_nsources = {}

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

        # Run optimization
        estimates, rmse = optimizer.estimate_sources(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            method=config["optimizer"]["method"],
            n_restarts=config["optimizer"]["n_restarts"],
            max_iter=config["optimizer"]["max_iter"],
            parallel=config["optimizer"]["parallel"],
            n_jobs=config["optimizer"]["n_jobs"],
        )

        # Track by n_sources
        if n_sources not in rmse_by_nsources:
            rmse_by_nsources[n_sources] = []
        rmse_by_nsources[n_sources].append(rmse)

        # Log to tracker
        result = tracker.log_source_estimates(
            sample_id=sample_id,
            estimates=estimates,
            rmse=rmse,
        )
        all_predictions.append(result)
        all_rmse.append(rmse)

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Log metrics by number of sources
    for n_src, rmse_list in rmse_by_nsources.items():
        tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

    # Summary metrics
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
    }

    print(f"\n{'='*50}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"{'='*50}")

    # Print breakdown by sources
    for n_src in sorted(rmse_by_nsources.keys()):
        rmse_list = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: {np.mean(rmse_list):.6f} (n={len(rmse_list)})")

    return results


if __name__ == "__main__":
    # Quick test without MLflow
    from tracking import load_config

    class DummyTracker:
        def log_params(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_predictions(self, *args, **kwargs): pass
        def log_source_estimates(self, sample_id, estimates, rmse, **kwargs):
            return {"sample_id": sample_id, "estimates": [{"x": e[0], "y": e[1], "q": e[2]} for e in estimates], "rmse": rmse}

    config = load_config(str(project_root / "configs" / "default.yaml"))
    config["n_samples"] = 2  # Quick test
    run(config, DummyTracker())
