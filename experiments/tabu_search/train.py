#!/usr/bin/env python
"""
Tabu Search optimizer experiment.

Tabu Search is a metaheuristic that uses memory structures to escape
local optima by preventing the search from revisiting recent solutions.

Key advantages:
- Escapes local optima via tabu list
- Accepts worse moves strategically
- Generates diverse candidates through multiple restarts
- Smart initialization using sensor data

Run with:
    python scripts/run_experiment.py --experiment tabu_search
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from src.tabu_optimizer import TabuSearchOptimizer


def run(config: dict, tracker) -> dict:
    """
    Run the Tabu Search experiment.

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

    print(f"Processing {len(samples)} samples with Tabu Search optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Tabu Search specific parameters
    tabu_config = config.get("tabu", {})
    max_iterations = tabu_config.get("max_iterations", 50)
    tabu_tenure = tabu_config.get("tabu_tenure", 10)
    tabu_radius = tabu_config.get("tabu_radius", 0.05)
    n_neighbors = tabu_config.get("n_neighbors", 20)
    initial_step = tabu_config.get("initial_step", 0.15)
    step_decay = tabu_config.get("step_decay", 0.98)
    n_restarts = tabu_config.get("n_restarts", config["optimizer"]["n_restarts"])
    use_smart_init = tabu_config.get("use_smart_init", True)

    # Log tabu-specific params
    tracker.log_params({
        "tabu.max_iterations": max_iterations,
        "tabu.tabu_tenure": tabu_tenure,
        "tabu.tabu_radius": tabu_radius,
        "tabu.n_neighbors": n_neighbors,
        "tabu.initial_step": initial_step,
        "tabu.step_decay": step_decay,
        "tabu.n_restarts": n_restarts,
        "tabu.use_smart_init": use_smart_init,
    })

    # Create optimizer
    optimizer = TabuSearchOptimizer(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        max_iterations=max_iterations,
        tabu_tenure=tabu_tenure,
        tabu_radius=tabu_radius,
        n_neighbors=n_neighbors,
        initial_step=initial_step,
        step_decay=step_decay,
        n_restarts=n_restarts,
    )

    # Process each sample
    all_predictions = []
    all_rmse = []
    rmse_by_nsources = {}
    total_candidates = 0

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

        # Run Tabu Search optimization with candidates
        estimates, rmse, candidates = optimizer.estimate_sources_with_candidates(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            use_smart_init=use_smart_init,
            verbose=False,
        )

        total_candidates += len(candidates)

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

        # Add candidate count to result
        result["n_candidates"] = len(candidates)
        all_predictions.append(result)
        all_rmse.append(rmse)

        # Log per-sample candidate count
        tracker.log_metric(f"candidates_{sample_id}", len(candidates))

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Log metrics by number of sources
    for n_src, rmse_list in rmse_by_nsources.items():
        tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

    # Log total candidates (diversity metric)
    avg_candidates = total_candidates / len(samples) if samples else 0
    tracker.log_metric("avg_candidates_per_sample", avg_candidates)
    tracker.log_metric("total_candidates", total_candidates)

    # Summary metrics
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
        "avg_candidates": avg_candidates,
    }

    print(f"\n{'='*50}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"Average candidates per sample: {avg_candidates:.1f}")
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
            return {
                "sample_id": sample_id,
                "estimates": [{"x": e[0], "y": e[1], "q": e[2]} for e in estimates],
                "rmse": rmse
            }

    config = load_config(str(project_root / "configs" / "default.yaml"))
    config["n_samples"] = 1  # Quick test
    config["tabu"] = {
        "max_iterations": 20,
        "n_restarts": 2,
    }
    run(config, DummyTracker())
