#!/usr/bin/env python
"""
Gradient-Informed Tabu Search experiment.

This variant learns gradient information from its own evaluations during
inference and uses it to bias the search toward promising directions.

Key learning mechanisms:
- Estimates gradient from recent (params, cost) evaluations
- Biases neighborhood generation toward descent direction
- Balances gradient-informed moves with exploratory moves

Run with:
    python scripts/run_experiment.py --experiment tabu_gradient
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from src.tabu_gradient_optimizer import GradientInformedTabuOptimizer


def run(config: dict, tracker) -> dict:
    """
    Run the Gradient-Informed Tabu Search experiment.

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

    print(f"Processing {len(samples)} samples with Gradient-Informed Tabu Search...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Tabu Search parameters (base)
    tabu_config = config.get("tabu", {})
    max_iterations = tabu_config.get("max_iterations", 15)
    tabu_tenure = tabu_config.get("tabu_tenure", 7)
    tabu_radius = tabu_config.get("tabu_radius", 0.05)
    n_neighbors = tabu_config.get("n_neighbors", 12)
    initial_step = tabu_config.get("initial_step", 0.15)
    step_decay = tabu_config.get("step_decay", 0.95)
    n_restarts = tabu_config.get("n_restarts", config["optimizer"]["n_restarts"])
    use_smart_init = tabu_config.get("use_smart_init", True)

    # Gradient-specific parameters
    gradient_config = config.get("gradient", {})
    gradient_buffer_size = gradient_config.get("buffer_size", 15)
    gradient_exploitation_ratio = gradient_config.get("exploitation_ratio", 0.6)
    min_gradient_samples = gradient_config.get("min_samples", 5)

    # Log all parameters
    tracker.log_params({
        "tabu.max_iterations": max_iterations,
        "tabu.tabu_tenure": tabu_tenure,
        "tabu.tabu_radius": tabu_radius,
        "tabu.n_neighbors": n_neighbors,
        "tabu.initial_step": initial_step,
        "tabu.step_decay": step_decay,
        "tabu.n_restarts": n_restarts,
        "tabu.use_smart_init": use_smart_init,
        "gradient.buffer_size": gradient_buffer_size,
        "gradient.exploitation_ratio": gradient_exploitation_ratio,
        "gradient.min_samples": min_gradient_samples,
    })

    # Create optimizer
    optimizer = GradientInformedTabuOptimizer(
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
        gradient_buffer_size=gradient_buffer_size,
        gradient_exploitation_ratio=gradient_exploitation_ratio,
        min_gradient_samples=min_gradient_samples,
    )

    # Process each sample
    all_predictions = []
    all_rmse = []
    all_gradient_usage = []
    rmse_by_nsources = {}
    total_candidates = 0

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

        # Run Gradient-Informed Tabu Search
        estimates, rmse, candidates, gradient_usage = optimizer.estimate_sources_with_candidates(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            use_smart_init=use_smart_init,
            verbose=False,
        )

        total_candidates += len(candidates)
        all_gradient_usage.append(gradient_usage)

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

        result["n_candidates"] = len(candidates)
        result["gradient_usage"] = gradient_usage
        all_predictions.append(result)
        all_rmse.append(rmse)

        # Log per-sample metrics
        tracker.log_metric(f"candidates_{sample_id}", len(candidates))
        tracker.log_metric(f"gradient_usage_{sample_id}", gradient_usage)

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Log aggregate metrics
    for n_src, rmse_list in rmse_by_nsources.items():
        tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

    avg_candidates = total_candidates / len(samples) if samples else 0
    avg_gradient_usage = np.mean(all_gradient_usage) if all_gradient_usage else 0

    tracker.log_metric("avg_candidates_per_sample", avg_candidates)
    tracker.log_metric("total_candidates", total_candidates)
    tracker.log_metric("avg_gradient_usage", avg_gradient_usage)

    # Summary metrics
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
        "avg_candidates": avg_candidates,
        "avg_gradient_usage": avg_gradient_usage,
    }

    print(f"\n{'='*50}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"Average candidates per sample: {avg_candidates:.1f}")
    print(f"Average gradient usage: {avg_gradient_usage:.1%}")
    print(f"{'='*50}")

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
    config["n_samples"] = 1
    config["tabu"] = {
        "max_iterations": 10,
        "n_restarts": 2,
        "n_neighbors": 10,
    }
    config["gradient"] = {
        "buffer_size": 10,
        "exploitation_ratio": 0.5,
        "min_samples": 3,
    }
    run(config, DummyTracker())
