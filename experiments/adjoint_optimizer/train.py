#!/usr/bin/env python
"""
Adjoint Optimizer experiment - exact gradients via adjoint method.

Uses adjoint method for O(1) gradient computation instead of O(n_params)
finite differences. This should enable more L-BFGS-B iterations within
the same time budget.

Run with:
    uv run python scripts/run_experiment.py --experiment adjoint_optimizer
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import time
import numpy as np

from src.adjoint_optimizer_fast import FastAdjointOptimizer


def run(config: dict, tracker) -> dict:
    """
    Run the Adjoint Optimizer experiment.

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

    print(f"\nProcessing {len(samples)} samples with Adjoint optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Adjoint-specific parameters
    adjoint_config = config.get("adjoint", {})
    n_smart_inits = adjoint_config.get("n_smart_inits", 1)
    n_random_inits = adjoint_config.get("n_random_inits", 1)
    min_candidate_distance = adjoint_config.get("min_candidate_distance", 0.15)
    n_max_candidates = adjoint_config.get("n_max_candidates", 3)
    max_iter = adjoint_config.get("max_iter", 50)

    tracker.log_params({
        "adjoint.n_smart_inits": n_smart_inits,
        "adjoint.n_random_inits": n_random_inits,
        "adjoint.min_candidate_distance": min_candidate_distance,
        "adjoint.n_max_candidates": n_max_candidates,
        "adjoint.max_iter": max_iter,
        "adjoint.total_inits": n_smart_inits + n_random_inits,
    })

    # Create optimizer with checkpointing
    optimizer = FastAdjointOptimizer(
        Lx=Lx, Ly=Ly, nx=nx, ny=ny,
        n_smart_inits=n_smart_inits,
        n_random_inits=n_random_inits,
        min_candidate_distance=min_candidate_distance,
        n_max_candidates=n_max_candidates,
    )

    q_range = tuple(config["optimizer"]["q_range"])

    # Process samples
    all_predictions = []
    all_rmse = []
    all_n_candidates = []
    rmse_by_nsources = {}
    sample_times = []

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)", end="", flush=True)

        start_time = time.time()
        estimates, rmse, candidates = optimizer.estimate_sources(
            sample, meta,
            q_range=q_range,
            max_iter=max_iter,
            verbose=False,
        )
        elapsed = time.time() - start_time
        sample_times.append(elapsed)

        n_candidates = len(candidates)
        all_n_candidates.append(n_candidates)

        print(f" -> RMSE={rmse:.4f}, candidates={n_candidates}, time={elapsed:.1f}s")

        if n_sources not in rmse_by_nsources:
            rmse_by_nsources[n_sources] = []
        rmse_by_nsources[n_sources].append(rmse)

        result = tracker.log_source_estimates(
            sample_id=sample_id,
            estimates=estimates,
            rmse=rmse,
            n_candidates=n_candidates,
        )
        all_predictions.append(result)
        all_rmse.append(rmse)

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Log metrics by number of sources
    for n_src, rmse_list in rmse_by_nsources.items():
        tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_std", np.std(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

    # Timing metrics
    avg_time = np.mean(sample_times)
    total_time = np.sum(sample_times)
    tracker.log_metric("avg_sample_time_sec", avg_time)
    tracker.log_metric("total_time_sec", total_time)
    tracker.log_metric("projected_400_samples_min", (avg_time * 400) / 60)

    # Aggregate metrics
    avg_candidates = np.mean(all_n_candidates)
    tracker.log_metric("avg_candidates_per_sample", avg_candidates)

    # Summary
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
        "avg_candidates": avg_candidates,
        "avg_sample_time": avg_time,
        "total_time": total_time,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"Average time per sample: {avg_time:.1f}s")
    print(f"Projected time for 400 samples: {(avg_time * 400) / 60:.1f} min")
    print(f"Average candidates per sample: {avg_candidates:.1f}")
    print(f"{'='*60}")

    for n_src in sorted(rmse_by_nsources.keys()):
        rmse_list = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: {np.mean(rmse_list):.6f} +/- {np.std(rmse_list):.6f} (n={len(rmse_list)})")

    return results


if __name__ == "__main__":
    # Quick test without MLflow
    sys.path.insert(0, str(project_root))
    from tracking import load_config

    class DummyTracker:
        def log_params(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_predictions(self, *args, **kwargs): pass
        def log_source_estimates(self, sample_id, estimates, rmse, n_candidates=1, **kwargs):
            return {
                "sample_id": sample_id,
                "estimates": [{"x": e[0], "y": e[1], "q": e[2]} for e in estimates],
                "rmse": rmse,
                "n_candidates": n_candidates,
            }

    config = load_config(str(project_root / "configs" / "default.yaml"))
    config["n_samples"] = 5  # Quick test
    config["adjoint"] = {
        "n_smart_inits": 1,
        "n_random_inits": 1,
        "max_iter": 20,
    }
    run(config, DummyTracker())
