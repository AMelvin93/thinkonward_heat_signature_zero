#!/usr/bin/env python
"""
JAX-based optimizer with automatic differentiation.

Uses JAX for:
- GPU acceleration of PDE solver
- Automatic differentiation for true gradients (not finite differences)
- JIT compilation for speed

Run with:
    python scripts/run_experiment.py --experiment jax_gradient
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np

# Check JAX availability
try:
    from src.jax_optimizer import JAXOptimizer
    from src.jax_simulator import JAXHeatSimulator, check_gpu
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    JAX_ERROR = str(e)


def run(config: dict, tracker) -> dict:
    """
    Run the JAX gradient-based experiment.

    Args:
        config: Configuration dictionary
        tracker: ExperimentTracker instance for logging

    Returns:
        Dictionary with summary metrics
    """
    if not JAX_AVAILABLE:
        raise ImportError(f"JAX not available: {JAX_ERROR}")

    # Check GPU
    gpu_available = check_gpu()
    tracker.log_metric("gpu_available", int(gpu_available))
    print(f"GPU available: {gpu_available}")

    # Load test data
    data_path = project_root / config["data"]["test_path"]
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    # Limit samples if specified
    n_samples = config.get("n_samples", len(samples))
    samples = samples[:n_samples]

    print(f"Processing {len(samples)} samples with JAX optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # JAX-specific parameters
    jax_config = config.get("jax", {})
    learning_rate = jax_config.get("learning_rate", 0.01)
    n_iterations = jax_config.get("n_iterations", 100)
    n_restarts = jax_config.get("n_restarts", config["optimizer"]["n_restarts"])

    tracker.log_params({
        "jax.learning_rate": learning_rate,
        "jax.n_iterations": n_iterations,
        "jax.n_restarts": n_restarts,
    })

    # Create JAX optimizer
    optimizer = JAXOptimizer(Lx, Ly, nx, ny)

    # Process each sample
    all_predictions = []
    all_rmse = []
    rmse_by_nsources = {}

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

        # Run JAX optimization
        estimates, rmse = optimizer.estimate_sources(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            n_restarts=n_restarts,
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
    config["n_samples"] = 2
    run(config, DummyTracker())
