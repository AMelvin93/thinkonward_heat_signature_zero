#!/usr/bin/env python
"""
Adaptive Learning Tabu Search experiment.

This optimizer combines multiple learning mechanisms that adapt during inference:
1. Observational learning: Smart init from sensor temperature patterns
2. Gradient learning: Descent direction from recent evaluations
3. Landscape learning: Adaptive step size from objective roughness
4. Curvature learning: L-BFGS-B polish learns local curvature
5. Search dynamics learning: Adaptive tabu tenure

Key philosophy: Learn from each simulator call, not from pre-computed strategies.

Run with:
    python scripts/run_experiment.py --experiment adaptive_learning
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from src.adaptive_learning_optimizer import AdaptiveLearningOptimizer


def run(config: dict, tracker) -> dict:
    """
    Run the Adaptive Learning Tabu Search experiment.

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

    print(f"Processing {len(samples)} samples with Adaptive Learning optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Base Tabu Search parameters
    tabu_config = config.get("tabu", {})
    max_iterations = tabu_config.get("max_iterations", 25)
    base_tabu_tenure = tabu_config.get("base_tabu_tenure", 8)
    tabu_radius = tabu_config.get("tabu_radius", 0.05)
    n_neighbors = tabu_config.get("n_neighbors", 16)
    initial_step = tabu_config.get("initial_step", 0.15)
    min_step = tabu_config.get("min_step", 0.02)
    n_restarts = tabu_config.get("n_restarts", config["optimizer"]["n_restarts"])
    use_smart_init = tabu_config.get("use_smart_init", True)

    # Gradient learning parameters
    gradient_config = config.get("gradient", {})
    gradient_buffer_size = gradient_config.get("buffer_size", 20)
    gradient_exploitation_ratio = gradient_config.get("exploitation_ratio", 0.5)
    min_gradient_samples = gradient_config.get("min_samples", 4)

    # Adaptive learning parameters
    adaptive_config = config.get("adaptive", {})
    enable_adaptive_step = adaptive_config.get("enable_step", True)
    enable_adaptive_tenure = adaptive_config.get("enable_tenure", True)
    enable_lbfgs_polish = adaptive_config.get("enable_polish", True)
    polish_max_iter = adaptive_config.get("polish_max_iter", 15)
    min_source_separation = adaptive_config.get("min_source_separation", 0.25)

    # Log all parameters
    tracker.log_params({
        "tabu.max_iterations": max_iterations,
        "tabu.base_tabu_tenure": base_tabu_tenure,
        "tabu.tabu_radius": tabu_radius,
        "tabu.n_neighbors": n_neighbors,
        "tabu.initial_step": initial_step,
        "tabu.min_step": min_step,
        "tabu.n_restarts": n_restarts,
        "tabu.use_smart_init": use_smart_init,
        "gradient.buffer_size": gradient_buffer_size,
        "gradient.exploitation_ratio": gradient_exploitation_ratio,
        "gradient.min_samples": min_gradient_samples,
        "adaptive.enable_step": enable_adaptive_step,
        "adaptive.enable_tenure": enable_adaptive_tenure,
        "adaptive.enable_polish": enable_lbfgs_polish,
        "adaptive.polish_max_iter": polish_max_iter,
        "adaptive.min_source_separation": min_source_separation,
    })

    # Create optimizer
    optimizer = AdaptiveLearningOptimizer(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        max_iterations=max_iterations,
        base_tabu_tenure=base_tabu_tenure,
        tabu_radius=tabu_radius,
        n_neighbors=n_neighbors,
        initial_step=initial_step,
        min_step=min_step,
        n_restarts=n_restarts,
        gradient_buffer_size=gradient_buffer_size,
        gradient_exploitation_ratio=gradient_exploitation_ratio,
        min_gradient_samples=min_gradient_samples,
        enable_adaptive_step=enable_adaptive_step,
        enable_adaptive_tenure=enable_adaptive_tenure,
        enable_lbfgs_polish=enable_lbfgs_polish,
        polish_max_iter=polish_max_iter,
        min_source_separation=min_source_separation,
    )

    # Process each sample
    all_predictions = []
    all_rmse = []
    rmse_by_nsources = {}

    # Aggregate learning metrics
    all_gradient_usage = []
    all_polish_improvements = []
    all_tenure_adaptations = []
    total_candidates = 0

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)")

        # Run Adaptive Learning optimizer
        estimates, rmse, candidates, metrics = optimizer.estimate_sources_with_metrics(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            use_smart_init=use_smart_init,
            verbose=False,
        )

        total_candidates += len(candidates)
        all_gradient_usage.append(metrics['gradient_usage'])
        all_polish_improvements.append(metrics['avg_polish_improvement'])
        all_tenure_adaptations.append(metrics['tenure_adaptations'])

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
        result["gradient_usage"] = metrics['gradient_usage']
        result["polish_improvement"] = metrics['avg_polish_improvement']
        result["tenure_adaptations"] = metrics['tenure_adaptations']
        all_predictions.append(result)
        all_rmse.append(rmse)

        # Log per-sample metrics
        tracker.log_metric(f"candidates_{sample_id}", len(candidates))
        tracker.log_metric(f"gradient_usage_{sample_id}", metrics['gradient_usage'])
        tracker.log_metric(f"polish_improvement_{sample_id}", metrics['avg_polish_improvement'])

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Log aggregate metrics
    for n_src, rmse_list in rmse_by_nsources.items():
        tracker.log_metric(f"rmse_{n_src}src_mean", np.mean(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_std", np.std(rmse_list))
        tracker.log_metric(f"rmse_{n_src}src_count", len(rmse_list))

    avg_candidates = total_candidates / len(samples) if samples else 0
    avg_gradient_usage = np.mean(all_gradient_usage) if all_gradient_usage else 0
    avg_polish_improvement = np.mean(all_polish_improvements) if all_polish_improvements else 0
    total_tenure_adaptations = sum(all_tenure_adaptations)

    tracker.log_metric("avg_candidates_per_sample", avg_candidates)
    tracker.log_metric("total_candidates", total_candidates)
    tracker.log_metric("avg_gradient_usage", avg_gradient_usage)
    tracker.log_metric("avg_polish_improvement", avg_polish_improvement)
    tracker.log_metric("total_tenure_adaptations", total_tenure_adaptations)

    # Summary metrics
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
        "avg_candidates": avg_candidates,
        "avg_gradient_usage": avg_gradient_usage,
        "avg_polish_improvement": avg_polish_improvement,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"{'='*60}")
    print(f"Learning Metrics:")
    print(f"  Gradient usage:       {avg_gradient_usage:.1%}")
    print(f"  Avg polish gain:      {avg_polish_improvement:.6f}")
    print(f"  Tenure adaptations:   {total_tenure_adaptations}")
    print(f"  Avg candidates:       {avg_candidates:.1f}")
    print(f"{'='*60}")

    for n_src in sorted(rmse_by_nsources.keys()):
        rmse_list = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: {np.mean(rmse_list):.6f} +/- {np.std(rmse_list):.6f} (n={len(rmse_list)})")

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
    config["n_samples"] = 2
    config["tabu"] = {
        "max_iterations": 10,
        "n_restarts": 2,
        "n_neighbors": 12,
    }
    config["gradient"] = {
        "buffer_size": 15,
        "exploitation_ratio": 0.5,
        "min_samples": 4,
    }
    config["adaptive"] = {
        "enable_step": True,
        "enable_tenure": True,
        "enable_polish": True,
        "polish_max_iter": 10,
    }
    run(config, DummyTracker())
