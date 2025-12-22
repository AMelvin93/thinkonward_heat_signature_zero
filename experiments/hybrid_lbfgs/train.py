#!/usr/bin/env python
"""
Hybrid Multi-Start L-BFGS-B experiment.

Combines smart initialization + L-BFGS-B accuracy + diverse candidates.

Run with:
    python scripts/run_experiment.py --experiment hybrid_lbfgs
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from joblib import Parallel, delayed
from src.hybrid_optimizer import HybridOptimizer


def process_single_sample(sample, meta, optimizer_params, q_range, max_iter):
    """Process a single sample - for parallel execution."""
    optimizer = HybridOptimizer(**optimizer_params)
    estimates, rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=q_range,
        max_iter=max_iter,
        parallel=False,  # Don't nest parallelism
        verbose=False,
    )
    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': estimates,
        'rmse': rmse,
        'n_candidates': len(candidates),
    }


def run(config: dict, tracker) -> dict:
    """
    Run the Hybrid L-BFGS-B experiment.

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

    print(f"Processing {len(samples)} samples with Hybrid L-BFGS-B optimizer...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Hybrid-specific parameters
    hybrid_config = config.get("hybrid", {})
    n_smart_inits = hybrid_config.get("n_smart_inits", 4)
    n_random_inits = hybrid_config.get("n_random_inits", 8)
    min_candidate_distance = hybrid_config.get("min_candidate_distance", 0.15)
    n_max_candidates = hybrid_config.get("n_max_candidates", 3)
    max_iter = hybrid_config.get("max_iter", 100)
    parallel = hybrid_config.get("parallel", True)

    # Log parameters
    tracker.log_params({
        "hybrid.n_smart_inits": n_smart_inits,
        "hybrid.n_random_inits": n_random_inits,
        "hybrid.min_candidate_distance": min_candidate_distance,
        "hybrid.n_max_candidates": n_max_candidates,
        "hybrid.max_iter": max_iter,
        "hybrid.parallel": parallel,
        "hybrid.total_inits": n_smart_inits + n_random_inits,
    })

    # Create optimizer
    optimizer = HybridOptimizer(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        n_smart_inits=n_smart_inits,
        n_random_inits=n_random_inits,
        min_candidate_distance=min_candidate_distance,
        n_max_candidates=n_max_candidates,
    )

    # Optimizer parameters for parallel execution
    optimizer_params = {
        'Lx': Lx, 'Ly': Ly, 'nx': nx, 'ny': ny,
        'n_smart_inits': n_smart_inits,
        'n_random_inits': n_random_inits,
        'min_candidate_distance': min_candidate_distance,
        'n_max_candidates': n_max_candidates,
    }
    q_range = tuple(config["optimizer"]["q_range"])

    # Check if sample-level parallelization is enabled
    parallel_samples = hybrid_config.get("parallel_samples", True)
    n_parallel_samples = hybrid_config.get("n_parallel_samples", 8)

    tracker.log_params({
        "hybrid.parallel_samples": parallel_samples,
        "hybrid.n_parallel_samples": n_parallel_samples,
    })

    all_predictions = []
    all_rmse = []
    all_n_candidates = []
    rmse_by_nsources = {}

    if parallel_samples:
        print(f"  Running {n_parallel_samples} samples in parallel...")

        # Process samples in parallel batches
        results = Parallel(n_jobs=n_parallel_samples, verbose=10)(
            delayed(process_single_sample)(
                sample, meta, optimizer_params, q_range, max_iter
            )
            for sample in samples
        )

        # Process results
        for r in results:
            sample_id = r['sample_id']
            n_sources = r['n_sources']
            estimates = r['estimates']
            rmse = r['rmse']
            n_candidates = r['n_candidates']

            all_rmse.append(rmse)
            all_n_candidates.append(n_candidates)

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

            print(f"  {sample_id}: RMSE={rmse:.4f}, candidates={n_candidates}")
    else:
        # Sequential processing
        for i, sample in enumerate(samples):
            sample_id = sample['sample_id']
            n_sources = sample['n_sources']

            print(f"  [{i+1}/{len(samples)}] {sample_id} ({n_sources} sources)", end="", flush=True)

            estimates, rmse, candidates = optimizer.estimate_sources(
                sample, meta,
                q_range=q_range,
                max_iter=max_iter,
                parallel=parallel,
                verbose=False,
            )

            n_candidates = len(candidates)
            all_n_candidates.append(n_candidates)

            print(f" -> RMSE={rmse:.4f}, candidates={n_candidates}")

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

    # Aggregate metrics
    avg_candidates = np.mean(all_n_candidates)
    tracker.log_metric("avg_candidates_per_sample", avg_candidates)
    tracker.log_metric("total_candidates", sum(all_n_candidates))

    # Summary metrics
    results = {
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
        "avg_candidates": avg_candidates,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: RMSE = {results['rmse_mean']:.6f} +/- {results['rmse_std']:.6f}")
    print(f"Average candidates per sample: {avg_candidates:.1f}")
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
        def log_source_estimates(self, sample_id, estimates, rmse, n_candidates=1, **kwargs):
            return {
                "sample_id": sample_id,
                "estimates": [{"x": e[0], "y": e[1], "q": e[2]} for e in estimates],
                "rmse": rmse,
                "n_candidates": n_candidates,
            }

    config = load_config(str(project_root / "configs" / "default.yaml"))
    config["n_samples"] = 3  # Quick test
    config["hybrid"] = {
        "n_smart_inits": 2,
        "n_random_inits": 4,
        "n_max_candidates": 3,
        "max_iter": 50,
        "parallel": True,
    }
    run(config, DummyTracker())
