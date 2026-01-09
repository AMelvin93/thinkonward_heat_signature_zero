#!/usr/bin/env python
"""
Run a model configuration and log to MLflow.

Usage:
    uv run python scripts/run_model.py --model hybrid --max-iter 3 --triangulation
    uv run python scripts/run_model.py --model jax_ultrafast
    uv run python scripts/run_model.py --model hybrid --max-iter 2 --no-triangulation

Only logs to MLflow when running with n_workers=7 (G4dn simulation).
Must be run on WSL for accurate timing.
"""

import sys
import os
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

# G4dn.2xlarge simulation settings
G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400


def calculate_sample_score(rmse, lambda_=0.3, n_max=3, max_rmse=1.0):
    """Calculate competition score for a single sample with 1 candidate."""
    if rmse > max_rmse:
        return 0.0
    accuracy_term = 1.0 / (1.0 + rmse)
    diversity_term = lambda_ * (1 / n_max)
    return accuracy_term + diversity_term


def process_sample_hybrid(sample, meta, q_range, max_iter, use_triangulation):
    """Process sample with HybridOptimizer."""
    from src.hybrid_optimizer import HybridOptimizer

    optimizer = HybridOptimizer(
        n_smart_inits=1,
        n_random_inits=0,
        use_triangulation=use_triangulation,
    )

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, max_iter=max_iter, verbose=False
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'score': calculate_sample_score(rmse),
        'time': elapsed,
    }


def process_sample_jax_ultrafast(sample, meta, q_range):
    """Process sample with JAX Ultrafast Optimizer."""
    from src.jax_ultrafast_optimizer import JAXUltraFastOptimizer

    optimizer = JAXUltraFastOptimizer()

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, verbose=False
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'score': calculate_sample_score(rmse),
        'time': elapsed,
    }


def process_sample_jax_coarse_to_fine(sample, meta, q_range):
    """Process sample with JAX Coarse-to-Fine Optimizer."""
    from src.jax_coarse_to_fine_optimizer import CoarseToFineOptimizer

    optimizer = CoarseToFineOptimizer()

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, verbose=False
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'score': calculate_sample_score(rmse),
        'time': elapsed,
    }


def process_sample_jax_pure(sample, meta, q_range):
    """Process sample with JAX Pure Optimizer."""
    from src.jax_pure_optimizer import PureJAXOptimizer

    optimizer = PureJAXOptimizer()

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, verbose=False
    )
    elapsed = time.time() - start

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'score': calculate_sample_score(rmse),
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Run model and log to MLflow')
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['hybrid', 'jax_ultrafast', 'jax_coarse_to_fine', 'jax_pure'],
                        help='Model to run')
    parser.add_argument('--max-iter', type=int, default=3,
                        help='Max iterations for hybrid optimizer')
    parser.add_argument('--triangulation', action='store_true', default=True,
                        help='Use triangulation init (default: True)')
    parser.add_argument('--no-triangulation', action='store_true',
                        help='Disable triangulation init')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of workers (7 for G4dn simulation)')
    args = parser.parse_args()

    use_triangulation = args.triangulation and not args.no_triangulation
    n_workers = args.workers

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    q_range = tuple(meta['q_range'])
    n_samples = len(samples)

    # Determine if this is a G4dn simulation run
    is_g4dn_simulation = (n_workers == G4DN_WORKERS)

    # Build model name for MLflow
    if args.model == 'hybrid':
        model_name = f"hybrid_iter{args.max_iter}"
        if use_triangulation:
            model_name += "_triang"
    else:
        model_name = args.model

    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Samples: {n_samples}")
    print(f"Workers: {n_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype)"))
    if args.model == 'hybrid':
        print(f"Config: max_iter={args.max_iter}, triangulation={use_triangulation}")
    print(f"MLflow logging: {'ENABLED' if is_g4dn_simulation else 'DISABLED'}")
    print("=" * 70)

    # Select processor function
    if args.model == 'hybrid':
        def process_fn(sample):
            return process_sample_hybrid(sample, meta, q_range, args.max_iter, use_triangulation)
    elif args.model == 'jax_ultrafast':
        def process_fn(sample):
            return process_sample_jax_ultrafast(sample, meta, q_range)
    elif args.model == 'jax_coarse_to_fine':
        def process_fn(sample):
            return process_sample_jax_coarse_to_fine(sample, meta, q_range)
    elif args.model == 'jax_pure':
        def process_fn(sample):
            return process_sample_jax_pure(sample, meta, q_range)

    # Process samples
    start_total = time.time()
    print(f"\nProcessing {n_samples} samples...")

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_fn)(sample) for sample in samples
    )

    total_time = time.time() - start_total

    # Aggregate results
    all_rmses = [r['rmse'] for r in results]
    all_scores = [r['score'] for r in results]

    rmse_by_nsources = {}
    for r in results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['rmse'])

    rmse_mean = np.mean(all_rmses)
    rmse_std = np.std(all_rmses)
    final_score = np.mean(all_scores)
    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    # Log to MLflow only for G4dn simulation runs
    if is_g4dn_simulation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("rmse", rmse_mean)
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("projected_runtime_min", projected_400)
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_nsources.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_nsources.get(2, [0])))

            mlflow.log_param("model", args.model)
            mlflow.log_param("max_iter", args.max_iter if args.model == 'hybrid' else 'N/A')
            mlflow.log_param("triangulation", use_triangulation if args.model == 'hybrid' else 'N/A')
            mlflow.log_param("n_workers", n_workers)

            # Save results
            output_path = project_root / "results" / f"{run_name}_results.pkl"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump({'results': results, 'config': vars(args)}, f)
            try:
                mlflow.log_artifact(str(output_path))
            except (PermissionError, OSError):
                pass

            print(f"\n[MLflow] Logged run: {run_name}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"RMSE:             {rmse_mean:.6f} +/- {rmse_std:.6f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Runtime (G4dn):   {projected_400:.1f} min (for {COMPETITION_SAMPLES} samples)")
    print()
    print("Per-source breakdown:")
    for n_src in sorted(rmse_by_nsources.keys()):
        rmses = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: {np.mean(rmses):.6f} +/- {np.std(rmses):.6f} (n={len(rmses)})")
    print("=" * 70)

    if projected_400 < 50:
        print("[OK] Under 50 min - ideal for submission!")
    elif projected_400 < 55:
        print("[OK] Under 55 min - good for submission")
    elif projected_400 < 60:
        print("[WARNING] 55-60 min - acceptable but tight")
    else:
        print(f"[FAIL] Over 60 min by {projected_400 - 60:.1f} min")


if __name__ == "__main__":
    main()
