#!/usr/bin/env python
"""
Run final competition submission with triangulation.

Uses the configuration from configs/final_submission.yaml.
Only logs to MLflow when running with n_workers=7 (simulating G4dn.2xlarge).
"""

import sys
import os
import time
import pickle
import yaml
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

from src.hybrid_optimizer import HybridOptimizer

# G4dn.2xlarge simulation settings
G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400


def load_config():
    """Load configuration from YAML file."""
    config_path = project_root / "configs" / "final_submission.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_sample_score(rmse, lambda_=0.3, n_max=3, max_rmse=1.0):
    """Calculate competition score for a single sample with 1 candidate."""
    if rmse > max_rmse:
        return 0.0
    accuracy_term = 1.0 / (1.0 + rmse)
    diversity_term = lambda_ * (1 / n_max)  # 1 candidate
    return accuracy_term + diversity_term


def process_sample(sample, meta, config):
    """Process a single sample."""
    opt_config = config['optimizer']
    grid_config = config['grid']
    q_range = tuple(config['bounds']['q_range'])

    optimizer = HybridOptimizer(
        Lx=grid_config['Lx'],
        Ly=grid_config['Ly'],
        nx=grid_config['nx'],
        ny=grid_config['ny'],
        n_smart_inits=opt_config['n_smart_inits'],
        n_random_inits=opt_config['n_random_inits'],
        min_candidate_distance=opt_config['min_candidate_distance'],
        n_max_candidates=opt_config['n_max_candidates'],
        use_triangulation=opt_config.get('use_triangulation', True),
    )

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta,
        q_range=q_range,
        max_iter=opt_config['max_iter'],
        parallel=opt_config['parallel'],
        verbose=False,
    )
    elapsed = time.time() - start

    score = calculate_sample_score(rmse)

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'n_candidates': len(candidates),
        'score': score,
        'time': elapsed,
    }


def main():
    # Load config and data
    config = load_config()

    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    n_samples = len(samples)
    n_workers = config['parallelization']['n_workers']
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers

    # Determine if this is a G4dn simulation run (should log to MLflow)
    is_g4dn_simulation = (n_workers == G4DN_WORKERS)

    print("=" * 70)
    print("FINAL SUBMISSION WITH TRIANGULATION")
    print("=" * 70)
    print(f"Samples: {n_samples}")
    print(f"Workers: {actual_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype mode)"))
    print(f"Config: max_iter={config['optimizer']['max_iter']}, "
          f"use_triangulation={config['optimizer'].get('use_triangulation', True)}")
    if is_g4dn_simulation:
        print("MLflow logging: ENABLED (G4dn simulation)")
    else:
        print("MLflow logging: DISABLED (use n_workers=7 for submission logging)")
    print("=" * 70)

    # Process samples in parallel
    start_total = time.time()

    print(f"\nProcessing {n_samples} samples with {actual_workers} workers...")
    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_sample)(sample, meta, config)
        for sample in samples
    )

    total_time = time.time() - start_total

    # Aggregate results
    all_rmses = [r['rmse'] for r in results]
    all_scores = [r['score'] for r in results]
    sample_times = [r['time'] for r in results]

    rmse_by_nsources = {}
    for r in results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['rmse'])

    # Calculate metrics
    rmse_mean = np.mean(all_rmses)
    rmse_std = np.std(all_rmses)
    avg_time = np.mean(sample_times)
    final_score = np.mean(all_scores)

    # Calculate projected time for 400 samples on G4dn
    if is_g4dn_simulation:
        # Direct projection - we're already simulating G4dn
        projected_400_g4dn = (total_time / n_samples) * COMPETITION_SAMPLES / 60
    else:
        # Scale projection based on worker ratio
        scaling_factor = actual_workers / G4DN_WORKERS
        projected_400_g4dn = (total_time / n_samples) * COMPETITION_SAMPLES / 60 * scaling_factor

    # Only log to MLflow for G4dn simulation runs
    if is_g4dn_simulation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"submission_g4dn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log key submission metrics
            mlflow.log_metric("rmse", rmse_mean)
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("projected_runtime_min", projected_400_g4dn)

            # Log additional metrics
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_nsources.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_nsources.get(2, [0])))
            mlflow.log_metric("total_time_sec", total_time)

            # Log config parameters
            mlflow.log_param("optimizer", "HybridOptimizer")
            mlflow.log_param("use_triangulation", config['optimizer'].get('use_triangulation', True))
            mlflow.log_param("max_iter", config['optimizer']['max_iter'])
            mlflow.log_param("n_workers", G4DN_WORKERS)
            mlflow.log_param("n_samples_tested", n_samples)

            # Save predictions
            output_path = project_root / "results" / f"{run_name}_results.pkl"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'config': config,
                    'final_score': final_score,
                    'total_time': total_time,
                    'projected_400_g4dn': projected_400_g4dn,
                }, f)
            # Log artifact (skip if path issues on WSL)
            try:
                mlflow.log_artifact(str(output_path))
            except (PermissionError, OSError) as e:
                print(f"[MLflow] Skipping artifact logging (path issue): {e}")

            print(f"\n[MLflow] Logged run: {run_name}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"RMSE:             {rmse_mean:.6f} +/- {rmse_std:.6f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Runtime (G4dn):   {projected_400_g4dn:.1f} min (for {COMPETITION_SAMPLES} samples)")
    print()
    print("Per-source breakdown:")
    for n_src in sorted(rmse_by_nsources.keys()):
        rmses = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: {np.mean(rmses):.6f} +/- {np.std(rmses):.6f} (n={len(rmses)})")
    print("=" * 70)

    if projected_400_g4dn < 60:
        print(f"[OK] Under 60 min target on G4dn!")
    else:
        print(f"[WARNING] Over 60 min target by {projected_400_g4dn - 60:.1f} min")

    return final_score, rmse_mean


if __name__ == "__main__":
    main()
