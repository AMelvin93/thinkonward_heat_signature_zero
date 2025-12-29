#!/usr/bin/env python
"""
Run CMA-ES optimizer experiment and log to MLflow.

CMA-ES is particularly effective for multi-modal problems like 2-source
heat identification where L-BFGS-B can get stuck in local minima.

Usage:
    # Prototype mode (all CPUs, no MLflow logging)
    uv run python experiments/cmaes/run.py --workers -1

    # G4dn simulation mode (7 workers, MLflow logging)
    uv run python experiments/cmaes/run.py --workers 7

Must be run on WSL for accurate timing (per CLAUDE.md).
"""

import sys
import os
import time
import pickle
import argparse
import platform
from pathlib import Path
from datetime import datetime

# experiments/cmaes/run.py -> project root is two levels up
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

# Import from local experiment folder
from experiments.cmaes.optimizer import CMAESOptimizer

# G4dn.2xlarge simulation settings
G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400


def detect_platform():
    """Detect if running on WSL, Linux, or Windows."""
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    return "wsl"
        except FileNotFoundError:
            pass
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return system


def calculate_sample_score(rmse, lambda_=0.3, n_max=3, max_rmse=1.0):
    """Calculate competition score for a single sample with 1 candidate."""
    if rmse > max_rmse:
        return 0.0
    accuracy_term = 1.0 / (1.0 + rmse)
    diversity_term = lambda_ * (1 / n_max)
    return accuracy_term + diversity_term


def process_sample(sample, meta, config):
    """Process a single sample with CMA-ES."""
    optimizer = CMAESOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        use_triangulation=config['use_triangulation'],
        use_lbfgsb_polish=config['use_polish'],
        polish_max_iter=config['polish_max_iter'],
    )

    q_range = tuple(meta['q_range'])

    start = time.time()
    sources, rmse, candidates = optimizer.estimate_sources(
        sample, meta, q_range=q_range, verbose=False
    )
    elapsed = time.time() - start

    n_evals = candidates[0].n_evals if candidates else 0

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'estimates': [{'x': x, 'y': y, 'q': q} for x, y, q in sources],
        'rmse': rmse,
        'score': calculate_sample_score(rmse),
        'time': elapsed,
        'n_evals': n_evals,
    }


def main():
    parser = argparse.ArgumentParser(description='Run CMA-ES optimizer')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of workers (7 for G4dn, -1 for all CPUs)')
    parser.add_argument('--max-fevals-1src', type=int, default=15,
                        help='Max evaluations for 1-source')
    parser.add_argument('--max-fevals-2src', type=int, default=25,
                        help='Max evaluations for 2-source')
    parser.add_argument('--sigma0-1src', type=float, default=0.10,
                        help='Initial step size for 1-source')
    parser.add_argument('--sigma0-2src', type=float, default=0.20,
                        help='Initial step size for 2-source')
    parser.add_argument('--polish', action='store_true', default=True,
                        help='Add L-BFGS-B polish step (default: True)')
    parser.add_argument('--no-polish', action='store_true',
                        help='Disable L-BFGS-B polish step')
    parser.add_argument('--polish-iter', type=int, default=5,
                        help='Max iterations for polish')
    parser.add_argument('--no-triangulation', action='store_true',
                        help='Disable triangulation init')
    args = parser.parse_args()

    # Build config
    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'use_triangulation': not args.no_triangulation,
        'use_polish': args.polish and not args.no_polish,
        'polish_max_iter': args.polish_iter,
    }

    n_workers = args.workers
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers
    is_g4dn_simulation = (n_workers == G4DN_WORKERS)

    # Detect platform
    current_platform = detect_platform()
    is_valid_platform = current_platform in ("wsl", "linux")

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    # Print header
    print("=" * 70)
    print("CMA-ES OPTIMIZER")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples}")
    print(f"Workers: {actual_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype)"))
    print(f"Config:")
    print(f"  max_fevals: 1-src={config['max_fevals_1src']}, 2-src={config['max_fevals_2src']}")
    print(f"  sigma0: 1-src={config['sigma0_1src']}, 2-src={config['sigma0_2src']}")
    print(f"  triangulation: {config['use_triangulation']}")
    print(f"  polish: {config['use_polish']} (iter={config['polish_max_iter']})")
    print(f"MLflow logging: {'ENABLED' if is_g4dn_simulation else 'DISABLED'}")

    if not is_valid_platform:
        print("")
        print("[WARNING] Running on Windows - timing will be ~35% slower than Linux!")
        print("[WARNING] For accurate submission timing, run on WSL:")
        print("          cd /mnt/c/Users/amelv/Repo/thinkonward_heat_signature_zero")
        print("          uv run python scripts/run_cmaes.py --workers 7")
    print("=" * 70)

    # Process samples
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
    all_evals = [r['n_evals'] for r in results]

    rmse_by_nsources = {}
    evals_by_nsources = {}
    for r in results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
            evals_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['rmse'])
        evals_by_nsources[n_src].append(r['n_evals'])

    rmse_mean = np.mean(all_rmses)
    rmse_std = np.std(all_rmses)
    final_score = np.mean(all_scores)
    avg_evals = np.mean(all_evals)

    # Project time for 400 samples
    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    # Log to MLflow for G4dn simulation runs
    if is_g4dn_simulation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"cmaes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Key metrics
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", rmse_mean)
            mlflow.log_metric("projected_400_samples_min", projected_400)

            # Additional metrics
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_nsources.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_nsources.get(2, [0])))
            mlflow.log_metric("avg_evals", avg_evals)
            mlflow.log_metric("total_time_sec", total_time)

            # Parameters
            mlflow.log_param("optimizer", "CMAESOptimizer")
            mlflow.log_param("max_fevals_1src", config['max_fevals_1src'])
            mlflow.log_param("max_fevals_2src", config['max_fevals_2src'])
            mlflow.log_param("sigma0_1src", config['sigma0_1src'])
            mlflow.log_param("sigma0_2src", config['sigma0_2src'])
            mlflow.log_param("use_triangulation", config['use_triangulation'])
            mlflow.log_param("use_polish", config['use_polish'])
            mlflow.log_param("n_workers", G4DN_WORKERS)
            mlflow.log_param("platform", current_platform)

            # Save results
            output_path = project_root / "results" / f"{run_name}_results.pkl"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'config': config,
                    'final_score': final_score,
                    'total_time': total_time,
                    'projected_400': projected_400,
                }, f)
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
    print(f"Avg Evaluations:  {avg_evals:.1f}")
    print(f"Total Time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    print("Per-source breakdown:")
    for n_src in sorted(rmse_by_nsources.keys()):
        rmses = rmse_by_nsources[n_src]
        evals = evals_by_nsources[n_src]
        print(f"  {n_src}-source: RMSE={np.mean(rmses):.6f} +/- {np.std(rmses):.6f}, "
              f"evals={np.mean(evals):.1f} (n={len(rmses)})")
    print("=" * 70)

    # Status message
    if projected_400 < 50:
        print("[OK] Under 50 min - ideal for submission!")
    elif projected_400 < 55:
        print("[OK] Under 55 min - good for submission")
    elif projected_400 < 60:
        print("[WARNING] 55-60 min - acceptable but tight")
    else:
        print(f"[FAIL] Over 60 min by {projected_400 - 60:.1f} min")

    return final_score, rmse_mean


if __name__ == "__main__":
    main()
