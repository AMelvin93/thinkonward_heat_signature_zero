#!/usr/bin/env python
"""
Run final competition submission using RobustFallbackOptimizer.

This uses the same optimizer that achieved the 1.1247 @ 57.2 min baseline.
Key config: threshold_1src=0.35, threshold_2src=0.45, fevals 20/36

Target: 400 samples in <60 min on G4dn.2xlarge (8 vCPUs, 7 workers)
"""

import sys
import os
import time
import pickle
import argparse
import platform
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'robust_fallback'))

import numpy as np

from optimizer import RobustFallbackOptimizer

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


def process_single_sample(args):
    """Process a single sample - must be module-level for pickling."""
    idx, sample, meta, config = args

    optimizer = RobustFallbackOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        early_fraction=config['early_fraction'],
        candidate_pool_size=config['candidate_pool_size'],
        nx_coarse=config['nx_coarse'],
        ny_coarse=config['ny_coarse'],
        refine_maxiter=config['refine_maxiter'],
        refine_top_n=config['refine_top_n'],
        rmse_threshold_1src=config['threshold_1src'],
        rmse_threshold_2src=config['threshold_2src'],
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        return {
            'idx': idx,
            'sample_id': sample.get('sample_id', idx),
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'sample_id': sample.get('sample_id', idx),
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'success': False,
            'error': str(e),
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    """
    Calculate competition score for a single sample.

    Score = (1/(1+RMSE)) + lambda * (N_candidates/N_max)
    """
    if n_candidates == 0:
        return 0.0
    accuracy_term = 1.0 / (1.0 + rmse)
    diversity_term = lambda_ * (n_candidates / n_max)
    return accuracy_term + diversity_term


def main():
    parser = argparse.ArgumentParser(description='Run Final Submission')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of parallel workers (7 for G4dn simulation)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle sample order')
    parser.add_argument('--seed', type=int, default=42)

    # Optimizer config - BASELINE that achieved 1.1247 @ 57.2 min
    parser.add_argument('--threshold-1src', type=float, default=0.35,
                        help='Fallback threshold for 1-source (baseline: 0.35)')
    parser.add_argument('--threshold-2src', type=float, default=0.45,
                        help='Fallback threshold for 2-source (baseline: 0.45)')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top', type=int, default=2)

    args = parser.parse_args()

    # Detect platform
    current_platform = detect_platform()
    is_valid_platform = current_platform in ("wsl", "linux")
    is_g4dn_simulation = (args.workers == G4DN_WORKERS)

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    np.random.seed(args.seed)
    if args.shuffle:
        indices = np.random.permutation(len(samples))
    else:
        indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    # Build config
    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'early_fraction': args.early_fraction,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top,
        'threshold_1src': args.threshold_1src,
        'threshold_2src': args.threshold_2src,
    }

    print("=" * 70)
    print("FINAL SUBMISSION - RobustFallbackOptimizer")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Fallback threshold: 1-src={args.threshold_1src}, 2-src={args.threshold_2src}")
    print(f"Refine: {args.refine_maxiter} iters on top-{args.refine_top}")
    if is_g4dn_simulation:
        print("Mode: G4dn.2xlarge simulation (MLflow logging enabled)")
    else:
        print(f"Mode: Prototype ({args.workers} workers)")
    if not is_valid_platform:
        print("")
        print("[WARNING] Running on Windows - timing will be ~35% slower than Linux!")
        print("[WARNING] For accurate submission timing, run on WSL")
    print("=" * 70)

    # Process samples
    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            flag = ""
            if result['n_sources'] == 1 and result['best_rmse'] > args.threshold_1src:
                flag = " [FB]"
            elif result['n_sources'] == 2 and result['best_rmse'] > args.threshold_2src:
                flag = " [FB]"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{flag}")

    total_time = time.time() - start_time

    # Calculate scores
    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    final_score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    # Count candidates
    avg_candidates = np.mean([r['n_candidates'] for r in results])

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f} +/- {rmse_std:.6f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Avg Candidates:   {avg_candidates:.2f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Baseline: 1.1247 @ 57.2 min")
    print(f"This run: {final_score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:    {final_score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print()

    if projected_400 <= 60:
        if final_score > 1.1247:
            print("SUCCESS! In budget with improved score!")
        else:
            print("IN BUDGET")
    else:
        print(f"OVER BUDGET by {projected_400 - 60:.1f} min")
    print(f"{'='*70}")

    # Count fallbacks
    n_fallbacks_1src = sum(1 for r in results if r['success'] and r['n_sources'] == 1 and r['best_rmse'] > args.threshold_1src)
    n_fallbacks_2src = sum(1 for r in results if r['success'] and r['n_sources'] == 2 and r['best_rmse'] > args.threshold_2src)
    print(f"Fallbacks triggered: 1-src={n_fallbacks_1src}, 2-src={n_fallbacks_2src}")

    # High RMSE outliers
    high_rmse = [r for r in results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:5]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f}")

    # Log to MLflow if G4dn simulation
    if is_g4dn_simulation and is_valid_platform:
        try:
            import mlflow
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("heat-signature-zero")

            run_name = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric("submission_score", final_score)
                mlflow.log_metric("rmse", rmse_mean)
                mlflow.log_metric("projected_runtime_min", projected_400)
                mlflow.log_metric("avg_candidates", avg_candidates)
                mlflow.log_param("optimizer", "RobustFallbackOptimizer")
                mlflow.log_param("threshold_1src", args.threshold_1src)
                mlflow.log_param("threshold_2src", args.threshold_2src)
                mlflow.log_param("fevals_1src", args.max_fevals_1src)
                mlflow.log_param("fevals_2src", args.max_fevals_2src)
                mlflow.log_param("platform", current_platform)
                mlflow.log_param("n_workers", args.workers)

            print(f"\n[MLflow] Logged run: {run_name}")
        except Exception as e:
            print(f"\n[MLflow] Failed to log: {e}")

    return final_score, projected_400


if __name__ == "__main__":
    main()
