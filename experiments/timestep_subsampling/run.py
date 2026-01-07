#!/usr/bin/env python
"""
Run Timestep Subsampling Optimizer experiment.

Key Innovation: Since Heat2D time is dominated by timesteps (nt), not grid size,
we use larger dt and fewer timesteps during CMA-ES exploration for 2-4x speedup.

Strategy:
    1. Init selection at SUBSAMPLED timesteps (fast)
    2. CMA-ES exploration at SUBSAMPLED timesteps (2-4x faster per sim)
    3. Polish top solutions at FULL resolution (accurate)

Combined with Smart Init Selection for maximum efficiency.

Expected Benefits:
    - 2-4x speedup per simulation during exploration
    - Enable 25-30+ 2-source fevals within budget
    - ADI method is unconditionally stable - larger dt is safe
    - Less accuracy loss than grid coarsening

Usage:
    # Test with default settings (2x subsampling)
    uv run python experiments/timestep_subsampling/run.py --workers 7 --shuffle

    # Test with higher subsampling (4x faster but less accurate)
    uv run python experiments/timestep_subsampling/run.py --subsample-factor 4 --workers 7 --shuffle
"""

import sys
import os
import time
import pickle
import argparse
import platform
from pathlib import Path
from datetime import datetime
from copy import deepcopy

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

from experiments.timestep_subsampling.optimizer import (
    TimestepSubsamplingOptimizer,
    extract_enhanced_features,
    N_MAX,
)

# G4dn.2xlarge simulation settings
G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400
LAMBDA = 0.3


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


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX, max_rmse=1.0):
    """Calculate competition score for a sample with multiple candidates."""
    valid_rmses = [r for r in rmses if r <= max_rmse]
    n_valid = len(valid_rmses)

    if n_valid == 0:
        return 0.0

    accuracy_sum = sum(1.0 / (1.0 + r) for r in valid_rmses)
    accuracy_term = accuracy_sum / n_valid
    diversity_term = lambda_ * (n_valid / n_max)

    return accuracy_term + diversity_term


def process_sample(sample, meta, config, history_1src, history_2src):
    """Process a single sample with timestep subsampling optimizer."""
    optimizer = TimestepSubsamplingOptimizer(
        subsample_factor=config['subsample_factor'],
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        polish_fevals_1src=config['polish_fevals_1src'],
        polish_fevals_2src=config['polish_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        use_triangulation=config['use_triangulation'],
        n_candidates=config['n_candidates'],
        candidate_pool_size=config['candidate_pool_size'],
        k_similar=config['k_similar'],
        use_enhanced_features=config['use_enhanced_features'],
    )

    q_range = tuple(meta['q_range'])

    start = time.time()
    candidates, best_rmse, results, features, best_positions, n_transferred = optimizer.estimate_sources(
        sample, meta, q_range=q_range,
        history_1src=history_1src,
        history_2src=history_2src,
        verbose=False
    )
    elapsed = time.time() - start

    candidate_rmses = [r.rmse for r in results]
    score = calculate_sample_score(candidate_rmses)
    n_evals = sum(r.n_evals for r in results)

    best_init_type = results[0].init_type if results else 'unknown'

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'n_candidates': len(candidates),
        'estimates': [
            [{'x': x, 'y': y, 'q': q} for x, y, q in cand]
            for cand in candidates
        ],
        'rmses': candidate_rmses,
        'best_rmse': best_rmse,
        'score': score,
        'time': elapsed,
        'n_evals': n_evals,
        'n_transferred': n_transferred,
        'best_init_type': best_init_type,
        'features': features,
        'best_positions': best_positions,
    }


def process_batch(samples, meta, config, history_1src, history_2src, n_workers):
    """Process a batch of samples in parallel with shared history."""
    h1_copy = deepcopy(history_1src)
    h2_copy = deepcopy(history_2src)

    results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(process_sample)(sample, meta, config, h1_copy, h2_copy)
        for sample in samples
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Timestep Subsampling Optimizer')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of workers (7 for G4dn, -1 for all CPUs)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Samples per batch for transfer learning')
    # Timestep subsampling specific
    parser.add_argument('--subsample-factor', type=int, default=2,
                        help='Timestep subsampling factor (2=2x faster, 4=4x faster)')
    # Default fevals: 12 for 1-src, 28 for 2-src (higher due to subsample speedup)
    parser.add_argument('--max-fevals-1src', type=int, default=12,
                        help='Max CMA-ES evaluations for 1-source (subsampled phase)')
    parser.add_argument('--max-fevals-2src', type=int, default=28,
                        help='Max CMA-ES evaluations for 2-source (subsampled phase)')
    parser.add_argument('--polish-fevals-1src', type=int, default=3,
                        help='Polish iterations for 1-source (full resolution)')
    parser.add_argument('--polish-fevals-2src', type=int, default=5,
                        help='Polish iterations for 2-source (full resolution)')
    parser.add_argument('--sigma0-1src', type=float, default=0.15,
                        help='Initial step size for 1-source')
    parser.add_argument('--sigma0-2src', type=float, default=0.20,
                        help='Initial step size for 2-source')
    parser.add_argument('--no-triangulation', action='store_true',
                        help='Disable triangulation init')
    parser.add_argument('--n-candidates', type=int, default=3,
                        help='Target number of candidates')
    parser.add_argument('--pool-size', type=int, default=10,
                        help='Number of top solutions to polish')
    parser.add_argument('--k-similar', type=int, default=1,
                        help='Number of similar samples to transfer from')
    parser.add_argument('--no-enhanced-features', action='store_true',
                        help='Use basic features instead of enhanced')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle samples before batching')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffle (default: 42)')
    args = parser.parse_args()

    # Build config
    config = {
        'subsample_factor': args.subsample_factor,
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'polish_fevals_1src': args.polish_fevals_1src,
        'polish_fevals_2src': args.polish_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'use_triangulation': not args.no_triangulation,
        'n_candidates': min(args.n_candidates, N_MAX),
        'candidate_pool_size': args.pool_size,
        'k_similar': args.k_similar,
        'use_enhanced_features': not args.no_enhanced_features,
    }

    n_workers = args.workers
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers
    is_g4dn_simulation = (n_workers == G4DN_WORKERS)
    batch_size = args.batch_size

    # Detect platform
    current_platform = detect_platform()
    is_valid_platform = current_platform in ("wsl", "linux")

    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']
    n_samples = len(samples)

    # Count sample types
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    # Shuffle samples if requested
    shuffle_enabled = args.shuffle
    shuffle_seed = args.seed
    if shuffle_enabled:
        np.random.seed(shuffle_seed)
        np.random.shuffle(samples)

    n_batches = (n_samples + batch_size - 1) // batch_size

    # Print header
    print("=" * 70)
    print("TIMESTEP SUBSAMPLING OPTIMIZER")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {actual_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype)"))
    print(f"Batch size: {batch_size} ({n_batches} batches)")
    print()
    print(f"KEY INNOVATION: Timestep Subsampling ({config['subsample_factor']}x)")
    print(f"  - CMA-ES exploration with dt*{config['subsample_factor']}, nt/{config['subsample_factor']} - {config['subsample_factor']}x faster per sim")
    print(f"  - Final polish at full resolution - maintains accuracy")
    print(f"  - ADI method is unconditionally stable - larger dt is safe")
    print()
    print(f"Config:")
    print(f"  subsample_factor: {config['subsample_factor']}")
    print(f"  max_fevals: 1-src={config['max_fevals_1src']}, 2-src={config['max_fevals_2src']} (subsampled)")
    print(f"  polish_fevals: 1-src={config['polish_fevals_1src']}, 2-src={config['polish_fevals_2src']} (full)")
    print(f"  triangulation: {config['use_triangulation']}")
    print(f"  n_candidates: {config['n_candidates']}")
    print(f"  pool_size: {config['candidate_pool_size']}")
    print(f"  k_similar: {config['k_similar']}")
    print(f"  enhanced_features: {config['use_enhanced_features']}")
    print(f"  shuffle: {shuffle_enabled}" + (f" (seed={shuffle_seed})" if shuffle_enabled else ""))
    print(f"MLflow logging: {'ENABLED' if is_g4dn_simulation else 'DISABLED'}")

    if not is_valid_platform:
        print("")
        print("[WARNING] Running on Windows - timing will be ~35% slower than Linux!")
    print("=" * 70)

    # === BATCH PROCESSING ===
    start_total = time.time()

    history_1src = []
    history_2src = []
    all_results = []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch_samples = samples[batch_start:batch_end]

        print(f"\nBatch {batch_idx + 1}/{n_batches}: samples {batch_start}-{batch_end-1} "
              f"(history: {len(history_1src)} 1-src, {len(history_2src)} 2-src)")

        batch_results = process_batch(
            batch_samples, meta, config,
            history_1src, history_2src,
            n_workers
        )

        # Update history
        for result in batch_results:
            if result['n_sources'] == 1:
                history_1src.append((result['features'], result['best_positions']))
            else:
                history_2src.append((result['features'], result['best_positions']))

        all_results.extend(batch_results)

        # Print batch summary
        batch_rmses = [r['best_rmse'] for r in batch_results]
        batch_transfers = [r['n_transferred'] for r in batch_results]
        batch_times = [r['time'] for r in batch_results]
        print(f"  -> Batch RMSE: {np.mean(batch_rmses):.4f}, "
              f"Avg transfers: {np.mean(batch_transfers):.1f}, "
              f"Avg time: {np.mean(batch_times):.2f}s")

    total_time = time.time() - start_total

    # === AGGREGATE RESULTS ===
    all_best_rmses = [r['best_rmse'] for r in all_results]
    all_scores = [r['score'] for r in all_results]
    all_n_candidates = [r['n_candidates'] for r in all_results]
    all_evals = [r['n_evals'] for r in all_results]
    all_transfers = [r['n_transferred'] for r in all_results]
    all_times = [r['time'] for r in all_results]

    # Track init types
    init_type_counts = {}
    for r in all_results:
        init_type = r['best_init_type']
        init_type_counts[init_type] = init_type_counts.get(init_type, 0) + 1

    rmse_by_nsources = {}
    evals_by_nsources = {}
    times_by_nsources = {}
    for r in all_results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
            evals_by_nsources[n_src] = []
            times_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['best_rmse'])
        evals_by_nsources[n_src].append(r['n_evals'])
        times_by_nsources[n_src].append(r['time'])

    rmse_mean = np.mean(all_best_rmses)
    rmse_std = np.std(all_best_rmses)
    final_score = np.mean(all_scores)
    avg_candidates = np.mean(all_n_candidates)
    avg_evals = np.mean(all_evals)
    avg_transfers = np.mean(all_transfers)
    avg_time_per_sample = np.mean(all_times)

    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    # Per-source RMSEs
    rmse_1src = np.mean(rmse_by_nsources.get(1, [0]))
    rmse_2src = np.mean(rmse_by_nsources.get(2, [0]))
    time_1src = np.mean(times_by_nsources.get(1, [0]))
    time_2src = np.mean(times_by_nsources.get(2, [0]))

    # Count transfer benefit
    transfer_benefit_count = sum(1 for r in all_results if 'transfer' in r['best_init_type'])
    transfer_benefit_pct = transfer_benefit_count / len(all_results) * 100

    # === MLFLOW LOGGING ===
    if is_g4dn_simulation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"timestep_subsample_{config['subsample_factor']}x_{config['max_fevals_1src']}_{config['max_fevals_2src']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", rmse_mean)
            mlflow.log_metric("projected_400_samples_min", projected_400)
            mlflow.log_metric("rmse_1src", rmse_1src)
            mlflow.log_metric("rmse_2src", rmse_2src)
            mlflow.log_metric("time_1src_avg", time_1src)
            mlflow.log_metric("time_2src_avg", time_2src)
            mlflow.log_metric("avg_n_candidates", avg_candidates)
            mlflow.log_metric("avg_evals", avg_evals)
            mlflow.log_metric("transfer_benefit_pct", transfer_benefit_pct)
            mlflow.log_metric("total_time_sec", total_time)

            mlflow.log_param("optimizer", "TimestepSubsamplingOptimizer")
            mlflow.log_param("subsample_factor", config['subsample_factor'])
            mlflow.log_param("max_fevals_1src", config['max_fevals_1src'])
            mlflow.log_param("max_fevals_2src", config['max_fevals_2src'])
            mlflow.log_param("polish_fevals_1src", config['polish_fevals_1src'])
            mlflow.log_param("polish_fevals_2src", config['polish_fevals_2src'])
            mlflow.log_param("use_triangulation", config['use_triangulation'])
            mlflow.log_param("n_candidates", config['n_candidates'])
            mlflow.log_param("pool_size", config['candidate_pool_size'])
            mlflow.log_param("k_similar", config['k_similar'])
            mlflow.log_param("n_workers", G4DN_WORKERS)
            mlflow.log_param("platform", current_platform)
            mlflow.log_param("shuffle", shuffle_enabled)

            # Save results
            output_path = project_root / "results" / f"{run_name}_results.pkl"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'results': all_results,
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

    # === PRINT SUMMARY ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Config:           {config['subsample_factor']}x subsampling, {config['max_fevals_1src']}/{config['max_fevals_2src']} + {config['polish_fevals_1src']}/{config['polish_fevals_2src']} polish")
    print(f"Best RMSE:        {rmse_mean:.6f} +/- {rmse_std:.6f}")
    print(f"Avg Candidates:   {avg_candidates:.1f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Avg Simulations:  {avg_evals:.1f}")
    print(f"Avg Transfers:    {avg_transfers:.1f}")
    print(f"Transfer Benefit: {transfer_benefit_pct:.1f}% of samples")
    print(f"Avg Time/Sample:  {avg_time_per_sample:.2f}s")
    print(f"Total Time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    print("Per-source breakdown:")
    for n_src in sorted(rmse_by_nsources.keys()):
        rmses = rmse_by_nsources[n_src]
        evals = evals_by_nsources[n_src]
        times = times_by_nsources[n_src]
        print(f"  {n_src}-source: RMSE={np.mean(rmses):.6f} +/- {np.std(rmses):.6f}, "
              f"sims={np.mean(evals):.1f}, time={np.mean(times):.2f}s (n={len(rmses)})")
    print()
    print("Best init types:")
    for init_type, count in sorted(init_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {init_type}: {count} samples ({count/len(all_results)*100:.1f}%)")
    print()

    # === COMPARISON ===
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Compare with baseline and smart init
    baseline_score = 0.9973
    baseline_time = 56.6
    smart_init_score = 1.0116
    smart_init_time = 58.6

    print(f"Baseline (15/20):            Score {baseline_score:.4f} @ {baseline_time:.1f} min")
    print(f"Smart Init (12/22):          Score {smart_init_score:.4f} @ {smart_init_time:.1f} min")
    print(f"Timestep Subsample ({config['subsample_factor']}x):   Score {final_score:.4f} @ {projected_400:.1f} min")
    print()

    score_vs_baseline = final_score - baseline_score
    score_vs_smart = final_score - smart_init_score

    if final_score > smart_init_score and projected_400 < 60:
        print(f"[SUCCESS] Score improved by {score_vs_smart:.4f} vs Smart Init, within budget!")
    elif final_score > 1.0 and projected_400 < 60:
        print(f"[SUCCESS] Score > 1.0 achieved within 60-min budget!")
    elif final_score > baseline_score and projected_400 < 60:
        print(f"[IMPROVED] Score +{score_vs_baseline:.4f} vs baseline, within budget")
    elif projected_400 < 60:
        print(f"[INFO] Within budget but score not improved ({score_vs_smart:+.4f} vs Smart Init)")
    else:
        print(f"[FAIL] Over budget by {projected_400 - 60:.1f} min")

    print("=" * 70)

    return final_score, rmse_mean


if __name__ == "__main__":
    main()
