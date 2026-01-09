#!/usr/bin/env python
"""
Run Asymmetric Budget Optimizer experiment.

Key Innovation: Reallocates compute from 1-source (already excellent) to
2-source (the main bottleneck) for better overall performance.

Baseline:     15/20 fevals -> Score 0.9973
Asymmetric:   12/24 fevals -> Expected +0.02-0.04 improvement

Configurations to test:
    --preset baseline   : 15/20 (current best)
    --preset mild       : 13/22 (mild reallocation)
    --preset moderate   : 12/24 (recommended)
    --preset aggressive : 10/26 (aggressive reallocation)
    --preset extreme    : 8/30  (extreme reallocation)

Usage:
    # Test moderate asymmetric config (recommended)
    uv run python experiments/asymmetric_budget/run.py --preset moderate --workers 7 --shuffle

    # Test aggressive config
    uv run python experiments/asymmetric_budget/run.py --preset aggressive --workers 7 --shuffle

    # Custom config
    uv run python experiments/asymmetric_budget/run.py --max-fevals-1src 11 --max-fevals-2src 25 --workers 7 --shuffle
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

from experiments.analytical_intensity.optimizer import (
    AnalyticalIntensityOptimizer,
    extract_enhanced_features,
    N_MAX,
)

# G4dn.2xlarge simulation settings
G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400
LAMBDA = 0.3

# Asymmetric budget presets
# Format: (1-source fevals, 2-source fevals, description)
PRESETS = {
    'baseline':   (15, 20, "Current best (symmetric-ish)"),
    'mild':       (13, 22, "Mild reallocation"),
    'moderate':   (12, 24, "Moderate reallocation (recommended)"),
    'aggressive': (10, 26, "Aggressive reallocation"),
    'extreme':    (8, 30,  "Extreme reallocation (risky)"),
}


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
    """Process a single sample with analytical intensity optimizer."""
    optimizer = AnalyticalIntensityOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
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
        # For history update (positions only)
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
    parser = argparse.ArgumentParser(description='Run Asymmetric Budget Optimizer')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a preset configuration')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of workers (7 for G4dn, -1 for all CPUs)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Samples per batch for transfer learning')
    parser.add_argument('--max-fevals-1src', type=int, default=None,
                        help='Max CMA-ES evaluations for 1-source (overrides preset)')
    parser.add_argument('--max-fevals-2src', type=int, default=None,
                        help='Max CMA-ES evaluations for 2-source (overrides preset)')
    parser.add_argument('--sigma0-1src', type=float, default=0.15,
                        help='Initial step size for 1-source')
    parser.add_argument('--sigma0-2src', type=float, default=0.20,
                        help='Initial step size for 2-source')
    parser.add_argument('--no-triangulation', action='store_true',
                        help='Disable triangulation init')
    parser.add_argument('--n-candidates', type=int, default=3,
                        help='Target number of candidates')
    parser.add_argument('--pool-size', type=int, default=10,
                        help='Number of top solutions to consider')
    parser.add_argument('--k-similar', type=int, default=1,
                        help='Number of similar samples to transfer from')
    parser.add_argument('--no-enhanced-features', action='store_true',
                        help='Use basic features instead of enhanced')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle samples before batching')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffle (default: 42)')
    args = parser.parse_args()

    # Determine fevals from preset or args
    if args.preset:
        fevals_1src, fevals_2src, preset_desc = PRESETS[args.preset]
        # Allow overrides
        if args.max_fevals_1src is not None:
            fevals_1src = args.max_fevals_1src
        if args.max_fevals_2src is not None:
            fevals_2src = args.max_fevals_2src
        preset_name = args.preset
    else:
        # Default to moderate if no preset and no args
        if args.max_fevals_1src is None and args.max_fevals_2src is None:
            fevals_1src, fevals_2src, preset_desc = PRESETS['moderate']
            preset_name = 'moderate (default)'
        else:
            fevals_1src = args.max_fevals_1src or 15
            fevals_2src = args.max_fevals_2src or 20
            preset_desc = "Custom configuration"
            preset_name = 'custom'

    # Build config
    config = {
        'max_fevals_1src': fevals_1src,
        'max_fevals_2src': fevals_2src,
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

    # Count sample types for time estimation
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    # Shuffle samples if requested
    shuffle_enabled = args.shuffle
    shuffle_seed = args.seed
    if shuffle_enabled:
        np.random.seed(shuffle_seed)
        np.random.shuffle(samples)

    n_batches = (n_samples + batch_size - 1) // batch_size

    # Calculate reallocation metrics
    baseline_1src, baseline_2src = 15, 20
    realloc_1src = fevals_1src - baseline_1src
    realloc_2src = fevals_2src - baseline_2src
    total_baseline = baseline_1src * n_1src + baseline_2src * n_2src
    total_new = fevals_1src * n_1src + fevals_2src * n_2src
    total_change_pct = (total_new - total_baseline) / total_baseline * 100

    # Print header
    print("=" * 70)
    print("ASYMMETRIC BUDGET OPTIMIZER")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {actual_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype)"))
    print(f"Batch size: {batch_size} ({n_batches} batches)")
    print()
    print(f"ASYMMETRIC BUDGET CONFIG: {preset_name}")
    print(f"  {preset_desc}")
    print(f"  1-source fevals: {fevals_1src} ({realloc_1src:+d} vs baseline 15)")
    print(f"  2-source fevals: {fevals_2src} ({realloc_2src:+d} vs baseline 20)")
    print(f"  Total fevals change: {total_change_pct:+.1f}%")
    print()
    print("Available presets:")
    for name, (f1, f2, desc) in PRESETS.items():
        marker = " <-- selected" if name == args.preset else ""
        print(f"  {name:12s}: {f1:2d}/{f2:2d} fevals - {desc}{marker}")
    print()
    print(f"Other settings:")
    print(f"  triangulation: {config['use_triangulation']}")
    print(f"  n_candidates: {config['n_candidates']}")
    print(f"  k_similar: {config['k_similar']}")
    print(f"  enhanced_features: {config['use_enhanced_features']}")
    print(f"  shuffle: {shuffle_enabled}" + (f" (seed={shuffle_seed})" if shuffle_enabled else ""))
    print(f"MLflow logging: {'ENABLED' if is_g4dn_simulation else 'DISABLED'}")

    if not is_valid_platform:
        print("")
        print("[WARNING] Running on Windows - timing will be ~35% slower than Linux!")
    print("=" * 70)

    # === BATCH PROCESSING WITH TRANSFER LEARNING ===
    start_total = time.time()

    history_1src = []  # [(features, positions), ...]
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

        # Update history with batch results (positions only)
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
    transfers_by_nsources = {}
    times_by_nsources = {}
    for r in all_results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
            evals_by_nsources[n_src] = []
            transfers_by_nsources[n_src] = []
            times_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['best_rmse'])
        evals_by_nsources[n_src].append(r['n_evals'])
        transfers_by_nsources[n_src].append(r['n_transferred'])
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

    # Count transfer benefit
    transfer_benefit_count = sum(1 for r in all_results if 'transfer' in r['best_init_type'])
    transfer_benefit_pct = transfer_benefit_count / len(all_results) * 100

    # === MLFLOW LOGGING ===
    if is_g4dn_simulation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"asymmetric_{fevals_1src}_{fevals_2src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Key metrics
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", rmse_mean)
            mlflow.log_metric("projected_400_samples_min", projected_400)

            # Transfer learning metrics
            mlflow.log_metric("avg_transfers_used", avg_transfers)
            mlflow.log_metric("transfer_benefit_pct", transfer_benefit_pct)

            # Per-source metrics
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.log_metric("rmse_1src", rmse_1src)
            mlflow.log_metric("rmse_2src", rmse_2src)
            mlflow.log_metric("avg_n_candidates", avg_candidates)
            mlflow.log_metric("avg_evals", avg_evals)
            mlflow.log_metric("avg_time_per_sample", avg_time_per_sample)
            mlflow.log_metric("total_time_sec", total_time)

            # Parameters
            mlflow.log_param("optimizer", "AsymmetricBudgetOptimizer")
            mlflow.log_param("preset", preset_name)
            mlflow.log_param("max_fevals_1src", config['max_fevals_1src'])
            mlflow.log_param("max_fevals_2src", config['max_fevals_2src'])
            mlflow.log_param("use_triangulation", config['use_triangulation'])
            mlflow.log_param("n_candidates", config['n_candidates'])
            mlflow.log_param("k_similar", config['k_similar'])
            mlflow.log_param("use_enhanced_features", config['use_enhanced_features'])
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("n_workers", G4DN_WORKERS)
            mlflow.log_param("platform", current_platform)
            mlflow.log_param("shuffle", shuffle_enabled)
            if shuffle_enabled:
                mlflow.log_param("shuffle_seed", shuffle_seed)

            # Save results
            output_path = project_root / "results" / f"{run_name}_results.pkl"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'results': all_results,
                    'config': config,
                    'preset': preset_name,
                    'final_score': final_score,
                    'total_time': total_time,
                    'projected_400': projected_400,
                    'history_sizes': (len(history_1src), len(history_2src)),
                    'shuffle': shuffle_enabled,
                    'shuffle_seed': shuffle_seed if shuffle_enabled else None,
                    'transfer_benefit_pct': transfer_benefit_pct,
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
    print(f"Configuration:    {preset_name} ({fevals_1src}/{fevals_2src} fevals)")
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
        transfers = transfers_by_nsources[n_src]
        times = times_by_nsources[n_src]
        print(f"  {n_src}-source: RMSE={np.mean(rmses):.6f} +/- {np.std(rmses):.6f}, "
              f"sims={np.mean(evals):.1f}, transfers={np.mean(transfers):.1f}, "
              f"time={np.mean(times):.2f}s (n={len(rmses)})")
    print()
    print("Best init types:")
    for init_type, count in sorted(init_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {init_type}: {count} samples ({count/len(all_results)*100:.1f}%)")
    print()

    # Comparison with baseline
    print("=" * 70)
    print("COMPARISON WITH BASELINE (15/20)")
    print("=" * 70)
    baseline_score = 0.9973  # Known baseline
    baseline_rmse_1src = 0.186
    baseline_rmse_2src = 0.348
    baseline_time = 56.6

    score_change = final_score - baseline_score
    rmse_1src_change = rmse_1src - baseline_rmse_1src
    rmse_2src_change = rmse_2src - baseline_rmse_2src
    time_change = projected_400 - baseline_time

    print(f"Score:      {final_score:.4f} ({score_change:+.4f} vs baseline {baseline_score})")
    print(f"1-src RMSE: {rmse_1src:.4f} ({rmse_1src_change:+.4f} vs baseline {baseline_rmse_1src})")
    print(f"2-src RMSE: {rmse_2src:.4f} ({rmse_2src_change:+.4f} vs baseline {baseline_rmse_2src})")
    print(f"Time (400): {projected_400:.1f} min ({time_change:+.1f} min vs baseline {baseline_time})")

    if score_change > 0 and projected_400 < 60:
        print("\n[SUCCESS] Improvement achieved within budget!")
    elif score_change > 0 and projected_400 >= 60:
        print(f"\n[WARNING] Score improved but over budget by {projected_400 - 60:.1f} min")
    elif score_change <= 0 and projected_400 < 60:
        print("\n[INFO] No improvement, but within budget")
    else:
        print("\n[FAIL] No improvement and over budget")

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
