#!/usr/bin/env python
"""
Run Bayesian Optimization experiment.

Key Innovation: Use GP surrogate for more sample-efficient optimization.

Hypothesis:
    - BO could match CMA-ES accuracy with fewer fevals
    - Model-based approach is smarter than population-based
    - Could save time by being strategic about where to sample

Target: Score >= 1.0116 AND time < 58.6 min (beat current best)

Usage:
    # Quick validation (10 samples, ~2 min)
    uv run python experiments/bayesian_optimization/run.py --workers 7 --shuffle --max-samples 10

    # Full run (80 samples)
    uv run python experiments/bayesian_optimization/run.py --workers 7 --shuffle
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

from experiments.bayesian_optimization.optimizer import (
    BayesianOptimizer,
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
    """Process a single sample with Bayesian optimization."""
    optimizer = BayesianOptimizer(
        n_initial_1src=config['n_initial_1src'],
        n_initial_2src=config['n_initial_2src'],
        n_bo_iter_1src=config['n_bo_iter_1src'],
        n_bo_iter_2src=config['n_bo_iter_2src'],
        use_triangulation=config['use_triangulation'],
        n_candidates=config['n_candidates'],
        candidate_pool_size=config['candidate_pool_size'],
        k_similar=config['k_similar'],
        use_enhanced_features=config['use_enhanced_features'],
        gp_kernel=config['gp_kernel'],
        xi=config['xi'],
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
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization')
    parser.add_argument('--workers', type=int, default=7,
                        help='Number of workers (7 for G4dn, -1 for all CPUs)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Samples per batch for transfer learning')
    parser.add_argument('--n-initial-1src', type=int, default=5,
                        help='Initial samples for 1-source (before BO)')
    parser.add_argument('--n-initial-2src', type=int, default=8,
                        help='Initial samples for 2-source (before BO)')
    parser.add_argument('--n-bo-iter-1src', type=int, default=7,
                        help='BO iterations for 1-source')
    parser.add_argument('--n-bo-iter-2src', type=int, default=14,
                        help='BO iterations for 2-source')
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
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick validation')
    parser.add_argument('--gp-kernel', type=str, default='matern',
                        choices=['rbf', 'matern'], help='GP kernel type')
    parser.add_argument('--xi', type=float, default=0.01,
                        help='EI exploration-exploitation trade-off')
    args = parser.parse_args()

    # Build config
    config = {
        'n_initial_1src': args.n_initial_1src,
        'n_initial_2src': args.n_initial_2src,
        'n_bo_iter_1src': args.n_bo_iter_1src,
        'n_bo_iter_2src': args.n_bo_iter_2src,
        'use_triangulation': not args.no_triangulation,
        'n_candidates': min(args.n_candidates, N_MAX),
        'candidate_pool_size': args.pool_size,
        'k_similar': args.k_similar,
        'use_enhanced_features': not args.no_enhanced_features,
        'gp_kernel': args.gp_kernel,
        'xi': args.xi,
    }

    # Total fevals equivalent
    total_fevals_1src = config['n_initial_1src'] + config['n_bo_iter_1src']
    total_fevals_2src = config['n_initial_2src'] + config['n_bo_iter_2src']

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

    # Limit samples for quick validation
    if args.max_samples and args.max_samples < len(samples):
        if args.shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(samples)
        samples = samples[:args.max_samples]
        is_quick_validation = True
    else:
        if args.shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(samples)
        is_quick_validation = False

    n_samples = len(samples)

    # Count sample types
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    n_batches = (n_samples + batch_size - 1) // batch_size

    # Print header
    print("=" * 70)
    print("BAYESIAN OPTIMIZATION")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    if is_quick_validation:
        print(f"[QUICK VALIDATION MODE - {args.max_samples} samples]")
    print(f"Workers: {actual_workers}" + (" (G4dn simulation)" if is_g4dn_simulation else " (prototype)"))
    print(f"Batch size: {batch_size} ({n_batches} batches)")
    print()
    print("KEY INNOVATION: GP surrogate with Expected Improvement")
    print(f"  - Initial samples: 1-src={config['n_initial_1src']}, 2-src={config['n_initial_2src']}")
    print(f"  - BO iterations: 1-src={config['n_bo_iter_1src']}, 2-src={config['n_bo_iter_2src']}")
    print(f"  - Total fevals: 1-src={total_fevals_1src}, 2-src={total_fevals_2src}")
    print(f"  - GP kernel: {config['gp_kernel']}")
    print(f"  - EI xi: {config['xi']}")
    print()
    print(f"Config:")
    print(f"  triangulation: {config['use_triangulation']}")
    print(f"  n_candidates: {config['n_candidates']}")
    print(f"  k_similar: {config['k_similar']}")
    print(f"  enhanced_features: {config['use_enhanced_features']}")
    print(f"  shuffle: {args.shuffle}" + (f" (seed={args.seed})" if args.shuffle else ""))
    print(f"MLflow logging: {'ENABLED' if is_g4dn_simulation and not is_quick_validation else 'DISABLED'}")

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
        # Simplify BO iteration types
        if init_type.startswith('bo_'):
            init_type = 'bo'
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

    # Per-source stats
    rmse_1src = np.mean(rmse_by_nsources.get(1, [0]))
    rmse_2src = np.mean(rmse_by_nsources.get(2, [0]))
    time_1src = np.mean(times_by_nsources.get(1, [0]))
    time_2src = np.mean(times_by_nsources.get(2, [0]))

    # Count transfer benefit
    transfer_benefit_count = sum(1 for r in all_results if 'transfer' in r['best_init_type'])
    transfer_benefit_pct = transfer_benefit_count / len(all_results) * 100

    # === MLFLOW LOGGING (only for full runs) ===
    if is_g4dn_simulation and not is_quick_validation:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"bo_{total_fevals_1src}_{total_fevals_2src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

            mlflow.log_param("optimizer", "BayesianOptimizer")
            mlflow.log_param("n_initial_1src", config['n_initial_1src'])
            mlflow.log_param("n_initial_2src", config['n_initial_2src'])
            mlflow.log_param("n_bo_iter_1src", config['n_bo_iter_1src'])
            mlflow.log_param("n_bo_iter_2src", config['n_bo_iter_2src'])
            mlflow.log_param("gp_kernel", config['gp_kernel'])
            mlflow.log_param("xi", config['xi'])
            mlflow.log_param("use_triangulation", config['use_triangulation'])
            mlflow.log_param("n_candidates", config['n_candidates'])
            mlflow.log_param("k_similar", config['k_similar'])
            mlflow.log_param("n_workers", G4DN_WORKERS)
            mlflow.log_param("platform", current_platform)
            mlflow.log_param("shuffle", args.shuffle)

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
    print(f"Config:           {total_fevals_1src}/{total_fevals_2src} fevals (Bayesian Optimization)")
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

    # Compare with baseline
    baseline_score = 1.0116
    baseline_time = 58.6
    baseline_fevals = "12/22"

    print(f"Current Best (SmartInit {baseline_fevals}): Score {baseline_score:.4f} @ {baseline_time:.1f} min")
    print(f"Bayesian Opt ({total_fevals_1src}/{total_fevals_2src}): Score {final_score:.4f} @ {projected_400:.1f} min")
    print()

    score_diff = final_score - baseline_score
    time_diff = projected_400 - baseline_time

    if final_score >= baseline_score and projected_400 < baseline_time:
        print(f"[SUCCESS] Same or better score ({score_diff:+.4f}), {-time_diff:.1f} min faster!")
    elif final_score >= baseline_score:
        print(f"[INFO] Same or better score ({score_diff:+.4f}), but {time_diff:+.1f} min slower")
    elif projected_400 < baseline_time:
        print(f"[TRADE-OFF] Faster ({-time_diff:.1f} min) but score dropped ({score_diff:.4f})")
    else:
        print(f"[FAIL] Both score ({score_diff:.4f}) and time ({time_diff:+.1f} min) worse")

    if projected_400 > 60:
        print(f"[WARNING] Over budget by {projected_400 - 60:.1f} min")

    print("=" * 70)

    return final_score, rmse_mean


if __name__ == "__main__":
    main()
