#!/usr/bin/env python
"""Run Adaptive Budget Optimizer experiment."""

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

from experiments.adaptive_budget.optimizer import (
    AdaptiveBudgetOptimizer,
    extract_enhanced_features,
    N_MAX,
)

G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400
LAMBDA = 0.3


def detect_platform():
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    return "wsl"
        except:
            pass
        return "linux"
    return system


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX, max_rmse=1.0):
    valid_rmses = [r for r in rmses if r <= max_rmse]
    n_valid = len(valid_rmses)
    if n_valid == 0:
        return 0.0
    accuracy_sum = sum(1.0 / (1.0 + r) for r in valid_rmses)
    return accuracy_sum / n_valid + lambda_ * (n_valid / n_max)


def process_sample(sample, meta, config, history_1src, history_2src):
    optimizer = AdaptiveBudgetOptimizer(
        min_fevals_1src=config['min_fevals_1src'],
        max_fevals_1src=config['max_fevals_1src'],
        min_fevals_2src=config['min_fevals_2src'],
        max_fevals_2src=config['max_fevals_2src'],
        easy_rmse_1src=config['easy_rmse_1src'],
        hard_rmse_1src=config['hard_rmse_1src'],
        easy_rmse_2src=config['easy_rmse_2src'],
        hard_rmse_2src=config['hard_rmse_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        use_triangulation=config.get('use_triangulation', True),
        n_candidates=config.get('n_candidates', 3),
        candidate_pool_size=config.get('candidate_pool_size', 10),
        k_similar=config.get('k_similar', 1),
    )

    q_range = tuple(meta['q_range'])

    start = time.time()
    candidates, best_rmse, results, features, best_positions, n_transferred, fevals_used = optimizer.estimate_sources(
        sample, meta, q_range=q_range,
        history_1src=history_1src,
        history_2src=history_2src,
        verbose=False
    )
    elapsed = time.time() - start

    candidate_rmses = [r.rmse for r in results]
    score = calculate_sample_score(candidate_rmses)
    n_evals = sum(r.n_evals for r in results) if results else 0
    best_init_type = results[0].init_type if results else 'unknown'

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'n_candidates': len(candidates),
        'rmses': candidate_rmses,
        'best_rmse': best_rmse,
        'score': score,
        'time': elapsed,
        'n_evals': n_evals,
        'fevals_used': fevals_used,
        'n_transferred': n_transferred,
        'best_init_type': best_init_type,
        'features': features,
        'best_positions': best_positions,
    }


def process_batch(samples, meta, config, history_1src, history_2src, n_workers):
    h1_copy = deepcopy(history_1src)
    h2_copy = deepcopy(history_2src)

    results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(process_sample)(sample, meta, config, h1_copy, h2_copy)
        for sample in samples
    )
    return results


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive Budget Optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-samples', type=int, default=None)
    # Feval ranges
    parser.add_argument('--min-fevals-1src', type=int, default=8)
    parser.add_argument('--max-fevals-1src', type=int, default=16)
    parser.add_argument('--min-fevals-2src', type=int, default=16)
    parser.add_argument('--max-fevals-2src', type=int, default=28)
    # RMSE thresholds
    parser.add_argument('--easy-rmse-1src', type=float, default=0.10)
    parser.add_argument('--hard-rmse-1src', type=float, default=0.25)
    parser.add_argument('--easy-rmse-2src', type=float, default=0.15)
    parser.add_argument('--hard-rmse-2src', type=float, default=0.35)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'min_fevals_1src': args.min_fevals_1src,
        'max_fevals_1src': args.max_fevals_1src,
        'min_fevals_2src': args.min_fevals_2src,
        'max_fevals_2src': args.max_fevals_2src,
        'easy_rmse_1src': args.easy_rmse_1src,
        'hard_rmse_1src': args.hard_rmse_1src,
        'easy_rmse_2src': args.easy_rmse_2src,
        'hard_rmse_2src': args.hard_rmse_2src,
        'use_triangulation': True,
        'n_candidates': 3,
        'candidate_pool_size': 10,
        'k_similar': 1,
    }

    n_workers = args.workers
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers
    is_g4dn_simulation = (n_workers == G4DN_WORKERS)

    current_platform = detect_platform()

    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(samples)

    n_batches = (n_samples + args.batch_size - 1) // args.batch_size

    print("=" * 70)
    print("ADAPTIVE BUDGET OPTIMIZER (A3)")
    print("=" * 70)
    print(f"Platform: {current_platform.upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-src, {n_2src} 2-src)")
    print(f"Workers: {actual_workers}" + (" (G4dn)" if is_g4dn_simulation else ""))
    print(f"1-src fevals: {config['min_fevals_1src']}-{config['max_fevals_1src']} "
          f"(easy<{config['easy_rmse_1src']}, hard>{config['hard_rmse_1src']})")
    print(f"2-src fevals: {config['min_fevals_2src']}-{config['max_fevals_2src']} "
          f"(easy<{config['easy_rmse_2src']}, hard>{config['hard_rmse_2src']})")
    print("=" * 70)

    start_total = time.time()

    history_1src = []
    history_2src = []
    all_results = []
    fevals_distribution = {'1src': [], '2src': []}

    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, n_samples)
        batch_samples = samples[batch_start:batch_end]

        print(f"\nBatch {batch_idx + 1}/{n_batches}: samples {batch_start}-{batch_end-1}")

        batch_results = process_batch(
            batch_samples, meta, config,
            history_1src, history_2src,
            n_workers
        )

        for result in batch_results:
            if result['n_sources'] == 1:
                history_1src.append((result['features'], result['best_positions']))
                fevals_distribution['1src'].append(result['fevals_used'])
            else:
                history_2src.append((result['features'], result['best_positions']))
                fevals_distribution['2src'].append(result['fevals_used'])

        all_results.extend(batch_results)

        batch_rmses = [r['best_rmse'] for r in batch_results]
        batch_fevals = [r['fevals_used'] for r in batch_results]
        print(f"  -> Batch RMSE: {np.mean(batch_rmses):.4f}, Avg fevals: {np.mean(batch_fevals):.1f}, "
              f"Avg time: {np.mean([r['time'] for r in batch_results]):.2f}s")

    total_time = time.time() - start_total

    all_best_rmses = [r['best_rmse'] for r in all_results]
    all_scores = [r['score'] for r in all_results]
    final_score = np.mean(all_scores)
    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    rmse_by_nsources = {}
    for r in all_results:
        n_src = r['n_sources']
        if n_src not in rmse_by_nsources:
            rmse_by_nsources[n_src] = []
        rmse_by_nsources[n_src].append(r['best_rmse'])

    if is_g4dn_simulation and not args.max_samples:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")

        run_name = f"adaptive_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", np.mean(all_best_rmses))
            mlflow.log_metric("projected_400_samples_min", projected_400)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_nsources.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_nsources.get(2, [0])))
            mlflow.log_metric("avg_fevals_1src", np.mean(fevals_distribution['1src']) if fevals_distribution['1src'] else 0)
            mlflow.log_metric("avg_fevals_2src", np.mean(fevals_distribution['2src']) if fevals_distribution['2src'] else 0)
            mlflow.log_param("optimizer", "AdaptiveBudgetOptimizer")
            mlflow.log_param("min_fevals_1src", config['min_fevals_1src'])
            mlflow.log_param("max_fevals_1src", config['max_fevals_1src'])
            mlflow.log_param("min_fevals_2src", config['min_fevals_2src'])
            mlflow.log_param("max_fevals_2src", config['max_fevals_2src'])
            mlflow.log_param("platform", current_platform)

        print(f"\n[MLflow] Logged: {run_name}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"RMSE:             {np.mean(all_best_rmses):.6f} +/- {np.std(all_best_rmses):.6f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()

    print("Feval Distribution:")
    if fevals_distribution['1src']:
        print(f"  1-src: avg={np.mean(fevals_distribution['1src']):.1f}, "
              f"min={min(fevals_distribution['1src'])}, max={max(fevals_distribution['1src'])}")
    if fevals_distribution['2src']:
        print(f"  2-src: avg={np.mean(fevals_distribution['2src']):.1f}, "
              f"min={min(fevals_distribution['2src'])}, max={max(fevals_distribution['2src'])}")
    print()

    for n_src in sorted(rmse_by_nsources.keys()):
        rmses = rmse_by_nsources[n_src]
        print(f"  {n_src}-source: RMSE={np.mean(rmses):.4f} (n={len(rmses)})")
    print()

    baseline_score = 1.0224
    print(f"Baseline: {baseline_score:.4f} @ 56.5 min")
    print(f"This run: {final_score:.4f} @ {projected_400:.1f} min")

    if final_score >= 1.15 and projected_400 < 60:
        print("[TARGET HIT!]")
    elif final_score > baseline_score and projected_400 < 60:
        print(f"[IMPROVED] +{final_score - baseline_score:.4f}")
    elif projected_400 >= 60:
        print(f"[OVER BUDGET] by {projected_400 - 60:.1f} min")
    else:
        print(f"[NO IMPROVEMENT]")
    print("=" * 70)

    return final_score, np.mean(all_best_rmses)


if __name__ == "__main__":
    main()
