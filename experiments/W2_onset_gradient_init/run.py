#!/usr/bin/env python
"""
Run script for W2_onset_gradient_init experiment.

Tests hybrid onset + gradient initialization for better source localization.
"""

import os
import sys
import time
import pickle
import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from experiments.W2_onset_gradient_init.optimizer import OnsetGradientOptimizer


def process_sample(args):
    idx, sample, meta, config = args
    optimizer = OnsetGradientOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        rmse_threshold_1src=config['threshold_1src'],
        rmse_threshold_2src=config['threshold_2src'],
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx,
            'sample_id': sample['sample_id'],
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'init_types': init_types,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'sample_id': sample['sample_id'],
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'init_types': [],
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    """Simplified scoring function."""
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def run_experiment(n_workers: int = 7, shuffle: bool = True, seed: int = 42):
    """Run the onset_gradient_init experiment."""

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    data_path = _project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    if shuffle:
        indices = list(range(len(samples)))
        random.shuffle(indices)
    else:
        indices = list(range(len(samples)))

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'threshold_1src': 0.35,
        'threshold_2src': 0.45,
        'refine_maxiter': 3,
        'refine_top_n': 2,
    }

    print(f"Running W2_onset_gradient_init on {n_samples} samples...")
    print(f"Config: onset_gradient hybrid init, threshold 0.35/0.45, refine 3 iters")
    print("-" * 60)

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]")

    elapsed = time.time() - start_time
    projected_400 = (elapsed / n_samples) * 400 / 60

    # Compute score
    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    # Compute metrics
    rmses_1src = [r['best_rmse'] for r in results if r['n_sources'] == 1 and r['success']]
    rmses_2src = [r['best_rmse'] for r in results if r['n_sources'] == 2 and r['success']]

    avg_rmse_1src = np.mean(rmses_1src) if rmses_1src else float('inf')
    avg_rmse_2src = np.mean(rmses_2src) if rmses_2src else float('inf')
    max_rmse = max(r['best_rmse'] for r in results if r['success'])

    # Count init types used
    init_counts = {}
    for r in results:
        for init_type in r['init_types']:
            init_counts[init_type] = init_counts.get(init_type, 0) + 1

    print("\n" + "=" * 60)
    print(f"RESULTS: W2_onset_gradient_init")
    print("=" * 60)
    print(f"Score: {score:.4f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Projected 400 samples: {projected_400:.1f} min")
    print(f"Avg RMSE 1-src: {avg_rmse_1src:.4f}")
    print(f"Avg RMSE 2-src: {avg_rmse_2src:.4f}")
    print(f"Max RMSE: {max_rmse:.4f}")
    print(f"Init type distribution: {init_counts}")

    # Compare with baseline
    print()
    print(f"Baseline: 1.1247 @ 57.2 min")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:    {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print()
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score > 1.1247:
        print("IMPROVED!")
    else:
        print("NO IMPROVEMENT")

    # Find worst samples
    sorted_results = sorted(results, key=lambda x: -x['best_rmse'])
    print("\nWorst 5 samples:")
    for r in sorted_results[:5]:
        print(f"  Sample {r['idx']}: RMSE={r['best_rmse']:.4f} ({r['n_sources']}-src)")

    # Log to MLflow
    mlflow.set_tracking_uri(str(_project_root / "mlruns"))
    mlflow.set_experiment("W2_init_strategies")

    with mlflow.start_run(run_name=f"onset_gradient_init_seed{seed}"):
        mlflow.log_param("experiment", "W2_onset_gradient_init")
        mlflow.log_param("init_strategy", "onset_gradient_hybrid")
        mlflow.log_param("threshold_1src", 0.35)
        mlflow.log_param("threshold_2src", 0.45)
        mlflow.log_param("refine_iters", 3)
        mlflow.log_param("refine_top", 2)
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_workers", n_workers)
        mlflow.log_param("platform", "wsl")

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("elapsed_sec", elapsed)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("avg_rmse_1src", avg_rmse_1src)
        mlflow.log_metric("avg_rmse_2src", avg_rmse_2src)
        mlflow.log_metric("max_rmse", max_rmse)

    return {
        'score': score,
        'elapsed': elapsed,
        'projected_400': projected_400,
        'avg_rmse_1src': avg_rmse_1src,
        'avg_rmse_2src': avg_rmse_2src,
        'max_rmse': max_rmse,
        'init_counts': init_counts,
        'worst_samples': sorted_results[:5]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(n_workers=args.workers, shuffle=args.shuffle, seed=args.seed)
