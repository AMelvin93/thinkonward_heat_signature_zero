#!/usr/bin/env python3
"""
Run script for Surrogate NN Pre-filter experiment.

Hypothesis: NN can identify bad CMA-ES candidates without full simulation,
saving ~50% of function evaluations while maintaining accuracy.

Success criteria: Score >= 1.12 AND time <= 55 min
Abort criteria: Score < 1.05 OR NN training fails
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import SurrogateNNOptimizer

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# Global optimizer with shared surrogate
_global_optimizer = None


def init_worker(config):
    """Initialize global optimizer for each worker process."""
    global _global_optimizer
    _global_optimizer = SurrogateNNOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        early_fraction=config['early_fraction'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        rmse_threshold_1src=config['rmse_threshold_1src'],
        rmse_threshold_2src=config['rmse_threshold_2src'],
        filter_ratio=config.get('filter_ratio', 0.5),
        use_surrogate=config.get('use_surrogate', True),
    )


def process_single_sample(args):
    idx, sample, meta, config = args

    # Create optimizer (each process has its own - surrogates trained per-process)
    optimizer = SurrogateNNOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        early_fraction=config['early_fraction'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        rmse_threshold_1src=config['rmse_threshold_1src'],
        rmse_threshold_2src=config['rmse_threshold_2src'],
        filter_ratio=config.get('filter_ratio', 0.5),
        use_surrogate=config.get('use_surrogate', True),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, n_filtered = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'n_filtered': n_filtered,
            'elapsed': elapsed, 'init_types': init_types, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'n_filtered': 0,
            'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e) + '\n' + traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Surrogate NN Pre-filter optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--threshold-1src', type=float, default=0.35)
    parser.add_argument('--threshold-2src', type=float, default=0.45)
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--filter-ratio', type=float, default=0.5,
                        help='Keep top N%% of candidates based on NN prediction')
    parser.add_argument('--no-surrogate', action='store_true',
                        help='Disable surrogate filtering (baseline comparison)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
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

    use_surrogate = not args.no_surrogate

    print(f"\n{'='*70}")
    print(f"Surrogate NN Pre-filter CMA-ES")
    print(f"{'='*70}")
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Filter ratio: {args.filter_ratio:.0%} (keep top candidates)")
    print(f"Surrogate enabled: {use_surrogate}")
    print(f"{'='*70}")

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'early_fraction': args.early_fraction,
        'candidate_pool_size': args.candidate_pool_size,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
        'filter_ratio': args.filter_ratio,
        'use_surrogate': use_surrogate,
    }

    start_time = time.time()
    results = []
    outliers = []
    total_filtered = 0
    total_sims = 0

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            total_filtered += result.get('n_filtered', 0)
            total_sims += result.get('n_sims', 0)

            # Track outliers
            if result['best_rmse'] > 0.4:
                outliers.append((result['idx'], result['best_rmse'], result['n_sources']))

            status = "✓" if result['best_rmse'] < 0.1 else ("!" if result['best_rmse'] > 0.4 else " ")
            filtered_str = f"filt={result.get('n_filtered', 0)}" if use_surrogate else ""
            print(f"[{len(results):3d}/{n_samples}] {status} Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} sims={result.get('n_sims', '?')} {filtered_str} "
                  f"time={result['elapsed']:.1f}s")

            if not result['success']:
                print(f"    ERROR: {result.get('error', 'Unknown error')[:100]}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses) if rmses else float('inf')
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    n_success = sum(1 for r in results if r['success'])
    filter_ratio_actual = total_filtered / max(1, total_sims) if total_sims > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: Surrogate NN Pre-filter")
    print(f"{'='*70}")
    print(f"Score: {score:.4f} | RMSE: {rmse_mean:.4f} | Projected: {projected_400:.1f} min")
    if rmses_1src:
        print(f"  1-src RMSE: {np.mean(rmses_1src):.4f} ({len(rmses_1src)} samples)")
    if rmses_2src:
        print(f"  2-src RMSE: {np.mean(rmses_2src):.4f} ({len(rmses_2src)} samples)")
    print(f"  Successful samples: {n_success}/{n_samples}")
    if use_surrogate:
        print(f"  Total sims: {total_sims}, Filtered: {total_filtered} ({filter_ratio_actual:.1%})")

    print(f"\nBaseline comparison:")
    print(f"  Baseline (no surrogate): 1.1247 @ 57.2 min")
    print(f"  This run:                {score:.4f} @ {projected_400:.1f} min")
    print(f"  Delta:                  {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")

    # Success/abort criteria
    print(f"\n{'='*70}")
    print(f"CRITERIA EVALUATION:")
    print(f"{'='*70}")

    success = score >= 1.12 and projected_400 <= 55
    abort = score < 1.05

    print(f"  Success criteria (score >= 1.12 AND time <= 55 min): ", end="")
    if success:
        print("PASS")
    else:
        if score < 1.12:
            print(f"FAIL (score {score:.4f} < 1.12)")
        else:
            print(f"FAIL (time {projected_400:.1f} > 55 min)")

    print(f"  Abort criteria (score < 1.05): ", end="")
    if abort:
        print(f"TRIGGERED (score {score:.4f} < 1.05)")
    else:
        print("NOT TRIGGERED")

    if outliers:
        print(f"\nOutliers (RMSE > 0.4):")
        for idx, rmse, n_src in sorted(outliers, key=lambda x: -x[1])[:10]:
            print(f"  Sample {idx}: {n_src}-src RMSE={rmse:.4f}")

    print(f"{'='*70}")

    # Final verdict
    if success:
        print("SUCCESS: Surrogate NN pre-filter works! Faster AND accurate!")
    elif abort:
        print("ABORT: Score below 1.05 threshold. Surrogate filtering hurts accuracy.")
    elif projected_400 > 60:
        print("OVER BUDGET: Need further optimization or reduce fevals.")
    elif score < 1.12:
        print("PARTIAL: Some improvement but didn't meet success criteria.")
    else:
        print("PARTIAL: Good score but timing needs work.")

    print(f"{'='*70}\n")

    return score, projected_400


if __name__ == '__main__':
    main()
