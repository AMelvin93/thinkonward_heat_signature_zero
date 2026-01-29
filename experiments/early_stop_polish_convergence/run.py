"""
Run script for early_stop_polish_convergence experiment.

Tests convergence-based early stopping for NM polish.
"""

import os
import sys
import pickle
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import EarlyStopPolishOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = EarlyStopPolishOptimizer(
        early_stop_threshold=config['early_stop_threshold'],
        min_nm_iters=config['min_nm_iters'],
        consecutive_iters=config.get('consecutive_iters', 2),
        use_saved_budget_for_2nd=config.get('use_saved_budget_for_2nd', False),
        max_fevals_1src=config.get('max_fevals_1src', 20),
        max_fevals_2src=config.get('max_fevals_2src', 36),
        timestep_fraction=config.get('timestep_fraction', 0.40),
        refine_maxiter=config.get('refine_maxiter', 8),
        refine_top_n=config.get('refine_top_n', 2),
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        # Get early stop stats for this sample
        early_stats = optimizer.get_early_stop_stats()

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'success': True,
            'early_stopped': early_stats['samples_early_stopped'] > 0,
            'iters_saved': early_stats['iters_saved'],
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'early_stopped': False,
            'iters_saved': 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (default: all)')
    parser.add_argument('--early-stop-threshold', type=float, default=0.001,
                        help='Stop if RMSE improvement < threshold')
    parser.add_argument('--min-nm-iters', type=int, default=3, help='Minimum NM iterations before early stop')
    parser.add_argument('--consecutive-iters', type=int, default=2,
                        help='Consecutive iterations with low improvement to stop')
    parser.add_argument('--use-saved-budget', action='store_true',
                        help='Use saved budget for second-best candidate')
    parser.add_argument('--refine-maxiter', type=int, default=8, help='Max NM iterations')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    np.random.seed(args.seed)
    indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    n_1src = sum(1 for s in samples_to_process if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"EARLY STOP POLISH CONVERGENCE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Early stop threshold: {args.early_stop_threshold}")
    print(f"Min NM iters: {args.min_nm_iters}")
    print(f"Consecutive iters: {args.consecutive_iters}")
    print(f"Max NM iters: {args.refine_maxiter}")
    print(f"Use saved budget for 2nd: {args.use_saved_budget}")
    print(f"{'='*60}")

    config = {
        'early_stop_threshold': args.early_stop_threshold,
        'min_nm_iters': args.min_nm_iters,
        'consecutive_iters': args.consecutive_iters,
        'use_saved_budget_for_2nd': args.use_saved_budget,
        'refine_maxiter': args.refine_maxiter,
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': 0.25,
        'refine_top_n': 2,
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            es_flag = " [EARLY_STOP]" if result['early_stopped'] else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{es_flag}")

    total_time = time.time() - start_time

    # Calculate score using simplified formula
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    # Statistics
    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    times_1src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 1]
    times_2src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 2]

    # Early stopping stats
    n_early_stopped = sum(1 for r in results if r['early_stopped'])
    total_iters_saved = sum(r['iters_saved'] for r in results)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Early Stop Polish Convergence")
    print(f"{'='*70}")
    print(f"Score:            {score:.4f}")
    print(f"RMSE mean:        {np.mean(rmses):.4f}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)}, time={np.mean(times_1src):.1f}s)")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)}, time={np.mean(times_2src):.1f}s)")
    print()
    print(f"Early stopping stats:")
    print(f"  Samples early stopped: {n_early_stopped}/{n_samples} ({100*n_early_stopped/n_samples:.1f}%)")
    print(f"  Total iterations saved: {total_iters_saved}")
    print(f"  Avg iters saved per sample: {total_iters_saved/n_samples:.2f}")
    print()
    print(f"Baseline:  1.1468 @ 54.2 min (perturbation_plus_verification)")
    print(f"This run:  {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:     {score - 1.1468:+.4f} score, {projected_400 - 54.2:+.1f} min")
    print()

    # Status check
    if projected_400 > 60:
        status_msg = "OVER BUDGET"
    elif score >= 1.15:
        status_msg = "SUCCESS - Meets target!"
    elif score > 1.1468:
        status_msg = "PARTIAL SUCCESS - Beats baseline!"
    else:
        status_msg = "FAILED - Score lower than baseline"
    print(f"STATUS: {status_msg}")
    print(f"{'='*70}\n")

    results_summary = {
        'score': score,
        'total_time_sec': total_time,
        'projected_400_min': projected_400,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else 0,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else 0,
        'time_mean_1src': np.mean(times_1src) if times_1src else 0,
        'time_mean_2src': np.mean(times_2src) if times_2src else 0,
        'n_early_stopped': n_early_stopped,
        'pct_early_stopped': 100 * n_early_stopped / n_samples,
        'total_iters_saved': total_iters_saved,
        'avg_iters_saved': total_iters_saved / n_samples,
    }

    # Update STATE.json
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)

        state['tuning_runs'].append({
            'run': len(state['tuning_runs']) + 1,
            'config': config,
            'results': results_summary,
            'timestamp': datetime.now().isoformat(),
        })

        if results_summary['projected_400_min'] <= 60:
            if (state['best_in_budget'] is None or
                results_summary['score'] > state['best_in_budget'].get('score', 0)):
                state['best_in_budget'] = {
                    'run': len(state['tuning_runs']),
                    'score': results_summary['score'],
                    'time_min': results_summary['projected_400_min'],
                    'config': config,
                }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    return results_summary


if __name__ == '__main__':
    main()
