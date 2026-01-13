#!/usr/bin/env python3
"""
Run script for Multi-Fidelity with 40x20 coarse grid.

Key idea: Coarser grid (~6x faster) allows more fevals within budget.
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import MultiFidelityCoarseRefineOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = MultiFidelityCoarseRefineOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        early_fraction=config['early_fraction'],
        refine_maxiter=config['refine_maxiter'],
        refine_top_n=config['refine_top_n'],
        sigma0_2src=config.get('sigma0_2src', 0.20),
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types, 'success': True,
        }
    except Exception as e:
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Multi-Fidelity 40x20')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=40)
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top-n', type=int, default=2)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
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

    print(f"\nMulti-Fidelity 40x20 Coarse Grid")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Refine: {args.refine_maxiter} iters on top {args.refine_top_n}")
    print(f"Coarse grid: 40x20 (~6x faster than fine)")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'early_fraction': args.early_fraction,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top_n,
        'sigma0_2src': args.sigma0_2src,
    }

    task_args = [(i, sample, meta, config) for i, sample in enumerate(samples_to_process)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, task): task for task in task_args}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress = len(results)
            sample_type = f"{result['n_sources']}-src"
            print(f"[{progress:3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{sample_type} RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s")

    total_time = time.time() - start_time

    # Simple score estimation based on RMSE (rough proxy)
    # True submission score requires full evaluation, this is approximate
    rmses_all = [r['best_rmse'] for r in results if r['success']]
    avg_rmse = np.mean(rmses_all)
    # Rough score approximation: 1/(1+RMSE) + diversity term
    # Calculate actual score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    # Calculate RMSE stats
    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_by_type = {1: [], 2: []}
    for r in results:
        if r['success']:
            rmse_by_type[r['n_sources']].append(r['best_rmse'])

    rmse_1src = np.mean(rmse_by_type[1]) if rmse_by_type[1] else float('inf')
    rmse_2src = np.mean(rmse_by_type[2]) if rmse_by_type[2] else float('inf')
    rmse_overall = np.mean(rmses)

    projected_400_min = (total_time / n_samples) * 400 / 60

    # Baseline for comparison (Session 14)
    baseline_score = 1.1233
    baseline_time = 55.3

    print()
    print("=" * 60)
    print(f"RMSE: {rmse_overall:.4f}, Score: {score:.4f}, Projected: {projected_400_min:.1f} min")
    print(f"  1-src: {rmse_1src:.4f}, 2-src: {rmse_2src:.4f}")
    print(f"Baseline: {baseline_score:.4f} @ {baseline_time:.1f} min (Session 14)")
    print(f"Delta: {score - baseline_score:+.4f} score, {projected_400_min - baseline_time:+.1f} min")
    if projected_400_min > 60:
        print("OVER BUDGET")
    elif score > baseline_score:
        print("IMPROVEMENT!")
    else:
        print("NO IMPROVEMENT")
    print("=" * 60)


if __name__ == '__main__':
    main()
