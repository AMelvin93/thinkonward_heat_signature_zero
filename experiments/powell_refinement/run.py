#!/usr/bin/env python3
"""
Run script for Multi-Fidelity + Powell Refinement experiment.

Powell's method may be more efficient than Nelder-Mead for smooth objectives.
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

from optimizer import PowellRefinementOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = PowellRefinementOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        early_fraction=config['early_fraction'],
        refine_maxfev=config['refine_maxfev'],
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
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Multi-Fidelity + Powell Refinement')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--refine-maxfev', type=int, default=15)
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

    print(f"\nMulti-Fidelity + Powell Refinement")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Powell refine: maxfev={args.refine_maxfev} on top {args.refine_top_n}")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'early_fraction': args.early_fraction,
        'refine_maxfev': args.refine_maxfev,
        'refine_top_n': args.refine_top_n,
        'sigma0_2src': args.sigma0_2src,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "" if result['success'] else " FAILED"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s{status}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*60}")
    print(f"RMSE: {rmse_mean:.4f}, Score: {score:.4f}, Projected: {projected_400:.1f} min")
    if rmses_1src and rmses_2src:
        print(f"  1-src: {np.mean(rmses_1src):.4f}, 2-src: {np.mean(rmses_2src):.4f}")
    print(f"Baseline: 1.1233 @ 55.3 min (Session 14)")
    print(f"Delta: {score - 1.1233:+.4f} score, {projected_400 - 55.3:+.1f} min")
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score > 1.1233:
        print("IMPROVEMENT!")
    else:
        print("NO IMPROVEMENT")
    print(f"{'='*60}\n")

    return score, projected_400


if __name__ == '__main__':
    main()
