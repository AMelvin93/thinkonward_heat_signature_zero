#!/usr/bin/env python3
"""
Run script for CMA-ES to Nelder-Mead Sequential Handoff optimizer.

Key idea: CMA-ES for global exploration (reduced budget) -> NM for local refinement (increased budget)
This is sequential, not parallel, to avoid doubling simulation count.

Success criteria: Score >= 1.13 AND time <= 55 min
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

from optimizer import SequentialHandoffOptimizer


def process_sample(args):
    """Process a single sample."""
    idx, sample, meta, config = args

    optimizer = SequentialHandoffOptimizer(
        cmaes_fevals_1src=config['cmaes_fevals_1src'],
        cmaes_fevals_2src=config['cmaes_fevals_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.20),
        sigma0_2src=config.get('sigma0_2src', 0.25),
        nm_maxiter_1src=config.get('nm_maxiter_1src', 40),
        nm_maxiter_2src=config.get('nm_maxiter_2src', 50),
        nm_top_n=config.get('nm_top_n', 3),
        nm_use_coarse_grid=config.get('nm_use_coarse_grid', False),
        use_triangulation=config.get('use_triangulation', True),
        n_candidates=config.get('n_candidates', 3),
        candidate_pool_size=config.get('candidate_pool_size', 8),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.35),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.45),
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
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'success': True,
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
    parser = argparse.ArgumentParser(description='Run Sequential Handoff optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    # CMA-ES parameters (slightly reduced for exploration)
    parser.add_argument('--cmaes-fevals-1src', type=int, default=15)
    parser.add_argument('--cmaes-fevals-2src', type=int, default=28)
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    # NM parameters (modest - NM on fine grid is expensive!)
    parser.add_argument('--nm-maxiter-1src', type=int, default=8)
    parser.add_argument('--nm-maxiter-2src', type=int, default=12)
    parser.add_argument('--nm-top-n', type=int, default=2)
    parser.add_argument('--nm-coarse', action='store_true', help='Use coarse grid for NM (faster)')
    # Thresholds
    parser.add_argument('--threshold-1src', type=float, default=0.35)
    parser.add_argument('--threshold-2src', type=float, default=0.45)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    np.random.seed(args.seed)
    if args.shuffle:
        perm = np.random.permutation(len(samples))
        samples = [samples[i] for i in perm]

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)

    print(f"\nCMA-ES to Nelder-Mead Sequential Handoff Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"CMA-ES: fevals={args.cmaes_fevals_1src}/{args.cmaes_fevals_2src}, "
          f"sigma={args.sigma0_1src}/{args.sigma0_2src}")
    nm_grid = "coarse" if args.nm_coarse else "fine"
    print(f"NM: maxiter={args.nm_maxiter_1src}/{args.nm_maxiter_2src}, top_n={args.nm_top_n}, grid={nm_grid}")
    print(f"Fallback threshold: 1-src={args.threshold_1src}, 2-src={args.threshold_2src}")
    print(f"=" * 70)

    config = {
        'cmaes_fevals_1src': args.cmaes_fevals_1src,
        'cmaes_fevals_2src': args.cmaes_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'nm_maxiter_1src': args.nm_maxiter_1src,
        'nm_maxiter_2src': args.nm_maxiter_2src,
        'nm_top_n': args.nm_top_n,
        'nm_use_coarse_grid': args.nm_coarse,
        'use_triangulation': True,
        'n_candidates': 3,
        'candidate_pool_size': 8,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
    }

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]
    all_results = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            status = "OK" if result['success'] else "ERR"
            nm_flag = "[NM]" if 'nm' in str(result['init_types']) else ""
            print(f"[{len(all_results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"sims={result['n_sims']:3d} time={result['elapsed']:.1f}s [{status}]{nm_flag}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in all_results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in all_results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in all_results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in all_results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    # Count NM usage
    nm_count = sum(1 for r in all_results if any('nm' in str(it) for it in r['init_types']))
    fallback_count = sum(1 for r in all_results if any('fallback' in str(it) for it in r['init_types']))

    print(f"\n{'='*70}")
    print(f"RESULTS: CMA-ES -> NM Sequential Handoff")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Total Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"NM refinement used: {nm_count}/{n_samples} samples")
    print(f"Fallback triggered: {fallback_count}/{n_samples} samples")
    print()
    print(f"Target: Score >= 1.13 AND time <= 55 min")
    print(f"Baseline (robust_fallback): 1.1247 @ 57.2 min")
    print(f"This run:                   {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                      {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print()

    if score >= 1.13 and projected_400 <= 55:
        print(f"STATUS: SUCCESS - Score >= 1.13 AND time <= 55 min!")
    elif score >= 1.1247 and projected_400 <= 60:
        print(f"STATUS: PARTIAL - Matches or exceeds baseline within budget")
    elif projected_400 <= 60:
        print(f"STATUS: PARTIAL - Time OK but score dropped")
    else:
        print(f"STATUS: FAILED - Over budget by {projected_400 - 60:.1f} min")

    print(f"{'='*70}\n")

    # Print outliers
    high_rmse = [r for r in all_results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:10]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f} "
                  f"sims={r['n_sims']} {r['init_types']}")

    return score, projected_400


if __name__ == '__main__':
    main()
