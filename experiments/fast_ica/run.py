#!/usr/bin/env python3
"""
Run script for Fast ICA Decomposition optimizer.

Accelerated version with speedup strategies:
1. FastICA max_iter=50 instead of 500
2. Coarse grid (50x25) for initial search
3. Reduced CMA-ES fevals (10/15)
4. No batched transfer learning

Target: Score >= 1.02 AND time <= 60 min
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

from optimizer import FastICAOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = FastICAOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        use_triangulation=config.get('use_triangulation', True),
        use_ica=config.get('use_ica', True),
        ica_max_iter=config.get('ica_max_iter', 50),
        n_candidates=config.get('n_candidates', 3),
        candidate_pool_size=config.get('candidate_pool_size', 8),
        use_coarse_search=config.get('use_coarse_search', True),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
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
    parser = argparse.ArgumentParser(description='Run Fast ICA optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=10)
    parser.add_argument('--max-fevals-2src', type=int, default=15)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--ica-max-iter', type=int, default=50)
    parser.add_argument('--no-coarse', action='store_true', help='Disable coarse grid search')
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--threshold-1src', type=float, default=0.35)
    parser.add_argument('--threshold-2src', type=float, default=0.45)
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

    print(f"\nFast ICA Decomposition Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"ICA max_iter: {args.ica_max_iter} (vs 500 default = 10x speedup)")
    print(f"Coarse search: {not args.no_coarse} ({args.nx_coarse}x{args.ny_coarse})")
    print(f"Fallback threshold: 1-src={args.threshold_1src}, 2-src={args.threshold_2src}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'use_triangulation': True,
        'use_ica': True,
        'ica_max_iter': args.ica_max_iter,
        'n_candidates': 3,
        'candidate_pool_size': 8,
        'use_coarse_search': not args.no_coarse,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
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
            fb_flag = "[FB]" if result['best_rmse'] > (0.35 if result['n_sources'] == 1 else 0.45) else "    "
            ica_flag = "[I]" if any('ica' in t for t in result['init_types']) else "   "
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} sims={result['n_sims']:3d} "
                  f"time={result['elapsed']:.1f}s [{status}]{fb_flag}{ica_flag}")

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

    # Count ICA wins
    ica_win_count = sum(1 for r in results if any('ica' in t for t in r.get('init_types', [])))
    n_2src = sum(1 for r in results if r['n_sources'] == 2)

    print(f"\n{'='*70}")
    print(f"RESULTS: Fast ICA Decomposition Optimizer")
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
    print(f"ICA best init: {ica_win_count}/{n_2src} 2-source samples ({ica_win_count/n_2src*100:.1f}%)" if n_2src > 0 else "No 2-source samples")
    print()
    print(f"Target: Score >= 1.02 AND time <= 60 min")
    print(f"Baseline (robust_fallback): 1.1247 @ 57.2 min")
    print(f"This fast ICA run:          {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                      {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print()

    if projected_400 <= 60 and score >= 1.02:
        print("STATUS: SUCCESS - Meets criteria!")
    elif projected_400 <= 60:
        print(f"STATUS: TIME OK but score below target (need {1.02 - score:.4f} more)")
    elif score >= 1.02:
        print(f"STATUS: SCORE OK but over budget by {projected_400 - 60:.1f} min")
    else:
        print("STATUS: FAILED - Both score and time miss targets")

    print(f"{'='*70}\n")

    # Print outliers
    high_rmse = [r for r in results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:10]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f} sims={r['n_sims']}")

    return score, projected_400


if __name__ == '__main__':
    main()
