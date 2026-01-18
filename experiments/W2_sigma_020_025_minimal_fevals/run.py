#!/usr/bin/env python3
"""
Run script for W2 Sigma 0.20/0.25 + Minimal Fevals optimizer.

EXP032: sigma_0.20_0.25_minimal_fevals
Config:
- sigma0_1src: 0.20 (best sigma for score)
- sigma0_2src: 0.25 (best sigma for score)
- threshold_1src: 0.35 (standard - proven optimal)
- threshold_2src: 0.45 (standard - proven optimal)
- fevals_1src: 18 (10% reduction from 20)
- fevals_2src: 33 (~8% reduction from 36)
- fallback_fevals: 16 (11% reduction from 18)

Hypothesis: Minimal fevals reduction with best sigma - preserve more score while reducing time.
Based on: EXP029 (14/26) and EXP030 (16/30) too aggressive. Try less reduction (~10%).
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

from optimizer import SigmaBaselineOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = SigmaBaselineOptimizer(
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        rmse_threshold_1src=config['threshold_1src'],
        rmse_threshold_2src=config['threshold_2src'],
        fallback_fevals=config['fallback_fevals'],
        max_fevals_1src=config.get('max_fevals_1src', 18),
        max_fevals_2src=config.get('max_fevals_2src', 33),
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
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
    parser = argparse.ArgumentParser(description='Run W2 Sigma 0.20/0.25 + Minimal Fevals')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # Key experiment parameters - EXP032 specific
    parser.add_argument('--sigma0-1src', type=float, default=0.20)  # Best sigma
    parser.add_argument('--sigma0-2src', type=float, default=0.25)  # Best sigma
    parser.add_argument('--threshold-1src', type=float, default=0.35)  # Standard
    parser.add_argument('--threshold-2src', type=float, default=0.45)  # Standard
    parser.add_argument('--fallback-fevals', type=int, default=16)  # Slight reduction
    # Minimal primary fevals reduction
    parser.add_argument('--max-fevals-1src', type=int, default=18)  # 10% reduction from 20
    parser.add_argument('--max-fevals-2src', type=int, default=33)  # ~8% reduction from 36
    # Other parameters
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top', type=int, default=2)
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

    print(f"\n[W2] Sigma 0.20/0.25 + Minimal Fevals Optimizer")
    print(f"=" * 70)
    print(f"EXP032: sigma_0.20_0.25_minimal_fevals")
    print(f"-" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Sigma: 1-src={args.sigma0_1src}, 2-src={args.sigma0_2src} (BEST)")
    print(f"Threshold: 1-src={args.threshold_1src}, 2-src={args.threshold_2src} (standard optimal)")
    print(f"PRIMARY fevals: 1-src={args.max_fevals_1src}, 2-src={args.max_fevals_2src} (MINIMAL reduction)")
    print(f"Fallback fevals: {args.fallback_fevals}")
    print(f"==" * 35)

    config = {
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'threshold_1src': args.threshold_1src,
        'threshold_2src': args.threshold_2src,
        'fallback_fevals': args.fallback_fevals,
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top,
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
            flag = ""
            if result['best_rmse'] > args.threshold_1src and result['n_sources'] == 1:
                flag = " [FB]"
            elif result['best_rmse'] > args.threshold_2src and result['n_sources'] == 2:
                flag = " [FB]"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{flag}")

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

    print(f"\n{'='*70}")
    print(f"RESULTS: [W2] Sigma 0.20/0.25 + Minimal Fevals (EXP032)")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Config: sigma={args.sigma0_1src}/{args.sigma0_2src} (BEST), fevals={args.max_fevals_1src}/{args.max_fevals_2src} (MINIMAL reduction)")
    print(f"Baseline: 1.1247 @ 57.2 min (sigma 0.15/0.20, fevals 20/36)")
    print(f"EXP002 (sigma 0.20/0.25, fevals 20/36): 1.1362 @ 69.2 min")
    print(f"EXP030 (sigma 0.20/0.25, fevals 16/30): 1.1188 @ 55.6 min")
    print(f"Target: ~60 min with score > baseline (1.1247)")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs baseline: {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print(f"Delta vs EXP002: {score - 1.1362:+.4f} score, {projected_400 - 69.2:+.1f} min")
    print()
    if projected_400 <= 60:
        if score > 1.1247:
            print(">>> IN BUDGET AND IMPROVED! <<<")
        else:
            print(">>> IN BUDGET <<<")
    else:
        print(">>> OVER BUDGET <<<")
    print(f"{'='*70}\n")

    # Print outliers
    high_rmse = [r for r in results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:5]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f}")

    # Print fallback samples
    fallback_samples = [r for r in results if
                        (r['n_sources'] == 1 and r['best_rmse'] > args.threshold_1src) or
                        (r['n_sources'] == 2 and r['best_rmse'] > args.threshold_2src)]
    if fallback_samples:
        print(f"\nFallback triggered ({len(fallback_samples)} samples):")
        for r in sorted(fallback_samples, key=lambda x: -x['best_rmse'])[:5]:
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f}")

    return score, projected_400


if __name__ == '__main__':
    main()
