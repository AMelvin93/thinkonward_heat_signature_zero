#!/usr/bin/env python3
"""
Run script for Ensemble Voting optimizer experiment.

Hypothesis: Different optimizers excel on different samples; ensemble captures best of each.

Success criteria: Score >= 1.13 AND time <= 60 min
Abort criteria: Time > 70 min consistently
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

from optimizer import EnsembleVotingOptimizer

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def process_single_sample(args):
    idx, sample, meta, config = args

    optimizer = EnsembleVotingOptimizer(
        cmaes_fevals_1src=config['cmaes_fevals_1src'],
        cmaes_fevals_2src=config['cmaes_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        nm_maxiter_1src=config['nm_maxiter_1src'],
        nm_maxiter_2src=config['nm_maxiter_2src'],
        nm_inits=config['nm_inits'],
        early_fraction=config['early_fraction'],
        candidate_pool_size=config.get('candidate_pool_size', 8),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, winner = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'winner': winner,
            'elapsed': elapsed, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'winner': 'error',
            'elapsed': time.time() - start,
            'success': False, 'error': str(e) + '\n' + traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Ensemble Voting optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    # CMA-ES settings (reduced budget)
    parser.add_argument('--cmaes-fevals-1src', type=int, default=15)
    parser.add_argument('--cmaes-fevals-2src', type=int, default=24)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    # Nelder-Mead settings
    parser.add_argument('--nm-maxiter-1src', type=int, default=30)
    parser.add_argument('--nm-maxiter-2src', type=int, default=50)
    parser.add_argument('--nm-inits', type=int, default=3)
    # General
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--candidate-pool-size', type=int, default=8)
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

    print(f"\n{'='*70}")
    print(f"Ensemble Voting Optimizer (CMA-ES + Nelder-Mead)")
    print(f"{'='*70}")
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"CMA-ES: {args.cmaes_fevals_1src}/{args.cmaes_fevals_2src} fevals, sigma={args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Nelder-Mead: {args.nm_maxiter_1src}/{args.nm_maxiter_2src} maxiter, {args.nm_inits} inits")
    print(f"{'='*70}")

    config = {
        'cmaes_fevals_1src': args.cmaes_fevals_1src,
        'cmaes_fevals_2src': args.cmaes_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'nm_maxiter_1src': args.nm_maxiter_1src,
        'nm_maxiter_2src': args.nm_maxiter_2src,
        'nm_inits': args.nm_inits,
        'early_fraction': args.early_fraction,
        'candidate_pool_size': args.candidate_pool_size,
    }

    start_time = time.time()
    results = []
    outliers = []
    optimizer_wins = {'cmaes': 0, 'nelder_mead': 0, 'other': 0}

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            # Track optimizer wins
            winner = result.get('winner', 'other')
            if winner in optimizer_wins:
                optimizer_wins[winner] += 1
            else:
                optimizer_wins['other'] += 1

            # Track outliers
            if result['best_rmse'] > 0.4:
                outliers.append((result['idx'], result['best_rmse'], result['n_sources']))

            status = "✓" if result['best_rmse'] < 0.1 else ("!" if result['best_rmse'] > 0.4 else " ")
            winner_str = f"[{result.get('winner', '?'):>6}]"
            print(f"[{len(results):3d}/{n_samples}] {status} Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} sims={result.get('n_sims', '?')} {winner_str} "
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

    print(f"\n{'='*70}")
    print(f"RESULTS: Ensemble Voting Optimizer")
    print(f"{'='*70}")
    print(f"Score: {score:.4f} | RMSE: {rmse_mean:.4f} | Projected: {projected_400:.1f} min")
    if rmses_1src:
        print(f"  1-src RMSE: {np.mean(rmses_1src):.4f} ({len(rmses_1src)} samples)")
    if rmses_2src:
        print(f"  2-src RMSE: {np.mean(rmses_2src):.4f} ({len(rmses_2src)} samples)")
    print(f"  Successful samples: {n_success}/{n_samples}")

    print(f"\nOptimizer Wins:")
    total_wins = sum(optimizer_wins.values())
    for opt, wins in sorted(optimizer_wins.items(), key=lambda x: -x[1]):
        pct = 100.0 * wins / total_wins if total_wins > 0 else 0
        print(f"  {opt}: {wins} ({pct:.1f}%)")

    print(f"\nBaseline comparison:")
    print(f"  Baseline (robust_fallback): 1.1247 @ 57.2 min")
    print(f"  This run:                   {score:.4f} @ {projected_400:.1f} min")
    print(f"  Delta:                     {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")

    # Success/abort criteria
    print(f"\n{'='*70}")
    print(f"CRITERIA EVALUATION:")
    print(f"{'='*70}")

    success = score >= 1.13 and projected_400 <= 60
    abort = projected_400 > 70

    print(f"  Success criteria (score >= 1.13 AND time <= 60 min): ", end="")
    if success:
        print("PASS")
    else:
        if score < 1.13:
            print(f"FAIL (score {score:.4f} < 1.13)")
        else:
            print(f"FAIL (time {projected_400:.1f} > 60 min)")

    print(f"  Abort criteria (time > 70 min): ", end="")
    if abort:
        print(f"TRIGGERED (time {projected_400:.1f} > 70 min)")
    else:
        print("NOT TRIGGERED")

    if outliers:
        print(f"\nOutliers (RMSE > 0.4):")
        for idx, rmse, n_src in sorted(outliers, key=lambda x: -x[1])[:10]:
            print(f"  Sample {idx}: {n_src}-src RMSE={rmse:.4f}")

    print(f"{'='*70}")

    # Final verdict
    if success:
        print("SUCCESS: Ensemble approach improves both score and stays in budget!")
    elif abort:
        print("ABORT: Consistently over 70 min budget. Need to reduce fevals.")
    elif projected_400 > 60:
        print("OVER BUDGET: Need to reduce NM iterations or CMA-ES fevals.")
    elif score < 1.13:
        print("PARTIAL: Ensemble didn't improve score over baseline.")
    else:
        print("PARTIAL: Good but didn't meet all success criteria.")

    print(f"{'='*70}\n")

    return score, projected_400, optimizer_wins


if __name__ == '__main__':
    main()
