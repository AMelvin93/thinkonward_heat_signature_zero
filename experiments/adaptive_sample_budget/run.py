#!/usr/bin/env python3
"""
Run script for Adaptive Budget optimizer.

Tests the hypothesis that adaptive early termination + bonus budget
for hard samples can improve efficiency without hurting accuracy.
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import AdaptiveBudgetOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = AdaptiveBudgetOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
        sigma_converge_thresh=config.get('sigma_converge_thresh', 0.01),
        stagnation_gens=config.get('stagnation_gens', 3),
        stagnation_tol=config.get('stagnation_tol', 1e-4),
        bonus_fevals_1src=config.get('bonus_fevals_1src', 10),
        bonus_fevals_2src=config.get('bonus_fevals_2src', 18),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.25),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.35),
        early_fraction=config.get('early_fraction', 0.3),
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, extra_info = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False,
            allow_bonus=config.get('allow_bonus', True)
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types, 'success': True,
            'early_terminated': extra_info['early_terminated'],
            'fevals_saved': extra_info['fevals_saved'],
            'used_bonus': extra_info['used_bonus'],
            'convergence_reasons': extra_info['convergence_reasons'],
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
            'early_terminated': False, 'fevals_saved': 0, 'used_bonus': False,
            'convergence_reasons': [],
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive Budget optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    # Base budget
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    # Sigma parameters
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    # Early termination
    parser.add_argument('--sigma-converge', type=float, default=0.01)
    parser.add_argument('--stagnation-gens', type=int, default=3)
    parser.add_argument('--stagnation-tol', type=float, default=1e-4)
    # Bonus budget
    parser.add_argument('--bonus-fevals-1src', type=int, default=10)
    parser.add_argument('--bonus-fevals-2src', type=int, default=18)
    parser.add_argument('--rmse-thresh-1src', type=float, default=0.25)
    parser.add_argument('--rmse-thresh-2src', type=float, default=0.35)
    parser.add_argument('--no-bonus', action='store_true', help='Disable bonus budget')
    # Standard parameters
    parser.add_argument('--early-fraction', type=float, default=0.3)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    # MLflow
    parser.add_argument('--no-mlflow', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
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

    print(f"\nAdaptive Budget Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Base fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Bonus fevals: {args.bonus_fevals_1src}/{args.bonus_fevals_2src} (enabled: {not args.no_bonus})")
    print(f"Early termination: sigma<{args.sigma_converge}, stagnation>{args.stagnation_gens} gens")
    print(f"RMSE thresholds: 1-src>{args.rmse_thresh_1src}, 2-src>{args.rmse_thresh_2src}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'sigma_converge_thresh': args.sigma_converge,
        'stagnation_gens': args.stagnation_gens,
        'stagnation_tol': args.stagnation_tol,
        'bonus_fevals_1src': args.bonus_fevals_1src,
        'bonus_fevals_2src': args.bonus_fevals_2src,
        'rmse_threshold_1src': args.rmse_thresh_1src,
        'rmse_threshold_2src': args.rmse_thresh_2src,
        'allow_bonus': not args.no_bonus,
        'early_fraction': args.early_fraction,
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
            flags = []
            if result.get('early_terminated'):
                flags.append('ET')  # Early Terminated
            if result.get('used_bonus'):
                flags.append('B')  # Bonus used
            flag_str = f" [{','.join(flags)}]" if flags else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{flag_str}")

    total_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    # Adaptive budget metrics
    early_terminated_count = sum(1 for r in results if r.get('early_terminated'))
    total_fevals_saved = sum(r.get('fevals_saved', 0) for r in results)
    bonus_used_count = sum(1 for r in results if r.get('used_bonus'))

    print(f"\n{'='*70}")
    print(f"RESULTS")
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
    print(f"Adaptive Budget Metrics:")
    print(f"  Early terminated: {early_terminated_count}/{n_samples} samples ({100*early_terminated_count/n_samples:.1f}%)")
    print(f"  Fevals saved: {total_fevals_saved}")
    print(f"  Bonus used: {bonus_used_count}/{n_samples} samples ({100*bonus_used_count/n_samples:.1f}%)")
    print()
    print(f"Baseline: 1.1247 @ 57 min")
    print(f"This run: {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:    {score - 1.1247:+.4f} score, {projected_400 - 57:+.1f} min")
    print()
    if projected_400 > 60:
        print("OVER BUDGET")
    elif score >= 1.13:
        print("SUCCESS - Met success criteria!")
    elif score > 1.1247:
        print("IMPROVED!")
    else:
        print("NO IMPROVEMENT")
    print(f"{'='*70}\n")

    # Print outliers
    high_rmse = [r for r in results if r['best_rmse'] > 0.4]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:5]:
            bonus_flag = " [BONUS]" if r.get('used_bonus') else ""
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f}{bonus_flag}")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment('adaptive_sample_budget')

        run_name = args.run_name or f"adaptive_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param('max_fevals_1src', args.max_fevals_1src)
            mlflow.log_param('max_fevals_2src', args.max_fevals_2src)
            mlflow.log_param('sigma_converge_thresh', args.sigma_converge)
            mlflow.log_param('stagnation_gens', args.stagnation_gens)
            mlflow.log_param('bonus_fevals_1src', args.bonus_fevals_1src)
            mlflow.log_param('bonus_fevals_2src', args.bonus_fevals_2src)
            mlflow.log_param('rmse_threshold_1src', args.rmse_thresh_1src)
            mlflow.log_param('rmse_threshold_2src', args.rmse_thresh_2src)
            mlflow.log_param('allow_bonus', not args.no_bonus)
            mlflow.log_param('n_samples', n_samples)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            # Log metrics
            mlflow.log_metric('submission_score', score)
            mlflow.log_metric('rmse_mean', rmse_mean)
            if rmses_1src:
                mlflow.log_metric('rmse_mean_1src', np.mean(rmses_1src))
            if rmses_2src:
                mlflow.log_metric('rmse_mean_2src', np.mean(rmses_2src))
            mlflow.log_metric('projected_400_samples_min', projected_400)
            mlflow.log_metric('total_time_sec', total_time)
            mlflow.log_metric('early_terminated_pct', 100 * early_terminated_count / n_samples)
            mlflow.log_metric('fevals_saved', total_fevals_saved)
            mlflow.log_metric('bonus_used_pct', 100 * bonus_used_count / n_samples)

            print(f"\nLogged to MLflow: {run_name}")

    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else None,
        'projected_400_min': projected_400,
        'early_terminated_pct': 100 * early_terminated_count / n_samples,
        'bonus_used_pct': 100 * bonus_used_count / n_samples,
        'in_budget': projected_400 <= 60,
    }


if __name__ == '__main__':
    main()
