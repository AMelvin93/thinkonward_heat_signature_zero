#!/usr/bin/env python3
"""
Correct Temporal Sweep - Test timestep fractions with CORRECT sigma 0.15/0.20.

The previous temporal_fidelity_sweep used sigma 0.18/0.22 by mistake, resulting
in all runs being over budget. This experiment uses the correct baseline sigma.

Test: 35%, 38%, 42%, 45% (skip 40% as that's the baseline)
Baseline: 40% timesteps + sigma 0.15/0.20 + 8 NM polish = 1.1688 @ 58.4 min
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

from optimizer import TemporalFidelityWithPolishOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = TemporalFidelityWithPolishOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
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


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def run_experiment(timestep_fraction, workers, config_base, samples, meta, indices):
    """Run a single experiment with given timestep fraction."""
    n_samples = len(indices)
    config = config_base.copy()
    config['timestep_fraction'] = timestep_fraction

    print(f"\n{'='*70}")
    print(f"Running timestep_fraction={timestep_fraction*100:.0f}%")
    print(f"sigma={config['sigma0_1src']}/{config['sigma0_2src']}, polish={config['final_polish_maxiter']}")
    print(f"{'='*70}")

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            if (len(results) % 10 == 0) or len(results) == n_samples:
                print(f"[{len(results):3d}/{n_samples}] {result['n_sources']}-src "
                      f"RMSE={result['best_rmse']:.4f} cands={result['n_candidates']} [{status}]")

    total_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\nResults for timestep_fraction={timestep_fraction*100:.0f}%:")
    print(f"  Score: {score:.4f}, Time: {projected_400:.1f} min")
    print(f"  1-src RMSE: {np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    print(f"  2-src RMSE: {np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")

    return {
        'timestep_fraction': timestep_fraction,
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': rmse_mean,
        'rmse_1src': np.mean(rmses_1src) if rmses_1src else None,
        'rmse_2src': np.mean(rmses_2src) if rmses_2src else None,
        'n_1src': len(rmses_1src),
        'n_2src': len(rmses_2src),
        'total_time': total_time,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Correct Temporal Sweep')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--final-polish-maxiter', type=int, default=8)
    # CORRECT sigma values from baseline
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timestep-fraction', type=float, default=None,
                        help='If specified, only run this fraction. Otherwise sweep all.')
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    np.random.seed(args.seed)
    indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]

    config_base = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'candidate_pool_size': 10,
        'nx_coarse': 50,
        'ny_coarse': 25,
        'refine_maxiter': 3,
        'refine_top_n': 2,
        'rmse_threshold_1src': 0.4,
        'rmse_threshold_2src': 0.5,
        'final_polish_maxiter': args.final_polish_maxiter,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
    }

    # Set up MLflow
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    # Fractions to test (skip 40% as that's the baseline)
    if args.timestep_fraction is not None:
        fractions = [args.timestep_fraction]
    else:
        fractions = [0.35, 0.38, 0.42, 0.45]

    print(f"\nCorrect Temporal Sweep Experiment")
    print(f"=" * 70)
    print(f"Samples: {len(samples_to_process)}, Workers: {args.workers}")
    print(f"Timestep fractions to test: {fractions}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src} (CORRECT baseline values)")
    print(f"Polish: {args.final_polish_maxiter} NM iters")
    print(f"=" * 70)

    all_results = []

    for fraction in fractions:
        run_name = f"correct_sweep_{int(fraction*100)}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("experiment_id", "EXP_CORRECT_TEMPORAL_SWEEP_001")
            mlflow.log_param("worker", "W1")
            mlflow.log_param("timestep_fraction", fraction)
            mlflow.log_param("final_polish_maxiter", args.final_polish_maxiter)
            mlflow.log_param("sigma0_1src", args.sigma0_1src)
            mlflow.log_param("sigma0_2src", args.sigma0_2src)
            mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
            mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
            mlflow.log_param("n_samples", len(samples_to_process))
            mlflow.log_param("platform", "wsl")

            result = run_experiment(
                fraction, args.workers, config_base,
                samples_to_process, meta, indices
            )
            result['mlflow_run_id'] = run.info.run_id
            all_results.append(result)

            mlflow.log_metric("submission_score", result['score'])
            mlflow.log_metric("projected_400_samples_min", result['projected_400_min'])
            mlflow.log_metric("rmse_mean", result['rmse_mean'])
            if result['rmse_1src']:
                mlflow.log_metric("rmse_1src", result['rmse_1src'])
            if result['rmse_2src']:
                mlflow.log_metric("rmse_2src", result['rmse_2src'])

    # Print summary
    print(f"\n{'='*70}")
    print("CORRECT SWEEP SUMMARY (sigma 0.15/0.20)")
    print(f"{'='*70}")
    print(f"{'Fraction':<10} {'Score':<10} {'Time (min)':<12} {'In Budget':<10} {'vs Baseline (1.1688)'}")
    print("-" * 70)

    for r in sorted(all_results, key=lambda x: -x['score']):
        in_budget = "YES" if r['projected_400_min'] <= 60 else "NO"
        delta = r['score'] - 1.1688
        frac_str = f"{r['timestep_fraction']*100:.0f}%"
        print(f"{frac_str:<10} {r['score']:.4f}     {r['projected_400_min']:.1f}          {in_budget:<10} {delta:+.4f}")

    print(f"\nBaseline (40%): 1.1688 @ 58.4 min")

    # Find best in-budget
    in_budget_results = [r for r in all_results if r['projected_400_min'] <= 60]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['timestep_fraction']*100:.0f}% = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")
        if best['score'] > 1.1688:
            print("SUCCESS: Better than 40% baseline!")
        else:
            print(f"No improvement over 40% baseline (1.1688)")
    else:
        print(f"\nNo runs completed within 60 min budget!")

    print(f"{'='*70}\n")

    return all_results


if __name__ == '__main__':
    main()
