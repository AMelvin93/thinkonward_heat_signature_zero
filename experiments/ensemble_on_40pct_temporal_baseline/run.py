#!/usr/bin/env python3
"""Run script for Ensemble on 40% Temporal Baseline optimizer with MLflow logging."""

import argparse
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

from optimizer import EnsembleOn40PctTemporalOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = EnsembleOn40PctTemporalOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.4),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.5),
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        ensemble_top_n=config['ensemble_top_n'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims, ensemble_win = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'success': True, 'ensemble_win': ensemble_win,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
            'ensemble_win': False,
        }


def main():
    parser = argparse.ArgumentParser(description='Run Ensemble on 40% Temporal Baseline')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--final-polish-maxiter', type=int, default=8)  # Key: 8 for 1.1688
    parser.add_argument('--ensemble-top-n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-mlflow', action='store_true')
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
    n_samples = len(samples_to_process)

    print(f"\nEnsemble on 40% Temporal Baseline")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Final polish: {args.final_polish_maxiter} NM iters (full timesteps)")
    print(f"Ensemble: top-{args.ensemble_top_n} weighted average")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'final_polish_maxiter': args.final_polish_maxiter,
        'ensemble_top_n': args.ensemble_top_n,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            ens_flag = " [E]" if result.get('ensemble_win', False) else ""
            polished = " [P]" if 'polished' in result.get('init_types', []) else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{ens_flag}{polished}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    ensemble_wins = sum(1 for r in results if r.get('ensemble_win', False))
    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Ensemble on 40% Temporal Baseline")
    print(f"{'='*70}")
    print(f"Submission Score: {score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print(f"Ensemble wins:    {ensemble_wins}/{n_samples} ({100*ensemble_wins/n_samples:.1f}%)")
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"40% Temporal Baseline (1.1688 @ 58.4 min):")
    print(f"  This run:      {score:.4f} @ {projected_400:.1f} min")
    print(f"  Delta:         {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print(f"{'='*70}\n")

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"ensemble_40pct_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_id', 'EXP_ENSEMBLE_40PCT_COMBINED_001')
            mlflow.log_param('worker', 'W2')
            mlflow.log_param('optimizer', 'EnsembleOn40PctTemporalOptimizer')
            mlflow.log_param('n_samples', n_samples)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('timestep_fraction', args.timestep_fraction)
            mlflow.log_param('final_polish_maxiter', args.final_polish_maxiter)
            mlflow.log_param('ensemble_top_n', args.ensemble_top_n)
            mlflow.log_param('max_fevals_1src', args.max_fevals_1src)
            mlflow.log_param('max_fevals_2src', args.max_fevals_2src)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('submission_score', score)
            mlflow.log_metric('projected_400_samples_min', projected_400)
            mlflow.log_metric('rmse_mean', np.mean(rmses) if rmses else float('inf'))
            mlflow.log_metric('rmse_1src_mean', np.mean(rmses_1src) if rmses_1src else float('inf'))
            mlflow.log_metric('rmse_2src_mean', np.mean(rmses_2src) if rmses_2src else float('inf'))
            mlflow.log_metric('ensemble_win_rate', ensemble_wins / n_samples)
            mlflow.log_metric('total_time_sec', total_time)

            print(f"MLflow run ID: {run.info.run_id}")

    return score, projected_400, run.info.run_id if not args.no_mlflow else None


if __name__ == '__main__':
    main()
