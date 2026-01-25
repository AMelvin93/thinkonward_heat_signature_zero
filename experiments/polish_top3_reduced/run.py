#!/usr/bin/env python3
"""
Run script for Polish Top-3 optimizer.

Tests the hypothesis that polishing ALL top-3 candidates (4 NM iters each)
instead of just the best (8 NM iters) improves average accuracy.
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import PolishTop3Optimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = PolishTop3Optimizer(
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
        polish_iters_per_candidate=config['polish_iters_per_candidate'],
        n_candidates_to_polish=config['n_candidates_to_polish'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        all_rmses = [r.rmse for r in results]
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'all_rmses': all_rmses,  # Track all candidate RMSEs
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'all_rmses': [],
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Polish Top-3 optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--candidate-pool-size', type=int, default=10)
    parser.add_argument('--nx-coarse', type=int, default=50)
    parser.add_argument('--ny-coarse', type=int, default=25)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top', type=int, default=2)
    parser.add_argument('--threshold-1src', type=float, default=0.4)
    parser.add_argument('--threshold-2src', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    # NEW: Polish parameters
    parser.add_argument('--polish-iters-per-candidate', type=int, default=4,
                        help='NM iterations per candidate to polish')
    parser.add_argument('--n-candidates-to-polish', type=int, default=3,
                        help='Number of top candidates to polish')
    # MLflow
    parser.add_argument('--experiment-id', type=str, default='EXP_POLISH_TOP3_001')
    parser.add_argument('--tuning-run', type=int, default=1)
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

    total_polish_iters = args.polish_iters_per_candidate * args.n_candidates_to_polish

    print(f"\nPolish Top-3 Optimizer")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Fevals: {args.max_fevals_1src}/{args.max_fevals_2src}")
    print(f"Polish: {args.polish_iters_per_candidate} iters x {args.n_candidates_to_polish} candidates = {total_polish_iters} total")
    print(f"Baseline: 8 iters x 1 candidate = 8 total")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'candidate_pool_size': args.candidate_pool_size,
        'nx_coarse': args.nx_coarse,
        'ny_coarse': args.ny_coarse,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
        'timestep_fraction': args.timestep_fraction,
        'polish_iters_per_candidate': args.polish_iters_per_candidate,
        'n_candidates_to_polish': args.n_candidates_to_polish,
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
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]")

    total_time = time.time() - start_time

    def calculate_sample_score(rmses, n_candidates, lambda_=0.3, n_max=3):
        """Score using all candidate RMSEs (not just best)."""
        if n_candidates == 0 or not rmses:
            return 0.0
        # Accuracy: average of 1/(1+RMSE) over candidates
        accuracy = np.mean([1.0 / (1.0 + rmse) for rmse in rmses[:n_max]])
        # Diversity bonus
        diversity = lambda_ * (min(n_candidates, n_max) / n_max)
        return accuracy + diversity

    # Calculate scores using all candidate RMSEs
    sample_scores = []
    for r in results:
        if r['success'] and r['all_rmses']:
            sample_scores.append(calculate_sample_score(r['all_rmses'], r['n_candidates']))
        else:
            sample_scores.append(0.0)
    score = np.mean(sample_scores)

    # Also calculate baseline-style score for comparison
    baseline_sample_scores = [1.0/(1.0+r['best_rmse']) + 0.3*(r['n_candidates']/3)
                               for r in results if r['success']]
    baseline_score = np.mean(baseline_sample_scores) if baseline_sample_scores else 0

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmse_mean = np.mean(rmses) if rmses else float('inf')
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    # Analyze all candidate RMSEs
    all_cand_rmses = []
    for r in results:
        if r['success'] and r['all_rmses']:
            all_cand_rmses.extend(r['all_rmses'])
    avg_all_cand_rmse = np.mean(all_cand_rmses) if all_cand_rmses else float('inf')

    # MLflow logging
    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"polish_top3_run{args.tuning_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", args.experiment_id)
        mlflow.log_param("worker", "W3")
        mlflow.log_param("tuning_run", args.tuning_run)
        mlflow.log_param("polish_iters_per_candidate", args.polish_iters_per_candidate)
        mlflow.log_param("n_candidates_to_polish", args.n_candidates_to_polish)
        mlflow.log_param("total_polish_iters", total_polish_iters)
        mlflow.log_param("timestep_fraction", args.timestep_fraction)
        mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
        mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
        mlflow.log_param("n_samples", n_samples)

        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("baseline_score", baseline_score)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("avg_all_cand_rmse", avg_all_cand_rmse)
        mlflow.log_metric("total_time_sec", total_time)

        mlflow_run_id = run.info.run_id

    print(f"\n{'='*70}")
    print(f"RESULTS - Polish Top-3 ({args.polish_iters_per_candidate} iters x {args.n_candidates_to_polish} candidates)")
    print(f"{'='*70}")
    print(f"RMSE (best):      {rmse_mean:.6f}")
    print(f"RMSE (all cand):  {avg_all_cand_rmse:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Baseline Score:   {baseline_score:.4f}")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Baseline:  1.1688 @ 58.4 min")
    print(f"This run:  {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:     {score - 1.1688:+.4f} score, {projected_400 - 58.4:+.1f} min")
    print()
    print(f"MLflow run ID: {mlflow_run_id}")
    print(f"{'='*70}\n")

    # Determine success/failure
    in_budget = projected_400 <= 60
    better_score = score > 1.1688

    if in_budget and better_score:
        print("SUCCESS! Better score within budget")
    elif in_budget:
        print("FAILED: In budget but worse/same score")
    elif better_score:
        print("PARTIAL: Better score but over budget")
    else:
        print("FAILED: Worse score and over budget")

    return score, projected_400, mlflow_run_id


if __name__ == '__main__':
    main()
