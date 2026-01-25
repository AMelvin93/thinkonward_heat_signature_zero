#!/usr/bin/env python3
"""
Weighted Centroid NM Experiment

Tests weighted centroid approach from 2023 paper for NM polish phase.
Based on: "Weighted Centroids in Adaptive Nelder–Mead Simplex:
With heat source locator applications" (Applied Soft Computing, 2024)
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

from optimizer import WeightedCentroidNMOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = WeightedCentroidNMOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config['sigma0_1src'],
        sigma0_2src=config['sigma0_2src'],
        timestep_fraction=config['timestep_fraction'],
        final_polish_maxiter=config['final_polish_maxiter'],
        weight_power=config.get('weight_power', 2.0),
        use_weighted_nm=config.get('use_weighted_nm', True),
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


def main():
    parser = argparse.ArgumentParser(description='Weighted Centroid NM')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--final-polish-maxiter', type=int, default=8)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--weight-power', type=float, default=2.0,
                        help='Power for weight calculation (higher = more weight on best vertices)')
    parser.add_argument('--use-weighted-nm', action='store_true', default=True)
    parser.add_argument('--no-weighted-nm', dest='use_weighted_nm', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
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

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'timestep_fraction': args.timestep_fraction,
        'final_polish_maxiter': args.final_polish_maxiter,
        'weight_power': args.weight_power,
        'use_weighted_nm': args.use_weighted_nm,
    }

    print(f"\nWeighted Centroid NM Experiment")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Use weighted NM: {args.use_weighted_nm}, Weight power: {args.weight_power}")
    print(f"Temporal: {args.timestep_fraction*100:.0f}%, Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"Polish: {args.final_polish_maxiter} NM iters")
    print(f"=" * 70)

    # Set up MLflow
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"weighted_centroid_nm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_WEIGHTED_CENTROID_NM_001")
        mlflow.log_param("worker", "W1")
        mlflow.log_param("use_weighted_nm", args.use_weighted_nm)
        mlflow.log_param("weight_power", args.weight_power)
        mlflow.log_param("timestep_fraction", args.timestep_fraction)
        mlflow.log_param("final_polish_maxiter", args.final_polish_maxiter)
        mlflow.log_param("sigma0_1src", args.sigma0_1src)
        mlflow.log_param("sigma0_2src", args.sigma0_2src)
        mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
        mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("platform", "wsl")

        start_time = time.time()
        results = []

        work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                status = "OK" if result['success'] else "ERR"
                polish_flag = "[WNM]" if args.use_weighted_nm else "[NM]"
                if (len(results) % 10 == 0) or len(results) == n_samples:
                    print(f"[{len(results):3d}/{n_samples}] {result['n_sources']}-src "
                          f"RMSE={result['best_rmse']:.4f} cands={result['n_candidates']} [{status}] {polish_flag}")

        total_time = time.time() - start_time

        # Calculate metrics
        sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
        score = np.mean(sample_scores)

        rmses = [r['best_rmse'] for r in results if r['success']]
        rmse_mean = np.mean(rmses)
        rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
        rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
        projected_400 = (total_time / n_samples) * 400 / 60

        # Log metrics
        mlflow.log_metric("submission_score", score)
        mlflow.log_metric("projected_400_samples_min", projected_400)
        mlflow.log_metric("rmse_mean", rmse_mean)
        if rmses_1src:
            mlflow.log_metric("rmse_1src", np.mean(rmses_1src))
        if rmses_2src:
            mlflow.log_metric("rmse_2src", np.mean(rmses_2src))

        print(f"\n{'='*70}")
        print(f"WEIGHTED CENTROID NM RESULTS")
        print(f"{'='*70}")
        print(f"Score:        {score:.4f}")
        print(f"Projected:    {projected_400:.1f} min")
        print(f"In budget:    {'YES' if projected_400 <= 60 else 'NO'}")
        print(f"")
        if rmses_1src:
            print(f"1-src RMSE:   {np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
        if rmses_2src:
            print(f"2-src RMSE:   {np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
        print(f"")
        print(f"Baseline (standard NM): 1.1688 @ 58.4 min")
        delta = score - 1.1688
        delta_time = projected_400 - 58.4
        print(f"Delta:        {delta:+.4f} score, {delta_time:+.1f} min")
        print(f"")
        if score > 1.1688 and projected_400 <= 60:
            print("SUCCESS: Better than baseline and within budget!")
        elif score > 1.1688:
            print("PARTIAL: Better score but over budget")
        elif projected_400 <= 60:
            print("PARTIAL: Within budget but worse score")
        else:
            print("FAILED: Worse score and over budget")
        print(f"{'='*70}\n")

        mlflow_run_id = run.info.run_id

    return {
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': rmse_mean,
        'mlflow_run_id': mlflow_run_id,
    }


if __name__ == '__main__':
    main()
