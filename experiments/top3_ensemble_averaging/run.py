#!/usr/bin/env python3
"""Run script for Top-3 Ensemble Averaging optimizer."""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import Top3EnsembleOptimizer


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    """Calculate score for a single sample."""
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = Top3EnsembleOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        timestep_fraction=config.get('timestep_fraction', 0.40),
        top_k=config['top_k'],
        add_ensemble_as_candidate=config['add_ensemble_as_candidate'],
    )
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]
        has_ensemble = 'ensemble' in init_types
        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'has_ensemble': has_ensemble, 'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'has_ensemble': False, 'success': False, 'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run Top-3 Ensemble Averaging')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable ensemble candidate (for comparison)')
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

    print(f"\nTop-3 Ensemble Averaging Optimizer")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Top-K for ensemble: {args.top_k}")
    print(f"Add ensemble as candidate: {not args.no_ensemble}")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'top_k': args.top_k,
        'add_ensemble_as_candidate': not args.no_ensemble,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0]
                   for item in work_items}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            idx = result['idx']
            if result['success']:
                ensemble_tag = " [E]" if result['has_ensemble'] else ""
                print(f"  [{idx:3d}] {result['n_sources']}-src: RMSE={result['best_rmse']:.4f}, "
                      f"{result['n_candidates']} cands, {result['n_sims']} sims, "
                      f"{result['elapsed']:.1f}s{ensemble_tag}")
            else:
                print(f"  [{idx:3d}] FAILED: {result.get('error', 'Unknown error')}")

    total_time = time.time() - start_time

    # Calculate statistics
    results_by_src = {1: [], 2: []}
    for r in results:
        if r['success'] and r['n_sources'] in results_by_src:
            results_by_src[r['n_sources']].append(r)

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Projected 400 samples: {(total_time / n_samples) * 400 / 60:.1f} min")

    # Count how often ensemble was selected as best
    n_ensemble_best = sum(1 for r in results if r['success'] and r.get('has_ensemble', False)
                         and any(t == 'ensemble' for t in r.get('init_types', [])))
    print(f"Ensemble in final candidates: {n_ensemble_best}/{len([r for r in results if r['success']])}")

    for ns in [1, 2]:
        rs = results_by_src[ns]
        if rs:
            rmses = [r['best_rmse'] for r in rs]
            times = [r['elapsed'] for r in rs]
            print(f"\n{ns}-source ({len(rs)} samples):")
            print(f"  RMSE: mean={np.mean(rmses):.4f}, median={np.median(rmses):.4f}")
            print(f"  Time: mean={np.mean(times):.1f}s, median={np.median(times):.1f}s")

    # Compute submission score
    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results if r['success']]
    submission_score = np.mean(sample_scores) if sample_scores else 0.0

    print(f"\n{'=' * 60}")
    print(f"SUBMISSION SCORE: {submission_score:.4f}")
    print(f"{'=' * 60}")

    # MLflow logging
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("heat-signature-zero")

            run_name = f"top3_ensemble_averaging_run1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_metric("submission_score", submission_score)
                mlflow.log_metric("projected_400_samples_min", (total_time / n_samples) * 400 / 60)
                mlflow.log_metric("total_time_sec", total_time)
                mlflow.log_metric("rmse_mean_1src", np.mean([r['best_rmse'] for r in results_by_src.get(1, [])]) if results_by_src.get(1) else 0)
                mlflow.log_metric("rmse_mean_2src", np.mean([r['best_rmse'] for r in results_by_src.get(2, [])]) if results_by_src.get(2) else 0)
                mlflow.log_metric("n_samples", n_samples)
                mlflow.log_metric("n_ensemble_best", n_ensemble_best)

                mlflow.log_param("experiment_id", "EXP_TOP3_ENSEMBLE_001")
                mlflow.log_param("worker", "W1")
                mlflow.log_param("tuning_run", 1)
                mlflow.log_param("optimizer", "top3_ensemble_averaging")
                mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
                mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
                mlflow.log_param("timestep_fraction", args.timestep_fraction)
                mlflow.log_param("top_k", args.top_k)
                mlflow.log_param("add_ensemble_as_candidate", not args.no_ensemble)
                mlflow.log_param("workers", args.workers)
                mlflow.log_param("seed", args.seed)
                mlflow.log_param("platform", "wsl")

                print(f"\nMLflow run logged: {run.info.run_id}")
                return {
                    'submission_score': submission_score,
                    'total_time': total_time,
                    'projected_400_min': (total_time / n_samples) * 400 / 60,
                    'n_samples': n_samples,
                    'mlflow_id': run.info.run_id
                }
        except ImportError:
            print("MLflow not available, skipping logging")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    return {
        'submission_score': submission_score,
        'total_time': total_time,
        'projected_400_min': (total_time / n_samples) * 400 / 60,
        'n_samples': n_samples
    }


if __name__ == '__main__':
    main()
