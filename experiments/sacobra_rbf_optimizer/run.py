#!/usr/bin/env python3
"""Run script for SACOBRA RBF Surrogate Optimizer."""

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

from optimizer import SACOBRAOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args
    optimizer = SACOBRAOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        initial_samples=config.get('initial_samples', 8),
        rbf_function=config.get('rbf_function', 'multiquadric'),
        candidate_pool_size=config.get('candidate_pool_size', 10),
        nx_coarse=config.get('nx_coarse', 50),
        ny_coarse=config.get('ny_coarse', 25),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        timestep_fraction=config.get('timestep_fraction', 0.40),
        final_polish_maxiter=config.get('final_polish_maxiter', 5),
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
    """Calculate score for a single sample using the scoring formula."""
    if n_candidates == 0:
        return 0.0
    # Score = 1/(1+RMSE) + lambda * (n_candidates/n_max)
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def main():
    parser = argparse.ArgumentParser(description='Run SACOBRA RBF Optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--initial-samples', type=int, default=8)
    parser.add_argument('--rbf-function', type=str, default='multiquadric')
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--final-polish-maxiter', type=int, default=5)
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

    print(f"\nSACOBRA RBF Surrogate Optimizer")
    print(f"=" * 60)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Initial samples: {args.initial_samples}, RBF: {args.rbf_function}")
    print(f"Timestep fraction: {args.timestep_fraction:.0%}")
    print(f"Final polish: {args.final_polish_maxiter} NM iters")
    print(f"=" * 60)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'initial_samples': args.initial_samples,
        'rbf_function': args.rbf_function,
        'timestep_fraction': args.timestep_fraction,
        'final_polish_maxiter': args.final_polish_maxiter,
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
                print(f"  [{idx:3d}] {result['n_sources']}-src: RMSE={result['best_rmse']:.4f}, "
                      f"{result['n_candidates']} cands, {result['n_sims']} sims, "
                      f"{result['elapsed']:.1f}s")
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

            run_name = f"sacobra_rbf_run1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_metric("submission_score", submission_score)
                mlflow.log_metric("projected_400_samples_min", (total_time / n_samples) * 400 / 60)
                mlflow.log_metric("total_time_sec", total_time)
                mlflow.log_metric("rmse_mean_1src", np.mean([r['best_rmse'] for r in results_by_src.get(1, [])]) if results_by_src.get(1) else 0)
                mlflow.log_metric("rmse_mean_2src", np.mean([r['best_rmse'] for r in results_by_src.get(2, [])]) if results_by_src.get(2) else 0)
                mlflow.log_metric("n_samples", n_samples)

                mlflow.log_param("experiment_id", "EXP_SACOBRA_RBF_001")
                mlflow.log_param("worker", "W1")
                mlflow.log_param("tuning_run", 1)
                mlflow.log_param("optimizer", "sacobra_rbf")
                mlflow.log_param("max_fevals_1src", args.max_fevals_1src)
                mlflow.log_param("max_fevals_2src", args.max_fevals_2src)
                mlflow.log_param("initial_samples", args.initial_samples)
                mlflow.log_param("rbf_function", args.rbf_function)
                mlflow.log_param("timestep_fraction", args.timestep_fraction)
                mlflow.log_param("final_polish_maxiter", args.final_polish_maxiter)
                mlflow.log_param("workers", args.workers)
                mlflow.log_param("seed", args.seed)
                mlflow.log_param("platform", "wsl")

                print(f"\nMLflow run logged: {run.info.run_id}")
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
