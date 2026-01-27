"""
Run Ensemble Weighted Solution experiment with MLflow logging.
"""

import os
import sys
import time
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from experiments.ensemble_weighted_solution.optimizer import EnsembleWeightedOptimizer


def process_sample(args):
    """Process a single sample."""
    sample_idx, sample, meta, q_range, config = args

    optimizer = EnsembleWeightedOptimizer(
        ensemble_top_n=config.get('ensemble_top_n', 5),
        polish_maxiter=config.get('polish_maxiter', 8),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=q_range
        )

        # Track if ensemble solution won
        ensemble_won = any(r.init_type == 'ensemble' or r.init_type == 'polished'
                          for r in results if r.rmse == best_rmse)

        elapsed = time.time() - start
        return {
            'sample_id': sample['sample_id'],
            'candidates': candidates,
            'best_rmse': best_rmse,
            'time': elapsed,
            'n_sims': n_sims,
            'n_sources': sample['n_sources'],
            'ensemble_won': ensemble_won,
            'success': True
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'sample_id': sample['sample_id'],
            'candidates': [],
            'best_rmse': float('inf'),
            'time': elapsed,
            'n_sims': 0,
            'n_sources': sample['n_sources'],
            'ensemble_won': False,
            'success': False,
            'error': str(e)
        }


def run_experiment(config=None, n_workers=None, data_path=None, n_samples=None):
    """Run the ensemble experiment."""

    if config is None:
        config = {'ensemble_top_n': 5}

    if n_workers is None:
        n_workers = os.cpu_count()

    if data_path is None:
        data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')

    with open(data_path, 'rb') as f:
        test_dataset = pickle.load(f)

    samples = test_dataset['samples']
    meta = test_dataset['meta']
    q_range = meta['q_range']

    if n_samples is not None:
        samples = samples[:n_samples]

    print(f"[Ensemble] Running with ensemble_top_n={config.get('ensemble_top_n', 5)}")
    print(f"[Ensemble] Processing {len(samples)} samples with {n_workers} workers")

    start_time = time.time()

    args_list = [
        (i, sample, meta, q_range, config)
        for i, sample in enumerate(samples)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                if len(results) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {len(results)}/{len(samples)} samples ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  Sample {idx} failed: {e}")
                results.append({
                    'sample_id': samples[idx]['sample_id'],
                    'candidates': [],
                    'best_rmse': float('inf'),
                    'time': 0,
                    'n_sims': 0,
                    'n_sources': samples[idx]['n_sources'],
                    'ensemble_won': False,
                    'success': False,
                    'error': str(e)
                })

    total_time = time.time() - start_time

    # Calculate score
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0 or rmse == float('inf'):
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = []
    for result in results:
        n_cands = len(result['candidates']) if result['candidates'] else 0
        sample_scores.append(calculate_sample_score(result['best_rmse'], n_cands))

    score = np.mean(sample_scores) if sample_scores else 0.0

    rmses = [r['best_rmse'] for r in results if r['success'] and r['best_rmse'] < float('inf')]
    rmse_mean = np.mean(rmses) if rmses else float('inf')

    n_samples_run = len(samples)
    projected_400 = (total_time / n_samples_run) * 400 / 60

    # Ensemble statistics
    ensemble_wins = sum(1 for r in results if r.get('ensemble_won', False))
    ensemble_pct = 100 * ensemble_wins / len(results) if results else 0

    print(f"\n[Ensemble] Results:")
    print(f"  Score: {score:.4f}")
    print(f"  RMSE mean: {rmse_mean:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Projected 400 samples: {projected_400:.1f} min")
    print(f"  In budget (<60 min): {projected_400 < 60}")
    print(f"  Ensemble wins: {ensemble_wins}/{len(results)} ({ensemble_pct:.1f}%)")

    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'total_time': total_time,
        'projected_400_min': projected_400,
        'in_budget': projected_400 < 60,
        'results': results,
        'config': config,
        'ensemble_wins': ensemble_wins,
        'ensemble_pct': ensemble_pct,
    }


def main():
    """Main entry point with MLflow logging."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble-top-n', type=int, default=5)
    parser.add_argument('--polish-maxiter', type=int, default=8)
    parser.add_argument('--refine-maxiter', type=int, default=3)
    parser.add_argument('--refine-top-n', type=int, default=2)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--no-mlflow', action='store_true')
    args = parser.parse_args()

    config = {
        'ensemble_top_n': args.ensemble_top_n,
        'polish_maxiter': args.polish_maxiter,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': args.refine_top_n,
    }

    if args.no_mlflow:
        result = run_experiment(config=config, n_workers=args.n_workers, n_samples=args.n_samples)
        return result, None

    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"ensemble_{args.ensemble_top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("experiment_id", "EXP_ENSEMBLE_SOLUTION_001")
        mlflow.log_param("worker", "W2")
        mlflow.log_param("ensemble_top_n", args.ensemble_top_n)
        mlflow.log_param("n_workers", args.n_workers or os.cpu_count())

        result = run_experiment(config=config, n_workers=args.n_workers, n_samples=args.n_samples)

        mlflow.log_metric("submission_score", result['score'])
        mlflow.log_metric("rmse_mean", result['rmse_mean'])
        mlflow.log_metric("total_time_sec", result['total_time'])
        mlflow.log_metric("projected_400_samples_min", result['projected_400_min'])
        mlflow.log_metric("in_budget", 1 if result['in_budget'] else 0)
        mlflow.log_metric("ensemble_wins", result['ensemble_wins'])
        mlflow.log_metric("ensemble_win_pct", result['ensemble_pct'])

        print(f"\n[Ensemble] MLflow run ID: {run.info.run_id}")

        return result, run.info.run_id


if __name__ == '__main__':
    main()
