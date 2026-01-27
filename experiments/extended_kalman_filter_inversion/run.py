"""
Run Extended Kalman Filter experiment with MLflow logging.
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

from experiments.extended_kalman_filter_inversion.optimizer import ExtendedKalmanFilterOptimizer


def process_sample(args):
    """Process a single sample."""
    sample_idx, sample, meta, q_range, config = args

    optimizer = ExtendedKalmanFilterOptimizer(
        n_timesteps_ekf=config.get('n_timesteps_ekf', 20),
        process_noise_std=config.get('process_noise_std', 0.01),
        measurement_noise_std=config.get('measurement_noise_std', 0.1),
        jacobian_eps=config.get('jacobian_eps', 0.01),
        nm_polish_iters=config.get('nm_polish_iters', 8),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=q_range
        )
        elapsed = time.time() - start
        return {
            'sample_id': sample['sample_id'],
            'candidates': candidates,
            'best_rmse': best_rmse,
            'time': elapsed,
            'n_sims': n_sims,
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
            'success': False,
            'error': str(e)
        }


def run_experiment(config=None, n_workers=None, data_path=None, n_samples=None):
    """Run the EKF experiment."""

    if config is None:
        config = {}

    if n_workers is None:
        n_workers = os.cpu_count()

    if data_path is None:
        data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')

    # Load data
    with open(data_path, 'rb') as f:
        test_dataset = pickle.load(f)

    samples = test_dataset['samples']
    meta = test_dataset['meta']
    q_range = meta['q_range']

    if n_samples is not None:
        samples = samples[:n_samples]

    print(f"[EKF] Running Extended Kalman Filter experiment")
    print(f"[EKF] Config: {config}")
    print(f"[EKF] Processing {len(samples)} samples with {n_workers} workers")

    start_time = time.time()

    # Process samples in parallel
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

    # Calculate metrics
    rmses = [r['best_rmse'] for r in results if r['success'] and r['best_rmse'] < float('inf')]
    rmse_mean = np.mean(rmses) if rmses else float('inf')

    avg_sims = np.mean([r['n_sims'] for r in results])

    n_samples_run = len(samples)
    projected_400 = (total_time / n_samples_run) * 400 / 60  # minutes

    print(f"\n[EKF] Results:")
    print(f"  Score: {score:.4f}")
    print(f"  RMSE mean: {rmse_mean:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Projected 400 samples: {projected_400:.1f} min")
    print(f"  In budget (<60 min): {projected_400 < 60}")
    print(f"  Avg simulations per sample: {avg_sims:.1f}")

    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'total_time': total_time,
        'projected_400_min': projected_400,
        'in_budget': projected_400 < 60,
        'results': results,
        'config': config,
        'avg_sims': avg_sims,
    }


def main():
    """Main entry point with MLflow logging."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-timesteps-ekf', type=int, default=20)
    parser.add_argument('--process-noise-std', type=float, default=0.01)
    parser.add_argument('--measurement-noise-std', type=float, default=0.1)
    parser.add_argument('--jacobian-eps', type=float, default=0.01)
    parser.add_argument('--nm-polish-iters', type=int, default=8)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples to run (for quick test)')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    args = parser.parse_args()

    config = {
        'n_timesteps_ekf': args.n_timesteps_ekf,
        'process_noise_std': args.process_noise_std,
        'measurement_noise_std': args.measurement_noise_std,
        'jacobian_eps': args.jacobian_eps,
        'nm_polish_iters': args.nm_polish_iters,
    }

    if args.no_mlflow:
        result = run_experiment(config=config, n_workers=args.n_workers, n_samples=args.n_samples)
        return result, None

    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")

    run_name = f"ekf_{args.n_timesteps_ekf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("experiment_id", "EXP_KALMAN_ESTIMATION_001")
        mlflow.log_param("worker", "W2")
        for k, v in config.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_workers", args.n_workers or os.cpu_count())

        # Run experiment
        result = run_experiment(config=config, n_workers=args.n_workers, n_samples=args.n_samples)

        # Log metrics
        mlflow.log_metric("submission_score", result['score'])
        mlflow.log_metric("rmse_mean", result['rmse_mean'])
        mlflow.log_metric("total_time_sec", result['total_time'])
        mlflow.log_metric("projected_400_samples_min", result['projected_400_min'])
        mlflow.log_metric("in_budget", 1 if result['in_budget'] else 0)
        mlflow.log_metric("avg_sims", result['avg_sims'])

        print(f"\n[EKF] MLflow run ID: {run.info.run_id}")

        return result, run.info.run_id


if __name__ == '__main__':
    main()
