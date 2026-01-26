"""
Run BOBYQA Polish Optimizer experiment with MLflow logging.
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

from experiments.bobyqa_polish.optimizer import BOBYQAPolishOptimizer


def process_sample(args):
    """Process a single sample."""
    sample_idx, sample, meta, q_range, polish_method, polish_maxiter = args
    
    optimizer = BOBYQAPolishOptimizer(
        polish_method=polish_method,
        polish_maxiter=polish_maxiter,
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


def run_experiment(
    polish_method='COBYLA',
    polish_maxiter=8,
    n_workers=None,
    data_path=None
):
    """Run the BOBYQA polish experiment."""
    
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
    
    print(f"[BOBYQA Polish] Running with polish_method={polish_method}, maxiter={polish_maxiter}")
    print(f"[BOBYQA Polish] Processing {len(samples)} samples with {n_workers} workers")
    
    start_time = time.time()
    
    # Process samples in parallel
    args_list = [
        (i, sample, meta, q_range, polish_method, polish_maxiter)
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
    
    # Build submission
    submission = []
    for result in results:
        submission.append({
            'sample_id': result['sample_id'],
            'estimated_sources': result['candidates'] if result['candidates'] else [[(1.0, 0.5, 1.0)]]
        })
    
    # Sort by sample_id to match ground truth order
    sample_id_to_idx = {s['sample_id']: i for i, s in enumerate(samples)}
    submission.sort(key=lambda x: sample_id_to_idx.get(x['sample_id'], 999))
    
    # Calculate score using simple formula (without ground truth)
    # Score = (1/(1+RMSE)) + 0.3 * (n_candidates/3)
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
    
    n_samples = len(samples)
    projected_400 = (total_time / n_samples) * 400 / 60  # minutes
    
    print(f"\n[BOBYQA Polish] Results:")
    print(f"  Score: {score:.4f}")
    print(f"  RMSE mean: {rmse_mean:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Projected 400 samples: {projected_400:.1f} min")
    print(f"  In budget (<60 min): {projected_400 < 60}")
    
    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'total_time': total_time,
        'projected_400_min': projected_400,
        'in_budget': projected_400 < 60,
        'results': results,
        'polish_method': polish_method,
        'polish_maxiter': polish_maxiter,
    }


def main():
    """Main entry point with MLflow logging."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--polish-method', default='COBYLA', choices=['COBYLA', 'trust-constr', 'Nelder-Mead'])
    parser.add_argument('--polish-maxiter', type=int, default=8)
    parser.add_argument('--n-workers', type=int, default=None)
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")
    
    run_name = f"bobyqa_polish_{args.polish_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("experiment_id", "EXP_BOBYQA_POLISH_001")
        mlflow.log_param("worker", "W2")
        mlflow.log_param("polish_method", args.polish_method)
        mlflow.log_param("polish_maxiter", args.polish_maxiter)
        mlflow.log_param("n_workers", args.n_workers or os.cpu_count())
        
        # Run experiment
        result = run_experiment(
            polish_method=args.polish_method,
            polish_maxiter=args.polish_maxiter,
            n_workers=args.n_workers,
        )
        
        # Log metrics
        mlflow.log_metric("submission_score", result['score'])
        mlflow.log_metric("rmse_mean", result['rmse_mean'])
        mlflow.log_metric("total_time_sec", result['total_time'])
        mlflow.log_metric("projected_400_samples_min", result['projected_400_min'])
        mlflow.log_metric("in_budget", 1 if result['in_budget'] else 0)
        
        print(f"\n[BOBYQA Polish] MLflow run ID: {run.info.run_id}")
        
        return result, run.info.run_id


if __name__ == '__main__':
    main()
