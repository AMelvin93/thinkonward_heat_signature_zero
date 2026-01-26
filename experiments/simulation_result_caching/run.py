"""
Run Simulation Result Caching Optimizer experiment with MLflow logging.
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

from experiments.simulation_result_caching.optimizer import CachingOptimizer


def process_sample(args):
    """Process a single sample."""
    sample_idx, sample, meta, q_range, cache_enabled, cache_precision = args
    
    optimizer = CachingOptimizer(
        cache_enabled=cache_enabled,
        cache_precision=cache_precision,
    )
    
    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=q_range
        )
        elapsed = time.time() - start
        cache_stats = optimizer.get_cache_stats()
        return {
            'sample_id': sample['sample_id'],
            'candidates': candidates,
            'best_rmse': best_rmse,
            'time': elapsed,
            'n_sims': n_sims,
            'success': True,
            'cache_stats': cache_stats
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
            'error': str(e),
            'cache_stats': {'total_hits': 0, 'total_misses': 0, 'hit_rate': 0}
        }


def run_experiment(
    cache_enabled=True,
    cache_precision=3,
    n_workers=None,
    data_path=None
):
    """Run the caching experiment."""
    
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
    
    print(f"[Caching] Running with cache_enabled={cache_enabled}, precision={cache_precision}")
    print(f"[Caching] Processing {len(samples)} samples with {n_workers} workers")
    
    start_time = time.time()
    
    # Process samples in parallel
    args_list = [
        (i, sample, meta, q_range, cache_enabled, cache_precision)
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
                    'error': str(e),
                    'cache_stats': {'total_hits': 0, 'total_misses': 0, 'hit_rate': 0}
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
    
    # Aggregate cache stats
    total_hits = sum(r['cache_stats']['total_hits'] for r in results)
    total_misses = sum(r['cache_stats']['total_misses'] for r in results)
    overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    
    n_samples = len(samples)
    projected_400 = (total_time / n_samples) * 400 / 60  # minutes
    
    print(f"\n[Caching] Results:")
    print(f"  Score: {score:.4f}")
    print(f"  RMSE mean: {rmse_mean:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Projected 400 samples: {projected_400:.1f} min")
    print(f"  In budget (<60 min): {projected_400 < 60}")
    print(f"\n[Caching] Cache Statistics:")
    print(f"  Total hits: {total_hits}")
    print(f"  Total misses: {total_misses}")
    print(f"  Hit rate: {overall_hit_rate:.2%}")
    
    return {
        'score': score,
        'rmse_mean': rmse_mean,
        'total_time': total_time,
        'projected_400_min': projected_400,
        'in_budget': projected_400 < 60,
        'results': results,
        'cache_enabled': cache_enabled,
        'cache_precision': cache_precision,
        'cache_hits': total_hits,
        'cache_misses': total_misses,
        'cache_hit_rate': overall_hit_rate,
    }


def main():
    """Main entry point with MLflow logging."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-enabled', action='store_true', default=True)
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for comparison')
    parser.add_argument('--cache-precision', type=int, default=3)
    parser.add_argument('--n-workers', type=int, default=None)
    args = parser.parse_args()
    
    cache_enabled = not args.no_cache
    
    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")
    
    cache_str = "enabled" if cache_enabled else "disabled"
    run_name = f"caching_{cache_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("experiment_id", "EXP_SIMULATOR_CACHING_001")
        mlflow.log_param("worker", "W2")
        mlflow.log_param("cache_enabled", cache_enabled)
        mlflow.log_param("cache_precision", args.cache_precision)
        mlflow.log_param("n_workers", args.n_workers or os.cpu_count())
        
        # Run experiment
        result = run_experiment(
            cache_enabled=cache_enabled,
            cache_precision=args.cache_precision,
            n_workers=args.n_workers,
        )
        
        # Log metrics
        mlflow.log_metric("submission_score", result['score'])
        mlflow.log_metric("rmse_mean", result['rmse_mean'])
        mlflow.log_metric("total_time_sec", result['total_time'])
        mlflow.log_metric("projected_400_samples_min", result['projected_400_min'])
        mlflow.log_metric("in_budget", 1 if result['in_budget'] else 0)
        mlflow.log_metric("cache_hits", result['cache_hits'])
        mlflow.log_metric("cache_misses", result['cache_misses'])
        mlflow.log_metric("cache_hit_rate", result['cache_hit_rate'])
        
        print(f"\n[Caching] MLflow run ID: {run.info.run_id}")
        
        return result, run.info.run_id


if __name__ == '__main__':
    main()
