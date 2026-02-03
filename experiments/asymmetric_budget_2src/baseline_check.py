"""
Quick baseline verification to check if timing is consistent.
"""

import os
import sys
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

sys.path.insert(0, os.path.join(_project_root, 'experiments', 'tighter_sigma_range'))
from optimizer import TabuBasinHoppingOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def calculate_sample_score(rmse, n_candidates=3, lambda_=0.3, n_max=3):
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def process_sample(args):
    sample_idx, sample, meta, config = args
    optimizer = TabuBasinHoppingOptimizer(**config)
    try:
        start = time.time()
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        n_candidates = len(candidates) if candidates else 0
        return {
            'idx': sample_idx,
            'rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': n_candidates,
            'time_s': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'idx': sample_idx,
            'rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'time_s': 0,
            'success': False,
        }


def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    # Exact baseline config
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,
        'perturb_nm_iters': 2,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("BASELINE CHECK: 20/44 fevals (4pert_nm2)")
    print("="*60)
    print(f"Expected: 1.1482 @ 51.7 min")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{n_samples}")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    time_1src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 1]
    time_2src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"\nResult: Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"  Time 1src: {np.mean(time_1src):.2f}s, Time 2src: {np.mean(time_2src):.2f}s")
    print(f"  Candidates: {avg_cands:.2f}")

    print(f"\nExpected vs Actual:")
    print(f"  Score: 1.1482 vs {score:.4f} (delta: {score - 1.1482:+.4f})")
    print(f"  Time: 51.7 vs {projected_400:.1f} (delta: {projected_400 - 51.7:+.1f} min)")

    if projected_400 > 60:
        print(f"\n*** WARNING: OVER BUDGET even for baseline! ***")
    elif abs(projected_400 - 51.7) > 5:
        print(f"\n*** WARNING: Timing variance > 5 min from expected! ***")


if __name__ == '__main__':
    main()
