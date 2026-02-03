"""Quick timing test - 2 perturbations (safer config)."""

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

    src1_indices = [i for i, s in enumerate(samples) if s['n_sources'] == 1]
    src2_indices = [i for i, s in enumerate(samples) if s['n_sources'] == 2]

    test_indices = src1_indices[:10] + src2_indices[:10]

    # 2-perturbation config (safer, less budget risk)
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,       # SAFER: 2 perturbations
        'perturb_nm_iters': 3,      # Standard NM iters
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("2-PERTURBATION CONFIG (safer)")
    print("-" * 60)

    args_list = [(idx, samples[idx], meta, config) for idx in test_indices]

    start = time.time()
    results = []
    times_1src = []
    times_2src = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            src_type = "1src" if result['n_sources'] == 1 else "2src"
            if result['n_sources'] == 1:
                times_1src.append(result['time_s'])
            else:
                times_2src.append(result['time_s'])
            print(f"  Sample {result['idx']} ({src_type}): rmse={result['rmse']:.4f}, time={result['time_s']:.1f}s")

    elapsed = time.time() - start

    n_1src_full = 32
    n_2src_full = 48
    avg_time_1src = np.mean(times_1src) if times_1src else 0
    avg_time_2src = np.mean(times_2src) if times_2src else 0

    projected_80 = (avg_time_1src * n_1src_full + avg_time_2src * n_2src_full) / MAX_WORKERS
    projected_400 = projected_80 * 5 / 60

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    avg_score = np.mean(scores) if scores else 0

    print("-" * 60)
    print(f"20 samples completed in {elapsed:.1f}s wall time")
    print(f"Average 1-source time: {avg_time_1src:.1f}s")
    print(f"Average 2-source time: {avg_time_2src:.1f}s")
    print(f"\nProjected for full 80 samples:")
    print(f"  With 7 workers: {projected_80:.1f}s = {projected_80/60:.1f} min")
    print(f"  Projected 400 samples: {projected_400:.1f} min")
    print(f"  Average score: {avg_score:.4f}")

    if projected_400 <= 55:
        print("\n  ✓ SAFELY IN BUDGET (<=55 min)")
    elif projected_400 <= 60:
        print("\n  ~ MARGINALLY IN BUDGET (55-60 min)")
    else:
        print("\n  ✗ OVER BUDGET (>60 min)")

if __name__ == "__main__":
    main()
