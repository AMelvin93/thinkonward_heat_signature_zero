"""Quick timing test - 20 samples only to estimate performance."""

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
    samples = data['samples'][:20]  # Only 20 samples
    meta = data['meta']

    # Test config
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

    print("Quick timing test: 20 samples, 4-pert + nm2")
    print("-" * 50)

    args_list = [(i, samples[i], meta, config) for i in range(20)]

    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  Sample {result['idx']}: rmse={result['rmse']:.4f}, time={result['time_s']:.1f}s")

    elapsed = time.time() - start

    # Calculate projected time
    projected_80 = (elapsed / 20) * 80
    projected_400 = (elapsed / 20) * 400 / 60

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    avg_score = np.mean(scores) if scores else 0

    print("-" * 50)
    print(f"20 samples completed in {elapsed:.1f}s")
    print(f"Average per sample: {elapsed/20:.1f}s")
    print(f"Projected 80 samples: {projected_80:.1f}s = {projected_80/60:.1f} min")
    print(f"Projected 400 samples: {projected_400:.1f} min")
    print(f"Average score: {avg_score:.4f}")

    # Check n_sources distribution
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = sum(1 for s in samples if s['n_sources'] == 2)
    print(f"\nSample distribution: {n_1src} 1-source, {n_2src} 2-source")

if __name__ == "__main__":
    main()
