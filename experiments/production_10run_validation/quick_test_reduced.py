"""Test reduced config that should be safely in budget."""

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

def run_test(config_name, config, samples, meta):
    n = 20  # Quick test
    test_samples = list(range(n))

    args_list = [(idx, samples[idx], meta, config) for idx in test_samples]

    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    elapsed = time.time() - start

    # Calculate projection (20 samples = 25% of 80)
    projected_80 = elapsed * 4
    projected_400 = projected_80 * 5 / 60

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    avg_score = np.mean(scores) if scores else 0

    return {
        'name': config_name,
        'time_20': elapsed,
        'projected_400': projected_400,
        'score': avg_score
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    configs = {
        'reduced_2pert_nm2': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturb_nm_iters': 2,  # Reduced
            'perturbation_scale': 0.05,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
        'no_perturb': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': False,  # No perturbations
        },
        'reduced_fevals': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 16,  # Reduced
            'max_fevals_2src': 36,  # Reduced
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturb_nm_iters': 3,
            'perturbation_scale': 0.05,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    }

    print("="*70)
    print("TESTING REDUCED CONFIGS (20-sample quick tests)")
    print("="*70)

    results = []
    for name, config in configs.items():
        print(f"\nTesting: {name}")
        result = run_test(name, config, samples, meta)
        print(f"  Score: {result['score']:.4f}")
        print(f"  Projected 400: {result['projected_400']:.1f} min")
        status = "IN BUDGET" if result['projected_400'] <= 60 else "OVER BUDGET"
        print(f"  Status: {status}")
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<25} {'Score':>8} {'Proj 400':>10} {'Status':>15}")
    print("-"*70)
    for r in results:
        status = "✓ IN" if r['projected_400'] <= 60 else "✗ OVER"
        print(f"{r['name']:<25} {r['score']:>8.4f} {r['projected_400']:>10.1f} {status:>15}")

if __name__ == "__main__":
    main()
