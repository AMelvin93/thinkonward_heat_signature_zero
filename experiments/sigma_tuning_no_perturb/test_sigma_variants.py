"""
Test: Sigma variants with 2 perturbations

The tighter_sigma_range experiment showed sigma 0.15/0.19 was optimal WITH perturbations.
Let's see if it beats sigma 0.18/0.22 with 2 perturbations.
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

def run_single_test(config_name, config, samples, meta):
    n_samples = len(samples)
    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    print(f"\n=== Testing: {config_name} ===")

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"  Result: Score={score:.4f}, Time={projected_400:.1f} min")

    return {
        'config_name': config_name,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'in_budget': projected_400 <= 60
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # Base 2-perturb config
    base_config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Sigma variants to test
    configs = {
        '2pert_sigma_014_018': {**base_config, 'sigma0_1src': 0.14, 'sigma0_2src': 0.18},
        '2pert_sigma_016_020': {**base_config, 'sigma0_1src': 0.16, 'sigma0_2src': 0.20},
    }

    print("=" * 70)
    print("TEST: Sigma variants with 2 perturbations")
    print("=" * 70)
    print("Current best (2pert_018_022): 1.1425 @ 51.3 min")
    print("Testing tighter sigma variants")

    results = []
    for config_name, config in configs.items():
        result = run_single_test(config_name, config, samples, meta)
        results.append(result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Score':>8} {'Delta':>8} {'Time':>10} {'Status':>8}")
    print("-" * 70)
    print(f"{'2pert_018_022 (best)':<25} {'1.1425':>8} {'---':>8} {'51.3 min':>10} {'IN':>8}")
    print("-" * 70)

    for r in results:
        status = "IN" if r['in_budget'] else "OVER"
        delta = r['score'] - 1.1425
        print(f"{r['config_name']:<25} {r['score']:>8.4f} {delta:>+8.4f} {r['projected_400_min']:>8.1f} m {status:>8}")

    # Check for improvements
    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        if best['score'] > 1.1425:
            print(f"\n*** NEW BEST: {best['config_name']} ***")
            print(f"    Score: {best['score']:.4f} (+{best['score'] - 1.1425:.4f})")
        else:
            print(f"\n*** No improvement over 2pert_018_022 ***")

    with open('test_sigma_variants_output.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

if __name__ == "__main__":
    main()
