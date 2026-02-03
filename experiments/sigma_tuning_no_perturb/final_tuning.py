"""
Final tuning: Try to close the gap to Top 10

Current best (nm4_scale05): 1.1487 mean, 1.1525 best
Top 10 threshold: 1.1585
Gap: 0.0098 mean, 0.0060 best

Try:
1. nm5_scale05 - more polish iterations
2. nm4_scale06 - combination of nm4 + larger scale
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

    # Base config
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Configurations to test
    configs = {
        'nm5_scale05': {**base_config, 'perturb_nm_iters': 5, 'perturbation_scale': 0.05},
        'nm4_scale06': {**base_config, 'perturb_nm_iters': 4, 'perturbation_scale': 0.06},
    }

    print("=" * 70)
    print("FINAL TUNING: Closing the gap to Top 10")
    print("=" * 70)
    print("Current best (nm4_scale05): Mean=1.1487, Best=1.1525")
    print("Top 10 threshold: 1.1585")
    print("Gap: 0.0098 mean, 0.0060 best")

    results = []
    for config_name, config in configs.items():
        result = run_single_test(config_name, config, samples, meta)
        results.append(result)

        if not result['in_budget']:
            print(f"  *** OVER BUDGET ***")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Score':>8} {'Delta':>8} {'Time':>10} {'Status':>8}")
    print("-" * 70)
    print(f"{'nm4_scale05 (best)':<20} {'1.1487':>8} {'---':>8} {'54.2 min':>10} {'IN':>8}")
    print("-" * 70)

    for r in results:
        status = "IN" if r['in_budget'] else "OVER"
        delta = r['score'] - 1.1487
        gap_to_top10 = 1.1585 - r['score']
        print(f"{r['config_name']:<20} {r['score']:>8.4f} {delta:>+8.4f} {r['projected_400_min']:>8.1f} m {status:>8} (gap: {gap_to_top10:.4f})")

    # Check for improvements
    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        if best['score'] > 1.1487:
            print(f"\n*** NEW BEST: {best['config_name']} ***")
            print(f"    Score: {best['score']:.4f}")
            print(f"    Improvement: +{best['score'] - 1.1487:.4f}")
            print(f"    Gap to Top 10: {1.1585 - best['score']:.4f}")
            print(f"    NEEDS VALIDATION!")
        else:
            print(f"\n*** nm4_scale05 remains the best ***")

    with open('final_tuning_output.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

if __name__ == "__main__":
    main()
