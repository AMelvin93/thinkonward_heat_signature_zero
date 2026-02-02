"""
Experiment: perturb_nm_4iters
Test 4 NM iterations per perturbation (vs 3 baseline).

Note: Will test both with specified sigma (0.14/0.19) and optimal sigma (0.18/0.22).
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


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== {config_name} ===")
    print(f"  sigma: {config['sigma0_1src']}/{config['sigma0_2src']}, perturb_nm_iters: {config['perturb_nm_iters']}")

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

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    BASELINE = 1.1464

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"vs Baseline (1.1464): {score - BASELINE:+.4f}")
    print(f"Budget remaining: {60.0 - projected_400:.1f} min")

    return {
        'config_name': config_name,
        'sigma0_1src': config['sigma0_1src'],
        'sigma0_2src': config['sigma0_2src'],
        'perturb_nm_iters': config['perturb_nm_iters'],
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'budget_remaining_min': 60.0 - projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.1464

    # Base config
    base_config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,  # Use optimal scale
    }

    results = []

    # Run 1: Specified config (sigma 0.14/0.19, nm_iters=4)
    config1 = {**base_config,
               'sigma0_1src': 0.14,
               'sigma0_2src': 0.19,
               'perturb_nm_iters': 4}
    result1 = run_experiment(config1, "Run1_sigma014_nm4", data)
    results.append(result1)

    # Run 2: Better sigma (0.18/0.22) with nm_iters=4
    config2 = {**base_config,
               'sigma0_1src': 0.18,
               'sigma0_2src': 0.22,
               'perturb_nm_iters': 4}
    result2 = run_experiment(config2, "Run2_sigma018_nm4", data)
    results.append(result2)

    # Run 3: Better sigma with nm_iters=5
    config3 = {**base_config,
               'sigma0_1src': 0.18,
               'sigma0_2src': 0.22,
               'perturb_nm_iters': 5}
    result3 = run_experiment(config3, "Run3_sigma018_nm5", data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Perturbation NM Iterations Tuning")
    print("="*60)
    print(f"Baseline (nm_iters=3, sigma 0.18/0.22): 1.1464 @ 51.2 min")
    print()
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    best = max(results, key=lambda x: x['score'])
    print()
    print(f"Best: {best['config_name']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    if best['score'] > BASELINE:
        print(f"\nNEW BEST FOUND! +{best['score'] - BASELINE:.4f}")
    else:
        print(f"\nNo improvement over baseline ({best['score'] - BASELINE:+.4f})")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
