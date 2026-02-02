"""
Experiment: larger_perturbation_pool
Test more perturbations (3-4) on the new best config.

Base: sigma 0.18/0.22 + fevals 20/44 (1.1405 @ 42.3 min)
Hypothesis: More perturbations may find better local optima.
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

    print(f"\nConfig: {config_name}")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"Perturb: {config.get('n_perturbations')}, NM: {config.get('refine_maxiter')}")

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

    NEW_BEST_BASELINE = 1.1405  # Our new best config

    print(f"\nScore: {score:.4f}, Avg cands: {avg_n_cands:.2f}, Time: {projected_400:.1f} min")
    print(f"vs New Best (1.1405): {score - NEW_BEST_BASELINE:+.4f}")

    return {
        'config_name': config_name,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
    }


def main():
    data = load_data()

    NEW_BEST_BASELINE = 1.1405

    # Base config is our new best
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    results = []

    # Run 1: 3 perturbations
    config1 = {**base_config, 'n_perturbations': 3}
    result1 = run_experiment(config1, '3_perturbations', data)
    results.append(result1)

    # Run 2: 4 perturbations
    config2 = {**base_config, 'n_perturbations': 4}
    result2 = run_experiment(config2, '4_perturbations', data)
    results.append(result2)

    # Run 3: 2 perturbations (baseline verify)
    config3 = {**base_config, 'n_perturbations': 2}
    result3 = run_experiment(config3, '2_perturbations_baseline', data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Perturbation Count Tuning")
    print("="*60)
    print(f"New best baseline: 1.1405 @ 42.3 min")
    print()
    for r in results:
        delta = r['score'] - NEW_BEST_BASELINE
        in_budget = "IN BUDGET" if r['projected_400_min'] < 60 else "OVER BUDGET"
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{in_budget}]")

    # Find best
    best = max(results, key=lambda x: x['score'] if x['projected_400_min'] < 60 else 0)
    print()
    print(f"Best in-budget: {best['config_name']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
