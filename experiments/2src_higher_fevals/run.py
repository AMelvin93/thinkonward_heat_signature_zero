"""
Experiment: 2src_higher_fevals
Test if 2-source problems benefit from even more CMA-ES evaluations.

2-source RMSE is ~0.19 (the bottleneck) vs 1-source ~0.13.
More exploration might help.
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
    print(f"  fevals 1src/2src: {config['max_fevals_1src']}/{config['max_fevals_2src']}")

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

    # Compute RMSE by source count
    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    BASELINE = 1.143

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"vs Baseline (1.143): {score - BASELINE:+.4f}")

    return {
        'config_name': config_name,
        'max_fevals_2src': config['max_fevals_2src'],
        'score': score,
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.143

    # Base config
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
    }

    print("="*60)
    print("2-SOURCE HIGHER FEVALS TEST")
    print("="*60)
    print(f"Baseline: fevals 20/44, score ~1.143")
    print(f"Goal: Improve 2-src RMSE (currently ~0.19)")

    results = []

    # Test 1: fevals 44 (baseline)
    config1 = {**base_config, 'max_fevals_2src': 44}
    result1 = run_experiment(config1, "fevals_44", data)
    results.append(result1)

    # Test 2: fevals 52
    config2 = {**base_config, 'max_fevals_2src': 52}
    result2 = run_experiment(config2, "fevals_52", data)
    results.append(result2)

    # Test 3: fevals 60
    config3 = {**base_config, 'max_fevals_2src': 60}
    result3 = run_experiment(config3, "fevals_60", data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: 2-Source Higher Fevals")
    print("="*60)
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  fevals={r['max_fevals_2src']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min")
        print(f"    RMSE 1src: {r['rmse_1src']:.4f}, 2src: {r['rmse_2src']:.4f} ({delta:+.4f}) [{status}]")

    best = max(results, key=lambda x: x['score'])
    print()
    print(f"Best: fevals={best['max_fevals_2src']} = {best['score']:.4f}")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
