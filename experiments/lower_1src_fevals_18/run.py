"""
Experiment: lower_1src_fevals_18
Test fevals 18/44 vs 20/44 - 1-source may need even fewer evals to preserve diversity.

Base: sigma 0.18/0.22 + fevals 20/44 + 4 perturbations = 1.1437 @ 42.2 min
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
    print(f"Fevals: 1src={config.get('max_fevals_1src')}, 2src={config.get('max_fevals_2src')}")

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

    # Breakdown by source count
    one_src = [r for r in results if r['success'] and r['n_sources'] == 1]
    two_src = [r for r in results if r['success'] and r['n_sources'] == 2]
    cands_1src = np.mean([r['n_candidates'] for r in one_src]) if one_src else 0
    cands_2src = np.mean([r['n_candidates'] for r in two_src]) if two_src else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    NEW_BEST = 1.1437

    print(f"\nScore: {score:.4f}, Avg cands: {avg_n_cands:.2f}, Time: {projected_400:.1f} min")
    print(f"1-src cands: {cands_1src:.2f}, 2-src cands: {cands_2src:.2f}")
    print(f"vs Current Best (1.1437): {score - NEW_BEST:+.4f}")

    return {
        'config_name': config_name,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'cands_1src': cands_1src,
        'cands_2src': cands_2src,
        'projected_400_min': projected_400,
    }


def main():
    data = load_data()

    NEW_BEST = 1.1437

    # Base config is current best
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 4,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    results = []

    # Run 1: 18 1src fevals
    config1 = {**base_config, 'max_fevals_1src': 18}
    result1 = run_experiment(config1, 'fevals_18_44', data)
    results.append(result1)

    # Run 2: 16 1src fevals (more aggressive)
    config2 = {**base_config, 'max_fevals_1src': 16}
    result2 = run_experiment(config2, 'fevals_16_44', data)
    results.append(result2)

    # Run 3: 20 1src fevals (baseline verify)
    config3 = {**base_config, 'max_fevals_1src': 20}
    result3 = run_experiment(config3, 'fevals_20_44_baseline', data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Lower 1-src Fevals Tuning")
    print("="*60)
    print(f"Current best: 1.1437 @ 42.2 min (fevals 20/44)")
    print()
    for r in results:
        delta = r['score'] - NEW_BEST
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f})")
        print(f"    1-src cands: {r['cands_1src']:.2f}, 2-src cands: {r['cands_2src']:.2f}")

    # Find best
    best = max(results, key=lambda x: x['score'] if x['projected_400_min'] < 60 else 0)
    print()
    print(f"Best: {best['config_name']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
