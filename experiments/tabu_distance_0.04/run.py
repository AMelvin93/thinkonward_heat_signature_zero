"""
Experiment: tabu_distance_0.04
Test larger tabu distance (0.04 and 0.05) vs baseline 0.03.
Larger distance may improve diversity of perturbation solutions.

IMPORTANT: Uses BOTH sigma configs (0.14/0.19 and 0.18/0.22) to validate results.
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

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"  Tabu distance: {config.get('tabu_distance')}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples}, elapsed: {elapsed/60:.1f} min")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    # Compute RMSE by source count
    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")

    in_budget = projected_400 <= 60.0
    budget_remaining = 60.0 - projected_400
    print(f"  In budget: {in_budget}, remaining: {budget_remaining:.1f} min")

    return {
        'config_name': config_name,
        'sigma0_1src': config.get('sigma0_1src'),
        'sigma0_2src': config.get('sigma0_2src'),
        'tabu_distance': config.get('tabu_distance'),
        'score': float(score),
        'avg_n_cands': float(avg_n_cands),
        'projected_400_min': float(projected_400),
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'in_budget': in_budget,
        'budget_remaining': float(budget_remaining),
    }


def main():
    data = load_data()

    BASELINE = 1.1464  # @ 51.2 min with tabu_distance=0.03

    print("="*60)
    print("TABU DISTANCE TUNING EXPERIMENT")
    print("="*60)
    print(f"Baseline: 1.1464 @ 51.2 min (tabu_distance=0.03)")
    print(f"Hypothesis: Larger tabu distance may improve diversity")

    # Base config
    base_config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'max_tabu_attempts': 10,
    }

    # Test configurations
    configs = [
        # Run 1: Queue default (sigma 0.14/0.19, tabu 0.04)
        {
            **base_config,
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'tabu_distance': 0.04,
        },
        # Run 2: Better sigma (0.18/0.22, tabu 0.04)
        {
            **base_config,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'tabu_distance': 0.04,
        },
        # Run 3: Larger tabu distance (0.18/0.22, tabu 0.05)
        {
            **base_config,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'tabu_distance': 0.05,
        },
    ]
    config_names = [
        'tabu_004_sigma_014_019',
        'tabu_004_sigma_018_022',
        'tabu_005_sigma_018_022',
    ]

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        # Save intermediate
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "="*70)
    print("TABU DISTANCE TUNING RESULTS")
    print("="*70)
    print(f"{'Config':<25} {'Sigma':>10} {'Tabu':>6} {'Score':>8} {'Proj 400':>10} {'vs Base':>8}")
    print("-"*75)
    for r in results:
        sigma_str = f"{r['sigma0_1src']}/{r['sigma0_2src']}"
        delta = r['score'] - BASELINE
        print(f"{r['config_name']:<25} {sigma_str:>10} {r['tabu_distance']:>6.2f} {r['score']:>8.4f} {r['projected_400_min']:>9.1f}m {delta:>+8.4f}")

    # Find best in-budget
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config_name']} ***")
        print(f"    Score: {best['score']:.4f} ({best['score'] - BASELINE:+.4f} vs baseline)")
        print(f"    Sigma: {best['sigma0_1src']}/{best['sigma0_2src']}")
        print(f"    Tabu distance: {best['tabu_distance']}")
        print(f"    Time: {best['projected_400_min']:.1f} min")

        if best['score'] > BASELINE:
            print(f"\n*** SUCCESS! Better than baseline! ***")
        else:
            print(f"\n*** FAILED: Did not beat baseline ***")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
