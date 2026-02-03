"""
Experiment: sigma_tuning_no_perturb

Test tighter sigma values WITHOUT perturbation to find optimal config
that fits within budget on this slower system.

Previous findings (from tighter_sigma_range):
- sigma_016_020_no_perturb: 1.1649 @ 45.0m
- sigma_017_021_no_perturb: 1.1570 @ 45.0m
- sigma_018_022_no_perturb: 1.1607 @ 43.3m

Current validated (on this system):
- sigma 0.18/0.22 no_perturb: 1.1386 @ 49.3m

Hypothesis: Tighter sigma (0.16/0.20 or 0.15/0.19) may improve score
without adding timing overhead.
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

def run_full_test(config_name, config, samples, meta):
    n_samples = len(samples)
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
                print(f"    Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    return {
        'config_name': config_name,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'elapsed_sec': float(elapsed_time),
        'rmse_1src': float(np.mean(rmse_1src)) if rmse_1src else 0,
        'rmse_2src': float(np.mean(rmse_2src)) if rmse_2src else 0,
        'in_budget': projected_400 <= 60
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # Base config - NO PERTURBATION
    base_config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,  # NO PERTURBATIONS
    }

    # Sigma configs to test
    sigma_configs = [
        ('sigma_016_020', 0.16, 0.20),
        ('sigma_015_019', 0.15, 0.19),
        ('sigma_017_021', 0.17, 0.21),
    ]

    print("=" * 70)
    print("EXPERIMENT: sigma_tuning_no_perturb")
    print("=" * 70)
    print("Testing tighter sigma values WITHOUT perturbation")
    print("\nBaseline: sigma 0.18/0.22 no_perturb = 1.1386 @ 49.3 min")

    results = []
    for config_name, sigma_1src, sigma_2src in sigma_configs:
        config = {**base_config, 'sigma0_1src': sigma_1src, 'sigma0_2src': sigma_2src}

        print(f"\n--- Testing: {config_name} ---")
        print(f"    sigma0_1src={sigma_1src}, sigma0_2src={sigma_2src}")

        result = run_full_test(config_name, config, samples, meta)
        results.append(result)

        status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
        delta = result['score'] - 1.1386
        print(f"    Result: Score={result['score']:.4f} ({delta:+.4f}), Time={result['projected_400_min']:.1f} min [{status}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Score':>8} {'Delta':>8} {'Time':>10} {'Status':>10}")
    print("-" * 70)
    print(f"{'baseline (0.18/0.22)':<20} {'1.1386':>8} {'---':>8} {'49.3 min':>10} {'IN':>10}")
    print("-" * 70)

    for r in results:
        status = "IN" if r['in_budget'] else "OVER"
        delta = r['score'] - 1.1386
        print(f"{r['config_name']:<20} {r['score']:>8.4f} {delta:>+8.4f} {r['projected_400_min']:>8.1f} m {status:>10}")

    # Find best in-budget config
    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config_name']} ***")
        print(f"    Score: {best['score']:.4f}, Time: {best['projected_400_min']:.1f} min")
        print(f"    Improvement over baseline: {best['score'] - 1.1386:+.4f}")
        print(f"    Gap to Top 10 (1.1585): {1.1585 - best['score']:.4f}")

    with open('run_output.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f"\nResults saved to run_output.json")

if __name__ == "__main__":
    main()
