"""
Run script for asymmetric_013_019 experiment.

Tests even tighter sigma for 1-source problems (0.13 vs 0.14).
Uses optimal perturbation scale (0.06) from prior findings.

Hypothesis: If 0.14 helped 1-source, 0.13 might help even more.
"""

import os
import sys
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import TabuBasinHoppingOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def process_sample(args):
    sample_idx, sample, meta, config = args
    optimizer = TabuBasinHoppingOptimizer(**config)
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        return sample_idx, best_rmse, n_sims, sample['n_sources'], None
    except Exception as e:
        return sample_idx, float('inf'), 0, sample['n_sources'], str(e)


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"Perturbation scale: {config.get('perturbation_scale')}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    rmses = {}
    source_counts = {}
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            sample_idx, best_rmse, n_sims, n_sources, error = future.result()
            rmses[sample_idx] = best_rmse
            source_counts[sample_idx] = n_sources

            if error:
                errors.append((sample_idx, error))

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{n_samples} samples, elapsed: {elapsed/60:.1f} min")

    elapsed_time = time.time() - start_time

    rmse_1src = []
    rmse_2src = []
    for idx, rmse in rmses.items():
        if rmse < float('inf'):
            n_sources = source_counts[idx]
            if n_sources == 1:
                rmse_1src.append(rmse)
            else:
                rmse_2src.append(rmse)

    avg_rmse_1src = sum(rmse_1src) / len(rmse_1src) if rmse_1src else float('inf')
    avg_rmse_2src = sum(rmse_2src) / len(rmse_2src) if rmse_2src else float('inf')

    overall_rmse = (avg_rmse_1src + avg_rmse_2src) / 2
    n_candidates = 3
    score = 1 / (1 + overall_rmse) + 0.3 * (n_candidates / 3)

    projected_400_min = (elapsed_time / n_samples) * 400 / 60

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f} (n={len(rmse_1src)})")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f} (n={len(rmse_2src)})")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  Projected 400: {projected_400_min:.1f} min")

    in_budget = projected_400_min <= 60
    budget_remaining = 60 - projected_400_min
    print(f"  In budget:     {in_budget} (remaining: {budget_remaining:.1f} min)")

    return {
        'config': config_name,
        'sigma0_1src': config.get('sigma0_1src'),
        'sigma0_2src': config.get('sigma0_2src'),
        'perturbation_scale': config.get('perturbation_scale'),
        'score': score,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'in_budget': in_budget,
        'budget_remaining': budget_remaining,
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    print(f"\nPRIOR BEST: asymmetric_014_019 = 1.1745 @ 50.4 min")
    print("Testing if even tighter 1-source sigma (0.13) improves further...")

    # Base config with optimal settings
    base_config = {
        'max_fevals_1src': 24,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.06,  # Optimal scale
        'perturb_nm_iters': 3,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
        'perturb_only_2src': False,
    }

    configs = [
        # Prior best: sigma 0.14/0.19
        {**base_config, 'sigma0_1src': 0.14, 'sigma0_2src': 0.19},
        # Test: sigma 0.13/0.19 (tighter 1-source)
        {**base_config, 'sigma0_1src': 0.13, 'sigma0_2src': 0.19},
        # Test: sigma 0.12/0.19 (even tighter)
        {**base_config, 'sigma0_1src': 0.12, 'sigma0_2src': 0.19},
    ]
    config_names = ['sigma_014_019', 'sigma_013_019', 'sigma_012_019']

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Analysis
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS: TIGHTER 1-SOURCE SIGMA")
    print("="*70)
    print(f"{'Config':<20} {'Sigma 1src':>10} {'Score':>8} {'Proj 400':>10} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<20} {r['sigma0_1src']:>10.2f} {r['score']:>8.4f} {r['projected_400_min']:>9.1f}m {r['rmse_1src']:>10.4f} {r['rmse_2src']:>10.4f}")

    # Find best
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config']} ***")
        print(f"    Score: {best['score']:.4f}")
        print(f"    Sigma 1-src: {best['sigma0_1src']}")
        print(f"    Time: {best['projected_400_min']:.1f} min")


if __name__ == '__main__':
    main()
