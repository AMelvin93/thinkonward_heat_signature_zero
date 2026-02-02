"""
Run script for perturbation_only_2source experiment.

Hypothesis: 2-source problems benefit more from perturbation than 1-source.
If true, we can save time by skipping perturbation for 1-source problems.

Uses new best config: asymmetric_014_019 (sigma 0.14/0.19, n_perturbations=2)
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
    print(f"Perturbation: {'2-SOURCE ONLY' if config.get('perturb_only_2src') else 'ALL'}")
    print(f"N perturbations: {config.get('n_perturbations')}")
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
    print(f"  Errors:        {len(errors)}")

    in_budget = projected_400_min <= 60
    budget_remaining = 60 - projected_400_min
    print(f"  In budget:     {in_budget} (remaining: {budget_remaining:.1f} min)")

    return {
        'config': config_name,
        'score': score,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'n_1src': len(rmse_1src),
        'n_2src': len(rmse_2src),
        'perturb_only_2src': config.get('perturb_only_2src'),
        'in_budget': in_budget,
        'budget_remaining': budget_remaining,
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Count 1-src vs 2-src
    n_1src = sum(1 for s in data['samples'] if s['n_sources'] == 1)
    n_2src = sum(1 for s in data['samples'] if s['n_sources'] == 2)
    print(f"  1-source problems: {n_1src}")
    print(f"  2-source problems: {n_2src}")

    print(f"\nBASELINE: asymmetric_014_019 = 1.1745 @ 50.4 min (perturbation for all)")
    print("Testing if perturbation only for 2-source can save time on 1-source...")

    # Base config from new best (asymmetric sigma + optimal perturbation scale)
    base_config = {
        'sigma0_1src': 0.14,  # Tighter for 1-source
        'sigma0_2src': 0.19,
        'max_fevals_1src': 24,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
    }

    configs = [
        # Baseline: perturbation for all
        {**base_config, 'perturb_only_2src': False},
        # Test: perturbation only for 2-source
        {**base_config, 'perturb_only_2src': True},
    ]
    config_names = ['perturb_all_baseline', 'perturb_2src_only']

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        # Save intermediate results
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Analysis
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS: PERTURBATION ONLY FOR 2-SOURCE")
    print("="*70)
    print(f"{'Config':<25} {'Score':>8} {'Time':>8} {'Proj 400':>10} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<25} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['projected_400_min']:>9.1f}m {r['rmse_1src']:>10.4f} {r['rmse_2src']:>10.4f}")

    # Compare results
    baseline = results[0]
    test = results[1]

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"Baseline (perturb all):     Score {baseline['score']:.4f} @ {baseline['projected_400_min']:.1f} min")
    print(f"Test (perturb 2-src only):  Score {test['score']:.4f} @ {test['projected_400_min']:.1f} min")
    print(f"\nScore delta: {test['score'] - baseline['score']:+.4f}")
    print(f"Time delta:  {test['projected_400_min'] - baseline['projected_400_min']:+.1f} min")

    print(f"\n1-source RMSE:")
    print(f"  Baseline: {baseline['rmse_1src']:.4f}")
    print(f"  Test:     {test['rmse_1src']:.4f}")
    print(f"  Delta:    {test['rmse_1src'] - baseline['rmse_1src']:+.4f}")

    print(f"\n2-source RMSE:")
    print(f"  Baseline: {baseline['rmse_2src']:.4f}")
    print(f"  Test:     {test['rmse_2src']:.4f}")
    print(f"  Delta:    {test['rmse_2src'] - baseline['rmse_2src']:+.4f}")

    # Conclusion
    if test['score'] >= baseline['score'] - 0.002:  # Within variance
        if test['projected_400_min'] < baseline['projected_400_min']:
            print(f"\n*** RECOMMENDATION: Use perturb_2src_only ***")
            print(f"    Same score, {baseline['projected_400_min'] - test['projected_400_min']:.1f} min faster!")
        else:
            print(f"\n*** RECOMMENDATION: Keep perturbation for all ***")
            print(f"    No time savings from conditional perturbation.")
    else:
        print(f"\n*** RECOMMENDATION: Keep perturbation for all ***")
        print(f"    Score loss ({baseline['score'] - test['score']:.4f}) outweighs time savings.")


if __name__ == '__main__':
    main()
