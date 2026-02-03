"""
Experiment: faster_nm_4pert

Test reduced NM iterations with 4 perturbations to find a config that fits budget.

Baseline: 4-pert nm2 @ 70.9 min (OVER BUDGET)
Target: Find NM config that achieves <60 min while maintaining high score

Test configs:
- refine_maxiter=6 (vs baseline 8)
- refine_maxiter=4
- Combined reductions
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
    """Run full 80-sample test."""
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
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'in_budget': projected_400 <= 60
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # Configs to test
    configs = {
        'nm6_4pert': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 6,  # REDUCED from 8
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.05,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
        'nm4_4pert': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 4,  # FURTHER REDUCED
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.05,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
        'nm6_4pert_nm1': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 6,  # REDUCED
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 1,  # ALSO REDUCED
            'perturbation_scale': 0.05,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    }

    print("=" * 70)
    print("EXPERIMENT: faster_nm_4pert")
    print("=" * 70)
    print("Testing reduced NM iterations with 4 perturbations")
    print("Target: <60 min projected while maintaining high score")
    print(f"\nBaseline: 4-pert nm2 (refine_maxiter=8, perturb_nm_iters=2)")
    print(f"  Score: 1.1546, Time: 70.9 min (OVER BUDGET)")
    print()

    results = []
    for config_name, config in configs.items():
        print(f"\n--- Testing: {config_name} ---")
        print(f"  refine_maxiter={config['refine_maxiter']}, perturb_nm_iters={config.get('perturb_nm_iters', 2)}")

        result = run_full_test(config_name, config, samples, meta)
        results.append(result)

        status = "✓ IN BUDGET" if result['in_budget'] else "✗ OVER BUDGET"
        print(f"  Result: Score={result['score']:.4f}, Time={result['projected_400_min']:.1f} min ({status})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Score':>8} {'Time':>10} {'Status':>15}")
    print("-" * 70)

    # Add baseline for comparison
    print(f"{'4pert_nm2 (base)':<20} {'1.1546':>8} {'70.9 min':>10} {'OVER':>15}")
    print(f"{'no_perturb (safe)':<20} {'1.1367':>8} {'56.3 min':>10} {'IN':>15}")
    print("-" * 70)

    for r in results:
        status = "✓ IN" if r['in_budget'] else "✗ OVER"
        print(f"{r['config_name']:<20} {r['score']:>8.4f} {r['projected_400_min']:>10.1f} {status:>15}")

    # Find best in-budget config
    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config_name']} ***")
        print(f"    Score: {best['score']:.4f}, Time: {best['projected_400_min']:.1f} min")
    else:
        print(f"\n*** NO CONFIGS FIT BUDGET ***")
        print("    Consider further reductions or different approach")

    # Save results
    with open('run_output.json', 'w') as f:
        json.dump({
            'baseline_4pert_nm2': {'score': 1.1546, 'time': 70.9},
            'safe_no_perturb': {'score': 1.1367, 'time': 56.3},
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")

if __name__ == "__main__":
    main()
