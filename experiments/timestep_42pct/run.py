"""
Experiment: timestep_42pct
Test higher temporal fidelity (42%, 44%) vs baseline 40%.
Using the validated config (sigma 0.18/0.22, tabu_distance=0.04).

Hypothesis: More timesteps may improve accuracy with modest time increase.
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
    print(f"  Timestep fraction: {config.get('timestep_fraction')}")
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
        'timestep_fraction': config.get('timestep_fraction'),
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

    # Updated baseline with tabu_distance=0.04
    BASELINE = 1.1496  # @ 55.4 min (validated tabu_004)

    print("="*60)
    print("TEMPORAL FIDELITY TUNING (42%, 44%)")
    print("="*60)
    print(f"Baseline: 1.1496 @ 55.4 min (40% temporal, tabu=0.04)")
    print(f"Hypothesis: More timesteps may improve accuracy")

    # Base config with validated settings
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,  # Validated improvement
        'max_tabu_attempts': 10,
    }

    # Test configurations
    configs = [
        # Run 1: 42% temporal (vs 40% baseline)
        {**base_config, 'timestep_fraction': 0.42},
        # Run 2: 44% temporal
        {**base_config, 'timestep_fraction': 0.44},
        # Run 3: Baseline for comparison (40%)
        {**base_config, 'timestep_fraction': 0.40},
    ]
    config_names = ['timestep_42pct', 'timestep_44pct', 'baseline_40pct']

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        # Save intermediate
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "="*70)
    print("TEMPORAL FIDELITY TUNING RESULTS")
    print("="*70)
    print(f"{'Config':<20} {'Timestep':>8} {'Score':>8} {'Proj 400':>10} {'vs Baseline':>12}")
    print("-"*65)
    for r in results:
        delta = r['score'] - BASELINE
        print(f"{r['config_name']:<20} {r['timestep_fraction']*100:>7.0f}% {r['score']:>8.4f} {r['projected_400_min']:>9.1f}m {delta:>+12.4f}")

    # Find best in-budget
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config_name']} ***")
        print(f"    Score: {best['score']:.4f} ({best['score'] - BASELINE:+.4f} vs baseline)")
        print(f"    Timestep fraction: {best['timestep_fraction']*100:.0f}%")
        print(f"    Time: {best['projected_400_min']:.1f} min")

        if best['score'] > BASELINE:
            print(f"\n*** SUCCESS! Better than baseline! ***")
        else:
            print(f"\n*** FAILED: Did not beat baseline ***")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
