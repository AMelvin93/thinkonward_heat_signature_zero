"""
Experiment: W2_budget_optimization

Current best: nm4_scale05 with 1.1487 @ 54.2 min
Budget remaining: ~6 min

Test configurations that use the remaining budget to improve score:
1. perturb_nm_iters=5 (more polish per perturbation)
2. perturb_nm_iters=6 (even more polish)
3. refine_maxiter=10 (more final polish)
4. n_perturbations=3, perturb_nm_iters=3 (more perturbations, less each)
"""

import os
import sys
import pickle
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

sys.path.insert(0, os.path.join(_project_root, 'experiments', 'tighter_sigma_range'))
from optimizer import TabuBasinHoppingOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7
STATE_FILE = os.path.join(os.path.dirname(__file__), 'STATE.json')

def load_state():
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

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
            'error': str(e)
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
    time_1src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 1]
    time_2src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    return {
        'config_name': config_name,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'elapsed_sec': float(elapsed_time),
        'rmse_1src': float(np.mean(rmse_1src)) if rmse_1src else 0,
        'rmse_2src': float(np.mean(rmse_2src)) if rmse_2src else 0,
        'time_1src': float(np.mean(time_1src)) if time_1src else 0,
        'time_2src': float(np.mean(time_2src)) if time_2src else 0,
        'in_budget': projected_400 <= 60
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    state = load_state()

    # Base config - current best (nm4_scale05)
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 4,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Configurations to test - using remaining budget
    configs_to_test = [
        ('nm5', {**base_config, 'perturb_nm_iters': 5}),
        ('nm6', {**base_config, 'perturb_nm_iters': 6}),
        ('refine10', {**base_config, 'refine_maxiter': 10}),
        ('3pert_nm3', {**base_config, 'n_perturbations': 3, 'perturb_nm_iters': 3}),
    ]

    print("=" * 70)
    print("EXPERIMENT: W2_budget_optimization")
    print("=" * 70)
    print(f"Baseline: nm4_scale05 = 1.1487 @ 54.2 min")
    print(f"Goal: Use remaining ~6 min budget to improve score")
    print(f"Target: > 1.1487 @ <= 60 min")
    print()

    all_results = []

    for config_name, config in configs_to_test:
        print(f"\n{'='*60}")
        print(f"RUN: {config_name}")
        print(f"{'='*60}")

        # Show what's different from base
        diff = {}
        for k, v in config.items():
            if k in base_config and base_config[k] != v:
                diff[k] = f"{base_config[k]} -> {v}"
        print(f"Changes: {diff}")

        result = run_full_test(config_name, config, samples, meta)
        all_results.append(result)

        status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
        delta = result['score'] - 1.1487
        print(f"\nResult: Score={result['score']:.4f} ({delta:+.4f}), Time={result['projected_400_min']:.1f} min [{status}]")
        print(f"RMSE: 1src={result['rmse_1src']:.4f}, 2src={result['rmse_2src']:.4f}")

        # Update state
        run_data = {
            'run': len(state['tuning_runs']) + 1,
            'config_name': config_name,
            'config': {k: v for k, v in config.items() if k in ['perturb_nm_iters', 'refine_maxiter', 'n_perturbations']},
            'score': result['score'],
            'time_min': result['projected_400_min'],
            'in_budget': result['in_budget'],
            'delta_vs_baseline': delta,
            'timestamp': datetime.now().isoformat()
        }
        state['tuning_runs'].append(run_data)
        save_state(state)

        # Decision logic
        budget_remaining = 60 - result['projected_400_min']
        print(f"\nBudget remaining: {budget_remaining:.1f} min")

        if not result['in_budget']:
            print("-> OVER BUDGET - this config not viable")
        elif delta > 0:
            print(f"-> IMPROVEMENT: +{delta:.4f} score!")
        else:
            print(f"-> No improvement ({delta:.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Score':>8} {'Delta':>8} {'Time':>8} {'Budget':>10}")
    print("-" * 55)
    print(f"{'baseline':<15} {1.1487:>8.4f} {0:>+8.4f} {54.2:>8.1f} {'IN':>10}")

    for r in all_results:
        delta = r['score'] - 1.1487
        status = "IN" if r['in_budget'] else "OVER"
        print(f"{r['config_name']:<15} {r['score']:>8.4f} {delta:>+8.4f} {r['projected_400_min']:>8.1f} {status:>10}")

    # Find best in-budget
    in_budget = [r for r in all_results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config_name']} with score {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

        if best['score'] > 1.1487:
            print(f"NEW BEST! Improvement: +{best['score'] - 1.1487:.4f}")
        else:
            print("No improvement over baseline")
    else:
        print("\nNo configs fit within budget!")

    # Gap to Top 10
    top_10_threshold = 1.1585
    if in_budget:
        best_score = max(r['score'] for r in in_budget)
        gap = top_10_threshold - best_score
        print(f"\nGap to Top 10 (1.1585): {gap:.4f}")

if __name__ == '__main__':
    main()
