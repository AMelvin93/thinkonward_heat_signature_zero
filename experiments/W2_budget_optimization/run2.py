"""
Experiment: W2_budget_optimization - Phase 2

Phase 1 found: 3pert_nm3 with 1.1499 @ 52.5 min (+0.0012)
Budget remaining: 7.5 min

Phase 2: Explore around 3 perturbations direction
1. 3pert_nm4 - 3 perturbations, more polish (nm=4)
2. 3pert_nm3_refine10 - 3 perturbations, more final polish
3. 4pert_nm2 - 4 perturbations, less polish (more diversity)
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

    state = load_state()

    # Base config
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Phase 2 configs - building on 3pert_nm3 success
    configs_to_test = [
        ('3pert_nm4', {**base_config, 'n_perturbations': 3, 'perturb_nm_iters': 4}),
        ('3pert_nm3_refine10', {**base_config, 'n_perturbations': 3, 'perturb_nm_iters': 3, 'refine_maxiter': 10}),
        ('4pert_nm2', {**base_config, 'n_perturbations': 4, 'perturb_nm_iters': 2}),
    ]

    print("=" * 70)
    print("EXPERIMENT: W2_budget_optimization - PHASE 2")
    print("=" * 70)
    print(f"Phase 1 best: 3pert_nm3 = 1.1499 @ 52.5 min")
    print(f"Goal: Explore around 3 perturbations direction")
    print()

    all_results = []

    for config_name, config in configs_to_test:
        print(f"\n{'='*60}")
        print(f"RUN: {config_name}")
        print(f"{'='*60}")
        print(f"Config: n_perturb={config['n_perturbations']}, nm_iters={config['perturb_nm_iters']}, refine={config['refine_maxiter']}")

        result = run_full_test(config_name, config, samples, meta)
        all_results.append(result)

        status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
        delta = result['score'] - 1.1499  # Compare to Phase 1 best
        print(f"\nResult: Score={result['score']:.4f} ({delta:+.4f} vs 3pert_nm3), Time={result['projected_400_min']:.1f} min [{status}]")
        print(f"RMSE: 1src={result['rmse_1src']:.4f}, 2src={result['rmse_2src']:.4f}")

        # Update state
        run_data = {
            'run': len(state['tuning_runs']) + 1,
            'config_name': config_name,
            'config': {k: v for k, v in config.items() if k in ['perturb_nm_iters', 'refine_maxiter', 'n_perturbations']},
            'score': result['score'],
            'time_min': result['projected_400_min'],
            'in_budget': result['in_budget'],
            'delta_vs_3pert_nm3': delta,
            'timestamp': datetime.now().isoformat()
        }
        state['tuning_runs'].append(run_data)
        save_state(state)

        budget_remaining = 60 - result['projected_400_min']
        print(f"\nBudget remaining: {budget_remaining:.1f} min")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<20} {'Score':>8} {'Delta':>8} {'Time':>8} {'Budget':>10}")
    print("-" * 60)
    print(f"{'3pert_nm3 (P1)':<20} {1.1499:>8.4f} {0:>+8.4f} {52.5:>8.1f} {'IN':>10}")

    for r in all_results:
        delta = r['score'] - 1.1499
        status = "IN" if r['in_budget'] else "OVER"
        print(f"{r['config_name']:<20} {r['score']:>8.4f} {delta:>+8.4f} {r['projected_400_min']:>8.1f} {status:>10}")

    # Find best in-budget from all phases
    all_phase_results = [
        {'config_name': '3pert_nm3', 'score': 1.1499, 'projected_400_min': 52.5, 'in_budget': True},
        {'config_name': 'nm5', 'score': 1.1494, 'projected_400_min': 51.4, 'in_budget': True},
    ] + all_results

    in_budget = [r for r in all_phase_results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nOverall best in-budget: {best['config_name']} with score {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

        top_10_threshold = 1.1585
        gap = top_10_threshold - best['score']
        print(f"Gap to Top 10 (1.1585): {gap:.4f}")

if __name__ == '__main__':
    main()
