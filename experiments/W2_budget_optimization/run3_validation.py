"""
Experiment: W2_budget_optimization - Phase 3: Validation

Phase 2 found: 3pert_nm3_refine10 with 1.1571 @ 56.7 min
Gap to Top 10: only 0.0014!

Phase 3:
1. Validate 3pert_nm3_refine10 with 3 runs
2. Try refine12 to see if we can squeeze more
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

    # Best config from Phase 2
    best_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 10,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("EXPERIMENT: W2_budget_optimization - PHASE 3: VALIDATION")
    print("=" * 70)
    print(f"Phase 2 best: 3pert_nm3_refine10 = 1.1571 @ 56.7 min")
    print(f"Gap to Top 10: 0.0014")
    print()

    # Validation: Run 3pert_nm3_refine10 three times
    print("=" * 70)
    print("VALIDATION: Running 3pert_nm3_refine10 three times")
    print("=" * 70)

    validation_results = []
    for run_num in range(1, 4):
        print(f"\n--- Validation Run {run_num}/3 ---")
        result = run_full_test(f'validation_run{run_num}', best_config, samples, meta)
        validation_results.append(result)

        status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
        print(f"Result: Score={result['score']:.4f}, Time={result['projected_400_min']:.1f} min [{status}]")

        run_data = {
            'run': len(state['tuning_runs']) + 1,
            'config_name': f'validation_run{run_num}',
            'config': {'n_perturbations': 3, 'perturb_nm_iters': 3, 'refine_maxiter': 10},
            'score': result['score'],
            'time_min': result['projected_400_min'],
            'in_budget': result['in_budget'],
            'timestamp': datetime.now().isoformat()
        }
        state['tuning_runs'].append(run_data)
        save_state(state)

    # Calculate validation statistics
    scores = [r['score'] for r in validation_results]
    times = [r['projected_400_min'] for r in validation_results]

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Run':<15} {'Score':>8} {'Time':>8}")
    print("-" * 35)
    for i, r in enumerate(validation_results, 1):
        print(f"{'Run '+str(i):<15} {r['score']:>8.4f} {r['projected_400_min']:>8.1f}")

    print("-" * 35)
    print(f"{'Mean':<15} {np.mean(scores):>8.4f} {np.mean(times):>8.1f}")
    print(f"{'Std':<15} {np.std(scores):>8.4f} {np.std(times):>8.1f}")
    print(f"{'Best':<15} {np.max(scores):>8.4f}")
    print(f"{'Worst':<15} {np.min(scores):>8.4f}")

    # Check if all runs are in budget
    in_budget_pct = sum(1 for r in validation_results if r['in_budget']) / len(validation_results) * 100
    print(f"\nIn-budget rate: {in_budget_pct:.0f}%")

    top_10_threshold = 1.1585
    gap_from_mean = top_10_threshold - np.mean(scores)
    gap_from_best = top_10_threshold - np.max(scores)
    print(f"\nGap to Top 10 (1.1585):")
    print(f"  From mean: {gap_from_mean:.4f}")
    print(f"  From best: {gap_from_best:.4f}")

    # Now try refine12 to see if we can squeeze more
    print("\n" + "=" * 70)
    print("EXPLORATORY: Testing refine12")
    print("=" * 70)

    refine12_config = {**best_config, 'refine_maxiter': 12}
    result = run_full_test('3pert_nm3_refine12', refine12_config, samples, meta)

    status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
    print(f"\nResult: Score={result['score']:.4f}, Time={result['projected_400_min']:.1f} min [{status}]")

    if result['in_budget'] and result['score'] > np.mean(scores):
        print(f"IMPROVEMENT: +{result['score'] - np.mean(scores):.4f} vs validation mean!")
    else:
        print("No improvement or over budget")

    run_data = {
        'run': len(state['tuning_runs']) + 1,
        'config_name': '3pert_nm3_refine12',
        'config': {'n_perturbations': 3, 'perturb_nm_iters': 3, 'refine_maxiter': 12},
        'score': result['score'],
        'time_min': result['projected_400_min'],
        'in_budget': result['in_budget'],
        'timestamp': datetime.now().isoformat()
    }
    state['tuning_runs'].append(run_data)
    save_state(state)

    # Final recommendation
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    best_score = np.max(scores)
    mean_score = np.mean(scores)
    mean_time = np.mean(times)

    print(f"\nConfig: 3pert_nm3_refine10")
    print(f"  n_perturbations: 3")
    print(f"  perturb_nm_iters: 3")
    print(f"  refine_maxiter: 10")
    print(f"  sigma0_1src: 0.18")
    print(f"  sigma0_2src: 0.22")
    print(f"  max_fevals: 20/44")
    print(f"  timestep_fraction: 0.40")

    print(f"\nValidated Performance:")
    print(f"  Mean: {mean_score:.4f} @ {mean_time:.1f} min")
    print(f"  Best: {best_score:.4f}")
    print(f"  Gap to Top 10: {top_10_threshold - mean_score:.4f} (mean) / {top_10_threshold - best_score:.4f} (best)")

if __name__ == '__main__':
    main()
