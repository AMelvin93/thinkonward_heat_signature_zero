"""
Test: 3 perturbations with nm_iters=3 and scale=0.06 (increased)

Previous 3pert nm3 (scale=0.05) result:
- Mean: 1.1492 @ 53.8 min
- 3/3 runs in budget (100%)
- Low variance (std 0.0020)

This test: Increase scale to 0.06 for more aggressive exploration
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

def run_single_test(run_num, config, samples, meta):
    n_samples = len(samples)
    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    print(f"\n=== Run {run_num} ===")

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"  Result: Score={score:.4f}, Time={projected_400:.1f} min")

    return {
        'run': run_num,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'in_budget': projected_400 <= 60
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # 3 perturbations with nm3 + increased scale
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,
        'perturbation_scale': 0.06,  # INCREASED from 0.05
        'perturb_nm_iters': 3,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("TEST: 3pert_nm3_scale06 (increased scale)")
    print("=" * 70)
    print("Config: 3 perturbations, nm_iters=3, scale=0.06")
    print()
    print("Previous 3pert nm3 (scale=0.05) result:")
    print("  Mean: 1.1492 @ 53.8 min")
    print("  3/3 runs in budget (100%)")
    print()
    print("Expectation: Similar timing, potentially better exploration")
    print("=" * 70)

    results = []
    for run_num in range(1, 4):
        result = run_single_test(run_num, config, samples, meta)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_time = np.mean(times)
    std_time = np.std(times)
    runs_in_budget = sum(1 for r in results if r['in_budget'])

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY: 3pert_nm3_scale06 (3 runs)")
    print("=" * 70)

    for r in results:
        status = "IN" if r['in_budget'] else "OVER"
        gap = 1.1585 - r['score']
        print(f"  Run {r['run']}: Score={r['score']:.4f} (gap: {gap:.4f}), Time={r['projected_400_min']:.1f} min [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Mean Time: {mean_time:.1f} +/- {std_time:.1f} min")
    print(f"  Runs in budget: {runs_in_budget}/3")

    print(f"\nComparison:")
    print(f"  vs 3pert nm3 scale=0.05 (1.1492): {mean_score - 1.1492:+.4f}")
    print(f"  vs 2pert nm4 (1.1487): {mean_score - 1.1487:+.4f}")
    print(f"  Gap to Top 10: {1.1585 - mean_score:.4f}")

    best_score = max(scores)
    print(f"\n  Best run: {best_score:.4f} (gap: {1.1585 - best_score:.4f})")
    print(f"  Max time: {max(times):.1f} min")

    with open('test_3pert_nm3_scale06_output.json', 'w') as f:
        json.dump({
            'config': '3pert_nm3_scale06',
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'mean_time': float(mean_time),
            'std_time': float(std_time),
            'runs_in_budget': runs_in_budget,
            'best_run': float(best_score),
            'gap_to_top10': float(1.1585 - mean_score),
            'runs': results
        }, f, indent=2)

    print("\nResults saved to test_3pert_nm3_scale06_output.json")

if __name__ == "__main__":
    main()
