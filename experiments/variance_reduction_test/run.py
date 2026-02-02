"""
Experiment: variance_reduction_test
Test if running the best config with 5 runs and taking the best gives consistent improvement.

Approach: For each sample, run optimization 2 times with different seeds and take best result.
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


def process_sample_multirun(args):
    """Run optimization n_runs times and take best result."""
    sample_idx, sample, meta, config, n_runs = args

    best_rmse = float('inf')
    best_candidates = None
    total_time = 0

    for run_idx in range(n_runs):
        # Set different seed for each run
        np.random.seed(sample_idx * 100 + run_idx)

        optimizer = TabuBasinHoppingOptimizer(**config)
        try:
            start = time.time()
            candidates, rmse, results, n_sims = optimizer.estimate_sources(
                sample, meta, q_range=(0.5, 2.0), verbose=False
            )
            elapsed = time.time() - start
            total_time += elapsed

            if rmse < best_rmse:
                best_rmse = rmse
                best_candidates = candidates
        except Exception as e:
            pass

    n_candidates = len(best_candidates) if best_candidates else 0
    return {
        'idx': sample_idx,
        'rmse': best_rmse,
        'n_sources': sample['n_sources'],
        'n_candidates': n_candidates,
        'time_s': total_time,
        'success': best_rmse < float('inf')
    }


def run_experiment(config, n_runs, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== {config_name} (n_runs={n_runs}) ===")

    args_list = [(i, samples[i], meta, config, n_runs) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample_multirun, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{n_samples}")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    BASELINE = 1.143

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"vs Baseline (1.143): {score - BASELINE:+.4f}")
    print(f"Budget remaining: {60.0 - projected_400:.1f} min")

    return {
        'config_name': config_name,
        'n_runs': n_runs,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.143

    # Best known config
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
    }

    print("="*60)
    print("VARIANCE REDUCTION: Multi-run per sample")
    print("="*60)
    print(f"Baseline: 1.143 @ ~45 min")

    results = []

    # Single run (baseline)
    result1 = run_experiment(config, n_runs=1, config_name="1_run", data=data)
    results.append(result1)

    # Two runs per sample
    result2 = run_experiment(config, n_runs=2, config_name="2_runs", data=data)
    results.append(result2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Multi-Run Variance Reduction")
    print("="*60)
    print(f"Baseline: 1.143 @ ~45 min")
    print()
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    best = max(results, key=lambda x: x['score'])
    print()
    print(f"Best: {best['config_name']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    improvement = result2['score'] - result1['score']
    time_overhead = result2['projected_400_min'] - result1['projected_400_min']
    print(f"\n2 runs vs 1 run: {improvement:+.4f} score, +{time_overhead:.1f} min time")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
