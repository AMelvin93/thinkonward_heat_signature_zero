"""
Experiment: intermediate_fevals_3pert
Test intermediate fevals (21/46) to find budget boundary.

Prior findings:
- 3-pert @ 20/44: 1.1475 @ 55.7 min (IN BUDGET)
- 3-pert @ 22/48: 1.1568 @ 63.4 min (OVER BUDGET)

Hypothesis: 21/46 might hit the sweet spot: improved score while staying in budget.
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


def run_single_test(run_num, config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Run {run_num}: {config_name} ===")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{n_samples}")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    BASELINE = 1.1475  # Our validated 3-pert baseline

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"vs 3-pert baseline (1.1475): {score - BASELINE:+.4f}")

    return {
        'run': run_num,
        'config_name': config_name,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE = 1.1475
    HIGHER = 1.1568

    # Intermediate fevals: 21/46 (between 20/44 and 22/48)
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 21,          # INTERMEDIATE
        'max_fevals_2src': 46,          # INTERMEDIATE
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("INTERMEDIATE FEVALS: 21/46 with 3 perturbations")
    print("="*60)
    print(f"20/44 baseline: {BASELINE} @ 55.7 min")
    print(f"22/48 (over budget): {HIGHER} @ 63.4 min")
    print(f"Goal: Find sweet spot in between")

    results = []

    # Run 3 validation tests
    for run_num in range(1, 4):
        result = run_single_test(run_num, config, "3pert_fevals_21_46", data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_time = float(np.mean(times))

    print("\n" + "="*60)
    print("SUMMARY: Intermediate Fevals (21/46) with 3 perturbations")
    print("="*60)
    print(f"\nRun Results:")
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Mean Time: {mean_time:.1f} min")
    print(f"  vs 20/44 baseline: {mean_score - BASELINE:+.4f}")
    print(f"  vs 22/48 over budget: {mean_score - HIGHER:+.4f}")

    in_budget = mean_time <= 60.0
    runs_in_budget = sum(1 for r in results if r['in_budget'])
    print(f"\n  Runs in budget: {runs_in_budget}/3")
    print(f"  Budget Status: {'IN BUDGET' if in_budget else 'OVER BUDGET'}")

    if in_budget and mean_score > BASELINE:
        print("\nCONCLUSION: SUCCESS - Sweet spot found!")
    elif in_budget:
        print("\nCONCLUSION: IN BUDGET but no improvement")
    else:
        print("\nCONCLUSION: STILL OVER BUDGET - try 20/45 instead")

    with open('run_output.json', 'w') as f:
        json.dump({
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_time': mean_time,
            'vs_baseline': mean_score - BASELINE,
            'vs_higher': mean_score - HIGHER,
            'in_budget': in_budget,
            'runs_in_budget': runs_in_budget,
            'runs': results
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
