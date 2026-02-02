"""
Experiment: 3pert_tabu004_budget
Test if 3 perturbations + tabu_distance=0.04 fits budget while preserving most improvement.

Prior finding from 4pert_tabu004_combined:
- 4 pert + tabu 0.04: Mean 1.1555 @ 61.9 min (OVER BUDGET)
- 3 pert alone: ~1.1671 @ 56.9 min (from asymmetric_014_019_4perturb run 3)

Hypothesis: 3 perturbations + tabu=0.04 should give ~1.15+ @ ~55-58 min
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

    BASELINE = 1.1337  # True baseline
    BEST_4PERT_TABU04 = 1.1555  # Combined 4-pert + tabu 0.04

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"vs True Baseline (1.1337): {score - BASELINE:+.4f}")
    print(f"vs 4-pert+tabu04 (1.1555): {score - BEST_4PERT_TABU04:+.4f}")

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

    BASELINE = 1.1337
    BEST_4PERT_TABU04 = 1.1555

    # Reduced config: 3 perturbations + tabu_distance=0.04
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,           # REDUCED from 4 to 3
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,          # KEEP 0.04
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("REDUCED CONFIG: 3 Perturbations + tabu_distance=0.04")
    print("="*60)
    print(f"True Baseline: {BASELINE}")
    print(f"4-pert + tabu 0.04 (over budget): {BEST_4PERT_TABU04} @ 61.9 min")
    print(f"Goal: Fit budget while preserving most improvement")

    results = []

    # Run 3 validation tests
    for run_num in range(1, 4):
        result = run_single_test(run_num, config, "3pert_tabu004", data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_time = float(np.mean(times))

    print("\n" + "="*60)
    print("SUMMARY: Reduced Configuration (3 pert + tabu 0.04)")
    print("="*60)
    print(f"\nRun Results:")
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Mean Time: {mean_time:.1f} min")
    print(f"  vs True Baseline: {mean_score - BASELINE:+.4f}")
    print(f"  vs 4-pert+tabu04: {mean_score - BEST_4PERT_TABU04:+.4f}")

    in_budget = mean_time <= 60.0
    print(f"\nBudget Status: {'IN BUDGET' if in_budget else 'OVER BUDGET'}")

    if in_budget:
        if mean_score > BASELINE + 0.01:
            print("\nCONCLUSION: SUCCESS - Significant improvement within budget!")
        else:
            print("\nCONCLUSION: MARGINAL - Within budget but limited improvement")
    else:
        print("\nCONCLUSION: STILL OVER BUDGET - Need further reduction")

    with open('run_output.json', 'w') as f:
        json.dump({
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_time': mean_time,
            'vs_baseline': mean_score - BASELINE,
            'vs_4pert_tabu04': mean_score - BEST_4PERT_TABU04,
            'in_budget': in_budget,
            'runs': results
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
