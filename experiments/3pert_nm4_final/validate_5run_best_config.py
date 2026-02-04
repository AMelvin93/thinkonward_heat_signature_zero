"""
5-RUN VALIDATION: 4pert nm2 scale06 (BEST CONFIG)

Previous 3-run validation:
- Mean: 1.1549 +/- 0.0058
- Best: 1.1612 (BEATS TOP 10!)
- 100% in budget

This validation: 5 runs for higher confidence statistics
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

    print(f"\n=== Validation Run {run_num} ===")

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

    # Breakdown by source count
    one_src = [r for r in results if r['n_sources'] == 1 and r['success']]
    two_src = [r for r in results if r['n_sources'] == 2 and r['success']]

    one_src_rmse = np.mean([r['rmse'] for r in one_src]) if one_src else 0
    two_src_rmse = np.mean([r['rmse'] for r in two_src]) if two_src else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"  Result: Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  1-src RMSE: {one_src_rmse:.4f}, 2-src RMSE: {two_src_rmse:.4f}")

    return {
        'run': run_num,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'in_budget': projected_400 <= 60,
        'one_src_rmse': float(one_src_rmse),
        'two_src_rmse': float(two_src_rmse),
    }

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # BEST CONFIG: 4pert nm2 scale06
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,
        'perturbation_scale': 0.06,
        'perturb_nm_iters': 2,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("5-RUN VALIDATION: 4pert nm2 scale06 (BEST CONFIG)")
    print("=" * 70)
    print()
    print("Config:")
    print("  n_perturbations: 4")
    print("  perturb_nm_iters: 2")
    print("  perturbation_scale: 0.06")
    print()
    print("Previous 3-run validation:")
    print("  Mean: 1.1549 +/- 0.0058")
    print("  Best: 1.1612 (BEATS TOP 10!)")
    print()
    print("Top 10 threshold: 1.1585")
    print("=" * 70)

    results = []
    for run_num in range(1, 6):  # 5 runs
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
    runs_beat_top10 = sum(1 for r in results if r['score'] >= 1.1585)

    print("\n" + "=" * 70)
    print("5-RUN VALIDATION SUMMARY")
    print("=" * 70)

    print("\nRun Results:")
    for r in results:
        status = "IN" if r['in_budget'] else "OVER"
        beats = "TOP 10!" if r['score'] >= 1.1585 else f"gap: {1.1585 - r['score']:.4f}"
        print(f"  Run {r['run']}: Score={r['score']:.4f} ({beats}), Time={r['projected_400_min']:.1f} min [{status}]")

    print(f"\n" + "-" * 40)
    print("STATISTICS:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Min Score: {min(scores):.4f}")
    print(f"  Max Score: {max(scores):.4f}")
    print(f"  Mean Time: {mean_time:.1f} +/- {std_time:.1f} min")
    print(f"  Runs in budget: {runs_in_budget}/5 ({runs_in_budget*100//5}%)")
    print(f"  Runs beating Top 10: {runs_beat_top10}/5 ({runs_beat_top10*100//5}%)")

    print(f"\n" + "-" * 40)
    print("TOP 10 ANALYSIS:")
    print(f"  Top 10 threshold: 1.1585")
    print(f"  Our mean: {mean_score:.4f}")
    print(f"  Gap to Top 10: {1.1585 - mean_score:.4f}")
    print(f"  Best run: {max(scores):.4f} ({'+' if max(scores) >= 1.1585 else ''}{max(scores) - 1.1585:.4f} vs Top 10)")

    # 95% confidence interval
    ci_95 = 1.96 * std_score / np.sqrt(len(scores))
    print(f"\n  95% CI: [{mean_score - ci_95:.4f}, {mean_score + ci_95:.4f}]")

    if mean_score + ci_95 >= 1.1585:
        print("  *** Upper CI bound REACHES Top 10! ***")

    # RMSE breakdown
    print(f"\n" + "-" * 40)
    print("RMSE BREAKDOWN:")
    avg_1src_rmse = np.mean([r['one_src_rmse'] for r in results])
    avg_2src_rmse = np.mean([r['two_src_rmse'] for r in results])
    print(f"  Avg 1-source RMSE: {avg_1src_rmse:.4f}")
    print(f"  Avg 2-source RMSE: {avg_2src_rmse:.4f}")

    with open('validate_5run_output.json', 'w') as f:
        json.dump({
            'config': '4pert_nm2_scale06',
            'n_runs': 5,
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
            'mean_time': float(mean_time),
            'std_time': float(std_time),
            'runs_in_budget': runs_in_budget,
            'runs_beat_top10': runs_beat_top10,
            'ci_95_lower': float(mean_score - ci_95),
            'ci_95_upper': float(mean_score + ci_95),
            'gap_to_top10': float(1.1585 - mean_score),
            'runs': results
        }, f, indent=2)

    print("\n" + "=" * 70)
    if runs_beat_top10 >= 2:
        print("*** MULTIPLE RUNS BEAT TOP 10! HIGH CONFIDENCE CONFIG! ***")
    elif runs_beat_top10 >= 1:
        print("*** AT LEAST ONE RUN BEATS TOP 10! ***")
    print("=" * 70)

    print("\nResults saved to validate_5run_output.json")

if __name__ == "__main__":
    main()
