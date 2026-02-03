"""
Experiment: production_10run_validation

10-run validation of the production config to establish high-confidence
performance metrics for final submission.

Best validated config: 4-pert + perturb_nm_iters=2 + sigma 0.18/0.22
Previous 3-run validation: 1.1482 +/- 0.0030 @ 51.7 min
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

    print(f"\n=== Validation Run {run_num}: {config_name} ===")

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
                print(f"  Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    status = "IN BUDGET" if projected_400 <= 60.0 else "OVER BUDGET"
    print(f"  Budget: {status}")

    return {
        'run': run_num,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    # Previous 3-run validation result
    BASELINE_3RUN = 1.1482
    TOP_10_THRESHOLD = 1.1585

    # PRODUCTION CONFIG: 4-pert + perturb_nm_iters=2 + sigma 0.18/0.22
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,
        'perturb_nm_iters': 2,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("10-RUN PRODUCTION VALIDATION")
    print("=" * 70)
    print(f"Config: 4-pert + perturb_nm_iters=2 + sigma 0.18/0.22")
    print(f"Previous 3-run result: {BASELINE_3RUN:.4f} +/- 0.0030 @ 51.7 min")
    print(f"Top 10 threshold: {TOP_10_THRESHOLD:.4f}")
    print(f"Goal: Establish high-confidence performance for final submission")

    results = []

    for run_num in range(1, 11):
        result = run_single_test(run_num, config, "production", data)
        results.append(result)

        # Interim statistics after each run
        scores = [r['score'] for r in results]
        print(f"  Interim mean after run {run_num}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Final statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]
    rmse_1src = [r['rmse_1src'] for r in results]
    rmse_2src = [r['rmse_2src'] for r in results]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    runs_in_budget = sum(1 for r in results if r['in_budget'])

    # 95% confidence interval
    ci_95 = 1.96 * std_score / np.sqrt(10)
    score_lower = mean_score - ci_95
    score_upper = mean_score + ci_95

    print("\n" + "=" * 70)
    print("10-RUN VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nRun Results:")
    for r in results:
        delta = r['score'] - BASELINE_3RUN
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  Run {r['run']:2d}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  95% Confidence Interval: [{score_lower:.4f}, {score_upper:.4f}]")
    print(f"  Score Range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"  Mean Time: {mean_time:.1f} +/- {std_time:.1f} min")
    print(f"  Runs in budget: {runs_in_budget}/10 ({runs_in_budget*10:.0f}%)")

    print(f"\nRMSE Breakdown:")
    print(f"  1-source mean: {np.mean(rmse_1src):.4f} +/- {np.std(rmse_1src):.4f}")
    print(f"  2-source mean: {np.mean(rmse_2src):.4f} +/- {np.std(rmse_2src):.4f}")

    print(f"\nComparisons:")
    print(f"  vs 3-run mean (1.1482): {mean_score - BASELINE_3RUN:+.4f}")
    print(f"  Gap to Top 10 (1.1585): {TOP_10_THRESHOLD - mean_score:+.4f}")

    # Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    if runs_in_budget == 10:
        print("  ✓ 100% of runs within budget")
    else:
        print(f"  ⚠ {10 - runs_in_budget}/10 runs over budget - RISK!")

    if score_upper < TOP_10_THRESHOLD:
        print(f"  × Even upper CI ({score_upper:.4f}) below Top 10 threshold")
        print(f"  → Novel approaches needed to reach Top 10")
    elif score_lower < TOP_10_THRESHOLD <= score_upper:
        print(f"  ~ Score range spans Top 10 threshold")
        print(f"  → Some runs may reach Top 10, depends on luck")
    else:
        print(f"  ✓ Lower CI ({score_lower:.4f}) exceeds Top 10 threshold!")
        print(f"  → Confident Top 10 finish")

    # Save results
    output = {
        'config': config,
        'mean_score': mean_score,
        'std_score': std_score,
        'ci_95_lower': float(score_lower),
        'ci_95_upper': float(score_upper),
        'min_score': min_score,
        'max_score': max_score,
        'mean_time_min': mean_time,
        'std_time_min': std_time,
        'runs_in_budget': runs_in_budget,
        'vs_3run_baseline': mean_score - BASELINE_3RUN,
        'gap_to_top10': TOP_10_THRESHOLD - mean_score,
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_1src_std': float(np.std(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'rmse_2src_std': float(np.std(rmse_2src)),
        'runs': results
    }

    with open('run_output.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to run_output.json")

    return output


if __name__ == "__main__":
    main()
