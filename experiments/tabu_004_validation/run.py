"""
Experiment: tabu_004_validation
Validate the finding that tabu_distance=0.04 with sigma 0.18/0.22 beats baseline.

Prior finding: 1.1535 @ 53.1 min (+0.0071 vs baseline 1.1464)

Run 3 validation tests to establish confidence intervals.
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


def run_single_validation(run_num, config, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Validation Run {run_num} ===")

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

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    # Compute RMSE by source count
    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")

    return {
        'run': run_num,
        'score': float(score),
        'avg_n_cands': float(avg_n_cands),
        'projected_400_min': float(projected_400),
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.1464
    PRIOR_RESULT = 1.1535

    # Config to validate: tabu_distance=0.04 with sigma 0.18/0.22
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,  # NEW: Increased from 0.03
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("TABU_DISTANCE=0.04 VALIDATION (3 Runs)")
    print("="*60)
    print(f"Config: sigma 0.18/0.22, tabu_distance=0.04")
    print(f"Prior result: {PRIOR_RESULT} @ 53.1 min (+0.0071 vs baseline)")
    print(f"Baseline: {BASELINE} @ 51.2 min (tabu_distance=0.03)")

    results = []

    for run_num in range(1, 4):  # 3 validation runs
        result = run_single_validation(run_num, config, data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY (3 Runs)")
    print("="*60)
    print(f"Prior finding: {PRIOR_RESULT} @ 53.1 min (+0.0071 vs baseline)")
    print(f"Baseline: {BASELINE} @ 51.2 min")
    print()
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER"
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    print()
    print(f"Statistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Mean Time:  {mean_time:.1f} +/- {std_time:.1f} min")
    print(f"  Score Range: [{min(scores):.4f}, {max(scores):.4f}]")
    print()

    # Compare to prior finding
    delta_vs_prior = mean_score - PRIOR_RESULT
    print(f"  vs Prior finding ({PRIOR_RESULT}): {delta_vs_prior:+.4f}")

    # Compare to baseline
    delta_vs_baseline = mean_score - BASELINE
    print(f"  vs Baseline ({BASELINE}): {delta_vs_baseline:+.4f}")

    if mean_score > BASELINE:
        print(f"\nCONCLUSION: VALIDATED! Mean {mean_score:.4f} > baseline {BASELINE}")
        print(f"  Improvement: +{delta_vs_baseline:.4f}")
    else:
        print(f"\nCONCLUSION: NOT VALIDATED. Mean {mean_score:.4f} <= baseline {BASELINE}")
        print(f"  Prior finding may have been lucky")

    # Save results
    summary = {
        'config': config,
        'validation_runs': results,
        'statistics': {
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'mean_time_min': float(mean_time),
            'std_time_min': float(std_time),
            'min_score': float(min(scores)),
            'max_score': float(max(scores))
        },
        'vs_baseline': float(delta_vs_baseline),
        'vs_prior_finding': float(delta_vs_prior),
        'conclusion': 'VALIDATED' if mean_score > BASELINE else 'NOT_VALIDATED'
    }

    with open('run_output.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
