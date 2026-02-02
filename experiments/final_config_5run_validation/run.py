"""
Experiment: final_config_5run_validation
Validate final config with 5 runs to establish confidence intervals.

Config:
- sigma 0.14/0.19
- fevals 20/44
- 2 perturbations
- perturbation_scale 0.06
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
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.1464

    # Config to validate (FIXED: enable_tabu_hopping must be True!)
    config = {
        'sigma0_1src': 0.14,
        'sigma0_2src': 0.19,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,  # FIXED: Was False, which disabled perturbations
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.06,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("FINAL CONFIG 5-RUN VALIDATION")
    print("="*60)
    print(f"Config: sigma 0.14/0.19, fevals 20/44, 2 perturb, scale 0.06")
    print(f"Baseline to beat: {BASELINE}")

    results = []

    for run_num in range(1, 6):
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
    print("VALIDATION SUMMARY (5 Runs)")
    print("="*60)
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

    if mean_score > BASELINE:
        print(f"\nCONCLUSION: Improvement confirmed! +{mean_score - BASELINE:.4f} vs baseline")
    else:
        print(f"\nCONCLUSION: No improvement ({mean_score - BASELINE:+.4f} vs baseline)")

    print("\n" + json.dumps(results, indent=2))

    # Save final summary
    summary = {
        'config': config,
        'validation_runs': results,
        'statistics': {
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_time_min': mean_time,
            'std_time_min': std_time,
            'min_score': min(scores),
            'max_score': max(scores)
        },
        'vs_baseline': mean_score - BASELINE,
        'conclusion': 'IMPROVED' if mean_score > BASELINE else 'NO_IMPROVEMENT'
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
