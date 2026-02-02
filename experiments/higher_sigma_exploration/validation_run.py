"""
Validate the new best: sigma 0.20/0.25 = 1.1474 @ 42.2 min
Run 3 more times to confirm reproducibility.
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


def run_validation(run_num, config, data):
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

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")

    return {
        'run': run_num,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    # New best config
    config = {
        'sigma0_1src': 0.20,
        'sigma0_2src': 0.25,
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
    print("VALIDATING NEW BEST: sigma 0.20/0.25")
    print("="*60)
    print(f"Initial result: 1.1474 @ 42.2 min")
    print(f"Baseline to beat: 1.1433 @ 46.5 min")

    results = []

    for run_num in range(1, 4):
        result = run_validation(run_num, config, data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_time = np.mean(times)

    # Add initial run
    all_scores = [1.1474] + scores
    all_mean = np.mean(all_scores)
    all_std = np.std(all_scores)

    print("\n" + "="*60)
    print("VALIDATION SUMMARY (4 Total Runs)")
    print("="*60)
    print(f"Baseline: 1.1433 @ 46.5 min")
    print()
    print(f"  Initial run: 1.1474 @ 42.2 min")
    for r in results:
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min")

    print()
    print(f"Statistics (4 runs):")
    print(f"  Mean Score: {all_mean:.4f} +/- {all_std:.4f}")
    print(f"  Score Range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    if all_mean > 1.1433:
        print(f"\nNEW BEST VALIDATED! +{all_mean - 1.1433:.4f} vs baseline")
    else:
        print(f"\nImprovement NOT validated ({all_mean - 1.1433:+.4f})")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
