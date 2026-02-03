"""
Test: sigma_015_019 with 1 perturbation

On faster system: 1.1556 @ 44.7 min
With 1.37x slowdown: ~61 min (borderline)

But our base runs faster (~42 min vs ~49 min for no_perturb),
so 1-perturb might fit in budget.
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

    # Config with 1 perturbation
    config = {
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,  # ENABLE perturbation
        'n_perturbations': 1,         # Only 1 perturbation
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 2,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("TEST: sigma_015_019 with 1 perturbation")
    print("=" * 70)
    print("Expected from faster system: 1.1556 @ 44.7 min")
    print("Our no_perturb baseline: 1.1386 @ 49.3 min")
    print("Our sigma_015_019 no_perturb: 1.1321 @ 42.0 min")

    result = run_single_test(1, config, samples, meta)

    status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
    print(f"\n*** RESULT: Score={result['score']:.4f}, Time={result['projected_400_min']:.1f} min [{status}] ***")

    if result['in_budget']:
        print(f"\nImprovement vs no_perturb baseline: {result['score'] - 1.1386:+.4f}")
        print(f"Gap to Top 10 (1.1585): {1.1585 - result['score']:.4f}")
    else:
        print(f"\nOver budget by {result['projected_400_min'] - 60:.1f} min")

    with open('test_1perturb_output.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
