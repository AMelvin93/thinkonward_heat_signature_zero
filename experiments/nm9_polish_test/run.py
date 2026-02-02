"""
Experiment: nm9_polish_test
Test 9 NM iterations (refine_maxiter=9) vs 8.

Current best: refine_maxiter=8 = 1.1437 @ 42.2 min
Previous tests: maxiter=6 (1.1391) < maxiter=8 (1.1437) > maxiter=10 (1.1389)
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


def run_experiment(config, run_num, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Run {run_num}: refine_maxiter=9 ===")

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

    CURRENT_BEST = 1.1437

    print(f"Run {run_num}: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"vs Current Best (1.1437): {score - CURRENT_BEST:+.4f}")

    return {
        'run': run_num,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
    }


def main():
    data = load_data()

    CURRENT_BEST = 1.1437

    # Config with refine_maxiter=9
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 9,
        'enable_tabu_hopping': False,
        'n_perturbations': 4,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    results = []
    for run_num in range(1, 4):
        result = run_experiment(config, run_num, data)
        results.append(result)

    # Summary
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    print("\n" + "="*60)
    print("SUMMARY: Refine Maxiter=9")
    print("="*60)
    print(f"Current best (maxiter=8): 1.1437 @ 42.2 min")
    print()
    for r in results:
        delta = r['score'] - CURRENT_BEST
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f})")

    print()
    print(f"Mean score: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    print(f"Mean time:  {np.mean(times):.1f} min")

    if np.mean(scores) > CURRENT_BEST:
        improvement = (np.mean(scores) - CURRENT_BEST) / CURRENT_BEST * 100
        print(f"\nImproved! {improvement:.2f}%")
    else:
        print(f"\nDid not improve over current best")

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
