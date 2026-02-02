"""
Experiment: adaptive_nm_by_source
Use different NM polish iterations based on source count:
- 1-source: fewer iterations (converges faster)
- 2-source: more iterations (higher complexity)
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


def process_sample_adaptive(args):
    """Process sample with adaptive NM iterations based on source count."""
    sample_idx, sample, meta, base_config, nm_1src, nm_2src = args

    n_sources = sample['n_sources']

    # Set adaptive refine_maxiter
    config = base_config.copy()
    if n_sources == 1:
        config['refine_maxiter'] = nm_1src
    else:
        config['refine_maxiter'] = nm_2src

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
            'n_sources': n_sources,
            'n_candidates': n_candidates,
            'time_s': elapsed,
            'refine_maxiter': config['refine_maxiter'],
            'success': True
        }
    except Exception as e:
        return {
            'idx': sample_idx,
            'rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'time_s': 0,
            'refine_maxiter': config['refine_maxiter'],
            'success': False,
        }


def run_experiment(base_config, nm_1src, nm_2src, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== {config_name} ===")
    print(f"  1-source: {nm_1src} NM iters, 2-source: {nm_2src} NM iters")

    args_list = [(i, samples[i], meta, base_config, nm_1src, nm_2src) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample_adaptive, args): args[0] for args in args_list}
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

    time_1src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 1]
    time_2src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 2]

    BASELINE = 1.1464

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"  1-src RMSE: {np.mean(rmse_1src):.4f}, avg time: {np.mean(time_1src):.1f}s")
    print(f"  2-src RMSE: {np.mean(rmse_2src):.4f}, avg time: {np.mean(time_2src):.1f}s")
    print(f"vs Baseline (1.1464): {score - BASELINE:+.4f}")
    print(f"Budget remaining: {60.0 - projected_400:.1f} min")

    return {
        'config_name': config_name,
        'nm_1src': nm_1src,
        'nm_2src': nm_2src,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'time_1src_mean_s': float(np.mean(time_1src)),
        'time_2src_mean_s': float(np.mean(time_2src)),
        'budget_remaining_min': 60.0 - projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.1464

    # Base config (without refine_maxiter - set adaptively)
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
    }

    results = []

    # Run 1: 1-src=4, 2-src=8 (as specified in experiment)
    result1 = run_experiment(base_config, nm_1src=4, nm_2src=8,
                             config_name="Run1_nm4_nm8", data=data)
    results.append(result1)

    # Run 2: 1-src=4, 2-src=10 (more polish for 2-src)
    result2 = run_experiment(base_config, nm_1src=4, nm_2src=10,
                             config_name="Run2_nm4_nm10", data=data)
    results.append(result2)

    # Run 3: 1-src=6, 2-src=8 (balanced)
    result3 = run_experiment(base_config, nm_1src=6, nm_2src=8,
                             config_name="Run3_nm6_nm8", data=data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Adaptive NM by Source Count")
    print("="*60)
    print(f"Baseline (uniform 8 NM): 1.1464 @ 51.2 min")
    print()
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")
        print(f"    1-src: RMSE={r['rmse_1src_mean']:.4f}, time={r['time_1src_mean_s']:.1f}s")
        print(f"    2-src: RMSE={r['rmse_2src_mean']:.4f}, time={r['time_2src_mean_s']:.1f}s")

    best = max(results, key=lambda x: x['score'])
    print()
    print(f"Best: {best['config_name']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    if best['score'] > BASELINE:
        print(f"\nNEW BEST FOUND! +{best['score'] - BASELINE:.4f}")
    else:
        print(f"\nNo improvement over baseline ({best['score'] - BASELINE:+.4f})")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
