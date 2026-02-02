"""
Experiment: adaptive_nm_by_source_count
Test adaptive NM iterations based on source count:
- 1-source: 4 NM iterations (simpler, converges faster)
- 2-source: 10 NM iterations (more complex, needs more polish)

Hypothesis: 2-source problems need more polish, 1-source converges faster.
Using validated config (sigma 0.18/0.22, tabu=0.04).
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
    # Adaptive NM iterations
    refine_maxiter = nm_1src if n_sources == 1 else nm_2src

    config = {**base_config, 'refine_maxiter': refine_maxiter}
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
            'nm_iters': refine_maxiter,
            'success': True
        }
    except Exception as e:
        return {
            'idx': sample_idx,
            'rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'time_s': 0,
            'nm_iters': refine_maxiter,
            'success': False,
        }


def run_adaptive_experiment(base_config, nm_1src, nm_2src, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  NM iterations: 1-source={nm_1src}, 2-source={nm_2src}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, base_config, nm_1src, nm_2src) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample_adaptive, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples}, elapsed: {elapsed/60:.1f} min")

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
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f} (n={len(rmse_1src)})")
    print(f"  RMSE 2src: {np.mean(rmse_2src):.4f} (n={len(rmse_2src)})")

    in_budget = projected_400 <= 60.0
    budget_remaining = 60.0 - projected_400
    print(f"  In budget: {in_budget}, remaining: {budget_remaining:.1f} min")

    return {
        'config_name': config_name,
        'nm_1src': nm_1src,
        'nm_2src': nm_2src,
        'score': float(score),
        'avg_n_cands': float(avg_n_cands),
        'projected_400_min': float(projected_400),
        'rmse_1src_mean': float(np.mean(rmse_1src)),
        'rmse_2src_mean': float(np.mean(rmse_2src)),
        'n_1src': len(rmse_1src),
        'n_2src': len(rmse_2src),
        'in_budget': in_budget,
        'budget_remaining': float(budget_remaining),
    }


def main():
    data = load_data()

    BASELINE = 1.1496  # @ 55.4 min (validated tabu_004, NM=8 for all)

    print("="*60)
    print("ADAPTIVE NM BY SOURCE COUNT")
    print("="*60)
    print(f"Baseline: 1.1496 @ 55.4 min (NM=8 for all)")
    print(f"Hypothesis: 1-source needs less polish, 2-source needs more")

    # Base config with validated settings
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Test configurations
    configs = [
        # Run 1: Adaptive 4/10 (less for 1-src, more for 2-src)
        (4, 10, 'adaptive_4_10'),
        # Run 2: Adaptive 6/10 (moderate for 1-src, more for 2-src)
        (6, 10, 'adaptive_6_10'),
        # Run 3: Baseline 8/8 for comparison
        (8, 8, 'baseline_8_8'),
    ]

    results = []
    for nm_1src, nm_2src, name in configs:
        result = run_adaptive_experiment(base_config, nm_1src, nm_2src, name, data)
        results.append(result)

        # Save intermediate
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "="*70)
    print("ADAPTIVE NM RESULTS")
    print("="*70)
    print(f"{'Config':<20} {'NM 1src':>8} {'NM 2src':>8} {'Score':>8} {'Proj 400':>10} {'vs Baseline':>12}")
    print("-"*75)
    for r in results:
        delta = r['score'] - BASELINE
        print(f"{r['config_name']:<20} {r['nm_1src']:>8} {r['nm_2src']:>8} {r['score']:>8.4f} {r['projected_400_min']:>9.1f}m {delta:>+12.4f}")

    # Find best in-budget
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config_name']} ***")
        print(f"    Score: {best['score']:.4f} ({best['score'] - BASELINE:+.4f} vs baseline)")
        print(f"    NM iterations: 1-src={best['nm_1src']}, 2-src={best['nm_2src']}")
        print(f"    Time: {best['projected_400_min']:.1f} min")

        if best['score'] > BASELINE:
            print(f"\n*** SUCCESS! Better than baseline! ***")
        else:
            print(f"\n*** FAILED: Did not beat baseline ***")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
