"""
Experiment: Asymmetric budget allocation for 2-source

Current 4pert_nm2: 1.1482 @ 51.7 min (20/44 fevals)
Budget remaining: 8.3 min

Hypothesis:
- 1-source problems are easier and might need fewer fevals
- 2-source problems are the accuracy bottleneck
- Trade-off: reduce 1-source fevals, increase 2-source fevals

Config variations:
1. 16/48 fevals (reduce 1src, increase 2src) - budget neutral
2. 18/46 fevals (slight redistribution)
3. 20/50 fevals (use 8 min budget for 2src)
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


def run_single_test(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Testing: {config_name} ===")

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
    time_1src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 1]
    time_2src = [r['time_s'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"  Time 1src: {np.mean(time_1src):.2f}s, Time 2src: {np.mean(time_2src):.2f}s")
    status = "IN BUDGET" if projected_400 <= 60.0 else "OVER BUDGET"
    print(f"  Budget: {status}")

    return {
        'config': config_name,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'time_1src_avg': float(np.mean(time_1src)),
        'time_2src_avg': float(np.mean(time_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE_SCORE = 1.1482
    BASELINE_TIME = 51.7

    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,
        'perturb_nm_iters': 2,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    configs = [
        {
            'name': '16/48 (redistribute)',
            'config': {**base_config, 'max_fevals_1src': 16, 'max_fevals_2src': 48}
        },
        {
            'name': '18/46 (slight shift)',
            'config': {**base_config, 'max_fevals_1src': 18, 'max_fevals_2src': 46}
        },
        {
            'name': '20/48 (use buffer)',
            'config': {**base_config, 'max_fevals_1src': 20, 'max_fevals_2src': 48}
        },
    ]

    print("="*60)
    print("EXPERIMENT: Asymmetric Budget Allocation for 2-Source")
    print("="*60)
    print(f"Baseline (20/44 fevals): {BASELINE_SCORE} @ {BASELINE_TIME} min")
    print(f"Budget remaining: {60 - BASELINE_TIME:.1f} min")

    results = []
    for cfg in configs:
        result = run_single_test(cfg['config'], cfg['name'], data)
        results.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n| Config | Score | Time | vs Baseline | Budget |")
    print("|--------|-------|------|-------------|--------|")
    print(f"| baseline (20/44) | {BASELINE_SCORE:.4f} | {BASELINE_TIME:.1f} | -- | IN |")

    for r in results:
        delta = r['score'] - BASELINE_SCORE
        status = "IN" if r['in_budget'] else "OVER"
        print(f"| {r['config']} | {r['score']:.4f} | {r['projected_400_min']:.1f} | {delta:+.4f} | {status} |")

    # Find best in-budget config
    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config']} @ {best['score']:.4f}")
        if best['score'] > BASELINE_SCORE:
            print("*** NEW BEST CONFIG FOUND! ***")

    with open('run_output.json', 'w') as f:
        json.dump({
            'baseline': {'score': BASELINE_SCORE, 'time_min': BASELINE_TIME},
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
