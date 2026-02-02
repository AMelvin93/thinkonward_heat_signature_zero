"""
Experiment: 4perturb_scale_006
Test 4 perturbations with scale 0.06 (validating prior findings).

Prior evidence: Scale 0.05 > 0.06, but we validate to be sure.
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


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    scale = config.get('perturbation_scale')
    print(f"\n=== {config_name} (scale={scale}) ===")

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

    BASELINE = 1.1464

    print(f"Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")
    print(f"vs Baseline (1.1464): {score - BASELINE:+.4f}")
    print(f"Budget remaining: {60.0 - projected_400:.1f} min")

    return {
        'config_name': config_name,
        'scale': scale,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
        'budget_remaining_min': 60.0 - projected_400,
        'in_budget': projected_400 <= 60.0
    }


def main():
    data = load_data()

    BASELINE = 1.1464

    # Base config
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 4,
        'perturb_nm_iters': 3,
    }

    results = []

    # Run 1: Scale 0.06 validation
    config1 = {**base_config, 'perturbation_scale': 0.06}
    result1 = run_experiment(config1, "Run1_scale_006", data)
    results.append(result1)

    # Decision after Run 1
    print("\n" + "="*60)
    print("TIME ANALYSIS after Run 1:")
    print(f"  Time: {result1['projected_400_min']:.1f} min")
    print(f"  Budget remaining: {result1['budget_remaining_min']:.1f} min")
    print(f"  Score: {result1['score']:.4f}")

    # Check if scale 0.06 is competitive
    if result1['score'] < BASELINE - 0.01:
        print("  Scale 0.06 significantly underperforms. Testing alternatives...")

    # Run 2: Scale 0.055 (intermediate)
    config2 = {**base_config, 'perturbation_scale': 0.055}
    result2 = run_experiment(config2, "Run2_scale_0055", data)
    results.append(result2)

    print("\n" + "="*60)
    print("TIME ANALYSIS after Run 2:")
    print(f"  Time: {result2['projected_400_min']:.1f} min")
    print(f"  Budget remaining: {result2['budget_remaining_min']:.1f} min")
    print(f"  Score: {result2['score']:.4f}")

    # Run 3: Scale 0.07 (larger)
    config3 = {**base_config, 'perturbation_scale': 0.07}
    result3 = run_experiment(config3, "Run3_scale_007", data)
    results.append(result3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: 4 Perturbations Scale Tuning")
    print("="*60)
    print(f"Baseline: 1.1464 @ 51.2 min")
    print()
    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  scale={r['scale']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    best = max(results, key=lambda x: x['score'])
    print()
    print(f"Best: scale={best['scale']} = {best['score']:.4f} @ {best['projected_400_min']:.1f} min")

    if best['score'] > BASELINE:
        print(f"\nNEW BEST FOUND! +{best['score'] - BASELINE:.4f}")
    else:
        print(f"\nNo improvement over baseline ({best['score'] - BASELINE:+.4f})")

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
