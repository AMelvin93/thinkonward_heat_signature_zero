"""
Experiment: tabu004_scale_sweep
Test perturbation scales 0.045, 0.05, 0.055 with tabu_distance=0.04.

Prior finding: perturbation_scale=0.05 is optimal (with tabu=0.03)
Hypothesis: Optimal scale may change with tabu_distance=0.04.
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


def run_config(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== {config_name} ===")
    print(f"  perturbation_scale: {config['perturbation_scale']}")

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
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    BASELINE = 1.1496  # tabu 0.04 with scale 0.05

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"vs Baseline (1.1496): {score - BASELINE:+.4f}")

    return {
        'config_name': config_name,
        'perturbation_scale': config['perturbation_scale'],
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE = 1.1496

    # Base config with tabu_distance=0.04
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("PERTURBATION SCALE SWEEP WITH TABU_DISTANCE=0.04")
    print("="*60)
    print(f"Baseline: 1.1496 @ 55.4 min (scale=0.05, tabu=0.04)")
    print(f"Testing: scales 0.045, 0.05, 0.055")

    results = []

    # Test 3 scales
    for scale in [0.045, 0.05, 0.055]:
        config = {**base_config, 'perturbation_scale': scale}
        result = run_config(config, f"scale_{scale}", data)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Perturbation Scale Sweep")
    print("="*60)
    print(f"\nBaseline: 1.1496 @ 55.4 min (scale=0.05)")
    print()

    for r in results:
        delta = r['score'] - BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        marker = "**" if r['score'] > BASELINE else ""
        print(f"  scale={r['perturbation_scale']}: {marker}{r['score']:.4f}{marker} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    best = max(results, key=lambda x: x['score'])
    print(f"\nBest: scale={best['perturbation_scale']} with score {best['score']:.4f}")

    if best['perturbation_scale'] != 0.05:
        print(f"\nNEW OPTIMAL SCALE FOUND: {best['perturbation_scale']}")
    else:
        print(f"\n0.05 CONFIRMED as optimal scale")

    with open('run_output.json', 'w') as f:
        json.dump({
            'baseline': BASELINE,
            'results': results,
            'best_scale': best['perturbation_scale'],
            'best_score': best['score']
        }, f, indent=2)


if __name__ == '__main__':
    main()
