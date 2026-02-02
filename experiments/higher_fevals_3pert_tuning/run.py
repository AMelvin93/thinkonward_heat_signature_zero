"""
Experiment: higher_fevals_3pert_tuning

With the validated 3-pert config running at 55.7 min average, we have ~4.3 min buffer.
Test if increasing CMA-ES fevals from 20/44 to 22/46 or 24/48 improves accuracy.

Current validated:
- 3-pert + tabu 0.04, fevals 20/44: 1.1475 @ 55.7 min

Hypothesis: More fevals = better CMA-ES convergence = higher score
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


def run_single_test(run_num, config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Run {run_num}: {config_name} ===")

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

    BASELINE = 1.1475  # Validated 3-pert config

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"  Fevals: {config.get('max_fevals_1src', 20)}/{config.get('max_fevals_2src', 44)}")
    print(f"vs Validated Baseline (1.1475): {score - BASELINE:+.4f}")

    return {
        'run': run_num,
        'config_name': config_name,
        'fevals_1src': config.get('max_fevals_1src', 20),
        'fevals_2src': config.get('max_fevals_2src', 44),
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    VALIDATED_BASELINE = 1.1475

    # Base config (validated 3-pert)
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    # Test configurations with increasing fevals
    configs = [
        ('fevals_20_44_baseline', {'max_fevals_1src': 20, 'max_fevals_2src': 44}),
        ('fevals_22_46', {'max_fevals_1src': 22, 'max_fevals_2src': 46}),
        ('fevals_24_48', {'max_fevals_1src': 24, 'max_fevals_2src': 48}),
    ]

    print("="*60)
    print("HIGHER FEVALS TUNING: 3-pert + tabu 0.04")
    print("="*60)
    print(f"Validated baseline: 1.1475 @ 55.7 min (fevals 20/44)")
    print(f"Time budget available: ~4.3 min")
    print(f"Testing: fevals 22/46, 24/48")

    results = []

    for config_name, fevals_config in configs:
        config = {**base_config, **fevals_config}
        result = run_single_test(len(results)+1, config, config_name, data)
        results.append(result)

        # Early stop if already over budget
        if not result['in_budget']:
            print(f"\n[WARNING] Config {config_name} OVER BUDGET ({result['projected_400_min']:.1f} min)")
            print("Remaining configs may also be over budget. Continuing anyway...")

    # Summary
    print("\n" + "="*60)
    print("FEVALS TUNING SUMMARY")
    print("="*60)

    print("\nResults:")
    for r in results:
        delta = r['score'] - VALIDATED_BASELINE
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  {r['config_name']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    # Find best in-budget
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config_name']} ({best['score']:.4f} @ {best['projected_400_min']:.1f} min)")
        improvement = best['score'] - VALIDATED_BASELINE
        print(f"vs Validated baseline: {improvement:+.4f}")
    else:
        print("\n[WARNING] No configs within budget!")

    # Save results
    with open('run_output.json', 'w') as f:
        json.dump({
            'validated_baseline': VALIDATED_BASELINE,
            'results': results,
            'best_in_budget': best['config_name'] if in_budget_results else None
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
