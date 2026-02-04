"""
Experiment: sigma014_019_4pert_nm2

Hypothesis: Combine the best sigma (0.14/0.19) with the best perturbation
strategy (4pert nm2 scale06) that we found in 3pert_nm4_final.

Previous results:
- sigma 0.14/0.19 + 2pert nm3: 1.1675 @ 53.4 min (single run, scoring method differs)
- sigma 0.14/0.19 + 4pert nm3: 1.1684 @ 61.3 min (OVER BUDGET by 1.3 min)
- sigma 0.18/0.22 + 4pert nm2: 1.1549 @ 57.5 min (3-run mean, per-sample scoring)

Key insight: The previous 4pert test used nm_iters=3. Our 3pert_nm4_final
experiment found nm_iters=2 is optimal (faster, fits budget). With nm2 instead
of nm3, 4perturb should save ~2-3 minutes, fitting in budget.

Also testing:
- sigma 0.14/0.19 + 3pert nm3 as a timing-safe alternative
- sigma 0.14/0.19 + 2pert nm4 as a baseline with correct scoring
"""

import os
import sys
import pickle
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    """Correct per-sample scoring matching competition formula."""
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


def run_single_test(run_num, config, config_name, samples, meta):
    n_samples = len(samples)
    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    print(f"\n=== {config_name} - Run {run_num} ===")

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

    # Correct per-sample scoring
    scores = [calculate_sample_score(r['rmse'], r['n_candidates'])
              for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    # Breakdown by source count
    one_src = [r for r in results if r['n_sources'] == 1 and r['success']]
    two_src = [r for r in results if r['n_sources'] == 2 and r['success']]
    one_src_rmse = np.mean([r['rmse'] for r in one_src]) if one_src else 0
    two_src_rmse = np.mean([r['rmse'] for r in two_src]) if two_src else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"  Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  1-src RMSE: {one_src_rmse:.4f}, 2-src RMSE: {two_src_rmse:.4f}")

    return {
        'run': run_num,
        'config': config_name,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'in_budget': projected_400 <= 60,
        'one_src_rmse': float(one_src_rmse),
        'two_src_rmse': float(two_src_rmse),
    }


def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']

    print("=" * 70)
    print("EXPERIMENT: sigma 0.14/0.19 + perturbation tuning")
    print("=" * 70)
    print()
    print("Previous best configs:")
    print("  sigma 0.14/0.19 + 2pert nm3: ~1.167 @ 53 min (SINGLE RUN, different scoring)")
    print("  sigma 0.18/0.22 + 4pert nm2: 1.1549 @ 57.5 min (3-run mean, per-sample scoring)")
    print("  sigma 0.14/0.19 + 4pert nm3: 1.168 @ 61.3 min (OVER BUDGET)")
    print()
    print("Hypothesis: sigma 0.14/0.19 + 4pert nm2 fits in budget and scores highest")
    print("=" * 70)

    # Configs to test
    configs = {
        '4pert_nm2_scale06': {
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturbation_scale': 0.06,
            'perturb_nm_iters': 2,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
        '2pert_nm3_baseline': {
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturbation_scale': 0.06,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
        '3pert_nm3_scale06': {
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 3,
            'perturbation_scale': 0.06,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    }

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n{'=' * 70}")
        print(f"CONFIG: {config_name}")
        print(f"  sigma: {config['sigma0_1src']}/{config['sigma0_2src']}")
        print(f"  perturbations: {config['n_perturbations']}, nm_iters: {config['perturb_nm_iters']}")
        print(f"  scale: {config['perturbation_scale']}, tabu_dist: {config['tabu_distance']}")
        print(f"{'=' * 70}")

        runs = []
        for run_num in range(1, 4):
            r = run_single_test(run_num, config, config_name, samples, meta)
            runs.append(r)

        scores = [r['score'] for r in runs]
        times = [r['projected_400_min'] for r in runs]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_time = np.mean(times)
        in_budget_count = sum(1 for r in runs if r['in_budget'])

        print(f"\n--- {config_name} SUMMARY ---")
        for r in runs:
            status = "IN" if r['in_budget'] else "OVER"
            print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min [{status}]")
        print(f"  Mean: {mean_score:.4f} +/- {std_score:.4f}")
        print(f"  Time: {mean_time:.1f} min")
        print(f"  In budget: {in_budget_count}/3")

        all_results[config_name] = {
            'runs': runs,
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'mean_time': float(mean_time),
            'in_budget_count': in_budget_count,
            'best_score': float(max(scores)),
        }

        # Save incrementally
        with open('run_output.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Config':<25} {'Mean':>8} {'Std':>8} {'Best':>8} {'Time':>8} {'Budget':>8}")
    print("-" * 70)
    for name, data in all_results.items():
        print(f"{name:<25} {data['mean_score']:>8.4f} {data['std_score']:>8.4f} "
              f"{data['best_score']:>8.4f} {data['mean_time']:>7.1f}m "
              f"{data['in_budget_count']}/3")

    # Find best
    in_budget = {k: v for k, v in all_results.items() if v['in_budget_count'] >= 2}
    if in_budget:
        best_name = max(in_budget, key=lambda k: in_budget[k]['mean_score'])
        best = in_budget[best_name]
        print(f"\n*** BEST CONFIG: {best_name} ***")
        print(f"    Mean: {best['mean_score']:.4f} +/- {best['std_score']:.4f}")
        print(f"    Best: {best['best_score']:.4f}")
        print(f"    Time: {best['mean_time']:.1f} min")

        # Compare to previous results
        gap_to_top10 = 1.1585 - best['mean_score']
        print(f"\n    Gap to Top 10 (1.1585): {gap_to_top10:.4f}")
        if best['best_score'] >= 1.1585:
            print(f"    *** Best run BEATS Top 10! ***")

    print("\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
