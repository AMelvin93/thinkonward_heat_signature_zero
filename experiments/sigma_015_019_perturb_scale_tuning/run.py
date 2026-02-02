"""
Run script for sigma_015_019_perturb_scale_tuning experiment.

Tests different perturbation scales with optimal sigma 0.15/0.19 config.
Current best: perturbation_scale=0.05, score=1.1730 @ 50.4 min

Hypothesis: Different perturbation scales might improve results
- Smaller scale (0.03): More local refinement, might help if best is close
- Larger scale (0.07): Wider exploration, might escape local optima better
"""

import os
import sys
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import TabuBasinHoppingOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def process_sample(args):
    sample_idx, sample, meta, config = args
    optimizer = TabuBasinHoppingOptimizer(**config)
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        return sample_idx, best_rmse, n_sims, None
    except Exception as e:
        return sample_idx, float('inf'), 0, str(e)


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"Perturbation scale: {config.get('perturbation_scale')}")
    print(f"N perturbations: {config.get('n_perturbations')}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    rmses = {}
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            sample_idx, best_rmse, n_sims, error = future.result()
            rmses[sample_idx] = best_rmse

            if error:
                errors.append((sample_idx, error))

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{n_samples} samples, elapsed: {elapsed/60:.1f} min")

    elapsed_time = time.time() - start_time

    rmse_1src = []
    rmse_2src = []
    for idx, rmse in rmses.items():
        if rmse < float('inf'):
            n_sources = samples[idx]['n_sources']
            if n_sources == 1:
                rmse_1src.append(rmse)
            else:
                rmse_2src.append(rmse)

    avg_rmse_1src = sum(rmse_1src) / len(rmse_1src) if rmse_1src else float('inf')
    avg_rmse_2src = sum(rmse_2src) / len(rmse_2src) if rmse_2src else float('inf')

    overall_rmse = (avg_rmse_1src + avg_rmse_2src) / 2
    n_candidates = 3
    score = 1 / (1 + overall_rmse) + 0.3 * (n_candidates / 3)

    projected_400_min = (elapsed_time / n_samples) * 400 / 60

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f} (n={len(rmse_1src)})")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f} (n={len(rmse_2src)})")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  Projected 400: {projected_400_min:.1f} min")
    print(f"  Errors:        {len(errors)}")

    in_budget = projected_400_min <= 60
    budget_remaining = 60 - projected_400_min
    print(f"  In budget:     {in_budget} (remaining: {budget_remaining:.1f} min)")

    return {
        'config': config_name,
        'score': score,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'perturbation_scale': config.get('perturbation_scale'),
        'in_budget': in_budget,
        'budget_remaining': budget_remaining,
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")
    print(f"\nBASELINE: sigma 0.15/0.19, scale=0.05, n_perturbations=2 = 1.1730 @ 50.4 min")
    print("Testing perturbation scales to find optimal value...")

    # Base config from best (sigma 0.15/0.19)
    base_config = {
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
    }

    # Phase 1: Test 0.03, 0.05 (baseline), 0.07
    configs = [
        {**base_config, 'perturbation_scale': 0.03},
        {**base_config, 'perturbation_scale': 0.05},  # Baseline for comparison
        {**base_config, 'perturbation_scale': 0.07},
    ]
    config_names = ['scale_003', 'scale_005_baseline', 'scale_007']

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        # Save intermediate results
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Analyze and determine if we need more runs
    print("\n" + "="*70)
    print("PHASE 1 RESULTS: PERTURBATION SCALE SWEEP")
    print("="*70)
    print(f"{'Config':<25} {'Scale':>6} {'Score':>8} {'Time':>8} {'Proj 400':>10} {'In Budget':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<25} {r['perturbation_scale']:>6.2f} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['projected_400_min']:>9.1f}m {'YES' if r['in_budget'] else 'NO':>10}")

    # Find best in-budget
    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config']} with score {best['score']:.4f}")

        # Compare to baseline
        baseline_result = next((r for r in results if r['perturbation_scale'] == 0.05), None)
        if baseline_result:
            delta = best['score'] - baseline_result['score']
            print(f"Delta vs baseline (0.05): {delta:+.4f}")

        delta_vs_best = best['score'] - 1.1730
        print(f"Delta vs claimed best (1.1730): {delta_vs_best:+.4f}")

    # Phase 2: If needed, explore further
    best_scale = max(results, key=lambda x: x['score'])['perturbation_scale']
    print(f"\nBest scale so far: {best_scale}")

    if best_scale == 0.03:
        # Try even smaller
        print("\nPHASE 2: Testing smaller scales (0.02)...")
        config = {**base_config, 'perturbation_scale': 0.02}
        result = run_experiment(config, 'scale_002', data)
        results.append(result)
    elif best_scale == 0.07:
        # Try even larger
        print("\nPHASE 2: Testing larger scales (0.09, 0.10)...")
        for scale in [0.09, 0.10]:
            config = {**base_config, 'perturbation_scale': scale}
            result = run_experiment(config, f'scale_{int(scale*100):03d}', data)
            results.append(result)
    else:
        # 0.05 is best, try fine-tuning around it
        print("\nPHASE 2: 0.05 is best, testing 0.04 and 0.06...")
        for scale in [0.04, 0.06]:
            config = {**base_config, 'perturbation_scale': scale}
            result = run_experiment(config, f'scale_{int(scale*100):03d}', data)
            results.append(result)

    # Final save
    with open('run_output.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Config':<25} {'Scale':>6} {'Score':>8} {'Time':>8} {'Proj 400':>10} {'In Budget':>10}")
    print("-"*70)
    for r in sorted(results, key=lambda x: x['perturbation_scale']):
        print(f"{r['config']:<25} {r['perturbation_scale']:>6.2f} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['projected_400_min']:>9.1f}m {'YES' if r['in_budget'] else 'NO':>10}")

    in_budget_results = [r for r in results if r['in_budget']]
    if in_budget_results:
        best = max(in_budget_results, key=lambda x: x['score'])
        print(f"\n*** BEST IN-BUDGET: {best['config']} ***")
        print(f"    Score: {best['score']:.4f}")
        print(f"    Perturbation scale: {best['perturbation_scale']}")
        print(f"    Time: {best['projected_400_min']:.1f} min projected")
        print(f"    Delta vs baseline (1.1730): {best['score'] - 1.1730:+.4f}")


if __name__ == '__main__':
    main()
