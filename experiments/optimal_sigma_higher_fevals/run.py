"""
Run script for optimal_sigma_higher_fevals experiment.

Tests combining optimal sigma (0.15/0.19) with higher fevals (22/40, 24/44).

Current best: sigma 0.15/0.19 + 20/36 fevals + 2 perturbations = 1.1730 @ 50.4 min

Hypothesis: Higher fevals improved 2-source RMSE in nm4_perturb1_fevals_22_40 experiment.
Combining with optimal sigma may push score even higher while staying in budget.
"""

import os
import sys
import pickle
import time
import json
from datetime import datetime
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
    print(f"Fevals: 1src={config.get('max_fevals_1src')}, 2src={config.get('max_fevals_2src')}")
    print(f"NM iters: {config.get('refine_maxiter')}, Perturbations: {config.get('n_perturbations')}")
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
                print(f"  Sample {sample_idx}: ERROR - {error}")

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{n_samples} samples, elapsed: {elapsed/60:.1f} min")

    elapsed_time = time.time() - start_time

    # Compute metrics
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

    # Competition formula
    overall_rmse = (avg_rmse_1src + avg_rmse_2src) / 2
    n_candidates = 3
    score = 1 / (1 + overall_rmse) + 0.3 * (n_candidates / 3)

    # Projected time for 400 samples
    projected_400 = (elapsed_time / 60 / n_samples) * 400

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f} (n={len(rmse_1src)})")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f} (n={len(rmse_2src)})")
    print(f"  Overall RMSE:  {overall_rmse:.6f}")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  Projected 400: {projected_400:.1f} min")
    print(f"  Errors:        {len(errors)}")

    return {
        'config': config_name,
        'params': config,
        'score': score,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'overall_rmse': overall_rmse,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400,
        'n_1src': len(rmse_1src),
        'n_2src': len(rmse_2src),
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")
    print(f"Baseline: sigma 0.15/0.19 + 20/36 fevals + 2 perturb + 8 NM = 1.1730 @ 50.4 min")

    # Base config from current best (tighter_sigma_range sigma_015_019)
    base_config = {
        'timestep_fraction': 0.40,
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    # Test configurations
    configs = [
        # Run 1: Current best baseline
        ('baseline_20_36', {**base_config}),

        # Run 2: Higher fevals 22/40
        ('sigma015_fevals_22_40', {
            **base_config,
            'max_fevals_1src': 22,
            'max_fevals_2src': 40,
        }),

        # Run 3: Even higher fevals 24/44
        ('sigma015_fevals_24_44', {
            **base_config,
            'max_fevals_1src': 24,
            'max_fevals_2src': 44,
        }),

        # Run 4: Higher fevals with reduced NM (budget trade-off)
        ('sigma015_fevals_24_44_nm6', {
            **base_config,
            'max_fevals_1src': 24,
            'max_fevals_2src': 44,
            'refine_maxiter': 6,  # Less NM to offset higher fevals
        }),
    ]

    results = []
    for config_name, config in configs:
        result = run_experiment(config, config_name, data)
        results.append(result)

        # Save intermediate results
        with open('experiments/optimal_sigma_higher_fevals/run_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Score':>8} {'Time':>8} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)

    best_in_budget = None
    for r in results:
        in_budget_str = "IN" if r['projected_400_min'] <= 60 else "OVER"
        print(f"{r['config']:<30} {r['score']:>8.4f} {r['projected_400_min']:>6.1f}m {r['rmse_1src']:>10.6f} {r['rmse_2src']:>10.6f} ({in_budget_str})")
        if r['projected_400_min'] <= 60:
            if best_in_budget is None or r['score'] > best_in_budget['score']:
                best_in_budget = r

    print("-"*70)
    if best_in_budget:
        print(f"Best in-budget: {best_in_budget['config']} with score {best_in_budget['score']:.4f} @ {best_in_budget['projected_400_min']:.1f} min")
        delta = best_in_budget['score'] - 1.1730
        print(f"Delta vs current best (1.1730): {delta:+.4f}")
        if delta > 0:
            print(f">>> NEW BEST FOUND! <<<")
    else:
        print("No configs finished within budget!")

    print("\nSaved results to experiments/optimal_sigma_higher_fevals/run_output.json")


if __name__ == '__main__':
    main()
