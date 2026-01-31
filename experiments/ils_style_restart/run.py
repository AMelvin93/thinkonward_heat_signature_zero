"""
Run script for ils_style_restart experiment.

Tests:
1. ILS with 3 iterations, 4 NM iter per step (standard)
2. ILS with 2 iterations, 3 NM iter per step (lighter)
3. No ILS (baseline comparison)
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

from optimizer import ILSStyleOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def process_sample(args):
    sample_idx, sample, meta, config = args
    optimizer = ILSStyleOptimizer(**config)
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
    print(f"Parameters: max_ils_iterations={config.get('max_ils_iterations', 3)}")
    print(f"            ils_nm_iters={config.get('ils_nm_iters', 4)}")
    print(f"            enable_ils={config.get('enable_ils', True)}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    rmses = {}
    n_1src = 0
    n_2src = 0
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            sample_idx, best_rmse, n_sims, error = future.result()
            rmses[sample_idx] = best_rmse

            if error:
                errors.append((sample_idx, error))
                print(f"  Sample {sample_idx}: ERROR - {error}")
            else:
                n_sources = samples[sample_idx]['n_sources']
                if n_sources == 1:
                    n_1src += 1
                else:
                    n_2src += 1

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
    n_candidates = 3  # Always 3 candidates
    score = 1 / (1 + overall_rmse) + 0.3 * (n_candidates / 3)

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f} (n={len(rmse_1src)})")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f} (n={len(rmse_2src)})")
    print(f"  Overall RMSE:  {overall_rmse:.6f}")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  Errors:        {len(errors)}")

    return {
        'config': config_name,
        'params': config,
        'score': score,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'overall_rmse': overall_rmse,
        'time_min': elapsed_time / 60,
        'n_1src': len(rmse_1src),
        'n_2src': len(rmse_2src),
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Configuration 1: Standard ILS (3 iterations, 4 NM iter per step)
    config1 = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_ils': True,
        'max_ils_iterations': 3,
        'ils_nm_iters': 4,
        'ils_perturbation_scale': 0.05,
        'ils_apply_to_top_n': 1,
    }

    # Configuration 2: Lighter ILS (2 iterations, 3 NM iter per step)
    config2 = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_ils': True,
        'max_ils_iterations': 2,
        'ils_nm_iters': 3,
        'ils_perturbation_scale': 0.05,
        'ils_apply_to_top_n': 1,
    }

    # Configuration 3: No ILS (baseline)
    config3 = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_ils': False,
    }

    configs = [
        (config1, 'ils_3iter_4nm'),
        (config2, 'ils_2iter_3nm'),
        (config3, 'no_ils_baseline'),
    ]

    results = []
    for config, name in configs:
        result = run_experiment(config, name, data)
        results.append(result)

        # Save intermediate results
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Config':<25} {'Score':>8} {'Time':>8} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<25} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['rmse_1src']:>10.6f} {r['rmse_2src']:>10.6f}")

    # Determine best in-budget config
    in_budget = [r for r in results if r['time_min'] <= 60]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget config: {best['config']} with score {best['score']:.4f} @ {best['time_min']:.1f} min")
    else:
        print("\nNo configs finished within budget!")


if __name__ == '__main__':
    main()
