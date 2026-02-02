"""
Run script for hopping_with_22_40_fevals experiment.

Apply the sweet spot fevals (22/40) to the best hopping_no_tabu config.

Base config:
- sigma 0.18/0.22 (confirmed optimal)
- 8 NM polish
- 2 perturbations
- 40% temporal fidelity

Test:
1. Base config with 20/36 fevals (baseline)
2. Base config with 22/40 fevals (sweet spot)
3. Base config with 24/44 fevals (even higher)
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
    print(f"Parameters: max_fevals_1src={config.get('max_fevals_1src', 20)}")
    print(f"            max_fevals_2src={config.get('max_fevals_2src', 36)}")
    print(f"            refine_maxiter={config.get('refine_maxiter', 8)}")
    print(f"            n_perturbations={config.get('n_perturbations', 2)}")
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
    n_candidates = 3
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

    # Base config - hopping_no_tabu best config
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
    }

    # Config 1: Baseline fevals 20/36
    config1 = {**base_config, 'max_fevals_1src': 20, 'max_fevals_2src': 36}

    # Config 2: Sweet spot fevals 22/40
    config2 = {**base_config, 'max_fevals_1src': 22, 'max_fevals_2src': 40}

    # Config 3: Higher fevals 24/44
    config3 = {**base_config, 'max_fevals_1src': 24, 'max_fevals_2src': 44}

    configs = [
        (config1, 'fevals_20_36_baseline'),
        (config2, 'fevals_22_40_sweetspot'),
        (config3, 'fevals_24_44_higher'),
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
    print(f"{'Config':<30} {'Score':>8} {'Time':>8} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<30} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['rmse_1src']:>10.6f} {r['rmse_2src']:>10.6f}")

    # Determine best in-budget config
    # Project 400 samples: time * 5
    print("\n" + "-"*70)
    print("Projected times for 400 samples:")
    for r in results:
        projected = r['time_min'] * 5
        in_budget = "IN budget" if projected <= 60 else "OVER budget"
        print(f"  {r['config']:<30} {projected:>6.1f} min ({in_budget})")

    in_budget = [r for r in results if r['time_min'] * 5 <= 60]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget config: {best['config']} with score {best['score']:.4f} @ {best['time_min']*5:.1f} min projected")
    else:
        print("\nNo configs projected within budget!")
        best = min(results, key=lambda x: x['time_min'])
        print(f"Fastest config: {best['config']} with score {best['score']:.4f} @ {best['time_min']*5:.1f} min projected")


if __name__ == '__main__':
    main()
