"""
Test higher CMA-ES function evaluations.

Based on nm4_perturb1 (1.1585 @ 54.9 min) which has 5.1 min budget remaining.
Testing if more CMA-ES evaluations can improve accuracy.

Configs tested:
- baseline (20/36): Current baseline
- higher_25_45 (25/45): 25% more evaluations for 1src, 25% more for 2src
- higher_30_50 (30/50): 50% more evaluations for 1src, 39% more for 2src
"""

import os
import sys
import pickle
import time
import json
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Import the optimizer using importlib to avoid circular import
base_optimizer_path = os.path.join(_project_root, 'experiments', 'hopping_with_tabu_memory', 'optimizer.py')
spec = importlib.util.spec_from_file_location("tabu_optimizer", base_optimizer_path)
tabu_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tabu_module)

TabuBasinHoppingOptimizer = tabu_module.TabuBasinHoppingOptimizer

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
            sample, meta, q_range=(0.5, 2.0), verbose=False)
        return sample_idx, best_rmse, n_sims, None
    except Exception as e:
        return sample_idx, float('inf'), 0, str(e)


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Parameters:")
    print(f"  max_fevals_1src={config.get('max_fevals_1src', 20)}")
    print(f"  max_fevals_2src={config.get('max_fevals_2src', 36)}")
    print(f"  refine_maxiter={config.get('refine_maxiter', 8)}")
    print(f"  n_perturbations={config.get('n_perturbations', 0)}")
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
    score = 1 / (1 + overall_rmse) + 0.3

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f}")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f}")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  In budget:     {'YES' if elapsed_time/60 <= 60 else 'NO'}")

    return {
        'config': config_name,
        'params': {k: v for k, v in config.items() if not callable(v)},
        'score': score,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'time_min': elapsed_time / 60,
        'in_budget': elapsed_time / 60 <= 60,
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Base config from successful nm4_perturb1
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 4,  # Reduced from 8
        'enable_tabu_hopping': True,
        'n_perturbations': 1,  # Single perturbation
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.0,  # No tabu checking
    }

    # Test different fevals counts
    configs = [
        ({**base_config, 'max_fevals_1src': 25, 'max_fevals_2src': 45}, 'fevals_25_45'),
        ({**base_config, 'max_fevals_1src': 30, 'max_fevals_2src': 50}, 'fevals_30_50'),
        ({**base_config, 'max_fevals_1src': 22, 'max_fevals_2src': 40}, 'fevals_22_40'),
    ]

    results = []
    for config, name in configs:
        result = run_experiment(config, name, data)
        results.append(result)
        with open('run_output.json', 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Score':>8} {'Time':>8} {'In Budget':>10} {'RMSE 1src':>10} {'RMSE 2src':>10}")
    print("-"*70)
    for r in results:
        in_budget_str = "YES" if r['in_budget'] else "NO"
        print(f"{r['config']:<20} {r['score']:>8.4f} {r['time_min']:>7.1f}m {in_budget_str:>10} {r['rmse_1src']:>10.6f} {r['rmse_2src']:>10.6f}")

    # Compare with baseline
    print("\n--- Comparison with baseline ---")
    print("nm4_perturb1 (20/36 fevals): 1.1585 @ 54.9 min")

    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config']} @ {best['score']:.4f}, {best['time_min']:.1f} min")
    else:
        print("\nNo configs within budget!")


if __name__ == '__main__':
    main()
