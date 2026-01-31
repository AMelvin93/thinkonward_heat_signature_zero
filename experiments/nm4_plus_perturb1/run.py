"""
Test combined config: reduced NM iterations + perturbation.

Based on findings:
- 4 NM iterations: 1.1528 @ 51.9 min, 8 min budget remaining
- Perturbation n=1: +0.0108 score, +3.4 min

Goal: Fit perturbation into the time saved by reducing NM iterations.
Expected: ~1.16+ @ ~55 min
"""

import os
import sys
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Import optimizer from hopping_with_tabu_memory
sys.path.insert(0, os.path.join(_project_root, 'experiments', 'hopping_with_tabu_memory'))
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
        'params': config,
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

    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'enable_tabu_hopping': True,
        'tabu_distance': 0.0,  # No tabu checking
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    # Test combined configs
    configs = [
        ({**base_config, 'refine_maxiter': 4, 'n_perturbations': 1}, 'nm4_perturb1'),
        ({**base_config, 'refine_maxiter': 4, 'n_perturbations': 2}, 'nm4_perturb2'),
        ({**base_config, 'refine_maxiter': 5, 'n_perturbations': 1}, 'nm5_perturb1'),
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

    # Prior results for comparison
    print("\n--- Comparison with prior results ---")
    print("nm_4iter (no perturb): 1.1528 @ 51.9 min")
    print("8nm + 1 perturb:       1.1622 @ 64.1 min (over budget)")
    print("8nm + 2 perturb:       1.1657 @ 68.1 min (over budget)")

    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config']} @ {best['score']:.4f}, {best['time_min']:.1f} min")
    else:
        print("\nNo configs within budget!")


if __name__ == '__main__':
    main()
