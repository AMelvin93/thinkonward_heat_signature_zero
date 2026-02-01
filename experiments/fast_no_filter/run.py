"""
Test: hopping optimizer with NO dissimilarity filtering.

Uses the fast hopping_with_tabu_memory code path but skips dissimilarity filtering,
taking top 3 by RMSE directly.

Hypothesis: Top candidates by RMSE are naturally distinct. Skipping filter may
improve accuracy by not discarding good candidates that happen to be similar.
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

# Import the modified optimizer
spec = importlib.util.spec_from_file_location("no_filter_optimizer",
    os.path.join(os.path.dirname(__file__), 'optimizer.py'))
opt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt_module)

TabuBasinHoppingOptimizer = opt_module.TabuBasinHoppingOptimizer

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
        return sample_idx, best_rmse, n_sims, len(candidates), None
    except Exception as e:
        return sample_idx, float('inf'), 0, 0, str(e)


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Parameters:")
    for k, v in config.items():
        print(f"  {k}={v}")
    print(f"{'='*60}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    rmses = {}
    n_candidates_list = []
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures)):
            sample_idx, best_rmse, n_sims, n_cands, error = future.result()
            rmses[sample_idx] = best_rmse
            n_candidates_list.append(n_cands)
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

    avg_n_candidates = sum(n_candidates_list) / len(n_candidates_list) if n_candidates_list else 0

    print(f"\n--- Results for {config_name} ---")
    print(f"  RMSE 1-source: {avg_rmse_1src:.6f}")
    print(f"  RMSE 2-source: {avg_rmse_2src:.6f}")
    print(f"  Score:         {score:.4f}")
    print(f"  Time:          {elapsed_time/60:.2f} min")
    print(f"  Avg N_cands:   {avg_n_candidates:.2f}")
    print(f"  In budget:     {'YES' if elapsed_time/60 <= 60 else 'NO'}")

    return {
        'config': config_name,
        'params': {k: v for k, v in config.items() if not callable(v)},
        'score': score,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
        'time_min': elapsed_time / 60,
        'avg_n_candidates': avg_n_candidates,
        'in_budget': elapsed_time / 60 <= 60,
        'errors': len(errors),
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Best known config from hopping_no_tabu: 1.1689 @ 58.18 min
    base_config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.0,  # No tabu checking
    }

    # Test configs
    configs = [
        ({**base_config}, 'nm8_perturb2_nofilter'),
        ({**base_config, 'refine_maxiter': 6}, 'nm6_perturb2_nofilter'),
        ({**base_config, 'n_perturbations': 1}, 'nm8_perturb1_nofilter'),
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
    print(f"{'Config':<25} {'Score':>8} {'Time':>8} {'In Budget':>10} {'N_cands':>8}")
    print("-"*70)
    for r in results:
        in_budget_str = "YES" if r['in_budget'] else "NO"
        print(f"{r['config']:<25} {r['score']:>8.4f} {r['time_min']:>7.1f}m {in_budget_str:>10} {r['avg_n_candidates']:>8.2f}")

    # Compare with baselines
    print("\n--- Comparison with baselines ---")
    print("hopping_no_tabu (WITH filter): 1.1689 @ 58.18 min")
    print("simple_top3_no_dissimilarity:  1.1825 @ 69.1 min (different code path)")

    in_budget = [r for r in results if r['in_budget']]
    if in_budget:
        best = max(in_budget, key=lambda x: x['score'])
        print(f"\nBest in-budget: {best['config']} @ {best['score']:.4f}, {best['time_min']:.1f} min")
        delta = best['score'] - 1.1689
        print(f"Delta vs hopping_no_tabu: {delta:+.4f}")
    else:
        print("\nNo configs within budget!")


if __name__ == '__main__':
    main()
