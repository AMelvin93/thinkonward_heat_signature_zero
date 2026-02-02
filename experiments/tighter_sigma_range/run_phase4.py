"""
Run script for tighter_sigma_range experiment - Phase 4.

Phase 3 showed sigma 0.15/0.19 is the sweet spot (1.1730).
Phase 4: Test if 3 perturbations can push even higher.
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
    print(f"Perturbations: {config.get('n_perturbations')}")
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

    return {
        'config': config_name,
        'score': score,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400_min,
        'rmse_1src': avg_rmse_1src,
        'rmse_2src': avg_rmse_2src,
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Base config from Phase 3 best (sigma 0.15/0.19)
    base_config = {
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
        'tabu_distance': 0.03,
        'max_tabu_attempts': 10,
    }

    configs = [
        # Test 3 perturbations
        {**base_config, 'n_perturbations': 3},
        # Test 1 perturbation (faster, maybe sufficient)
        {**base_config, 'n_perturbations': 1},
    ]

    config_names = ['sigma_015_019_3perturb', 'sigma_015_019_1perturb']

    results = []
    for config, name in zip(configs, config_names):
        result = run_experiment(config, name, data)
        results.append(result)

        with open('run_phase4_output.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("PHASE 4 EXPERIMENT SUMMARY (PERTURBATION COUNT)")
    print("="*70)
    print(f"{'Config':<30} {'Score':>8} {'Time':>8} {'Proj 400':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<30} {r['score']:>8.4f} {r['time_min']:>7.1f}m {r['projected_400_min']:>9.1f}m")

    print(f"\nPhase 3 Best: sigma_015_019 (n_perturbations=2) = 1.1730 @ 50.4 min projected")

    best = max(results, key=lambda x: x['score'])
    print(f"\nBest Phase 4: {best['config']} with score {best['score']:.4f}")

    if best['score'] > 1.1730:
        print(f"*** NEW BEST! +{best['score'] - 1.1730:.4f} vs Phase 3 ***")
    else:
        print(f"Delta vs Phase 3 best: {best['score'] - 1.1730:+.4f}")


if __name__ == '__main__':
    main()
