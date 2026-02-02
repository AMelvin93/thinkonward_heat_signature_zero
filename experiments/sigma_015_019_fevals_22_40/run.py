"""
Run script for sigma_015_019_fevals_22_40 experiment.

Combines optimal sigma (0.15/0.19) with sweet spot fevals (22/40).
"""

import os
import sys
import pickle
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Use the optimizer from tighter_sigma_range (proven best)
sys.path.insert(0, os.path.join(_project_root, 'experiments', 'tighter_sigma_range'))
from optimizer import TabuBasinHoppingOptimizer

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def calculate_sample_score(rmse, n_candidates=3, lambda_=0.3, n_max=3):
    """Competition scoring formula per sample."""
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
            'n_sims': n_sims,
            'time_s': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'idx': sample_idx,
            'rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'time_s': 0,
            'success': False,
            'error': str(e)
        }


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"Fevals: 1src={config.get('max_fevals_1src')}, 2src={config.get('max_fevals_2src')}")
    print(f"NM Polish: {config.get('refine_maxiter')} iterations")
    print(f"Perturbations: {config.get('n_perturbations')}")
    print(f"{'='*70}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            status = "OK" if result['success'] else "ERR"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['time_s']:.1f}s [{status}]")

    elapsed_time = time.time() - start_time

    # Calculate score
    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    # RMSE breakdown
    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*70}")
    print(f"Score:            {score:.4f}")
    print(f"RMSE 1-source:    {np.mean(rmse_1src):.6f} (n={len(rmse_1src)})" if rmse_1src else "RMSE 1-source: N/A")
    print(f"RMSE 2-source:    {np.mean(rmse_2src):.6f} (n={len(rmse_2src)})" if rmse_2src else "RMSE 2-source: N/A")
    print(f"Total time:       {elapsed_time/60:.2f} min")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()

    # Compare to baseline
    baseline_score = 1.173
    baseline_time = 50.4
    print(f"Baseline:         {baseline_score:.4f} @ {baseline_time:.1f} min")
    print(f"Delta:            {score - baseline_score:+.4f} score, {projected_400 - baseline_time:+.1f} min")

    if projected_400 > 60:
        print("\nOVER BUDGET")
    elif score > baseline_score:
        print("\nNEW BEST!")
    elif score >= 1.16:
        print("\nGOOD - competitive")
    else:
        print("\nNO IMPROVEMENT")

    print(f"{'='*70}")

    return {
        'config_name': config_name,
        'config': config,
        'score': score,
        'rmse_1src': np.mean(rmse_1src) if rmse_1src else None,
        'rmse_2src': np.mean(rmse_2src) if rmse_2src else None,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
        'n_1src': len(rmse_1src),
        'n_2src': len(rmse_2src),
    }


def main():
    print("="*70)
    print("Experiment: sigma_015_019_fevals_22_40")
    print("Hypothesis: Combine optimal sigma with sweet spot fevals")
    print("="*70)

    print("\nLoading data...")
    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    # Configuration combining:
    # - sigma 0.15/0.19 (from tighter_sigma_range best)
    # - fevals 22/40 (from higher_fevals_test sweet spot)
    # - 8 NM polish, 2 perturbations (proven optimal)
    config = {
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'max_fevals_1src': 22,
        'max_fevals_2src': 40,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    result = run_experiment(config, 'sigma_015_019_fevals_22_40', data)

    # Save result
    with open('run_output.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == '__main__':
    main()
