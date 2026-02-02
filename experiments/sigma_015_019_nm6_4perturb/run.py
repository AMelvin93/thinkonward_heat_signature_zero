"""
Experiment: sigma_015_019_nm6_4perturb

Trade 2 NM iterations for 2 more perturbations.
- Baseline: 8 NM, 2 perturbations
- This: 6 NM, 4 perturbations

Hypothesis: More perturbations may catch more local minima.
"""

import os
import sys
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

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


def run_experiment(config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
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

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*70}")
    print(f"Score:            {score:.4f}")
    print(f"Avg candidates:   {avg_n_cands:.2f}")
    print(f"RMSE 1-source:    {np.mean(rmse_1src):.6f} (n={len(rmse_1src)})")
    print(f"RMSE 2-source:    {np.mean(rmse_2src):.6f} (n={len(rmse_2src)})")
    print(f"Total time:       {elapsed_time/60:.2f} min")
    print(f"Projected (400):  {projected_400:.1f} min")

    baseline_score = 1.173
    baseline_time = 50.4
    print(f"\nBaseline:         {baseline_score:.4f} @ {baseline_time:.1f} min")
    print(f"Delta:            {score - baseline_score:+.4f} score, {projected_400 - baseline_time:+.1f} min")

    if projected_400 > 60:
        print("\nOVER BUDGET")
    elif score > baseline_score:
        print("\nNEW BEST!")
    else:
        print("\nNO IMPROVEMENT" if score < baseline_score else "\nMATCHES BASELINE")

    return {
        'config_name': config_name,
        'config': config,
        'score': score,
        'avg_n_candidates': avg_n_cands,
        'rmse_1src': np.mean(rmse_1src) if rmse_1src else None,
        'rmse_2src': np.mean(rmse_2src) if rmse_2src else None,
        'time_min': elapsed_time / 60,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
    }


def main():
    print("="*70)
    print("Experiment: sigma_015_019_nm6_4perturb")
    print("Trade 2 NM iterations for 2 more perturbations")
    print("="*70)

    data = load_data()
    print(f"Loaded {len(data['samples'])} samples")

    results = []

    # Run 1: 6 NM, 4 perturbations
    config1 = {
        'sigma0_1src': 0.15,
        'sigma0_2src': 0.19,
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': 0.40,
        'refine_maxiter': 6,
        'enable_tabu_hopping': False,
        'n_perturbations': 4,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }
    result1 = run_experiment(config1, 'nm6_perturb4', data)
    results.append(result1)

    with open('run_output.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
