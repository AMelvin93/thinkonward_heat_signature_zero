"""
Experiment: sigma_018_022_fevals_20_44_validation
Validate the 1.1457 result from sigma_018_022_fevals_24_44 Run 2.

This config outperformed the TRUE baseline (1.1337) by 1.2%.
Need to confirm with 3 runs.
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


def run_experiment(config, run_num, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Run {run_num} ===")
    print(f"Sigma: 1src={config.get('sigma0_1src')}, 2src={config.get('sigma0_2src')}")
    print(f"Fevals: 1src={config.get('max_fevals_1src')}, 2src={config.get('max_fevals_2src')}")

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{n_samples}")

    elapsed_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(sample_scores) if sample_scores else 0

    all_n_cands = [r['n_candidates'] for r in results if r['success']]
    avg_n_cands = np.mean(all_n_cands) if all_n_cands else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"Run {run_num} Result: Score={score:.4f}, Avg cands={avg_n_cands:.2f}, Time={projected_400:.1f} min")

    return {
        'run': run_num,
        'score': score,
        'avg_n_cands': avg_n_cands,
        'projected_400_min': projected_400,
    }


def main():
    data = load_data()

    # Config from sigma_018_022_fevals_24_44 Run 2 that scored 1.1457
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': False,
        'n_perturbations': 2,
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 3,
    }

    results = []
    for run_num in range(1, 4):  # 3 runs
        result = run_experiment(config, run_num, data)
        results.append(result)

    # Summary
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    TRUE_BASELINE = 1.1337  # From sigma_015_019_validation

    print("\n" + "="*60)
    print("SUMMARY: Validation of New Best Candidate")
    print("="*60)
    print(f"True baseline (sigma 0.15/0.19): {TRUE_BASELINE}")
    print()
    for r in results:
        delta = r['score'] - TRUE_BASELINE
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min (vs baseline: {delta:+.4f})")

    print()
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std dev:    {np.std(scores):.4f}")
    print(f"Min score:  {min(scores):.4f}")
    print(f"Max score:  {max(scores):.4f}")
    print(f"Mean time:  {np.mean(times):.1f} min")
    print()

    improvement = (np.mean(scores) - TRUE_BASELINE) / TRUE_BASELINE * 100
    if np.mean(scores) > TRUE_BASELINE:
        print(f"CONCLUSION: Config is {improvement:.2f}% BETTER than baseline!")
        print("THIS IS A NEW BEST CONFIGURATION")
    else:
        print(f"CONCLUSION: Config is {-improvement:.2f}% worse than baseline")

    # Save results
    output = {
        'config': config,
        'runs': results,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'mean_time': float(np.mean(times)),
        'vs_baseline': float(np.mean(scores) - TRUE_BASELINE),
        'improvement_pct': float(improvement),
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
