"""
Experiment: Reduced fevals + 5 perturbations

Hypothesis: Lower fevals but more perturbations might maintain accuracy
while fitting budget. Perturbations explore local region around best,
so might compensate for fewer global exploration steps.

Config: fevals 15/35 + 5 perturbations + perturb_nm_iters=1
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


def run_single_test(run_num, config, config_name, data):
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    print(f"\n=== Run {run_num}: {config_name} ===")

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

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    status = "IN BUDGET" if projected_400 <= 60.0 else "OVER BUDGET"
    print(f"  Budget: {status}")

    return {
        'run': run_num,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE_4PERT = 1.1482
    BASELINE_TIME = 51.7

    # Reduced fevals + 5 perturbations
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 15,            # Reduced from 20
        'max_fevals_2src': 35,            # Reduced from 44
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 5,             # 5 perturbations
        'perturb_nm_iters': 1,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("EXPERIMENT: Reduced fevals (15/35) + 5 perturbations")
    print("="*60)
    print(f"Baseline (4pert_nm2): {BASELINE_4PERT} @ {BASELINE_TIME} min (fevals 20/44)")
    print(f"Testing: fevals 15/35 + 5 pert + nm_iters=1")

    result = run_single_test(1, config, "reduced_fevals_5pert", data)

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)

    delta_score = result['score'] - BASELINE_4PERT
    delta_time = result['projected_400_min'] - BASELINE_TIME

    print(f"\nScore: {result['score']:.4f} (vs baseline: {delta_score:+.4f})")
    print(f"Time: {result['projected_400_min']:.1f} min (vs baseline: {delta_time:+.1f} min)")
    print(f"Budget: {'IN' if result['in_budget'] else 'OVER'}")
    print(f"Candidates: {result['avg_candidates']:.2f}")

    if result['in_budget'] and result['score'] > BASELINE_4PERT:
        print("\n*** SUCCESS! Better score within budget. ***")
        decision = "SUCCESS"
    elif result['in_budget'] and result['score'] >= BASELINE_4PERT - 0.002:
        print("\n*** MARGINAL - In budget with similar score. ***")
        decision = "MARGINAL"
    elif not result['in_budget']:
        print("\n*** OVER BUDGET. ***")
        decision = "OVER_BUDGET"
    else:
        print("\n*** FAILED - Worse score. ***")
        decision = "FAILED"

    with open('run3_output.json', 'w') as f:
        json.dump({
            'config': 'reduced_fevals_5pert',
            'fevals': '15/35',
            'n_perturbations': 5,
            'score': result['score'],
            'time_min': result['projected_400_min'],
            'delta_score': delta_score,
            'delta_time': delta_time,
            'in_budget': result['in_budget'],
            'avg_candidates': result['avg_candidates'],
            'decision': decision
        }, f, indent=2)

    print(f"\nResults saved to run3_output.json")


if __name__ == '__main__':
    main()
