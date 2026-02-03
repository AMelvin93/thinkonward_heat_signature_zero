"""
Experiment: tau=0.15 with 4pert_nm2

Previous finding: tau=0.15 gave +0.0053 score but was over budget (67.2 min)
on the old config that was already 58.4 min.

The current 4pert_nm2 is faster (51.7 min). With 8.3 min buffer,
tau=0.15 might fit in budget.

Hypothesis: tau=0.15 + faster base config could achieve higher score within budget.
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

# Override TAU globally for this experiment
import experiments.tighter_sigma_range.optimizer as opt_module
original_tau = opt_module.TAU
opt_module.TAU = 0.15  # Lower threshold


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

    # Candidate distribution
    n_3cand = sum(1 for r in results if r['n_candidates'] == 3)
    n_2cand = sum(1 for r in results if r['n_candidates'] == 2)
    n_1cand = sum(1 for r in results if r['n_candidates'] == 1)

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"Result: Score={score:.4f}, Time={projected_400:.1f} min, Candidates={avg_cands:.2f}")
    print(f"  RMSE 1src: {np.mean(rmse_1src):.4f}, RMSE 2src: {np.mean(rmse_2src):.4f}")
    print(f"  Candidate distribution: 3-cand={n_3cand}, 2-cand={n_2cand}, 1-cand={n_1cand}")
    status = "IN BUDGET" if projected_400 <= 60.0 else "OVER BUDGET"
    print(f"  Budget: {status}")

    return {
        'run': run_num,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'n_3cand': n_3cand,
        'n_2cand': n_2cand,
        'n_1cand': n_1cand,
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE_4PERT = 1.1482
    BASELINE_TIME = 51.7

    # Use 4pert_nm2 config but with tau=0.15 (already overridden globally)
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,
        'perturb_nm_iters': 2,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("EXPERIMENT: tau=0.15 with 4pert_nm2")
    print("="*60)
    print(f"Baseline (tau=0.20, 4pert_nm2): {BASELINE_4PERT} @ {BASELINE_TIME} min")
    print(f"Prior tau=0.15 result: +0.0053 but 67.2 min (over budget)")
    print(f"Testing: tau=0.15 with faster 4pert_nm2 config")

    result = run_single_test(1, config, "tau015_4pert_nm2", data)

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)

    delta_score = result['score'] - BASELINE_4PERT
    delta_time = result['projected_400_min'] - BASELINE_TIME

    print(f"\nScore: {result['score']:.4f} (vs baseline: {delta_score:+.4f})")
    print(f"Time: {result['projected_400_min']:.1f} min (vs baseline: {delta_time:+.1f} min)")
    print(f"Budget: {'IN' if result['in_budget'] else 'OVER'}")
    print(f"Candidates: {result['avg_candidates']:.2f}")
    print(f"  3-cand: {result['n_3cand']}, 2-cand: {result['n_2cand']}, 1-cand: {result['n_1cand']}")

    if result['in_budget'] and result['score'] > BASELINE_4PERT:
        print("\n*** SUCCESS! tau=0.15 improves score within budget. ***")
        decision = "SUCCESS"
    elif result['in_budget']:
        print("\n*** MARGINAL - In budget but no improvement. ***")
        decision = "MARGINAL"
    elif not result['in_budget']:
        print("\n*** OVER BUDGET. ***")
        decision = "OVER_BUDGET"
    else:
        print("\n*** FAILED. ***")
        decision = "FAILED"

    with open('run_output.json', 'w') as f:
        json.dump({
            'config': 'tau015_4pert_nm2',
            'tau': 0.15,
            'score': result['score'],
            'time_min': result['projected_400_min'],
            'delta_score': delta_score,
            'delta_time': delta_time,
            'in_budget': result['in_budget'],
            'avg_candidates': result['avg_candidates'],
            'n_3cand': result['n_3cand'],
            'n_2cand': result['n_2cand'],
            'n_1cand': result['n_1cand'],
            'decision': decision
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")

    # Restore original tau
    opt_module.TAU = original_tau


if __name__ == '__main__':
    main()
