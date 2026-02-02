"""
Experiment: 4pert_nm2_scale06_validation

VALIDATE NEW BEST: 4pert + perturb_nm_iters=2 + scale=0.06

Prior finding: 1.1563 @ 52.2 min
Gap to top 10: only +0.0022!
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

    print(f"\n=== Validation Run {run_num}: {config_name} ===")

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

    PRIOR_BEST = 1.1482  # 4pert_nm2 scale=0.05 validated
    INITIAL_RESULT = 1.1563
    TOP_10 = 1.1585

    # NEW BEST config: 4-pert + perturb_nm_iters=2 + scale=0.06
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
        'perturbation_scale': 0.06,       # KEY: Scale 0.06
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("VALIDATION: 4pert_nm2 + scale=0.06 (NEW BEST CANDIDATE)")
    print("="*60)
    print(f"Initial result: {INITIAL_RESULT} @ 52.2 min")
    print(f"Prior validated: {PRIOR_BEST} @ 51.7 min")
    print(f"Top 10 threshold: {TOP_10}")
    print(f"Gap to top 10: {TOP_10 - INITIAL_RESULT:+.4f}")
    print(f"Goal: 3-run validation")

    results = []

    for run_num in range(1, 4):
        result = run_single_test(run_num, config, "4pert_nm2_scale06", data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_time = float(np.mean(times))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    runs_in_budget = sum(1 for r in results if r['in_budget'])

    print("\n" + "="*60)
    print("VALIDATION SUMMARY: 4pert_nm2 + scale=0.06 (3 runs)")
    print("="*60)

    print(f"\nRun Results:")
    for r in results:
        delta = r['score'] - PRIOR_BEST
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min ({delta:+.4f}) [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Score Range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"  Mean Time: {mean_time:.1f} min")
    print(f"  vs Prior validated (1.1482): {mean_score - PRIOR_BEST:+.4f}")
    print(f"  Runs in budget: {runs_in_budget}/3 ({runs_in_budget*100/3:.0f}%)")
    print(f"\n  Gap to Top 10 (1.1585): {TOP_10 - mean_score:+.4f}")

    # Decision
    if mean_score >= TOP_10:
        print("\n*** BREAKTHROUGH! WE MATCH OR BEAT TOP 10! ***")
    elif mean_score > PRIOR_BEST + 0.003:
        print("\n*** VALIDATED IMPROVEMENT! Promote to production. ***")
    elif mean_score > PRIOR_BEST:
        print("\n*** MARGINAL IMPROVEMENT. Consider for production. ***")
    else:
        print("\n*** NO IMPROVEMENT. Keep prior config. ***")

    # Save
    with open('run_output.json', 'w') as f:
        json.dump({
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'mean_time': mean_time,
            'vs_prior_best': mean_score - PRIOR_BEST,
            'gap_to_top10': TOP_10 - mean_score,
            'runs_in_budget': runs_in_budget,
            'runs': results
        }, f, indent=2)

    print(f"\nResults saved to run_output.json")


if __name__ == '__main__':
    main()
