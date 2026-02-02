"""
Experiment: reduced_fevals_4pert
Test reduced fevals (18/40) with 4 perturbations + tabu_distance=0.04 to fit budget.

Prior finding:
- 4 pert + tabu 0.04 + 20/44 fevals: 1.1535 @ 61.3 min (OVER BUDGET)

Hypothesis: Reducing fevals to 18/40 might save ~5 min to fit in budget while
preserving most of the score improvement from 4 perturbations.
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

    return {
        'run': run_num,
        'config_name': config_name,
        'score': float(score),
        'rmse_1src': float(np.mean(rmse_1src)),
        'rmse_2src': float(np.mean(rmse_2src)),
        'avg_candidates': float(avg_cands),
        'projected_400_min': float(projected_400),
        'in_budget': bool(projected_400 <= 60.0)
    }


def main():
    data = load_data()

    BASELINE = 1.1496  # 2 pert + tabu 0.04 + 20/44 fevals
    BEST_4PERT = 1.1535  # 4 pert + tabu 0.04 + 20/44 fevals (over budget)

    # Reduced fevals with 4 perturbations
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 18,         # REDUCED from 20
        'max_fevals_2src': 40,         # REDUCED from 44
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 4,          # HIGH (for best score)
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*60)
    print("REDUCED FEVALS (18/40) WITH 4 PERTURBATIONS")
    print("="*60)
    print(f"Baseline (2 pert): {BASELINE} @ 55.4 min")
    print(f"Best 4 pert (over budget): {BEST_4PERT} @ 61.3 min")
    print(f"Target: ~{BEST_4PERT} @ <60 min")

    results = []

    for run_num in range(1, 4):
        result = run_single_test(run_num, config, "reduced_fevals_4pert", data)
        results.append(result)

    # Statistics
    scores = [r['score'] for r in results]
    times = [r['projected_400_min'] for r in results]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_time = float(np.mean(times))

    print("\n" + "="*60)
    print("SUMMARY: Reduced Fevals with 4 Perturbations")
    print("="*60)
    print(f"\nRun Results:")
    for r in results:
        delta_base = r['score'] - BASELINE
        delta_4pert = r['score'] - BEST_4PERT
        status = "IN BUDGET" if r['in_budget'] else "OVER BUDGET"
        print(f"  Run {r['run']}: {r['score']:.4f} @ {r['projected_400_min']:.1f} min (vs base: {delta_base:+.4f}, vs 4pert: {delta_4pert:+.4f}) [{status}]")

    print(f"\nStatistics:")
    print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"  Mean Time: {mean_time:.1f} min")
    print(f"  vs Baseline (2 pert): {mean_score - BASELINE:+.4f}")
    print(f"  vs Best 4 pert (over budget): {mean_score - BEST_4PERT:+.4f}")

    in_budget = mean_time <= 60.0
    runs_in_budget = sum(1 for r in results if r['in_budget'])
    print(f"\nBudget Status: {'IN BUDGET' if in_budget else 'OVER BUDGET'}")
    print(f"  Runs in budget: {runs_in_budget}/{len(results)}")

    if in_budget and mean_score > BASELINE:
        print(f"\nSUCCESS: Config is IN BUDGET and beats baseline!")
    elif in_budget:
        print(f"\nPARTIAL: Config is in budget but doesn't beat baseline")
    else:
        print(f"\nFAILED: Config is still over budget")

    with open('run_output.json', 'w') as f:
        json.dump({
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_time': mean_time,
            'vs_baseline': mean_score - BASELINE,
            'vs_4pert': mean_score - BEST_4PERT,
            'in_budget': in_budget,
            'runs_in_budget': runs_in_budget,
            'runs': results
        }, f, indent=2)


if __name__ == '__main__':
    main()
