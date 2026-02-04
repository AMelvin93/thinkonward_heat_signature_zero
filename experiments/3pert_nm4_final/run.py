"""
Experiment: 3pert_nm4_final

Hypothesis: With 2pert nm4 at 54.2 min, adding a 3rd perturbation might
fit within budget and improve accuracy.

Best 2pert nm4 results:
- Mean: 1.1487 @ 54.2 min
- Best: 1.1525
- Gap to Top 10: 0.0098 from mean

Target: 1.1585 (Top 10 threshold)
Budget remaining: ~6 min (60 - 54.2)
"""

import os
import sys
import pickle
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

sys.path.insert(0, os.path.join(_project_root, 'experiments', 'tighter_sigma_range'))
from optimizer import TabuBasinHoppingOptimizer
from src.seed_manager import SeedManager

DATA_PATH = '/workspace/data/heat-signature-zero-test-data.pkl'
MAX_WORKERS = 7
DEFAULT_SEED = 42

def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)

def calculate_sample_score(rmse, n_candidates=3, lambda_=0.3, n_max=3):
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

def process_sample(args):
    sample_idx, sample, meta, config, sample_seed = args
    # Seed worker for reproducibility
    np.random.seed(sample_seed)
    # Pass seed to optimizer
    config_with_seed = {**config, 'seed': sample_seed}
    optimizer = TabuBasinHoppingOptimizer(**config_with_seed)
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

def run_single_test(run_num, config, samples, meta, seed_manager):
    n_samples = len(samples)
    # Create work items with per-sample seeds for reproducibility
    args_list = [
        (i, samples[i], meta, config, seed_manager.get_sample_seed(i))
        for i in range(n_samples)
    ]

    print(f"\n=== Run {run_num} (seed={seed_manager.master_seed}) ===")

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print(f"  Result: Score={score:.4f}, Time={projected_400:.1f} min")

    # Breakdown by source count
    one_src = [r for r in results if r['n_sources'] == 1 and r['success']]
    two_src = [r for r in results if r['n_sources'] == 2 and r['success']]

    one_src_rmse = np.mean([r['rmse'] for r in one_src]) if one_src else 0
    two_src_rmse = np.mean([r['rmse'] for r in two_src]) if two_src else 0
    one_src_time = np.mean([r['time_s'] for r in one_src]) if one_src else 0
    two_src_time = np.mean([r['time_s'] for r in two_src]) if two_src else 0

    print(f"  1-src: RMSE={one_src_rmse:.4f}, avg_time={one_src_time:.1f}s (n={len(one_src)})")
    print(f"  2-src: RMSE={two_src_rmse:.4f}, avg_time={two_src_time:.1f}s (n={len(two_src)})")

    return {
        'run': run_num,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'in_budget': projected_400 <= 60,
        'one_src_rmse': float(one_src_rmse),
        'two_src_rmse': float(two_src_rmse),
        'one_src_time': float(one_src_time),
        'two_src_time': float(two_src_time),
    }

def main():
    parser = argparse.ArgumentParser(description='3pert_nm4_final experiment')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Initialize seed manager for reproducibility
    seed_manager = SeedManager(master_seed=args.seed)
    np.random.seed(args.seed)

    data = load_data()
    samples = data['samples']
    meta = data['meta']

    # 3 perturbations with nm4 config
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 3,  # INCREASED FROM 2
        'perturbation_scale': 0.05,
        'perturb_nm_iters': 4,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("=" * 70)
    print("EXPERIMENT: 3pert_nm4_final")
    print("=" * 70)
    print("Config: 3 perturbations, nm_iters=4, scale=0.05")
    print()
    print("Baseline (2pert nm4):")
    print("  Mean: 1.1487 @ 54.2 min")
    print("  Best: 1.1525")
    print("  Gap to Top 10: 0.0098")
    print()
    print("Hypothesis: 3rd perturbation adds ~3-4 min, fits in budget, improves score")
    print("=" * 70)

    print(f"Seed: {args.seed}")

    # Single run first to check timing
    result = run_single_test(1, config, samples, meta, seed_manager)

    print("\n" + "=" * 70)
    print("INITIAL RESULT")
    print("=" * 70)

    status = "IN BUDGET" if result['in_budget'] else "OVER BUDGET"
    gap = 1.1585 - result['score']
    print(f"Score: {result['score']:.4f} (gap to Top 10: {gap:.4f})")
    print(f"Time: {result['projected_400_min']:.1f} min [{status}]")

    delta_vs_baseline = result['score'] - 1.1487
    print(f"Delta vs 2pert nm4 baseline (1.1487): {delta_vs_baseline:+.4f}")

    with open('run_output.json', 'w') as f:
        json.dump({
            'config': '3pert_nm4_scale05',
            'seed': args.seed,
            'result': result,
            'delta_vs_baseline': float(delta_vs_baseline),
            'gap_to_top10': float(gap),
        }, f, indent=2)

    # If in budget and promising, run validation
    if result['in_budget'] and result['score'] >= 1.1487:
        print("\n*** PROMISING! Running 2 more validation runs... ***")
        all_results = [result]
        for run_num in [2, 3]:
            # Use different seed for each validation run
            val_seed_manager = SeedManager(master_seed=args.seed + run_num * 1000)
            r = run_single_test(run_num, config, samples, meta, val_seed_manager)
            all_results.append(r)

        scores = [r['score'] for r in all_results]
        times = [r['projected_400_min'] for r in all_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_time = np.mean(times)
        runs_in_budget = sum(1 for r in all_results if r['in_budget'])

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY (3 runs)")
        print("=" * 70)
        for r in all_results:
            status = "IN" if r['in_budget'] else "OVER"
            print(f"  Run {r['run']}: Score={r['score']:.4f}, Time={r['projected_400_min']:.1f} min [{status}]")

        print(f"\nMean: {mean_score:.4f} +/- {std_score:.4f}")
        print(f"Time: {mean_time:.1f} min")
        print(f"Runs in budget: {runs_in_budget}/3")

        delta = mean_score - 1.1487
        print(f"\nDelta vs 2pert nm4 (1.1487): {delta:+.4f}")
        print(f"Gap to Top 10: {1.1585 - mean_score:.4f}")

        with open('validation_output.json', 'w') as f:
            json.dump({
                'config': '3pert_nm4_scale05',
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'mean_time': float(mean_time),
                'runs_in_budget': runs_in_budget,
                'best_run': float(max(scores)),
                'runs': all_results,
            }, f, indent=2)
    elif not result['in_budget']:
        print("\n*** OVER BUDGET - stopping experiment ***")
        print("3 perturbations with nm4 does NOT fit in budget.")
    else:
        print("\n*** Score worse than baseline - not worth validating ***")

    print("\nResults saved to run_output.json")

if __name__ == "__main__":
    main()
