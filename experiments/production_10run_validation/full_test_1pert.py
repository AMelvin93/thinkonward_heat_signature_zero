"""Full 80-sample test with 1 perturbation."""

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

def main():
    data = load_data()
    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    # 1-PERTURBATION CONFIG
    config = {
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'max_fevals_1src': 20,
        'max_fevals_2src': 44,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
        'enable_tabu_hopping': True,
        'n_perturbations': 1,       # JUST 1 PERTURBATION
        'perturb_nm_iters': 3,      # Standard
        'perturbation_scale': 0.05,
        'tabu_distance': 0.04,
        'max_tabu_attempts': 10,
    }

    print("="*70)
    print("FULL 80-SAMPLE TEST: 1 PERTURBATION")
    print("="*70)

    args_list = [(i, samples[i], meta, config) for i in range(n_samples)]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {len(results)}/{n_samples} ({elapsed:.0f}s)")

    elapsed_time = time.time() - start_time

    scores = [calculate_sample_score(r['rmse'], r['n_candidates']) for r in results if r['success']]
    score = np.mean(scores) if scores else 0

    rmse_1src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    times_1src = [r['time_s'] for r in results if r['n_sources'] == 1]
    times_2src = [r['time_s'] for r in results if r['n_sources'] == 2]
    avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

    projected_400 = (elapsed_time / n_samples) * 400 / 60

    print("="*70)
    print("RESULTS: 1 PERTURBATION")
    print("="*70)
    print(f"Total time: {elapsed_time:.1f}s = {elapsed_time/60:.1f} min")
    print(f"Projected 400 samples: {projected_400:.1f} min")
    print(f"Score: {score:.4f}")
    print(f"Average candidates: {avg_cands:.2f}")
    print(f"\nRMSE:")
    print(f"  1-source: {np.mean(rmse_1src):.4f} (n={len(rmse_1src)})")
    print(f"  2-source: {np.mean(rmse_2src):.4f} (n={len(rmse_2src)})")
    print(f"\nTiming by source type:")
    print(f"  1-source avg: {np.mean(times_1src):.1f}s")
    print(f"  2-source avg: {np.mean(times_2src):.1f}s")

    if projected_400 <= 55:
        print(f"\n✓ SAFELY IN BUDGET ({projected_400:.1f} <= 55 min)")
    elif projected_400 <= 60:
        print(f"\n~ MARGINALLY IN BUDGET ({projected_400:.1f} <= 60 min)")
    else:
        print(f"\n✗ OVER BUDGET ({projected_400:.1f} > 60 min)")

    print(f"\nComparison:")
    print(f"  No perturb: Score=1.1367, Time=56.3 min")
    print(f"  1 perturb:  Score={score:.4f}, Time={projected_400:.1f} min")
    print(f"  2 perturb:  Score=1.1452, Time=67.4 min")
    print(f"  4 perturb:  Score=1.1546, Time=70.9 min")

    with open('full_test_1pert_output.json', 'w') as f:
        json.dump({
            'config': config,
            'score': float(score),
            'elapsed_sec': elapsed_time,
            'projected_400_min': projected_400,
            'rmse_1src_mean': float(np.mean(rmse_1src)),
            'rmse_2src_mean': float(np.mean(rmse_2src)),
            'avg_candidates': float(avg_cands),
            'in_budget': projected_400 <= 60
        }, f, indent=2)

    print(f"\nResults saved to full_test_1pert_output.json")

if __name__ == "__main__":
    main()
