"""
Run script for Multi-Round Perturbation experiment.

Tests multi-round perturbation: perturb -> NM -> perturb -> NM
Hypothesis: Multiple rounds of perturbation+optimization may escape deeper local minima.

Usage:
    python run.py [--workers N] [--perturb-rounds N] [--nm-iters N]
"""

import os
import sys
import time
import pickle
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import MultiRoundPerturbationOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = MultiRoundPerturbationOptimizer(**config)

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        # Count multi-perturbed candidates
        n_multi_perturbed = sum(1 for r in results if r.init_type == 'multi_perturbed')

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'n_multi_perturbed_selected': n_multi_perturbed,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': 0,
            'elapsed': time.time() - start,
            'n_multi_perturbed_selected': 0,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perturb-rounds', type=int, default=2, help='Number of perturbation rounds')
    parser.add_argument('--nm-iters', type=int, default=2, help='NM iterations per perturbation round')
    parser.add_argument('--perturb-top-n', type=int, default=1, help='Number of candidates to perturb')
    parser.add_argument('--perturbation-scale', type=float, default=0.05, help='Perturbation scale')
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']
    n_samples = len(samples)

    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    np.random.seed(args.seed)

    # Config based on perturbed_extended_polish baseline
    config = {
        "enable_perturbation": True,
        "perturb_top_n": args.perturb_top_n,
        "perturbation_scale": args.perturbation_scale,
        "perturb_rounds": args.perturb_rounds,
        "perturb_nm_iters_per_round": args.nm_iters,
        "max_fevals_1src": 20,
        "max_fevals_2src": 36,
        "timestep_fraction": 0.4,
        "refine_maxiter": 8,
        "refine_top_n": 2,
        "sigma0_1src": 0.18,
        "sigma0_2src": 0.22,
    }

    print(f"\n{'='*60}")
    print(f"MULTI-ROUND PERTURBATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Perturb rounds: {config['perturb_rounds']}")
    print(f"NM iters per round: {config['perturb_nm_iters_per_round']}")
    print(f"Perturb top N: {config['perturb_top_n']}")
    print(f"Perturbation scale: {config['perturbation_scale']}")
    print(f"Sigma: {config['sigma0_1src']}/{config['sigma0_2src']}")
    print(f"NM polish iters: {config['refine_maxiter']}")
    print(f"Timestep fraction: {config['timestep_fraction']}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            flag_str = ""
            if result['n_multi_perturbed_selected'] > 0:
                flag_str = f" [MULTI_PERT:{result['n_multi_perturbed_selected']}]"
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{flag_str}")

    total_time = time.time() - start_time

    # Calculate score using simplified formula
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    # Statistics
    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    times_1src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 1]
    times_2src = [r['elapsed'] for r in results if r['success'] and r['n_sources'] == 2]
    n_multi_perturbed = sum(r['n_multi_perturbed_selected'] for r in results)
    n_samples_with_multi_perturbed = sum(1 for r in results if r['n_multi_perturbed_selected'] > 0)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Multi-Round Perturbation")
    print(f"{'='*70}")
    print(f"Score:            {score:.4f}")
    print(f"RMSE mean:        {np.mean(rmses):.4f}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)}, time={np.mean(times_1src):.1f}s)")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)}, time={np.mean(times_2src):.1f}s)")
    print()
    print(f"Multi-perturbed candidates selected: {n_multi_perturbed} across {n_samples_with_multi_perturbed}/{n_samples} samples")
    print()
    print(f"Baseline (perturbed_extended_polish): 1.1464 @ 51.2 min")
    print(f"This run:                             {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs baseline:                    {score - 1.1464:+.4f} score, {projected_400 - 51.2:+.1f} min")
    print()

    # Status check
    if projected_400 > 60:
        status_msg = "OVER BUDGET"
    elif score >= 1.17:
        status_msg = "SUCCESS - Meets target!"
    elif score > 1.1464:
        status_msg = "SUCCESS - Beats current best!"
    elif score > 1.1373:
        status_msg = "PARTIAL - Beats old baseline but not new best"
    else:
        status_msg = "FAILED - Score lower than baseline"
    print(f"STATUS: {status_msg}")
    print(f"{'='*70}\n")

    # Save STATE.json
    state_path = Path(__file__).parent / 'STATE.json'

    # Load existing state if present
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment': 'multi_round_perturbation',
            'experiment_id': 'EXP_MULTI_ROUND_PERTURB_001',
            'worker': 'W2',
            'status': 'in_progress',
            'tuning_runs': [],
            'best_in_budget': None,
        }

    # Add this run's results
    run_number = len(state.get('tuning_runs', [])) + 1
    run_result = {
        'run': run_number,
        'config': config,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'total_time_sec': float(total_time),
        'rmse_mean': float(np.mean(rmses)),
        'rmse_mean_1src': float(np.mean(rmses_1src)) if rmses_1src else None,
        'rmse_mean_2src': float(np.mean(rmses_2src)) if rmses_2src else None,
        'n_multi_perturbed_selected': int(n_multi_perturbed),
        'n_samples_with_multi_perturbed': int(n_samples_with_multi_perturbed),
        'in_budget': projected_400 <= 60,
        'timestamp': datetime.now().isoformat(),
    }
    state['tuning_runs'].append(run_result)

    # Update best in-budget
    in_budget_runs = [r for r in state['tuning_runs'] if r['in_budget']]
    if in_budget_runs:
        best = max(in_budget_runs, key=lambda r: r['score'])
        state['best_in_budget'] = {
            'run': best['run'],
            'score': best['score'],
            'time_min': best['projected_400_min'],
            'config': best['config'],
        }

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"State saved to {state_path}")

    return state


if __name__ == '__main__':
    main()
