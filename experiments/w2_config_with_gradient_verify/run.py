"""
Run script for W2 Config with Gradient Verification experiment.

Tests 40% temporal fidelity with gradient verification and reduced NM polish.

Usage:
    python run.py [--workers N] [--refine-maxiter N] [--timestep-fraction F]
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

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import W2ConfigWithVerificationOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = W2ConfigWithVerificationOptimizer(**config)

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        n_verified = sum(1 for r in results if r.init_type == 'verified')
        n_perturbed = sum(1 for r in results if r.init_type == 'perturbed')

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'n_verified': n_verified,
            'n_perturbed': n_perturbed,
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
            'n_verified': 0,
            'n_perturbed': 0,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--refine-maxiter', type=int, default=5)
    parser.add_argument('--timestep-fraction', type=float, default=0.4)
    parser.add_argument('--disable-perturbation', action='store_true')
    parser.add_argument('--disable-verification', action='store_true')
    args = parser.parse_args()

    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']
    n_samples = len(samples)

    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    np.random.seed(args.seed)

    config = {
        "refine_maxiter": args.refine_maxiter,
        "timestep_fraction": args.timestep_fraction,
        "enable_perturbation": not args.disable_perturbation,
        "enable_verification": not args.disable_verification,
        "perturb_top_n": 1,
        "n_perturbations": 2,
        "perturbation_scale": 0.05,
        "perturb_nm_iters": 2,
        "max_fevals_1src": 20,
        "max_fevals_2src": 36,
        "sigma0_1src": 0.18,
        "sigma0_2src": 0.22,
    }

    print(f"\n{'='*60}")
    print(f"W2 CONFIG WITH GRADIENT VERIFICATION")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Timestep fraction: {config['timestep_fraction']}")
    print(f"NM polish iters: {config['refine_maxiter']}")
    print(f"Sigma: {config['sigma0_1src']}/{config['sigma0_2src']}")
    print(f"Perturbation: {'ENABLED' if config['enable_perturbation'] else 'DISABLED'}")
    print(f"Verification: {'ENABLED' if config['enable_verification'] else 'DISABLED'}")
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
            flags = []
            if result['n_verified'] > 0:
                flags.append(f"V:{result['n_verified']}")
            if result['n_perturbed'] > 0:
                flags.append(f"P:{result['n_perturbed']}")
            flag_str = f" [{'/'.join(flags)}]" if flags else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{flag_str}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    n_verified = sum(r['n_verified'] for r in results)
    n_perturbed = sum(r['n_perturbed'] for r in results)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - W2 Config with Verification")
    print(f"{'='*70}")
    print(f"Score:            {score:.4f}")
    print(f"RMSE mean:        {np.mean(rmses):.4f}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Verified candidates: {n_verified}")
    print(f"Perturbed candidates: {n_perturbed}")
    print()
    print(f"Baseline (perturbation_plus_verification, 25% temporal): 1.1468 @ 54.2 min")
    print(f"This run ({config['timestep_fraction']*100:.0f}% temporal):              {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                                                    {score - 1.1468:+.4f} score, {projected_400 - 54.2:+.1f} min")
    print()

    if projected_400 > 60:
        status_msg = "OVER BUDGET"
    elif score >= 1.17:
        status_msg = "SUCCESS - Meets target!"
    elif score > 1.1468:
        status_msg = "SUCCESS - Beats 25% temporal baseline!"
    else:
        status_msg = "FAILED - Score lower than baseline"
    print(f"STATUS: {status_msg}")
    print(f"{'='*70}\n")

    state_path = Path(__file__).parent / 'STATE.json'

    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment': 'w2_config_with_gradient_verify',
            'experiment_id': 'EXP_W2_CONFIG_WITH_VERIFICATION_001',
            'worker': 'W1',
            'status': 'in_progress',
            'tuning_runs': [],
            'best_in_budget': None,
        }

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
        'n_verified': int(n_verified),
        'n_perturbed': int(n_perturbed),
        'in_budget': projected_400 <= 60,
        'timestamp': datetime.now().isoformat(),
    }
    state['tuning_runs'].append(run_result)

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
