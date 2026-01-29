"""
Run Tikhonov regularized RMSE experiment.
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import TikhonovRegularizedOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args

    optimizer = TikhonovRegularizedOptimizer(
        enable_regularization=config.get('enable_regularization', True),
        regularization_lambda=config.get('regularization_lambda', 0.01),
        enable_perturbation=config.get('enable_perturbation', True),
        perturb_top_n=config.get('perturb_top_n', 1),
        n_perturbations=config.get('n_perturbations', 2),
        perturb_nm_iters=config.get('perturb_nm_iters', 3),
        perturbation_scale=config.get('perturbation_scale', 0.05),
        max_fevals_1src=config.get('max_fevals_1src', 20),
        max_fevals_2src=config.get('max_fevals_2src', 36),
        timestep_fraction=config.get('timestep_fraction', 0.40),
        refine_maxiter=config.get('refine_maxiter', 8),
        refine_top_n=config.get('refine_top_n', 2),
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        n_perturbed = sum(1 for r in results if 'perturbed' in r.init_type)

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'n_perturbed_selected': n_perturbed,
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
            'n_perturbed_selected': 0,
            'success': False,
            'error': str(e),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--regularization-lambda', type=float, default=0.01)
    parser.add_argument('--disable-regularization', action='store_true')
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    np.random.seed(args.seed)
    n_samples = len(samples)

    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"TIKHONOV REGULARIZED RMSE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Regularization: lambda={args.regularization_lambda if not args.disable_regularization else 'DISABLED'}")
    print(f"Timestep fraction: {args.timestep_fraction}")
    print(f"{'='*60}")

    config = {
        'enable_regularization': not args.disable_regularization,
        'regularization_lambda': args.regularization_lambda,
        'enable_perturbation': True,
        'perturb_top_n': 1,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'perturbation_scale': 0.05,
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': args.timestep_fraction,
        'refine_maxiter': 8,
        'refine_top_n': 2,
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
    }

    start_time = time.time()
    results = []

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            perturb_flag = f" [PERT]" if result['n_perturbed_selected'] > 0 else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"time={result['elapsed']:.1f}s [{status}]{perturb_flag}")

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
    n_perturbed = sum(r['n_perturbed_selected'] for r in results)
    n_samples_with_perturbed = sum(1 for r in results if r['n_perturbed_selected'] > 0)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Tikhonov Regularized RMSE")
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
    print(f"Perturbed selected: {n_perturbed} across {n_samples_with_perturbed}/{n_samples} samples")
    print()
    print(f"Baseline (perturbed_extended_polish): 1.1464 @ 51.2 min (no regularization)")
    print(f"This run:                             {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs baseline:                    {score - 1.1464:+.4f} score, {projected_400 - 51.2:+.1f} min")
    print()

    if projected_400 > 60:
        status_msg = "OVER BUDGET"
    elif score > 1.1464:
        status_msg = "SUCCESS - Beats baseline!"
    else:
        status_msg = "FAILED - Score at or below baseline"
    print(f"STATUS: {status_msg}")
    print(f"{'='*70}\n")

    # Update STATE.json
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment_id': 'EXP_TIKHONOV_REGULARIZED_001',
            'experiment_name': 'tikhonov_regularized_rmse',
            'worker': 'W1',
            'status': 'in_progress',
            'started_at': datetime.now().isoformat(),
            'tuning_runs': [],
        }

    state['tuning_runs'].append({
        'run': len(state.get('tuning_runs', [])) + 1,
        'config': config,
        'score': float(score),
        'projected_400_min': float(projected_400),
        'total_time_sec': float(total_time),
        'rmse_mean': float(np.mean(rmses)),
        'rmse_mean_1src': float(np.mean(rmses_1src)) if rmses_1src else None,
        'rmse_mean_2src': float(np.mean(rmses_2src)) if rmses_2src else None,
        'n_perturbed_selected': n_perturbed,
        'n_samples_with_perturbed': n_samples_with_perturbed,
        'in_budget': projected_400 <= 60,
        'timestamp': datetime.now().isoformat(),
    })

    # Update best in budget
    in_budget_runs = [r for r in state['tuning_runs'] if r.get('in_budget', False)]
    if in_budget_runs:
        best = max(in_budget_runs, key=lambda x: x['score'])
        state['best_in_budget'] = {
            'run': best['run'],
            'score': best['score'],
            'time_min': best['projected_400_min'],
        }
    else:
        state['best_in_budget'] = None

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    return {'score': score, 'time_min': projected_400}


if __name__ == '__main__':
    main()
