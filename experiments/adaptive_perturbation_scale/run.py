"""
Run adaptive_perturbation_scale experiment.
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

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import AdaptivePerturbationOptimizer


def process_single_sample(args):
    idx, sample, meta, config = args

    optimizer = AdaptivePerturbationOptimizer(
        enable_perturbation=config['enable_perturbation'],
        perturb_top_n=config.get('perturb_top_n', 1),
        n_perturbations=config.get('n_perturbations', 2),
        perturb_nm_iters=config.get('perturb_nm_iters', 3),
        max_fevals_1src=config.get('max_fevals_1src', 20),
        max_fevals_2src=config.get('max_fevals_2src', 36),
        timestep_fraction=config.get('timestep_fraction', 0.40),
        refine_maxiter=config.get('refine_maxiter', 8),
        refine_top_n=config.get('refine_top_n', 2),
        sigma0_1src=config.get('sigma0_1src', 0.18),
        sigma0_2src=config.get('sigma0_2src', 0.22),
        perturbation_scale_high=config.get('perturbation_scale_high', 0.10),
        perturbation_scale_mid=config.get('perturbation_scale_mid', 0.05),
        perturbation_scale_low=config.get('perturbation_scale_low', 0.02),
        rmse_high_threshold=config.get('rmse_high_threshold', 0.30),
        rmse_mid_threshold=config.get('rmse_mid_threshold', 0.15),
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
    parser.add_argument('--perturbation-scale-high', type=float, default=0.10)
    parser.add_argument('--perturbation-scale-mid', type=float, default=0.05)
    parser.add_argument('--perturbation-scale-low', type=float, default=0.02)
    parser.add_argument('--rmse-high-threshold', type=float, default=0.30)
    parser.add_argument('--rmse-mid-threshold', type=float, default=0.15)
    parser.add_argument('--timestep-fraction', type=float, default=0.40)
    parser.add_argument('--refine-maxiter', type=int, default=8)
    parser.add_argument('--sigma0-1src', type=float, default=0.18)
    parser.add_argument('--sigma0-2src', type=float, default=0.22)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load test data
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
    print(f"ADAPTIVE PERTURBATION SCALE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Adaptive perturbation scales:")
    print(f"  High (RMSE>{args.rmse_high_threshold}): {args.perturbation_scale_high}")
    print(f"  Mid ({args.rmse_mid_threshold}<RMSE<={args.rmse_high_threshold}): {args.perturbation_scale_mid}")
    print(f"  Low (RMSE<={args.rmse_mid_threshold}): {args.perturbation_scale_low}")
    print(f"Sigma: {args.sigma0_1src}/{args.sigma0_2src}")
    print(f"{'='*60}")

    config = {
        'enable_perturbation': True,
        'perturb_top_n': 1,
        'n_perturbations': 2,
        'perturb_nm_iters': 3,
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': args.timestep_fraction,
        'refine_maxiter': args.refine_maxiter,
        'refine_top_n': 2,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'perturbation_scale_high': args.perturbation_scale_high,
        'perturbation_scale_mid': args.perturbation_scale_mid,
        'perturbation_scale_low': args.perturbation_scale_low,
        'rmse_high_threshold': args.rmse_high_threshold,
        'rmse_mid_threshold': args.rmse_mid_threshold,
    }

    start_time = time.time()
    results = []

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            perturb_flag = f" [PERT:{result['n_perturbed_selected']}]" if result['n_perturbed_selected'] > 0 else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{perturb_flag}")

    total_time = time.time() - start_time

    # Calculate score
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
    n_perturbed = sum(r['n_perturbed_selected'] for r in results)
    n_samples_with_perturbed = sum(1 for r in results if r['n_perturbed_selected'] > 0)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Adaptive Perturbation Scale")
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
    print(f"Perturbed candidates selected: {n_perturbed} across {n_samples_with_perturbed}/{n_samples} samples")
    print()
    print(f"Baseline (perturbed_extended_polish): 1.1464 @ 51.2 min (fixed scale 0.05)")
    print(f"This run:                             {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs baseline:                    {score - 1.1464:+.4f} score, {projected_400 - 51.2:+.1f} min")
    print()

    if projected_400 > 60:
        status_msg = "OVER BUDGET"
    elif score > 1.1464:
        status_msg = "SUCCESS - Beats current best!"
    elif score > 1.1373:
        status_msg = "PARTIAL - Above simple baseline"
    else:
        status_msg = "FAILED - Score below baseline"
    print(f"STATUS: {status_msg}")
    print(f"{'='*70}\n")

    # Update STATE.json
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {'tuning_runs': []}

    state['tuning_runs'].append({
        'run': len(state.get('tuning_runs', [])) + 1,
        'config': config,
        'score': score,
        'time_min': projected_400,
        'rmse_mean': float(np.mean(rmses)),
        'rmse_mean_1src': float(np.mean(rmses_1src)) if rmses_1src else None,
        'rmse_mean_2src': float(np.mean(rmses_2src)) if rmses_2src else None,
        'n_perturbed_selected': n_perturbed,
        'n_samples_with_perturbed': n_samples_with_perturbed,
        'timestamp': datetime.now().isoformat(),
    })

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    return {'score': score, 'time_min': projected_400}


if __name__ == '__main__':
    main()
