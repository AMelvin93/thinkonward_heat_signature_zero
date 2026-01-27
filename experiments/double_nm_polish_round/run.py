"""
Run script for double_nm_polish_round experiment.

Usage:
    python run.py [--workers N] [--max-samples N]
"""

import os
import sys
import pickle
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mlflow

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import DoubleNMPolishOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = DoubleNMPolishOptimizer(
        max_fevals_1src=config.get('max_fevals_1src', 20),
        max_fevals_2src=config.get('max_fevals_2src', 36),
        timestep_fraction=config.get('timestep_fraction', 0.25),
        polish1_maxiter=config.get('polish1_maxiter', 6),
        polish2_maxiter=config.get('polish2_maxiter', 4),
        perturb_scale=config.get('perturb_scale', 0.02),
        refine_top_n=config.get('refine_top_n', 2),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        # Count double_polished candidates
        n_polished = sum(1 for r in results if r.init_type == 'double_polished')

        return {
            'idx': idx,
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sources': sample['n_sources'],
            'n_candidates': len(candidates),
            'n_sims': n_sims,
            'elapsed': elapsed,
            'n_polished': n_polished,
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
            'n_polished': 0,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (default: all)')
    parser.add_argument('--polish1-iter', type=int, default=6, help='First polish iterations')
    parser.add_argument('--polish2-iter', type=int, default=4, help='Second polish iterations')
    parser.add_argument('--perturb-scale', type=float, default=0.02, help='Perturbation scale')
    parser.add_argument('--timestep-fraction', type=float, default=0.25, help='Temporal fidelity fraction')
    parser.add_argument('--fevals-1src', type=int, default=20, help='Max fevals for 1-source')
    parser.add_argument('--fevals-2src', type=int, default=36, help='Max fevals for 2-source')
    parser.add_argument('--refine-top-n', type=int, default=2, help='Number of candidates to polish')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    np.random.seed(args.seed)
    indices = np.arange(len(samples))

    if args.max_samples:
        indices = indices[:args.max_samples]

    samples_to_process = [samples[i] for i in indices]
    n_samples = len(samples_to_process)

    n_1src = sum(1 for s in samples_to_process if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"DOUBLE NM POLISH EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Polish rounds: {args.polish1_iter} + {args.polish2_iter} iterations")
    print(f"Perturb scale: {args.perturb_scale}")
    print(f"Timestep fraction: {args.timestep_fraction}")
    print(f"{'='*60}")

    config = {
        'max_fevals_1src': args.fevals_1src,
        'max_fevals_2src': args.fevals_2src,
        'timestep_fraction': args.timestep_fraction,
        'polish1_maxiter': args.polish1_iter,
        'polish2_maxiter': args.polish2_iter,
        'perturb_scale': args.perturb_scale,
        'refine_top_n': args.refine_top_n,
    }

    start_time = time.time()
    results = []

    work_items = [(indices[i], samples_to_process[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            polish_flag = f" [POL:{result['n_polished']}]" if result['n_polished'] > 0 else ""
            print(f"[{len(results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} time={result['elapsed']:.1f}s [{status}]{polish_flag}")

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
    n_polished = sum(r['n_polished'] for r in results)
    n_samples_polished = sum(1 for r in results if r['n_polished'] > 0)

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print(f"RESULTS - Double NM Polish")
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
    print(f"Polished candidates selected: {n_polished} across {n_samples_polished}/{n_samples} samples")
    print()
    print(f"Baseline (best):     1.1373 @ 42.6 min (solution_verification_pass)")
    print(f"Old baseline:        1.1688 @ 58.4 min (early_timestep_filtering)")
    print(f"This run:            {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta vs best:       {score - 1.1373:+.4f} score, {projected_400 - 42.6:+.1f} min")
    print()

    # Status check
    if projected_400 > 60:
        print("STATUS: OVER BUDGET")
    elif score >= 1.17:
        print("STATUS: SUCCESS - Meets target!")
    elif score > 1.1373:
        print("STATUS: SUCCESS - Beats current best!")
    elif score > 1.1246:
        print("STATUS: PARTIAL - Beats old baseline but not new best")
    else:
        print("STATUS: FAILED - Score lower than baseline")
    print(f"{'='*70}\n")

    results_summary = {
        'score': score,
        'total_time_sec': total_time,
        'projected_400_min': projected_400,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else 0,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else 0,
        'time_mean_1src': np.mean(times_1src) if times_1src else 0,
        'time_mean_2src': np.mean(times_2src) if times_2src else 0,
        'n_polished': n_polished,
        'n_samples_polished': n_samples_polished,
    }

    # Save STATE.json
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment_id': 'EXP_DOUBLE_NM_POLISH_001',
            'worker': 'W2',
            'status': 'in_progress',
            'baseline_to_beat': {
                'score': 1.1373,
                'time_min': 42.6,
                'source': 'solution_verification_pass'
            },
            'hypothesis': 'First NM polish may get stuck in local basin. Second round from different simplex may escape.',
            'tuning_runs': [],
            'best_in_budget': None,
        }

    state['tuning_runs'].append({
        'run': len(state['tuning_runs']) + 1,
        'config': config,
        'results': results_summary,
        'timestamp': datetime.now().isoformat(),
    })

    if projected_400 <= 60:
        if state['best_in_budget'] is None or score > state['best_in_budget'].get('score', 0):
            state['best_in_budget'] = {
                'run': len(state['tuning_runs']),
                'score': score,
                'time_min': projected_400,
                'config': config,
            }

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    # Log to MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(str(project_root / 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"double_nm_polish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_name', 'double_nm_polish_round')
            mlflow.log_param('experiment_id', 'EXP_DOUBLE_NM_POLISH_001')
            mlflow.log_param('worker', 'W2')
            mlflow.log_param('polish1_maxiter', args.polish1_iter)
            mlflow.log_param('polish2_maxiter', args.polish2_iter)
            mlflow.log_param('perturb_scale', args.perturb_scale)
            mlflow.log_param('timestep_fraction', args.timestep_fraction)
            mlflow.log_param('n_samples', n_samples)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('submission_score', results_summary['score'])
            mlflow.log_metric('projected_400_samples_min', results_summary['projected_400_min'])
            mlflow.log_metric('total_time_sec', results_summary['total_time_sec'])
            mlflow.log_metric('rmse_mean', results_summary['rmse_mean'])
            mlflow.log_metric('rmse_mean_1src', results_summary['rmse_mean_1src'])
            mlflow.log_metric('rmse_mean_2src', results_summary['rmse_mean_2src'])
            mlflow.log_metric('n_polished', results_summary['n_polished'])
            mlflow.log_metric('n_samples_polished', results_summary['n_samples_polished'])

            print(f"MLflow run ID: {run.info.run_id}")

    return results_summary


if __name__ == '__main__':
    main()
