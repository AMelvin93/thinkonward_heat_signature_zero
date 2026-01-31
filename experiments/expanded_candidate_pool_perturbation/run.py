"""
Run script for expanded candidate pool perturbation experiment.

Tests:
1. Baseline: expand_top_n=3, n_perturbations=1, perturb_nm_iters=4
2. More perturbations: expand_top_n=3, n_perturbations=2, perturb_nm_iters=3
3. Deeper polish: expand_top_n=2, n_perturbations=2, perturb_nm_iters=5
"""

import os
import sys
import time
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from optimizer import ExpandedPoolOptimizer


def process_single_sample(args):
    """Process a single sample (for parallel execution)."""
    idx, sample, meta, config = args

    optimizer = ExpandedPoolOptimizer(
        expand_top_n=config['expand_top_n'],
        n_perturbations_per_candidate=config['n_perturbations_per_candidate'],
        perturbation_scale=config.get('perturbation_scale', 0.05),
        perturb_nm_iters=config.get('perturb_nm_iters', 4),
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

        # Count perturbed candidates
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
            'traceback': traceback.format_exc(),
        }


def run_configuration(samples, meta, config, workers=7):
    """Run a single configuration with parallel processing."""
    n_samples = len(samples)
    indices = np.arange(n_samples)

    work_items = [(indices[i], samples[i], meta, config) for i in range(n_samples)]

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
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
    n_perturbed = sum(r['n_perturbed_selected'] for r in results)
    n_samples_with_perturbed = sum(1 for r in results if r['n_perturbed_selected'] > 0)

    projected_400 = (total_time / n_samples) * 400 / 60

    return {
        'score': score,
        'total_time_sec': total_time,
        'projected_400_min': projected_400,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src) if rmses_1src else 0,
        'rmse_mean_2src': np.mean(rmses_2src) if rmses_2src else 0,
        'time_mean_1src': np.mean(times_1src) if times_1src else 0,
        'time_mean_2src': np.mean(times_2src) if times_2src else 0,
        'n_perturbed_selected': n_perturbed,
        'n_samples_with_perturbed': n_samples_with_perturbed,
        'n_samples': n_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7, help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    np.random.seed(args.seed)

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"EXPANDED CANDIDATE POOL PERTURBATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Baseline: 1.1464 @ 51.2 min")
    print(f"{'='*60}")

    # Load state
    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'experiment': 'expanded_candidate_pool_perturbation',
            'status': 'in_progress',
            'claimed_by': 'W1',
            'tuning_runs': [],
        }

    all_results = []

    # ============ RUN 1: Baseline expansion ============
    print(f"\n{'='*50}")
    print("RUN 1: Baseline expansion (top 3 x 1 perturbation)")
    print(f"{'='*50}")

    config1 = {
        'expand_top_n': 3,
        'n_perturbations_per_candidate': 1,
        'perturb_nm_iters': 4,
        'perturbation_scale': 0.05,
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'timestep_fraction': 0.40,
        'refine_maxiter': 8,
    }

    result1 = run_configuration(samples, meta, config1, args.workers)
    all_results.append(('expand_3x1_nm4', config1, result1))

    budget_remaining = 60.0 - result1['projected_400_min']
    in_budget = result1['projected_400_min'] <= 60.0

    print(f"\n{'='*40}")
    print(f"RUN 1 Results: expand_3x1_nm4")
    print(f"  Score: {result1['score']:.4f}")
    print(f"  Projected time: {result1['projected_400_min']:.1f} min")
    print(f"  Budget remaining: {budget_remaining:.1f} min")
    print(f"  In budget: {in_budget}")
    print(f"  RMSE 1-src: {result1['rmse_mean_1src']:.4f}")
    print(f"  RMSE 2-src: {result1['rmse_mean_2src']:.4f}")
    print(f"  Perturbed selected: {result1['n_perturbed_selected']} in {result1['n_samples_with_perturbed']} samples")
    print(f"{'='*40}")

    state['tuning_runs'].append({
        'run': 1,
        'config': 'expand_3x1_nm4',
        'params': config1,
        'score': result1['score'],
        'time_min': result1['projected_400_min'],
        'budget_remaining_min': budget_remaining,
        'in_budget': in_budget,
        'rmse_1src': result1['rmse_mean_1src'],
        'rmse_2src': result1['rmse_mean_2src'],
        'n_perturbed_selected': result1['n_perturbed_selected'],
    })

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    # ============ RUN 2: Adjust based on Run 1 ============
    if result1['projected_400_min'] > 60.0:
        print("\nRun 1 over budget, trying reduced configuration...")
        config2 = {
            'expand_top_n': 2,
            'n_perturbations_per_candidate': 1,
            'perturb_nm_iters': 3,
            'perturbation_scale': 0.05,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
        }
        config_name = "expand_2x1_nm3_reduced"
    elif result1['score'] >= 1.1464:  # Beats baseline
        print("\nRun 1 beats baseline! Trying to improve further with more perturbations...")
        config2 = {
            'expand_top_n': 3,
            'n_perturbations_per_candidate': 2,
            'perturb_nm_iters': 3,
            'perturbation_scale': 0.05,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
        }
        config_name = "expand_3x2_nm3"
    else:
        print("\nRun 1 below baseline. Trying deeper polish...")
        config2 = {
            'expand_top_n': 2,
            'n_perturbations_per_candidate': 2,
            'perturb_nm_iters': 5,
            'perturbation_scale': 0.05,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
        }
        config_name = "expand_2x2_nm5"

    print(f"\n{'='*50}")
    print(f"RUN 2: {config_name}")
    print(f"{'='*50}")

    result2 = run_configuration(samples, meta, config2, args.workers)
    all_results.append((config_name, config2, result2))

    budget_remaining = 60.0 - result2['projected_400_min']
    in_budget = result2['projected_400_min'] <= 60.0

    print(f"\n{'='*40}")
    print(f"RUN 2 Results: {config_name}")
    print(f"  Score: {result2['score']:.4f}")
    print(f"  Projected time: {result2['projected_400_min']:.1f} min")
    print(f"  Budget remaining: {budget_remaining:.1f} min")
    print(f"  In budget: {in_budget}")
    print(f"{'='*40}")

    state['tuning_runs'].append({
        'run': 2,
        'config': config_name,
        'params': config2,
        'score': result2['score'],
        'time_min': result2['projected_400_min'],
        'budget_remaining_min': budget_remaining,
        'in_budget': in_budget,
        'rmse_1src': result2['rmse_mean_1src'],
        'rmse_2src': result2['rmse_mean_2src'],
        'n_perturbed_selected': result2['n_perturbed_selected'],
    })

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    # ============ RUN 3: Final tuning ============
    best_so_far = max(all_results, key=lambda x: x[2]['score'] if x[2]['projected_400_min'] <= 60 else 0)
    best_name, best_config, best_result = best_so_far

    print(f"\nBest in-budget so far: {best_name} with score {best_result['score']:.4f}")

    if best_result['projected_400_min'] < 55 and best_result['score'] >= 1.14:
        # Time to spare, try increasing polish
        config3 = best_config.copy()
        config3['refine_maxiter'] = 10
        config_name = f"{best_name}_more_polish"
    elif best_result['projected_400_min'] > 60:
        # Need to reduce
        config3 = {
            'expand_top_n': 2,
            'n_perturbations_per_candidate': 1,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.05,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'timestep_fraction': 0.40,
            'refine_maxiter': 6,
        }
        config_name = "minimal_expand"
    else:
        # Try different scale
        config3 = best_config.copy()
        config3['perturbation_scale'] = 0.08
        config_name = f"{best_name}_scale08"

    print(f"\n{'='*50}")
    print(f"RUN 3: {config_name}")
    print(f"{'='*50}")

    result3 = run_configuration(samples, meta, config3, args.workers)
    all_results.append((config_name, config3, result3))

    budget_remaining = 60.0 - result3['projected_400_min']
    in_budget = result3['projected_400_min'] <= 60.0

    print(f"\n{'='*40}")
    print(f"RUN 3 Results: {config_name}")
    print(f"  Score: {result3['score']:.4f}")
    print(f"  Projected time: {result3['projected_400_min']:.1f} min")
    print(f"  Budget remaining: {budget_remaining:.1f} min")
    print(f"  In budget: {in_budget}")
    print(f"{'='*40}")

    state['tuning_runs'].append({
        'run': 3,
        'config': config_name,
        'params': config3,
        'score': result3['score'],
        'time_min': result3['projected_400_min'],
        'budget_remaining_min': budget_remaining,
        'in_budget': in_budget,
        'rmse_1src': result3['rmse_mean_1src'],
        'rmse_2src': result3['rmse_mean_2src'],
        'n_perturbed_selected': result3['n_perturbed_selected'],
    })

    # ============ FINAL SUMMARY ============
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    best_in_budget = None
    for name, cfg, res in all_results:
        status = "IN BUDGET" if res['projected_400_min'] <= 60 else "OVER BUDGET"
        print(f"  {name}: Score {res['score']:.4f} @ {res['projected_400_min']:.1f} min [{status}]")
        if res['projected_400_min'] <= 60 and (best_in_budget is None or res['score'] > best_in_budget[2]['score']):
            best_in_budget = (name, cfg, res)

    print(f"\nBaseline: 1.1464 @ 51.2 min")
    if best_in_budget:
        name, cfg, res = best_in_budget
        delta = res['score'] - 1.1464
        print(f"Best in-budget: {name} = {res['score']:.4f} @ {res['projected_400_min']:.1f} min")
        print(f"Delta vs baseline: {delta:+.4f}")
        if delta > 0:
            print(">>> SUCCESS: Beat baseline!")
            state['result'] = 'SUCCESS'
        else:
            print(">>> FAILED: Did not beat baseline")
            state['result'] = 'FAILED'
    else:
        print(">>> FAILED: No configuration within budget")
        state['result'] = 'FAILED'

    # Update final state
    state['status'] = 'completed'
    state['completed_at'] = datetime.now().isoformat()
    if best_in_budget:
        name, cfg, res = best_in_budget
        state['best_in_budget'] = {
            'config': name,
            'params': cfg,
            'score': res['score'],
            'time_min': res['projected_400_min'],
        }

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
