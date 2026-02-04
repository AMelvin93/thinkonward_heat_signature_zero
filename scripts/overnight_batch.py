#!/usr/bin/env python
"""
Overnight Batch Runner

Runs 8 optimizer configurations sequentially, producing submission-ready .npz
files for each. Includes crash recovery via batch_state.json.

Two key improvements tested:
  - Multi-basin hopping: perturb from top-K solutions, not just best
  - Quality gating: drop candidates that hurt the competition score

Usage:
    # Quick smoke test (5 samples, 1 config):
    uv run python scripts/overnight_batch.py --max-samples 5 --max-configs 1

    # Full overnight run on WSL:
    nohup uv run python scripts/overnight_batch.py --workers 7 > submissions/overnight/batch_log.txt 2>&1 &
"""

import os
import sys
import pickle
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'overnight_batch'))

from optimizer import TabuBasinHoppingOptimizer
from src.seed_manager import SeedManager

# Competition constants
N_MAX = 3
LAMBDA = 0.3
TAU = 0.2
COMPETITION_SAMPLES = 400
DEFAULT_WORKERS = 7


# ============================================================================
# BATCH CONFIGURATIONS (sorted by priority)
# ============================================================================

CONFIGS = [
    {
        'name': 'multibasin3_qgate_4pert_nm2',
        'description': 'Multi-basin(3) + quality gate + current best config',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 3,
            'quality_gate_enabled': True,
        },
    },
    {
        'name': 'multibasin3_4pert_nm2',
        'description': 'Multi-basin(3) only, no quality gate',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 3,
            'quality_gate_enabled': False,
        },
    },
    {
        'name': 'qgate_4pert_nm2',
        'description': 'Quality gate only, no multi-basin (n_perturb_bases=1)',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 1,
            'quality_gate_enabled': True,
        },
    },
    {
        'name': 'multibasin3_qgate_3pert_nm3_ref10',
        'description': 'Multi-basin + gate + 3pert/nm3/refine10 base',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 10,
            'enable_tabu_hopping': True,
            'n_perturbations': 3,
            'perturb_nm_iters': 3,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 3,
            'quality_gate_enabled': True,
        },
    },
    {
        'name': 'multibasin3_qgate_sigma020_024',
        'description': 'Multi-basin + gate + higher sigma range',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.20,
            'sigma0_2src': 0.24,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 3,
            'quality_gate_enabled': True,
        },
    },
    {
        'name': 'multibasin3_larger_scale',
        'description': 'Multi-basin + perturbation_scale=0.10',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.10,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 3,
            'quality_gate_enabled': False,
        },
    },
    {
        'name': 'baseline_4pert_nm2_seed99',
        'description': 'Current best config, different seed (variance check)',
        'seed': 99,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 1,
            'quality_gate_enabled': False,
        },
    },
    {
        'name': 'multibasin2_6pert_nm2',
        'description': '2 bases, 6 perturbations',
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'max_fevals_1src': 20,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 6,
            'perturb_nm_iters': 2,
            'perturbation_scale': 0.06,
            'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
            'n_perturb_bases': 2,
            'quality_gate_enabled': False,
        },
    },
]


# ============================================================================
# SCORING
# ============================================================================

def score_with_competition_formula(candidate_rmses, lambda_=LAMBDA, n_max=N_MAX):
    """
    Score using the ACTUAL competition formula:
    P = (1/N_valid) * sum(1/(1+L_i)) + lambda * (N_valid/N_max)
    """
    n_valid = len(candidate_rmses)
    if n_valid == 0:
        return 0.0
    accuracy = sum(1.0 / (1.0 + r) for r in candidate_rmses) / n_valid
    diversity = lambda_ * (n_valid / n_max)
    return accuracy + diversity


def simulate_candidate(candidate_sources, sample, meta):
    """
    Simulate a candidate and compute RMSE against Y_noisy.
    candidate_sources: list of (x, y, q) tuples
    """
    Lx, Ly = 2.0, 1.0
    nx, ny = 100, 50
    dt = meta['dt']
    nt = sample['sample_metadata']['nt']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']
    sensors_xy = np.array(sample['sensors_xy'])
    Y_observed = sample['Y_noisy']

    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)
    sources = [{'x': s[0], 'y': s[1], 'q': s[2], 'on': (0, nt * dt)} for s in candidate_sources]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

    n_steps = min(len(Y_pred), len(Y_observed))
    rmse = float(np.sqrt(np.mean((Y_pred[:n_steps] - Y_observed[:n_steps]) ** 2)))
    return rmse


# ============================================================================
# SAMPLE PROCESSING
# ============================================================================

def process_sample(args):
    """Process a single sample. Returns result dict with candidates and RMSEs."""
    sample_idx, sample, meta, config, sample_seed = args

    # Seed worker for reproducibility
    np.random.seed(sample_seed)
    config_with_seed = {**config, 'seed': sample_seed}

    optimizer = TabuBasinHoppingOptimizer(**config_with_seed)

    try:
        start = time.time()
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        # Format candidates for submission
        estimated_sources = []
        candidate_rmses = []
        for i, candidate in enumerate(candidates):
            source_list = [(float(x), float(y), float(q)) for x, y, q in candidate]
            estimated_sources.append(source_list)
            # Get per-candidate RMSE from results
            if i < len(results):
                candidate_rmses.append(float(results[i].rmse))
            else:
                candidate_rmses.append(float(best_rmse))

        return {
            'idx': sample_idx,
            'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
            'n_sources': sample['n_sources'],
            'estimated_sources': estimated_sources,
            'candidate_rmses': candidate_rmses,
            'best_rmse': float(best_rmse),
            'n_candidates': len(estimated_sources),
            'time_s': elapsed,
            'success': True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'idx': sample_idx,
            'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
            'n_sources': sample.get('n_sources', 0),
            'estimated_sources': [],
            'candidate_rmses': [],
            'best_rmse': float('inf'),
            'n_candidates': 0,
            'time_s': 0,
            'success': False,
            'error': str(e),
        }


# ============================================================================
# BATCH STATE (crash recovery)
# ============================================================================

def load_batch_state(state_path):
    """Load batch state from disk for crash recovery."""
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return {'completed': {}, 'started_at': datetime.now().isoformat()}


def save_batch_state(state_path, state):
    """Save batch state to disk."""
    state['updated_at'] = datetime.now().isoformat()
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Overnight batch runner')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per config (for testing)')
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Limit number of configs to run')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to test data pickle')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for submissions')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from batch_state.json (skip completed configs)')
    args = parser.parse_args()

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'submissions' / 'overnight'
    output_dir.mkdir(parents=True, exist_ok=True)

    # State file for crash recovery
    state_path = output_dir / 'batch_state.json'
    if args.resume:
        batch_state = load_batch_state(state_path)
        print(f"Resuming from batch state: {len(batch_state['completed'])} configs already done")
    else:
        batch_state = {'completed': {}, 'started_at': datetime.now().isoformat()}

    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        candidates_paths = [
            project_root / 'data' / 'heat-signature-zero-test-data.pkl',
            Path('/workspace/data/heat-signature-zero-test-data.pkl'),
        ]
        data_path = None
        for p in candidates_paths:
            if p.exists():
                data_path = p
                break
        if data_path is None:
            print("ERROR: Could not find test data. Use --data-path to specify.")
            sys.exit(1)

    # Load data once
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)

    # Select configs to run
    configs_to_run = CONFIGS
    if args.max_configs:
        configs_to_run = configs_to_run[:args.max_configs]

    # Print header
    print("=" * 80)
    print("OVERNIGHT BATCH RUNNER")
    print("=" * 80)
    print(f"Configs:    {len(configs_to_run)}")
    print(f"Samples:    {n_samples}")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {output_dir}")
    print(f"Resume:     {args.resume}")
    print("=" * 80)

    for i, cfg in enumerate(configs_to_run):
        print(f"  [{i+1}] {cfg['name']}: {cfg['description']}")
    print("=" * 80)

    # Track all results for final comparison
    all_results_summary = {}
    batch_start = time.time()

    for config_idx, cfg in enumerate(configs_to_run):
        config_name = cfg['name']

        # Skip if already completed (crash recovery)
        if config_name in batch_state['completed']:
            print(f"\n{'='*80}")
            print(f"[{config_idx+1}/{len(configs_to_run)}] SKIPPING {config_name} (already completed)")
            prev = batch_state['completed'][config_name]
            all_results_summary[config_name] = {
                'score': prev.get('score', 0),
                'projected_400_min': prev.get('projected_400_min', 0),
                'status': 'skipped (cached)',
            }
            continue

        print(f"\n{'='*80}")
        print(f"[{config_idx+1}/{len(configs_to_run)}] RUNNING: {config_name}")
        print(f"  {cfg['description']}")
        print(f"  seed={cfg['seed']}")
        print(f"{'='*80}")

        # Initialize seed manager for this config
        seed_manager = SeedManager(master_seed=cfg['seed'])
        np.random.seed(cfg['seed'])

        optimizer_config = cfg['optimizer']

        # Create work items
        work_items = [
            (i, samples[i], meta, optimizer_config, seed_manager.get_sample_seed(i))
            for i in range(n_samples)
        ]

        # Process all samples
        config_start = time.time()
        results = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, item): item[0] for item in work_items}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "OK" if result['success'] else "ERR"
                elapsed_so_far = time.time() - config_start
                print(f"  [{len(results):3d}/{n_samples}] "
                      f"Sample {result['idx']:3d}: "
                      f"{result['n_sources']}-src "
                      f"RMSE={result['best_rmse']:.4f} "
                      f"cands={result['n_candidates']} "
                      f"time={result['time_s']:.1f}s [{status}]")

        config_time = time.time() - config_start

        # Sort results by sample index
        results.sort(key=lambda r: r['idx'])

        # ---- Compute TRUE competition score ----
        # Re-simulate each candidate on fine grid to get true RMSEs
        # (The optimizer already does fine-grid eval, so candidate_rmses are fine-grid)
        sample_scores = []
        for r in results:
            if r['success'] and r['candidate_rmses']:
                score = score_with_competition_formula(r['candidate_rmses'])
                sample_scores.append(score)
            elif r['success']:
                # Fallback: use best_rmse with n_candidates
                score = score_with_competition_formula([r['best_rmse']] * r['n_candidates'])
                sample_scores.append(score)

        final_score = float(np.mean(sample_scores)) if sample_scores else 0.0
        projected_400 = (config_time / n_samples) * COMPETITION_SAMPLES / 60

        # Per-type stats
        rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
        rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
        avg_candidates = np.mean([r['n_candidates'] for r in results if r['success']])

        # ---- Save submission .npz ----
        submission_list = []
        for r in results:
            submission_list.append({
                'sample_id': r['sample_id'],
                'estimated_sources': r['estimated_sources'],
            })

        npz_path = output_dir / f'{config_name}.npz'
        submission = {'samples': submission_list}
        np.savez(str(npz_path), **submission)

        # ---- Save metrics JSON ----
        metrics = {
            'name': config_name,
            'description': cfg['description'],
            'seed': cfg['seed'],
            'config': optimizer_config,
            'n_samples': n_samples,
            'n_workers': args.workers,
            'total_time_s': config_time,
            'projected_400_min': projected_400,
            'submission_score': final_score,
            'avg_candidates': float(avg_candidates),
            'rmse_1src_mean': float(np.mean(rmses_1src)) if rmses_1src else None,
            'rmse_2src_mean': float(np.mean(rmses_2src)) if rmses_2src else None,
            'rmse_overall_mean': float(np.mean([r['best_rmse'] for r in results if r['success']])),
            'n_successful': sum(1 for r in results if r['success']),
            'n_failed': sum(1 for r in results if not r['success']),
            'timestamp': datetime.now().isoformat(),
        }

        json_path = output_dir / f'{config_name}.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # ---- Update batch state ----
        batch_state['completed'][config_name] = {
            'score': final_score,
            'projected_400_min': projected_400,
            'time_s': config_time,
            'avg_candidates': float(avg_candidates),
            'timestamp': datetime.now().isoformat(),
        }
        save_batch_state(state_path, batch_state)

        # ---- Print config summary ----
        budget_status = "IN BUDGET" if projected_400 <= 60 else f"OVER by {projected_400 - 60:.1f}m"
        print(f"\n  --- {config_name} RESULTS ---")
        print(f"  Score:           {final_score:.4f}")
        print(f"  Time:            {config_time/60:.1f} min ({n_samples} samples)")
        print(f"  Projected (400): {projected_400:.1f} min [{budget_status}]")
        print(f"  Avg candidates:  {avg_candidates:.2f}")
        if rmses_1src:
            print(f"  RMSE 1-src:      {np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
        if rmses_2src:
            print(f"  RMSE 2-src:      {np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
        print(f"  Saved:           {npz_path}")

        # Store for comparison table
        all_results_summary[config_name] = {
            'score': final_score,
            'projected_400_min': projected_400,
            'avg_candidates': float(avg_candidates),
            'rmse_1src': float(np.mean(rmses_1src)) if rmses_1src else None,
            'rmse_2src': float(np.mean(rmses_2src)) if rmses_2src else None,
            'status': budget_status,
        }

        # ---- Print running comparison table ----
        if len(all_results_summary) > 1:
            print(f"\n  {'='*70}")
            print(f"  RUNNING COMPARISON (sorted by score)")
            print(f"  {'='*70}")
            print(f"  {'Config':<40s} {'Score':>7s} {'Proj':>6s} {'Cands':>6s} {'Status':>10s}")
            print(f"  {'-'*40} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")
            for name, summary in sorted(all_results_summary.items(),
                                         key=lambda x: -x[1]['score']):
                cands = summary.get('avg_candidates', 0)
                proj = summary.get('projected_400_min', 0)
                print(f"  {name:<40s} {summary['score']:7.4f} {proj:5.1f}m {cands:5.2f} {summary['status']:>10s}")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    total_batch_time = time.time() - batch_start

    print(f"\n{'='*80}")
    print("FINAL BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total batch time: {total_batch_time/3600:.2f} hours")
    print(f"Configs completed: {len(all_results_summary)}")
    print()
    print(f"{'Config':<40s} {'Score':>7s} {'Proj':>6s} {'Cands':>6s} {'Status':>10s}")
    print(f"{'-'*40} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")

    best_name = None
    best_score = -1

    for name, summary in sorted(all_results_summary.items(),
                                 key=lambda x: -x[1]['score']):
        cands = summary.get('avg_candidates', 0)
        proj = summary.get('projected_400_min', 0)
        marker = ""
        if summary['score'] > best_score and proj <= 60:
            best_score = summary['score']
            best_name = name
        print(f"{name:<40s} {summary['score']:7.4f} {proj:5.1f}m {cands:5.2f} {summary['status']:>10s}")

    if best_name:
        print(f"\nRECOMMENDATION: Submit {best_name} (score={best_score:.4f})")
        print(f"  File: {output_dir / best_name}.npz")
    else:
        print("\nWARNING: No configs finished within budget!")

    print(f"\nBatch state: {state_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
