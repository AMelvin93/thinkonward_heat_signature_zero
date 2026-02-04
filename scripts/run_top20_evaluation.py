#!/usr/bin/env python
"""
Top-20 Experiment Evaluation with Accurate Per-Candidate RMSE Scoring.

Runs 20 optimizer configs on the full 80-sample test set using the actual
competition scoring formula (per-candidate RMSE, not simplified best-only).

Phase 1: Run all 20 configs (seed=42 unless noted)
Phase 2: Re-run top 5 from Phase 1 with seed=99 for variance checking

Usage:
    uv run python scripts/run_top20_evaluation.py --phase 1
    uv run python scripts/run_top20_evaluation.py --phase 2
    uv run python scripts/run_top20_evaluation.py --phase both
    uv run python scripts/run_top20_evaluation.py --max-samples 5 --config-index 0  # smoke test
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
sys.path.insert(0, str(project_root / 'experiments' / 'tighter_sigma_range'))

from optimizer import TabuBasinHoppingOptimizer
from src.seed_manager import SeedManager

COMPETITION_SAMPLES = 400
LAMBDA = 0.3
N_MAX = 3

# ============================================================================
# Top 20 Configs (ranked by experiment score, all in-budget)
# ============================================================================
CONFIGS = [
    {  # 0
        'name': 'asym_014_019_f24_44',
        'exp_score': 1.1745,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.19,
            'max_fevals_1src': 24, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 1
        'name': 'sigma_015_019_phase3',
        'exp_score': 1.1730,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.15, 'sigma0_2src': 0.19,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 2
        'name': 'sigma_015_019_scale006',
        'exp_score': 1.1709,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.15, 'sigma0_2src': 0.19,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 3
        'name': 'asym_014_019_scale006',
        'exp_score': 1.1703,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.19,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 4
        'name': 'sigma_016_020_perturb',
        'exp_score': 1.1699,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.16, 'sigma0_2src': 0.20,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 5
        'name': 'sigma_015_019_f24_44',
        'exp_score': 1.1694,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.15, 'sigma0_2src': 0.19,
            'max_fevals_1src': 24, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 6
        'name': 'perturb_baseline_014_019',
        'exp_score': 1.1680,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.19,
            'max_fevals_1src': 24, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 7
        'name': 'validated_014_019_scale006',
        'exp_score': 1.1675,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.19,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 8
        'name': 'sigma_014_018_phase3',
        'exp_score': 1.1709,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.18,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 9
        'name': 'sigma_016_020_scale006',
        'exp_score': 1.1657,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.16, 'sigma0_2src': 0.20,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 10
        'name': 'sigma_012_019_scale006',
        'exp_score': 1.1641,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.12, 'sigma0_2src': 0.19,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 11
        'name': 'looser_018_022_nm7',
        'exp_score': 1.1640,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 12
        'name': 'sigma_015_019_nm6',
        'exp_score': 1.1602,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.15, 'sigma0_2src': 0.19,
            'max_fevals_1src': 22, 'max_fevals_2src': 40,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 13
        'name': '4pert_nm2_scale006',
        'exp_score': 1.1563,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 2, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
    {  # 14
        'name': '4pert_nm2_scale006_s99',
        'exp_score': 1.1549,
        'seed': 99,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 2, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
    {  # 15
        'name': 'tabu004_018_022',
        'exp_score': 1.1535,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
    {  # 16
        'name': '4pert_nm2_scale005',
        'exp_score': 1.1524,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 2, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 17
        'name': 'baseline_018_022',
        'exp_score': 1.1504,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 36,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {  # 18
        'name': '4pert_nm2_production',
        'exp_score': 1.1482,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 2, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
    {  # 19
        'name': 'sigma_014_019_f24_44_s99',
        'exp_score': None,
        'seed': 99,
        'optimizer': {
            'sigma0_1src': 0.14, 'sigma0_2src': 0.19,
            'max_fevals_1src': 24, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2, 'perturbation_scale': 0.05,
            'perturb_nm_iters': 3, 'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
]


# ============================================================================
# Scoring
# ============================================================================

def score_with_competition_formula(candidate_rmses, lambda_=LAMBDA, n_max=N_MAX):
    """Per-candidate RMSE scoring (accurate competition formula)."""
    n_valid = len(candidate_rmses)
    if n_valid == 0:
        return 0.0
    accuracy = sum(1.0 / (1.0 + r) for r in candidate_rmses) / n_valid
    diversity = lambda_ * (n_valid / n_max)
    return accuracy + diversity


def score_simplified(best_rmse, n_candidates, lambda_=LAMBDA, n_max=N_MAX):
    """Simplified scoring (assumes all candidates have best RMSE)."""
    accuracy = 1.0 / (1.0 + best_rmse)
    diversity = lambda_ * (n_candidates / n_max)
    return accuracy + diversity


# ============================================================================
# Sample processing
# ============================================================================

def process_sample(args):
    """Process a single sample - returns per-candidate RMSE data."""
    sample_idx, sample, meta, config, sample_seed = args
    np.random.seed(sample_seed)
    config_with_seed = {**config, 'seed': sample_seed}
    optimizer = TabuBasinHoppingOptimizer(**config_with_seed)

    try:
        start = time.time()
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start

        estimated_sources = []
        candidate_rmses = []
        for i, candidate in enumerate(candidates):
            source_list = [(float(x), float(y), float(q)) for x, y, q in candidate]
            estimated_sources.append(source_list)
            if i < len(results):
                candidate_rmses.append(float(results[i].rmse))
            else:
                candidate_rmses.append(float(best_rmse))

        actual_score = score_with_competition_formula(candidate_rmses)
        simplified = score_simplified(best_rmse, len(candidates))

        return {
            'idx': sample_idx,
            'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
            'n_sources': sample['n_sources'],
            'estimated_sources': estimated_sources,
            'candidate_rmses': candidate_rmses,
            'best_rmse': float(best_rmse),
            'n_candidates': len(estimated_sources),
            'actual_score': actual_score,
            'simplified_score': simplified,
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
            'actual_score': 0.0,
            'simplified_score': 0.0,
            'time_s': 0,
            'success': False,
            'error': str(e),
        }


# ============================================================================
# Config runner
# ============================================================================

def run_single_config(cfg, samples, meta, n_workers, output_dir, skip_existing=False):
    """Run a single config on all samples, save .npz + .json."""
    config_name = cfg['name']
    seed = cfg['seed']

    npz_path = output_dir / f'{config_name}.npz'
    json_path = output_dir / f'{config_name}.json'

    if skip_existing and npz_path.exists() and json_path.exists():
        print(f"  SKIP {config_name} (already exists)")
        with open(json_path) as f:
            return json.load(f)

    seed_manager = SeedManager(master_seed=seed)
    np.random.seed(seed)
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"  Config: {config_name} (seed={seed})")
    print(f"  Samples: {n_samples}, Workers: {n_workers}")
    print(f"{'='*60}")

    work_items = [
        (i, samples[i], meta, cfg['optimizer'], seed_manager.get_sample_seed(i))
        for i in range(n_samples)
    ]

    config_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            print(f"  [{len(results):3d}/{n_samples}] "
                  f"Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src "
                  f"RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} "
                  f"actual={result['actual_score']:.4f} "
                  f"t={result['time_s']:.1f}s [{status}]")

    config_time = time.time() - config_start
    results.sort(key=lambda r: r['idx'])

    # Compute aggregate scores
    actual_scores = [r['actual_score'] for r in results if r['success']]
    simplified_scores = [r['simplified_score'] for r in results if r['success']]

    actual_mean = float(np.mean(actual_scores)) if actual_scores else 0.0
    simplified_mean = float(np.mean(simplified_scores)) if simplified_scores else 0.0
    score_delta = actual_mean - simplified_mean
    projected_400 = (config_time / n_samples) * COMPETITION_SAMPLES / 60

    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
    avg_cands = float(np.mean([r['n_candidates'] for r in results if r['success']]))

    # Save .npz
    submission_list = []
    for r in results:
        submission_list.append({
            'sample_id': r['sample_id'],
            'estimated_sources': r['estimated_sources'],
        })
    np.savez(str(npz_path), samples=submission_list)

    # Build metrics
    metrics = {
        'name': config_name,
        'seed': seed,
        'config': cfg['optimizer'],
        'exp_score': cfg['exp_score'],
        'n_samples': n_samples,
        'n_workers': n_workers,
        'total_time_s': config_time,
        'projected_400_min': projected_400,
        'actual_competition_score': actual_mean,
        'simplified_score': simplified_mean,
        'score_delta': score_delta,
        'avg_candidates': avg_cands,
        'rmse_1src_mean': float(np.mean(rmses_1src)) if rmses_1src else None,
        'rmse_2src_mean': float(np.mean(rmses_2src)) if rmses_2src else None,
        'n_successful': sum(1 for r in results if r['success']),
        'per_sample_scores': [
            {
                'idx': r['idx'],
                'n_sources': r['n_sources'],
                'actual_score': r['actual_score'],
                'simplified_score': r['simplified_score'],
                'best_rmse': r['best_rmse'],
                'candidate_rmses': r['candidate_rmses'],
                'n_candidates': r['n_candidates'],
                'time_s': r['time_s'],
            }
            for r in results if r['success']
        ],
        'timestamp': datetime.now().isoformat(),
    }

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    budget = "IN BUDGET" if projected_400 <= 60 else f"OVER by {projected_400-60:.1f}m"
    exp_str = f" (exp: {cfg['exp_score']:.4f})" if cfg['exp_score'] else ""
    print(f"\n  {config_name} DONE")
    print(f"  Actual score:     {actual_mean:.4f}")
    print(f"  Simplified score: {simplified_mean:.4f}{exp_str}")
    print(f"  Delta:            {score_delta:+.4f}")
    print(f"  Time:  {config_time/60:.1f} min -> projected {projected_400:.1f} min [{budget}]")
    print(f"  Cands: {avg_cands:.2f}")
    print(f"  Saved: {npz_path}")

    return metrics


# ============================================================================
# Phase logic
# ============================================================================

def run_phase1(samples, meta, n_workers, output_dir, config_indices=None,
               skip_existing=False):
    """Phase 1: Run all 20 configs."""
    print("\n" + "=" * 70)
    print("PHASE 1: RUNNING ALL 20 CONFIGS")
    print("=" * 70)

    configs_to_run = CONFIGS
    if config_indices is not None:
        configs_to_run = [CONFIGS[i] for i in config_indices]

    for i, cfg in enumerate(CONFIGS):
        marker = " <-- running" if cfg in configs_to_run else ""
        exp = f" (exp: {cfg['exp_score']:.4f})" if cfg['exp_score'] else ""
        print(f"  [{i:2d}] {cfg['name']:<35s}{exp}{marker}")

    all_metrics = []
    phase_start = time.time()

    for cfg in configs_to_run:
        metrics = run_single_config(cfg, samples, meta, n_workers, output_dir,
                                    skip_existing=skip_existing)
        all_metrics.append(metrics)

    phase_time = time.time() - phase_start

    # Sort by actual competition score
    all_metrics.sort(key=lambda m: -m['actual_competition_score'])

    # Save master ranking
    ranking = []
    for rank, m in enumerate(all_metrics, 1):
        ranking.append({
            'rank': rank,
            'name': m['name'],
            'actual_score': m['actual_competition_score'],
            'simplified_score': m['simplified_score'],
            'delta': m['score_delta'],
            'exp_score': m.get('exp_score'),
            'projected_400_min': m['projected_400_min'],
            'avg_candidates': m['avg_candidates'],
            'seed': m['seed'],
        })

    ranking_path = output_dir / 'master_ranking.json'
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)

    # Print ranking table
    print(f"\n{'='*90}")
    print(f"PHASE 1 RANKING (by actual competition score)")
    print(f"Total time: {phase_time/60:.1f} min")
    print(f"{'='*90}")
    print(f"{'Rank':>4s}  {'Config':<35s} {'Actual':>7s} {'Simpl':>7s} "
          f"{'Delta':>7s} {'Exp':>7s} {'Proj':>6s} {'Cands':>5s}")
    print(f"{'-'*4}  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
    for r in ranking:
        exp = f"{r['exp_score']:.4f}" if r['exp_score'] else "   N/A"
        print(f"{r['rank']:4d}  {r['name']:<35s} {r['actual_score']:7.4f} "
              f"{r['simplified_score']:7.4f} {r['delta']:+7.4f} {exp:>7s} "
              f"{r['projected_400_min']:5.1f}m {r['avg_candidates']:5.2f}")
    print(f"\nRanking saved: {ranking_path}")

    return all_metrics


def run_phase2(samples, meta, n_workers, output_dir, top_n=5):
    """Phase 2: Re-run top N configs with seed=99 for variance checking."""
    ranking_path = output_dir / 'master_ranking.json'
    if not ranking_path.exists():
        print("ERROR: Phase 1 master_ranking.json not found. Run Phase 1 first.")
        sys.exit(1)

    with open(ranking_path) as f:
        ranking = json.load(f)

    # Take top N, but skip any that already used seed=99
    top_configs = []
    for r in ranking:
        if len(top_configs) >= top_n:
            break
        if r['seed'] == 99:
            continue  # already a seed=99 variant
        top_configs.append(r)

    print("\n" + "=" * 70)
    print(f"PHASE 2: VARIANCE CHECK (seed=99) ON TOP {len(top_configs)} CONFIGS")
    print("=" * 70)
    for r in top_configs:
        print(f"  {r['rank']:2d}. {r['name']:<35s} actual={r['actual_score']:.4f}")

    phase2_results = []
    phase_start = time.time()

    for r in top_configs:
        # Find original config
        orig_cfg = None
        for c in CONFIGS:
            if c['name'] == r['name']:
                orig_cfg = c
                break
        if orig_cfg is None:
            print(f"  WARNING: Could not find config for {r['name']}, skipping")
            continue

        # Create seed=99 variant
        s99_cfg = {
            'name': f"{r['name']}_s99_phase2",
            'exp_score': orig_cfg['exp_score'],
            'seed': 99,
            'optimizer': orig_cfg['optimizer'].copy(),
        }

        metrics = run_single_config(s99_cfg, samples, meta, n_workers, output_dir)

        phase2_results.append({
            'name': r['name'],
            'seed42_actual': r['actual_score'],
            'seed42_simplified': r['simplified_score'],
            'seed99_actual': metrics['actual_competition_score'],
            'seed99_simplified': metrics['simplified_score'],
            'actual_diff': metrics['actual_competition_score'] - r['actual_score'],
            'mean_actual': (r['actual_score'] + metrics['actual_competition_score']) / 2,
            'min_actual': min(r['actual_score'], metrics['actual_competition_score']),
            'max_actual': max(r['actual_score'], metrics['actual_competition_score']),
        })

    phase_time = time.time() - phase_start

    # Save phase2 results
    variance_path = output_dir / 'phase2_variance.json'
    with open(variance_path, 'w') as f:
        json.dump(phase2_results, f, indent=2)

    # Print variance table
    print(f"\n{'='*90}")
    print(f"PHASE 2 VARIANCE RESULTS")
    print(f"Total time: {phase_time/60:.1f} min")
    print(f"{'='*90}")
    print(f"{'Config':<35s} {'s42 Actual':>10s} {'s99 Actual':>10s} "
          f"{'Diff':>7s} {'Mean':>7s} {'Min':>7s}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")
    for p in sorted(phase2_results, key=lambda x: -x['mean_actual']):
        print(f"{p['name']:<35s} {p['seed42_actual']:10.4f} {p['seed99_actual']:10.4f} "
              f"{p['actual_diff']:+7.4f} {p['mean_actual']:7.4f} {p['min_actual']:7.4f}")

    # Recommendation
    best_by_mean = max(phase2_results, key=lambda x: x['mean_actual'])
    best_by_min = max(phase2_results, key=lambda x: x['min_actual'])
    print(f"\n  Best by mean score: {best_by_mean['name']} ({best_by_mean['mean_actual']:.4f})")
    print(f"  Most robust (max min): {best_by_min['name']} ({best_by_min['min_actual']:.4f})")
    print(f"\nVariance data saved: {variance_path}")

    return phase2_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Top-20 Experiment Evaluation with Per-Candidate RMSE Scoring')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples (default: all 80)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Workers (default: all CPUs)')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--config-index', type=int, default=None,
                        help='Run only this config index (0-19)')
    parser.add_argument('--phase', type=str, default='both',
                        choices=['1', '2', 'both'],
                        help='Phase 1 only, Phase 2 only, or both')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip configs with existing .npz files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Override seed for all configs')
    args = parser.parse_args()

    n_workers = args.workers or os.cpu_count()

    # Find data
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

    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    if args.max_samples:
        samples = samples[:args.max_samples]
    n_samples = len(samples)

    output_dir = project_root / 'submissions' / 'top20'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOP-20 EXPERIMENT EVALUATION")
    print("=" * 70)
    print(f"Samples: {n_samples}, Workers: {n_workers}, Phase: {args.phase}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    config_indices = None
    if args.config_index is not None:
        config_indices = [args.config_index]

    total_start = time.time()

    if args.phase in ('1', 'both'):
        run_phase1(samples, meta, n_workers, output_dir,
                   config_indices=config_indices,
                   skip_existing=args.skip_existing)

    if args.phase in ('2', 'both'):
        run_phase2(samples, meta, n_workers, output_dir)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE in {total_time/60:.1f} min")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
