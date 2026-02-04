#!/usr/bin/env python
"""
Generate .npz submission files for the top 4 in-budget experiment configs.

Based on existing experiment results:
  1. sigma_014_019 (scored 1.1703, projected 52.8 min)
  2. sigma_016_020_with_perturb (scored 1.1699, projected 49.0 min)
  3. sigma_018_022_with_perturb (scored 1.1653, projected 51.8 min)
  4. sigma_014_019 seed=99 (variance check on #1)

Usage:
    uv run python scripts/generate_top4.py
    uv run python scripts/generate_top4.py --max-samples 5  # quick test
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

# Top 4 configs derived from existing experiment results
CONFIGS = [
    {
        'name': 'sigma_014_019',
        'description': 'Best scorer: asymmetric sigma 0.14/0.19, fevals 24/44, scale 0.06',
        'original_score': 1.1703,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'max_fevals_1src': 24,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturbation_scale': 0.06,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {
        'name': 'sigma_016_020_perturb',
        'description': 'Runner-up: sigma 0.16/0.20, scale 0.05',
        'original_score': 1.1699,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.16,
            'sigma0_2src': 0.20,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturbation_scale': 0.05,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {
        'name': 'sigma_018_022_perturb',
        'description': 'Third: sigma 0.18/0.22, scale 0.05',
        'original_score': 1.1653,
        'seed': 42,
        'optimizer': {
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturbation_scale': 0.05,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
    {
        'name': 'sigma_014_019_seed99',
        'description': 'Variance check: same as #1 with seed=99',
        'original_score': None,
        'seed': 99,
        'optimizer': {
            'sigma0_1src': 0.14,
            'sigma0_2src': 0.19,
            'max_fevals_1src': 24,
            'max_fevals_2src': 44,
            'timestep_fraction': 0.40,
            'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 2,
            'perturbation_scale': 0.06,
            'perturb_nm_iters': 3,
            'tabu_distance': 0.03,
            'max_tabu_attempts': 10,
        },
    },
]


def score_with_competition_formula(candidate_rmses, lambda_=LAMBDA, n_max=N_MAX):
    n_valid = len(candidate_rmses)
    if n_valid == 0:
        return 0.0
    accuracy = sum(1.0 / (1.0 + r) for r in candidate_rmses) / n_valid
    diversity = lambda_ * (n_valid / n_max)
    return accuracy + diversity


def process_sample(args):
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


def main():
    parser = argparse.ArgumentParser(description='Generate top 4 submission .npz files')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='Workers (default: all CPUs)')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--config-index', type=int, default=None,
                        help='Run only this config index (0-3)')
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

    output_dir = project_root / 'submissions' / 'top4'
    output_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = CONFIGS
    if args.config_index is not None:
        configs_to_run = [CONFIGS[args.config_index]]

    print("=" * 70)
    print("GENERATING TOP 4 SUBMISSION FILES")
    print("=" * 70)
    print(f"Samples: {n_samples}, Workers: {n_workers}")
    for i, cfg in enumerate(CONFIGS):
        marker = " <-- running" if cfg in configs_to_run else ""
        orig = f" (prev: {cfg['original_score']:.4f})" if cfg['original_score'] else ""
        print(f"  [{i}] {cfg['name']}{orig}{marker}")
    print("=" * 70)

    all_summaries = {}
    total_start = time.time()

    for cfg in configs_to_run:
        config_name = cfg['name']
        seed_manager = SeedManager(master_seed=cfg['seed'])
        np.random.seed(cfg['seed'])

        print(f"\n--- Running {config_name} (seed={cfg['seed']}) ---")

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
                      f"t={result['time_s']:.1f}s [{status}]")

        config_time = time.time() - config_start
        results.sort(key=lambda r: r['idx'])

        # Compute score
        sample_scores = []
        for r in results:
            if r['success'] and r['candidate_rmses']:
                score = score_with_competition_formula(r['candidate_rmses'])
                sample_scores.append(score)

        final_score = float(np.mean(sample_scores)) if sample_scores else 0.0
        projected_400 = (config_time / n_samples) * COMPETITION_SAMPLES / 60

        rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
        rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]
        avg_cands = np.mean([r['n_candidates'] for r in results if r['success']])

        # Save .npz
        submission_list = []
        for r in results:
            submission_list.append({
                'sample_id': r['sample_id'],
                'estimated_sources': r['estimated_sources'],
            })

        npz_path = output_dir / f'{config_name}.npz'
        np.savez(str(npz_path), samples=submission_list)

        # Save metrics
        metrics = {
            'name': config_name,
            'description': cfg['description'],
            'seed': cfg['seed'],
            'original_score': cfg['original_score'],
            'config': cfg['optimizer'],
            'n_samples': n_samples,
            'n_workers': n_workers,
            'total_time_s': config_time,
            'projected_400_min': projected_400,
            'submission_score': final_score,
            'avg_candidates': float(avg_cands),
            'rmse_1src_mean': float(np.mean(rmses_1src)) if rmses_1src else None,
            'rmse_2src_mean': float(np.mean(rmses_2src)) if rmses_2src else None,
            'n_successful': sum(1 for r in results if r['success']),
            'timestamp': datetime.now().isoformat(),
        }
        json_path = output_dir / f'{config_name}.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        budget = "IN BUDGET" if projected_400 <= 60 else f"OVER by {projected_400-60:.1f}m"
        prev_str = f" (prev: {cfg['original_score']:.4f})" if cfg['original_score'] else ""
        print(f"\n  {config_name} DONE")
        print(f"  Score: {final_score:.4f}{prev_str}")
        print(f"  Time:  {config_time/60:.1f} min -> projected {projected_400:.1f} min [{budget}]")
        print(f"  Cands: {avg_cands:.2f}")
        print(f"  Saved: {npz_path}")

        all_summaries[config_name] = {
            'score': final_score,
            'projected': projected_400,
            'avg_cands': float(avg_cands),
            'budget': budget,
            'original': cfg['original_score'],
        }

    # Final comparison
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE in {total_time/60:.1f} min")
    print(f"{'='*70}")
    print(f"{'Config':<30s} {'Score':>7s} {'Prev':>7s} {'Proj':>6s} {'Cands':>6s}")
    print(f"{'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    for name, s in sorted(all_summaries.items(), key=lambda x: -x[1]['score']):
        prev = f"{s['original']:.4f}" if s['original'] else "   N/A"
        print(f"{name:<30s} {s['score']:7.4f} {prev:>7s} {s['projected']:5.1f}m {s['avg_cands']:5.2f}")
    print(f"\nFiles in: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
