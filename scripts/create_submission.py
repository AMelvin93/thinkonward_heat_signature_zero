#!/usr/bin/env python
"""
Create competition submission .npz file.

Uses the best validated config (4pert_nm2_scale06) with TabuBasinHoppingOptimizer.
Produces a .npz file in the exact format required by the competition.

Usage:
    # On WSL for accurate timing:
    uv run python scripts/create_submission.py

    # Quick test (10 samples):
    uv run python scripts/create_submission.py --max-samples 10

    # With custom output:
    uv run python scripts/create_submission.py --output submissions/my_submission.npz
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

# Competition constants
N_MAX = 3
LAMBDA = 0.3
TAU = 0.2
COMPETITION_SAMPLES = 400
DEFAULT_WORKERS = 7  # G4dn.2xlarge simulation


# Best validated config: 4pert_nm2_scale06
# Mean: 1.1549, Best: 1.1612 @ 57.5 min
BEST_CONFIG = {
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
}


def calculate_sample_score(rmse, n_candidates=3, lambda_=LAMBDA, n_max=N_MAX):
    """Calculate competition score for a single sample."""
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def process_sample(args):
    """
    Process a single sample and return both metrics AND source candidates.

    Returns dict with candidates formatted for submission.
    """
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
        # Each candidate is a list of (x, y, q) tuples
        estimated_sources = []
        for candidate in candidates:
            # candidate is a list of (x, y, q) tuples
            source_list = [(float(x), float(y), float(q)) for x, y, q in candidate]
            estimated_sources.append(source_list)

        return {
            'idx': sample_idx,
            'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
            'n_sources': sample['n_sources'],
            'estimated_sources': estimated_sources,
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
            'best_rmse': float('inf'),
            'n_candidates': 0,
            'time_s': 0,
            'success': False,
            'error': str(e),
        }


def format_submission(results):
    """
    Convert results to competition submission format.

    Returns list of dicts with 'sample_id' and 'estimated_sources'.
    """
    # Sort by sample index to maintain order
    results_sorted = sorted(results, key=lambda r: r['idx'])

    submission_list = []
    for r in results_sorted:
        submission_list.append({
            'sample_id': r['sample_id'],
            'estimated_sources': r['estimated_sources'],
        })

    return submission_list


def main():
    parser = argparse.ArgumentParser(description='Create competition submission .npz')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Number of parallel workers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick testing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz path')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to test data pickle')
    args = parser.parse_args()

    # Initialize seed manager
    seed_manager = SeedManager(master_seed=args.seed)
    np.random.seed(args.seed)

    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Try common locations
        candidates = [
            project_root / 'data' / 'heat-signature-zero-test-data.pkl',
            Path('/workspace/data/heat-signature-zero-test-data.pkl'),
        ]
        data_path = None
        for p in candidates:
            if p.exists():
                data_path = p
                break
        if data_path is None:
            print("ERROR: Could not find test data. Use --data-path to specify.")
            sys.exit(1)

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = project_root / 'submissions' / f'submission_{timestamp}.npz'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print config
    print("=" * 70)
    print("CREATING COMPETITION SUBMISSION")
    print("=" * 70)
    print(f"Samples:    {n_samples}")
    print(f"Workers:    {args.workers}")
    print(f"Seed:       {args.seed}")
    print(f"Config:     4pert_nm2_scale06 (best validated)")
    print(f"Output:     {output_path}")
    print("=" * 70)

    # Create work items with per-sample seeds
    work_items = [
        (i, samples[i], meta, BEST_CONFIG, seed_manager.get_sample_seed(i))
        for i in range(n_samples)
    ]

    # Process all samples
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "ERR"
            elapsed_so_far = time.time() - start_time
            print(f"[{len(results):3d}/{n_samples}] "
                  f"Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src "
                  f"RMSE={result['best_rmse']:.4f} "
                  f"cands={result['n_candidates']} "
                  f"time={result['time_s']:.1f}s [{status}]")

    total_time = time.time() - start_time

    # Format submission
    submission_list = format_submission(results)

    # Save .npz
    submission = {'samples': submission_list}
    np.savez(str(output_path), **submission)

    # Also save a JSON with metrics for reference
    metrics_path = output_path.with_suffix('.json')

    # Calculate scores
    sample_scores = []
    for r in results:
        if r['success']:
            score = calculate_sample_score(r['best_rmse'], r['n_candidates'])
            sample_scores.append(score)

    final_score = np.mean(sample_scores) if sample_scores else 0
    projected_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    metrics = {
        'seed': args.seed,
        'config': BEST_CONFIG,
        'n_samples': n_samples,
        'n_workers': args.workers,
        'total_time_s': total_time,
        'projected_400_min': projected_400,
        'submission_score': float(final_score),
        'rmse_1src': float(np.mean(rmses_1src)) if rmses_1src else None,
        'rmse_2src': float(np.mean(rmses_2src)) if rmses_2src else None,
        'n_successful': sum(1 for r in results if r['success']),
        'n_failed': sum(1 for r in results if not r['success']),
        'output_file': str(output_path),
        'timestamp': datetime.now().isoformat(),
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*70}")
    print(f"Score:           {final_score:.4f}")
    print(f"Time:            {total_time/60:.1f} min ({n_samples} samples)")
    print(f"Projected (400): {projected_400:.1f} min")
    if rmses_1src:
        print(f"RMSE 1-src:      {np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"RMSE 2-src:      {np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print(f"Successful:      {metrics['n_successful']}/{n_samples}")
    print(f"\nSubmission saved: {output_path}")
    print(f"Metrics saved:   {metrics_path}")

    budget_status = "IN BUDGET" if projected_400 <= 60 else f"OVER BUDGET by {projected_400 - 60:.1f} min"
    print(f"Budget:          {budget_status}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
