#!/usr/bin/env python
"""
Post-process a submission .npz to create filtered variants.

Loads an existing submission, re-simulates each candidate to compute
per-candidate RMSE, then creates filtered variants:
  - best_1: Only the best candidate per sample
  - best_2: Best 2 candidates per sample
  - rmse_filtered: Only candidates with RMSE below a threshold

Usage:
    uv run python scripts/filter_submission.py --input submissions/best_4pert_nm2_scale06.npz
"""

import os
import sys
import pickle
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data' / 'Heat_Signature_zero-starter_notebook'))

from simulator import Heat2D


def simulate_candidate(candidate_sources, sample, meta):
    """
    Simulate a candidate and compute RMSE against Y_noisy.

    candidate_sources: list of (x, y, q) tuples
    Returns: RMSE float
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

    # Match shapes
    n_steps = min(len(Y_pred), len(Y_observed))
    rmse = float(np.sqrt(np.mean((Y_pred[:n_steps] - Y_observed[:n_steps]) ** 2)))
    return rmse


def score_with_competition_formula(candidate_rmses, lambda_=0.3, n_max=3):
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


def evaluate_sample(args):
    """Evaluate all candidates for a single sample."""
    sample_idx, sample_pred, sample_data, meta = args

    sample_id = sample_pred['sample_id']
    candidates = sample_pred['estimated_sources']

    candidate_rmses = []
    for candidate in candidates:
        rmse = simulate_candidate(candidate, sample_data, meta)
        candidate_rmses.append(rmse)

    return {
        'sample_idx': sample_idx,
        'sample_id': sample_id,
        'candidates': candidates,
        'rmses': candidate_rmses,
    }


def create_filtered_submission(evaluated_results, strategy, rmse_threshold=None):
    """
    Create a filtered submission based on strategy.

    strategy: 'best_1', 'best_2', 'all_3', 'rmse_filtered'
    """
    submission_list = []

    for r in evaluated_results:
        candidates = r['candidates']
        rmses = r['rmses']

        # Sort candidates by RMSE (best first)
        paired = sorted(zip(rmses, candidates), key=lambda x: x[0])

        if strategy == 'best_1':
            kept = [paired[0][1]]
        elif strategy == 'best_2':
            kept = [p[1] for p in paired[:2]]
        elif strategy == 'all_3':
            kept = [p[1] for p in paired[:3]]
        elif strategy == 'rmse_filtered':
            kept = [p[1] for p in paired if p[0] < rmse_threshold]
            if not kept:
                # Always keep at least the best one
                kept = [paired[0][1]]
        else:
            kept = [p[1] for p in paired]

        submission_list.append({
            'sample_id': r['sample_id'],
            'estimated_sources': kept,
        })

    return submission_list


def main():
    parser = argparse.ArgumentParser(description='Filter submission candidates')
    parser.add_argument('--input', type=str, required=True, help='Input .npz submission')
    parser.add_argument('--data-path', type=str, default=None, help='Test data path')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--rmse-threshold', type=float, default=0.30,
                        help='RMSE threshold for filtered variant')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    # Load submission
    print(f"Loading submission: {input_path}")
    sub_data = np.load(str(input_path), allow_pickle=True)
    pred_samples = list(sub_data['samples'])
    print(f"  {len(pred_samples)} samples, candidates per sample: "
          f"{[len(s['estimated_sources']) for s in pred_samples[:5]]}...")

    # Load test data
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
            print("ERROR: Could not find test data.")
            sys.exit(1)

    print(f"Loading test data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    # Map sample_id -> sample data
    sample_map = {s['sample_id']: s for s in samples}

    # Evaluate all candidates
    print(f"\nEvaluating {len(pred_samples)} samples with {args.workers} workers...")
    work_items = []
    for i, pred in enumerate(pred_samples):
        sample_data = sample_map[pred['sample_id']]
        work_items.append((i, pred, sample_data, meta))

    start = time.time()
    evaluated = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            evaluated.append(result)
            n_done = len(evaluated)
            rmses_str = ', '.join(f'{r:.3f}' for r in result['rmses'])
            print(f"[{n_done:3d}/{len(pred_samples)}] {result['sample_id']}: "
                  f"RMSE=[{rmses_str}]")

    eval_time = time.time() - start
    print(f"\nEvaluation complete in {eval_time:.1f}s")

    # Sort by sample index
    evaluated.sort(key=lambda x: x['sample_idx'])

    # Print per-candidate RMSE statistics
    all_rmses_by_rank = [[], [], []]
    for r in evaluated:
        sorted_rmses = sorted(r['rmses'])
        for i, rmse in enumerate(sorted_rmses):
            if i < 3:
                all_rmses_by_rank[i].append(rmse)

    print(f"\nPer-candidate RMSE statistics:")
    for i, rmses in enumerate(all_rmses_by_rank):
        if rmses:
            print(f"  Candidate {i+1} (rank by RMSE): "
                  f"mean={np.mean(rmses):.4f}, "
                  f"median={np.median(rmses):.4f}, "
                  f"max={np.max(rmses):.4f}")

    # Create variants
    strategies = {
        'best_1': {'strategy': 'best_1'},
        'best_2': {'strategy': 'best_2'},
        'rmse_filtered': {'strategy': 'rmse_filtered', 'rmse_threshold': args.rmse_threshold},
    }

    print(f"\n{'='*70}")
    print("CREATING SUBMISSION VARIANTS")
    print(f"{'='*70}")

    base_name = input_path.stem

    for name, params in strategies.items():
        threshold = params.get('rmse_threshold', None)
        filtered = create_filtered_submission(evaluated, params['strategy'], threshold)

        # Compute expected competition score
        scores = []
        for r, f_sample in zip(evaluated, filtered):
            # Get RMSEs for kept candidates
            kept_sources = f_sample['estimated_sources']
            kept_rmses = []
            for kept_cand in kept_sources:
                for orig_cand, orig_rmse in zip(r['candidates'], r['rmses']):
                    if kept_cand == orig_cand:
                        kept_rmses.append(orig_rmse)
                        break
            score = score_with_competition_formula(kept_rmses)
            scores.append(score)

        avg_score = np.mean(scores)
        avg_cands = np.mean([len(s['estimated_sources']) for s in filtered])

        # Save
        out_path = output_dir / f'{base_name}_{name}.npz'
        submission = {'samples': filtered}
        np.savez(str(out_path), **submission)

        threshold_str = f" (threshold={threshold})" if threshold else ""
        print(f"\n  {name}{threshold_str}:")
        print(f"    Expected score: {avg_score:.4f}")
        print(f"    Avg candidates: {avg_cands:.2f}")
        print(f"    Saved: {out_path}")

    # Also compute score for original (all candidates)
    orig_scores = []
    for r in evaluated:
        score = score_with_competition_formula(r['rmses'])
        orig_scores.append(score)

    print(f"\n  original (all candidates):")
    print(f"    Expected score: {np.mean(orig_scores):.4f}")
    print(f"    Avg candidates: {np.mean([len(r['candidates']) for r in evaluated]):.2f}")
    print(f"    (Already submitted: scored 1.1056)")

    # Save evaluation data for analysis
    eval_path = output_dir / f'{base_name}_evaluation.json'
    eval_data = {
        'per_sample': [
            {
                'sample_id': r['sample_id'],
                'rmses': r['rmses'],
                'rmses_sorted': sorted(r['rmses']),
            }
            for r in evaluated
        ],
        'variants': {
            name: {
                'expected_score': float(np.mean(scores)),
                'avg_candidates': float(np.mean([len(s['estimated_sources']) for s in filtered])),
            }
            for name, scores, filtered in []  # populated above
        }
    }
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"\n  Evaluation data: {eval_path}")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    # Find best variant
    all_variants = {
        'original': np.mean(orig_scores),
    }
    for name, params in strategies.items():
        threshold = params.get('rmse_threshold', None)
        filtered = create_filtered_submission(evaluated, params['strategy'], threshold)
        scores = []
        for r, f_sample in zip(evaluated, filtered):
            kept_sources = f_sample['estimated_sources']
            kept_rmses = []
            for kept_cand in kept_sources:
                for orig_cand, orig_rmse in zip(r['candidates'], r['rmses']):
                    if kept_cand == orig_cand:
                        kept_rmses.append(orig_rmse)
                        break
            score = score_with_competition_formula(kept_rmses)
            scores.append(score)
        all_variants[name] = np.mean(scores)

    for name, score in sorted(all_variants.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if score == max(all_variants.values()) else ""
        print(f"  {name:20s}: {score:.4f}{marker}")


if __name__ == '__main__':
    main()
