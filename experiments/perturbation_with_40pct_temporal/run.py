#!/usr/bin/env python
"""
Run experiment: Perturbation with 40% Temporal Fidelity

Combines W2's 40% temporal config (1.1688) with perturbed local restart (1.1452).

Usage:
    cd /workspace
    uv run python experiments/perturbation_with_40pct_temporal/run.py
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime

import numpy as np
import mlflow

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from experiments.perturbation_with_40pct_temporal.optimizer import PerturbationTemporalOptimizer
from utils import score_submission


def load_test_data():
    data_path = os.path.join(project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['samples'], data['meta']


def run_experiment(
    timestep_fraction: float = 0.40,
    sigma0_1src: float = 0.18,
    sigma0_2src: float = 0.22,
    max_fevals_1src: int = 20,
    max_fevals_2src: int = 36,
    refine_maxiter: int = 8,
    perturb_top_n: int = 1,
    n_perturbations: int = 2,
    perturbation_scale: float = 0.05,
    perturb_nm_iters: int = 3,
    enable_perturbation: bool = True,
    run_name: str = "perturbation_40pct_temporal",
    verbose: bool = True,
):
    """Run the experiment with given configuration."""

    test_data, meta = load_test_data()
    n_samples = len(test_data)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: Perturbation with 40% Temporal Fidelity")
        print(f"{'='*60}")
        print(f"Config:")
        print(f"  timestep_fraction: {timestep_fraction}")
        print(f"  sigma0_1src: {sigma0_1src}, sigma0_2src: {sigma0_2src}")
        print(f"  max_fevals: {max_fevals_1src}/{max_fevals_2src}")
        print(f"  refine_maxiter: {refine_maxiter}")
        print(f"  perturbation: {enable_perturbation}")
        if enable_perturbation:
            print(f"    perturb_top_n: {perturb_top_n}")
            print(f"    n_perturbations: {n_perturbations}")
            print(f"    perturbation_scale: {perturbation_scale}")
            print(f"    perturb_nm_iters: {perturb_nm_iters}")
        print(f"{'='*60}")

    # Create optimizer
    optimizer = PerturbationTemporalOptimizer(
        timestep_fraction=timestep_fraction,
        sigma0_1src=sigma0_1src,
        sigma0_2src=sigma0_2src,
        max_fevals_1src=max_fevals_1src,
        max_fevals_2src=max_fevals_2src,
        refine_maxiter=refine_maxiter,
        enable_perturbation=enable_perturbation,
        perturb_top_n=perturb_top_n,
        n_perturbations=n_perturbations,
        perturbation_scale=perturbation_scale,
        perturb_nm_iters=perturb_nm_iters,
    )

    # Run optimization
    predictions = []
    sample_times = []
    sample_rmses = []
    perturbed_selections = 0

    start_time = time.time()

    for i, sample in enumerate(test_data):
        sample_start = time.time()

        candidate_sources, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, verbose=False
        )

        sample_time = time.time() - sample_start
        sample_times.append(sample_time)
        sample_rmses.append(best_rmse)

        # Check if perturbed candidate was selected
        for r in results:
            if r.init_type == 'perturbed':
                perturbed_selections += 1
                break

        predictions.append({
            'sample_id': sample.get('sample_id', i),
            'candidates': candidate_sources,
            'n_sources': sample['n_sources'],
            'best_rmse': best_rmse,
        })

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            projected = avg_time * n_samples
            print(f"  Sample {i+1}/{n_samples}: RMSE={best_rmse:.4f}, "
                  f"Time={sample_time:.1f}s, Projected={projected/60:.1f}min")

    total_time = time.time() - start_time
    projected_400 = (total_time / n_samples) * 400 / 60

    # Calculate score
    gt_dataset = [{'sources': s['sources'], 'sample_metadata': s['sample_metadata']}
                  for s in test_data]
    pred_dataset = [{'sources': p['candidates']} for p in predictions]

    score = score_submission(
        gt_dataset, pred_dataset, N_max=3, lambda_=0.3, tau=0.2,
        scale_factors=(2.0, 1.0, 2.0), forward_loss="rmse",
        solver_kwargs={"Lx": 2.0, "Ly": 1.0, "nx": 100, "ny": 50}
    )

    # Calculate metrics
    mean_rmse = np.mean(sample_rmses)
    median_rmse = np.median(sample_rmses)

    # Count n_valid candidates
    n_valid_list = [len(p['candidates']) for p in predictions]
    mean_n_valid = np.mean(n_valid_list)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Score: {score:.4f}")
        print(f"  Mean RMSE: {mean_rmse:.4f}")
        print(f"  Median RMSE: {median_rmse:.4f}")
        print(f"  Mean N_valid: {mean_n_valid:.2f}")
        print(f"  Perturbed selections: {perturbed_selections}/{n_samples} ({100*perturbed_selections/n_samples:.1f}%)")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.2f}min)")
        print(f"  Projected 400 samples: {projected_400:.1f}min")
        print(f"  Budget status: {'IN BUDGET' if projected_400 <= 60 else 'OVER BUDGET'}")
        print(f"{'='*60}")

    return {
        'score': score,
        'mean_rmse': mean_rmse,
        'median_rmse': median_rmse,
        'mean_n_valid': mean_n_valid,
        'perturbed_selections': perturbed_selections,
        'total_time_sec': total_time,
        'total_time_min': total_time / 60,
        'projected_400_min': projected_400,
        'in_budget': projected_400 <= 60,
        'predictions': predictions,
    }


def run_with_mlflow(
    timestep_fraction: float = 0.40,
    sigma0_1src: float = 0.18,
    sigma0_2src: float = 0.22,
    max_fevals_1src: int = 20,
    max_fevals_2src: int = 36,
    refine_maxiter: int = 8,
    perturb_top_n: int = 1,
    n_perturbations: int = 2,
    perturbation_scale: float = 0.05,
    perturb_nm_iters: int = 3,
    enable_perturbation: bool = True,
    run_name: str = "perturbation_40pct_temporal",
):
    """Run experiment with MLflow logging."""

    mlflow.set_tracking_uri(f"file://{os.path.join(project_root, 'mlruns')}")
    mlflow.set_experiment("perturbation_with_40pct_temporal")

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("optimizer", "PerturbationTemporalOptimizer")
        mlflow.log_param("timestep_fraction", timestep_fraction)
        mlflow.log_param("sigma0_1src", sigma0_1src)
        mlflow.log_param("sigma0_2src", sigma0_2src)
        mlflow.log_param("max_fevals_1src", max_fevals_1src)
        mlflow.log_param("max_fevals_2src", max_fevals_2src)
        mlflow.log_param("refine_maxiter", refine_maxiter)
        mlflow.log_param("enable_perturbation", enable_perturbation)
        mlflow.log_param("perturb_top_n", perturb_top_n)
        mlflow.log_param("n_perturbations", n_perturbations)
        mlflow.log_param("perturbation_scale", perturbation_scale)
        mlflow.log_param("perturb_nm_iters", perturb_nm_iters)
        mlflow.log_param("platform", "wsl")

        # Run experiment
        results = run_experiment(
            timestep_fraction=timestep_fraction,
            sigma0_1src=sigma0_1src,
            sigma0_2src=sigma0_2src,
            max_fevals_1src=max_fevals_1src,
            max_fevals_2src=max_fevals_2src,
            refine_maxiter=refine_maxiter,
            perturb_top_n=perturb_top_n,
            n_perturbations=n_perturbations,
            perturbation_scale=perturbation_scale,
            perturb_nm_iters=perturb_nm_iters,
            enable_perturbation=enable_perturbation,
            run_name=run_name,
        )

        # Log metrics
        mlflow.log_metric("submission_score", results['score'])
        mlflow.log_metric("mean_rmse", results['mean_rmse'])
        mlflow.log_metric("median_rmse", results['median_rmse'])
        mlflow.log_metric("mean_n_valid", results['mean_n_valid'])
        mlflow.log_metric("perturbed_selections", results['perturbed_selections'])
        mlflow.log_metric("total_time_min", results['total_time_min'])
        mlflow.log_metric("projected_400_samples_min", results['projected_400_min'])
        mlflow.log_metric("in_budget", 1 if results['in_budget'] else 0)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep_fraction', type=float, default=0.40)
    parser.add_argument('--sigma0_1src', type=float, default=0.18)
    parser.add_argument('--sigma0_2src', type=float, default=0.22)
    parser.add_argument('--max_fevals_1src', type=int, default=20)
    parser.add_argument('--max_fevals_2src', type=int, default=36)
    parser.add_argument('--refine_maxiter', type=int, default=8)
    parser.add_argument('--perturb_top_n', type=int, default=1)
    parser.add_argument('--n_perturbations', type=int, default=2)
    parser.add_argument('--perturbation_scale', type=float, default=0.05)
    parser.add_argument('--perturb_nm_iters', type=int, default=3)
    parser.add_argument('--no_perturbation', action='store_true')
    parser.add_argument('--no_mlflow', action='store_true')
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    enable_perturbation = not args.no_perturbation

    if args.no_mlflow:
        results = run_experiment(
            timestep_fraction=args.timestep_fraction,
            sigma0_1src=args.sigma0_1src,
            sigma0_2src=args.sigma0_2src,
            max_fevals_1src=args.max_fevals_1src,
            max_fevals_2src=args.max_fevals_2src,
            refine_maxiter=args.refine_maxiter,
            perturb_top_n=args.perturb_top_n,
            n_perturbations=args.n_perturbations,
            perturbation_scale=args.perturbation_scale,
            perturb_nm_iters=args.perturb_nm_iters,
            enable_perturbation=enable_perturbation,
            run_name=run_name,
        )
    else:
        results = run_with_mlflow(
            timestep_fraction=args.timestep_fraction,
            sigma0_1src=args.sigma0_1src,
            sigma0_2src=args.sigma0_2src,
            max_fevals_1src=args.max_fevals_1src,
            max_fevals_2src=args.max_fevals_2src,
            refine_maxiter=args.refine_maxiter,
            perturb_top_n=args.perturb_top_n,
            n_perturbations=args.n_perturbations,
            perturbation_scale=args.perturbation_scale,
            perturb_nm_iters=args.perturb_nm_iters,
            enable_perturbation=enable_perturbation,
            run_name=run_name,
        )

    # Print final summary
    print(f"\nFINAL: Score={results['score']:.4f} @ {results['projected_400_min']:.1f}min projected")
