"""
Run higher_temporal_45pct experiment.

Tests 45% temporal fidelity with the perturbed_extended_polish optimizer.
"""

import os
import sys
import time
import pickle
import mlflow
from datetime import datetime

import numpy as np

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Import the perturbed_extended_polish optimizer
sys.path.insert(0, os.path.join(_project_root, 'experiments', 'perturbed_extended_polish'))
from optimizer import PerturbedLocalRestartOptimizer

# Import utils
from utils import score_submission


def run_experiment(
    timestep_fraction: float = 0.45,
    sigma0_1src: float = 0.18,
    sigma0_2src: float = 0.22,
    refine_maxiter: int = 8,
    enable_perturbation: bool = True,
    perturb_top_n: int = 1,
    n_perturbations: int = 2,
    perturb_nm_iters: int = 3,
    n_workers: int = -1,
):
    """Run experiment with given configuration."""
    
    # Load test data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    meta = data['meta']
    q_range = (0.5, 2.0)
    
    n_samples = len(samples)
    print(f"Running on {n_samples} samples")
    print(f"Configuration:")
    print(f"  timestep_fraction: {timestep_fraction}")
    print(f"  sigma: {sigma0_1src}/{sigma0_2src}")
    print(f"  refine_maxiter: {refine_maxiter}")
    print(f"  perturbation: {enable_perturbation} (top_n={perturb_top_n}, n={n_perturbations}, iters={perturb_nm_iters})")
    
    # Create optimizer
    optimizer = PerturbedLocalRestartOptimizer(
        Lx=2.0,
        Ly=1.0,
        nx_fine=100,
        ny_fine=50,
        nx_coarse=50,
        ny_coarse=25,
        max_fevals_1src=20,
        max_fevals_2src=36,
        sigma0_1src=sigma0_1src,
        sigma0_2src=sigma0_2src,
        use_triangulation=True,
        n_candidates=3,
        candidate_pool_size=10,
        refine_maxiter=refine_maxiter,
        refine_top_n=2,
        rmse_threshold_1src=0.4,
        rmse_threshold_2src=0.5,
        timestep_fraction=timestep_fraction,
        enable_perturbation=enable_perturbation,
        perturb_top_n=perturb_top_n,
        n_perturbations=n_perturbations,
        perturbation_scale=0.05,
        perturb_nm_iters=perturb_nm_iters,
    )
    
    # Process samples
    from joblib import Parallel, delayed
    
    if n_workers == -1:
        n_workers = os.cpu_count()
    
    def process_sample(idx, sample):
        try:
            candidate_sources, best_rmse, results, n_sims = optimizer.estimate_sources(
                sample, meta, q_range=q_range, verbose=False
            )
            return {
                'sample_idx': idx,
                'candidate_sources': candidate_sources,
                'best_rmse': best_rmse,
                'n_sources': sample['n_sources'],
                'n_sims': n_sims,
            }
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            return {
                'sample_idx': idx,
                'candidate_sources': [],
                'best_rmse': float('inf'),
                'n_sources': sample.get('n_sources', 1),
                'n_sims': 0,
            }
    
    start_time = time.time()
    
    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_sample)(idx, sample) for idx, sample in enumerate(samples)
    )
    
    total_time = time.time() - start_time
    total_time_min = total_time / 60
    projected_400_min = (total_time / n_samples) * 400 / 60
    
    print(f"\nTotal time: {total_time_min:.1f} min")
    print(f"Projected time for 400 samples: {projected_400_min:.1f} min")
    
    # Calculate score using simplified formula (without ground truth)
    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], len(r['candidate_sources'])) for r in results]
    submission_score = np.mean(sample_scores) if sample_scores else 0.0
    
    print(f"\n=== RESULTS ===")
    print(f"Submission Score: {submission_score:.4f}")
    print(f"Total Time: {total_time_min:.1f} min")
    print(f"Projected 400 samples: {projected_400_min:.1f} min")
    print(f"In Budget: {'YES' if projected_400_min <= 60 else 'NO'}")
    
    # Calculate RMSE stats
    rmses_1src = [r['best_rmse'] for r in results if r['n_sources'] == 1 and r['best_rmse'] < float('inf')]
    rmses_2src = [r['best_rmse'] for r in results if r['n_sources'] == 2 and r['best_rmse'] < float('inf')]
    
    if rmses_1src:
        print(f"1-source mean RMSE: {sum(rmses_1src)/len(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"2-source mean RMSE: {sum(rmses_2src)/len(rmses_2src):.4f} (n={len(rmses_2src)})")
    
    return {
        'score': submission_score,
        'time_min': total_time_min,
        'projected_400_min': projected_400_min,
        'in_budget': projected_400_min <= 60,
        'rmse_1src_mean': sum(rmses_1src)/len(rmses_1src) if rmses_1src else None,
        'rmse_2src_mean': sum(rmses_2src)/len(rmses_2src) if rmses_2src else None,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep_fraction', type=float, default=0.45)
    parser.add_argument('--sigma0_1src', type=float, default=0.18)
    parser.add_argument('--sigma0_2src', type=float, default=0.22)
    parser.add_argument('--refine_maxiter', type=int, default=8)
    parser.add_argument('--enable_perturbation', type=bool, default=True)
    parser.add_argument('--perturb_top_n', type=int, default=1)
    parser.add_argument('--n_perturbations', type=int, default=2)
    parser.add_argument('--perturb_nm_iters', type=int, default=3)
    parser.add_argument('--n_workers', type=int, default=-1)
    parser.add_argument('--mlflow', action='store_true', help='Log to MLflow')
    
    args = parser.parse_args()
    
    if args.mlflow:
        mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
        mlflow.set_experiment('higher_temporal_45pct')
        
        with mlflow.start_run():
            mlflow.log_params({
                'timestep_fraction': args.timestep_fraction,
                'sigma0_1src': args.sigma0_1src,
                'sigma0_2src': args.sigma0_2src,
                'refine_maxiter': args.refine_maxiter,
                'enable_perturbation': args.enable_perturbation,
                'perturb_top_n': args.perturb_top_n,
                'n_perturbations': args.n_perturbations,
                'perturb_nm_iters': args.perturb_nm_iters,
                'platform': 'wsl',
            })
            
            results = run_experiment(
                timestep_fraction=args.timestep_fraction,
                sigma0_1src=args.sigma0_1src,
                sigma0_2src=args.sigma0_2src,
                refine_maxiter=args.refine_maxiter,
                enable_perturbation=args.enable_perturbation,
                perturb_top_n=args.perturb_top_n,
                n_perturbations=args.n_perturbations,
                perturb_nm_iters=args.perturb_nm_iters,
                n_workers=args.n_workers,
            )
            
            mlflow.log_metrics({
                'submission_score': results['score'],
                'total_time_min': results['time_min'],
                'projected_400_samples_min': results['projected_400_min'],
            })
    else:
        results = run_experiment(
            timestep_fraction=args.timestep_fraction,
            sigma0_1src=args.sigma0_1src,
            sigma0_2src=args.sigma0_2src,
            refine_maxiter=args.refine_maxiter,
            enable_perturbation=args.enable_perturbation,
            perturb_top_n=args.perturb_top_n,
            n_perturbations=args.n_perturbations,
            perturb_nm_iters=args.perturb_nm_iters,
            n_workers=args.n_workers,
        )
