#!/usr/bin/env python
"""Run A13: Temperature-Weighted Centroid experiment."""

import sys
import os
import time
import pickle
import argparse
import platform
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

from experiments.weighted_centroid.optimizer import WeightedCentroidOptimizer, N_MAX

G4DN_WORKERS = 7
COMPETITION_SAMPLES = 400
LAMBDA = 0.3


def detect_platform():
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    return "wsl"
        except:
            pass
        return "linux"
    return system


def calculate_sample_score(rmses, lambda_=LAMBDA, n_max=N_MAX, max_rmse=1.0):
    valid_rmses = [r for r in rmses if r <= max_rmse]
    n_valid = len(valid_rmses)
    if n_valid == 0:
        return 0.0
    accuracy_sum = sum(1.0 / (1.0 + r) for r in valid_rmses)
    return accuracy_sum / n_valid + lambda_ * (n_valid / n_max)


def process_sample(sample, meta, config):
    optimizer = WeightedCentroidOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        use_triangulation=config.get('use_triangulation', True),
        use_gradient_refinement=config.get('use_gradient_refinement', True),
        n_candidates=3,
    )

    q_range = tuple(meta['q_range'])
    start = time.time()
    candidates, best_rmse, results, n_evals = optimizer.estimate_sources(
        sample, meta, q_range=q_range, verbose=False
    )
    elapsed = time.time() - start

    candidate_rmses = [r.rmse for r in results]
    score = calculate_sample_score(candidate_rmses)

    init_type = results[0].init_type if results else 'unknown'

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'n_candidates': len(candidates),
        'rmses': candidate_rmses,
        'best_rmse': best_rmse,
        'score': score,
        'time': elapsed,
        'n_evals': n_evals,
        'init_type': init_type,
    }


def main():
    parser = argparse.ArgumentParser(description='Run Weighted Centroid Optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=12)
    parser.add_argument('--max-fevals-2src', type=int, default=23)
    parser.add_argument('--no-triangulation', action='store_true')
    parser.add_argument('--no-gradient-refinement', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'use_triangulation': not args.no_triangulation,
        'use_gradient_refinement': not args.no_gradient_refinement,
    }

    n_workers = args.workers
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers
    is_g4dn = (n_workers == G4DN_WORKERS)

    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(samples)

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print("=" * 70)
    print("A13: TEMPERATURE-WEIGHTED CENTROID OPTIMIZER")
    print("=" * 70)
    print(f"Platform: {detect_platform().upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-src, {n_2src} 2-src)")
    print(f"Workers: {actual_workers}" + (" (G4dn)" if is_g4dn else ""))
    print(f"Config: {config['max_fevals_1src']}/{config['max_fevals_2src']} fevals")
    print(f"Triangulation: {config['use_triangulation']}")
    print(f"Gradient refinement: {config['use_gradient_refinement']}")
    print()
    print("KEY INNOVATION: Physics-based position estimation")
    print("  - Temperature-weighted centroid of sensors")
    print("  - Gradient refinement (move toward source)")
    print("  - Analytical intensity computation")
    print("  - Minimal CMA-ES polish")
    print("=" * 70)

    start_total = time.time()

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_sample)(s, meta, config) for s in samples
    )

    total_time = time.time() - start_total

    all_rmses = [r['best_rmse'] for r in results]
    all_scores = [r['score'] for r in results]
    all_n_cands = [r['n_candidates'] for r in results]
    final_score = np.mean(all_scores)
    proj_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    rmse_by_src = {}
    init_type_counts = {}
    for r in results:
        rmse_by_src.setdefault(r['n_sources'], []).append(r['best_rmse'])
        init_type_counts[r['init_type']] = init_type_counts.get(r['init_type'], 0) + 1

    if is_g4dn and not args.max_samples:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")
        run_name = f"weighted_centroid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", np.mean(all_rmses))
            mlflow.log_metric("projected_400_samples_min", proj_400)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_src.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_src.get(2, [0])))
            mlflow.log_metric("avg_candidates", np.mean(all_n_cands))
            mlflow.log_param("optimizer", "WeightedCentroidOptimizer")
            mlflow.log_param("config", f"{config['max_fevals_1src']}/{config['max_fevals_2src']}")
        print(f"\n[MLflow] Logged: {run_name}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"RMSE:             {np.mean(all_rmses):.6f} +/- {np.std(all_rmses):.6f}")
    print(f"Avg Candidates:   {np.mean(all_n_cands):.2f}")
    print(f"Submission Score: {final_score:.4f}")
    print(f"Projected (400):  {proj_400:.1f} min")
    print()
    for ns in sorted(rmse_by_src.keys()):
        rmses = rmse_by_src[ns]
        print(f"  {ns}-source: RMSE={np.mean(rmses):.4f} (n={len(rmses)})")
    print()
    print("Best init types:")
    for it, cnt in sorted(init_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {it}: {cnt} ({100*cnt/n_samples:.1f}%)")
    print()

    baseline = 1.0224
    print(f"Baseline: {baseline:.4f} @ 56.5 min")
    print(f"This run: {final_score:.4f} @ {proj_400:.1f} min")

    if final_score >= 1.15 and proj_400 < 60:
        print("[TARGET HIT!]")
    elif final_score > baseline and proj_400 < 60:
        print(f"[IMPROVED] +{final_score - baseline:.4f}")
    elif proj_400 >= 60:
        print(f"[OVER BUDGET] by {proj_400 - 60:.1f} min")
    else:
        print("[NO IMPROVEMENT]")
    print("=" * 70)

    return final_score, np.mean(all_rmses)


if __name__ == "__main__":
    main()
