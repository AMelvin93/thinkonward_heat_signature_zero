#!/usr/bin/env python
"""Run A12b: Perturbation-Based Diversity experiment."""

import sys
import os
import time
import pickle
import argparse
import platform
from pathlib import Path
from datetime import datetime
from copy import deepcopy

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from joblib import Parallel, delayed

from experiments.perturbation_diversity.optimizer import PerturbationDiversityOptimizer, extract_enhanced_features, N_MAX

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


def process_sample(sample, meta, config, history_1src, history_2src):
    optimizer = PerturbationDiversityOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        use_triangulation=True,
        n_candidates=3,
        k_similar=1,
        perturbation_scale=config.get('perturbation_scale', 0.25),
    )

    q_range = tuple(meta['q_range'])
    start = time.time()
    candidates, best_rmse, results, features, best_params, n_transferred = optimizer.estimate_sources(
        sample, meta, q_range=q_range,
        history_1src=history_1src, history_2src=history_2src, verbose=False
    )
    elapsed = time.time() - start

    candidate_rmses = [r.rmse for r in results]
    score = calculate_sample_score(candidate_rmses)

    init_types = [r.init_type for r in results]

    return {
        'sample_id': sample['sample_id'],
        'n_sources': sample['n_sources'],
        'n_candidates': len(candidates),
        'rmses': candidate_rmses,
        'best_rmse': best_rmse,
        'score': score,
        'time': elapsed,
        'n_evals': sum(r.n_evals for r in results),
        'n_transferred': n_transferred,
        'init_types': init_types,
        'features': features,
        'best_params': best_params,
    }


def process_batch(samples, meta, config, history_1src, history_2src, n_workers):
    h1, h2 = deepcopy(history_1src), deepcopy(history_2src)
    return Parallel(n_jobs=n_workers, verbose=0)(
        delayed(process_sample)(s, meta, config, h1, h2) for s in samples
    )


def main():
    parser = argparse.ArgumentParser(description='Run Perturbation Diversity Optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-fevals-1src', type=int, default=12)
    parser.add_argument('--max-fevals-2src', type=int, default=23)
    parser.add_argument('--perturbation-scale', type=float, default=0.25)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'perturbation_scale': args.perturbation_scale,
    }

    n_workers = args.workers
    actual_workers = os.cpu_count() if n_workers == -1 else n_workers
    is_g4dn = (n_workers == G4DN_WORKERS)

    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(samples)

    n_batches = (n_samples + args.batch_size - 1) // args.batch_size

    print("=" * 70)
    print("A12b: PERTURBATION-BASED DIVERSITY")
    print("=" * 70)
    print(f"Platform: {detect_platform().upper()}")
    print(f"Samples: {n_samples} ({n_1src} 1-src, {n_2src} 2-src)")
    print(f"Workers: {actual_workers}" + (" (G4dn)" if is_g4dn else ""))
    print(f"Config: {config['max_fevals_1src']}/{config['max_fevals_2src']} fevals")
    print(f"Perturbation scale: {config['perturbation_scale']}")
    print()
    print("KEY INNOVATION: Post-optimization perturbation for diversity")
    print("  - Full CMA-ES on best init (accuracy)")
    print("  - Perturb best solution for diversity")
    print("  - Only 2 extra simulations for perturbations")
    print("=" * 70)

    start_total = time.time()
    history_1src, history_2src, all_results = [], [], []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, n_samples)
        batch_samples = samples[batch_start:batch_end]

        print(f"\nBatch {batch_idx + 1}/{n_batches}: samples {batch_start}-{batch_end-1}")

        batch_results = process_batch(batch_samples, meta, config, history_1src, history_2src, n_workers)

        for r in batch_results:
            if r['n_sources'] == 1:
                history_1src.append((r['features'], r['best_params']))
            else:
                history_2src.append((r['features'], r['best_params']))

        all_results.extend(batch_results)
        batch_rmses = [r['best_rmse'] for r in batch_results]
        batch_n_cands = [r['n_candidates'] for r in batch_results]
        print(f"  -> RMSE: {np.mean(batch_rmses):.4f}, Avg candidates: {np.mean(batch_n_cands):.1f}, Time: {np.mean([r['time'] for r in batch_results]):.2f}s")

    total_time = time.time() - start_total

    all_rmses = [r['best_rmse'] for r in all_results]
    all_scores = [r['score'] for r in all_results]
    all_n_cands = [r['n_candidates'] for r in all_results]
    final_score = np.mean(all_scores)
    proj_400 = (total_time / n_samples) * COMPETITION_SAMPLES / 60

    rmse_by_src = {}
    cands_by_src = {}
    for r in all_results:
        rmse_by_src.setdefault(r['n_sources'], []).append(r['best_rmse'])
        cands_by_src.setdefault(r['n_sources'], []).append(r['n_candidates'])

    init_type_counts = {}
    for r in all_results:
        for init_type in r['init_types']:
            init_type_counts[init_type] = init_type_counts.get(init_type, 0) + 1

    if is_g4dn and not args.max_samples:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("heat-signature-zero")
        run_name = f"perturbation_diversity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("submission_score", final_score)
            mlflow.log_metric("rmse", np.mean(all_rmses))
            mlflow.log_metric("projected_400_samples_min", proj_400)
            mlflow.log_metric("rmse_1src", np.mean(rmse_by_src.get(1, [0])))
            mlflow.log_metric("rmse_2src", np.mean(rmse_by_src.get(2, [0])))
            mlflow.log_metric("avg_candidates", np.mean(all_n_cands))
            mlflow.log_param("optimizer", "PerturbationDiversityOptimizer")
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
        cands = cands_by_src[ns]
        print(f"  {ns}-source: RMSE={np.mean(rmses):.4f}, Candidates={np.mean(cands):.1f} (n={len(rmses)})")
    print()
    print("Init type distribution (of final candidates):")
    total_inits = sum(init_type_counts.values())
    for it, cnt in sorted(init_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {it}: {cnt} ({100*cnt/total_inits:.1f}%)")
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
