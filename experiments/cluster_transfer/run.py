#!/usr/bin/env python3
"""
Run script for Cluster Transfer optimizer.

Workflow:
1. Extract features from ALL samples
2. Cluster samples by n_sources (1-src and 2-src separately)
3. Identify cluster representatives (closest to centroid)
4. Fully optimize representatives
5. Use representative solutions to warm-start other samples

Success criteria: Same score with 20%+ time reduction
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import ClusterTransferOptimizer, extract_features


def process_single_sample(args):
    """Process a single sample with optional transfer initialization."""
    idx, sample, meta, config, transfer_init, is_representative = args

    optimizer = ClusterTransferOptimizer(
        max_fevals_1src=config['max_fevals_1src'],
        max_fevals_2src=config['max_fevals_2src'],
        max_fevals_transfer=config['max_fevals_transfer'],
        sigma0_1src=config.get('sigma0_1src', 0.15),
        sigma0_2src=config.get('sigma0_2src', 0.20),
        sigma0_transfer=config.get('sigma0_transfer', 0.08),
        use_triangulation=config.get('use_triangulation', True),
        n_candidates=config.get('n_candidates', 3),
        candidate_pool_size=config.get('candidate_pool_size', 10),
        refine_maxiter=config.get('refine_maxiter', 3),
        refine_top_n=config.get('refine_top_n', 2),
        rmse_threshold_1src=config.get('rmse_threshold_1src', 0.35),
        rmse_threshold_2src=config.get('rmse_threshold_2src', 0.45),
    )

    start = time.time()
    try:
        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, q_range=(0.5, 2.0),
            transfer_init=transfer_init,
            is_representative=is_representative,
            verbose=False
        )
        elapsed = time.time() - start
        init_types = [r.init_type for r in results]

        # Extract best position for potential transfer
        if results:
            best_result = min(results, key=lambda r: r.rmse)
            n_sources = sample['n_sources']
            if n_sources == 1:
                best_positions = best_result.params[:2]
            else:
                best_positions = np.array([
                    best_result.params[0], best_result.params[1],
                    best_result.params[3], best_result.params[4]
                ])
        else:
            best_positions = None

        return {
            'idx': idx, 'candidates': candidates, 'best_rmse': best_rmse,
            'n_sources': sample['n_sources'], 'n_candidates': len(candidates),
            'n_sims': n_sims, 'elapsed': elapsed, 'init_types': init_types,
            'is_representative': is_representative,
            'best_positions': best_positions,
            'success': True,
        }
    except Exception as e:
        import traceback
        return {
            'idx': idx, 'candidates': [], 'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0), 'n_candidates': 0,
            'n_sims': 0, 'elapsed': time.time() - start, 'init_types': [],
            'is_representative': is_representative,
            'best_positions': None,
            'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
        }


def cluster_samples(samples, n_clusters_1src=4, n_clusters_2src=6):
    """
    Cluster samples by features and n_sources.

    Returns:
        cluster_assignments: dict mapping idx -> cluster_id
        representatives: dict mapping cluster_id -> sample_idx (closest to centroid)
    """
    # Separate by n_sources
    indices_1src = [i for i, s in enumerate(samples) if s['n_sources'] == 1]
    indices_2src = [i for i, s in enumerate(samples) if s['n_sources'] == 2]

    # Extract features
    features_1src = np.array([extract_features(samples[i]) for i in indices_1src])
    features_2src = np.array([extract_features(samples[i]) for i in indices_2src])

    cluster_assignments = {}
    representatives = {}

    # Cluster 1-source samples
    if len(indices_1src) > n_clusters_1src:
        scaler_1src = StandardScaler()
        features_1src_scaled = scaler_1src.fit_transform(features_1src)

        kmeans_1src = KMeans(n_clusters=n_clusters_1src, random_state=42, n_init=10)
        labels_1src = kmeans_1src.fit_predict(features_1src_scaled)

        for i, label in enumerate(labels_1src):
            cluster_id = f'1src_{label}'
            cluster_assignments[indices_1src[i]] = cluster_id

        # Find representatives (closest to centroid)
        for cluster_label in range(n_clusters_1src):
            cluster_mask = labels_1src == cluster_label
            cluster_indices = [indices_1src[i] for i, m in enumerate(cluster_mask) if m]
            cluster_features = features_1src_scaled[cluster_mask]

            if len(cluster_indices) > 0:
                centroid = kmeans_1src.cluster_centers_[cluster_label]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                rep_idx = cluster_indices[np.argmin(distances)]
                representatives[f'1src_{cluster_label}'] = rep_idx
    else:
        # All samples are representatives if too few
        for i in indices_1src:
            cluster_id = f'1src_{i}'
            cluster_assignments[i] = cluster_id
            representatives[cluster_id] = i

    # Cluster 2-source samples
    if len(indices_2src) > n_clusters_2src:
        scaler_2src = StandardScaler()
        features_2src_scaled = scaler_2src.fit_transform(features_2src)

        kmeans_2src = KMeans(n_clusters=n_clusters_2src, random_state=42, n_init=10)
        labels_2src = kmeans_2src.fit_predict(features_2src_scaled)

        for i, label in enumerate(labels_2src):
            cluster_id = f'2src_{label}'
            cluster_assignments[indices_2src[i]] = cluster_id

        # Find representatives
        for cluster_label in range(n_clusters_2src):
            cluster_mask = labels_2src == cluster_label
            cluster_indices = [indices_2src[i] for i, m in enumerate(cluster_mask) if m]
            cluster_features = features_2src_scaled[cluster_mask]

            if len(cluster_indices) > 0:
                centroid = kmeans_2src.cluster_centers_[cluster_label]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                rep_idx = cluster_indices[np.argmin(distances)]
                representatives[f'2src_{cluster_label}'] = rep_idx
    else:
        for i in indices_2src:
            cluster_id = f'2src_{i}'
            cluster_assignments[i] = cluster_id
            representatives[cluster_id] = i

    return cluster_assignments, representatives


def main():
    parser = argparse.ArgumentParser(description='Run Cluster Transfer optimizer')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max-fevals-1src', type=int, default=20)
    parser.add_argument('--max-fevals-2src', type=int, default=36)
    parser.add_argument('--max-fevals-transfer', type=int, default=10)
    parser.add_argument('--sigma0-1src', type=float, default=0.15)
    parser.add_argument('--sigma0-2src', type=float, default=0.20)
    parser.add_argument('--sigma0-transfer', type=float, default=0.08)
    parser.add_argument('--n-clusters-1src', type=int, default=4)
    parser.add_argument('--n-clusters-2src', type=int, default=6)
    parser.add_argument('--threshold-1src', type=float, default=0.35)
    parser.add_argument('--threshold-2src', type=float, default=0.45)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    np.random.seed(args.seed)
    if args.shuffle:
        perm = np.random.permutation(len(samples))
        samples = [samples[i] for i in perm]
        original_indices = perm
    else:
        original_indices = np.arange(len(samples))

    if args.max_samples:
        samples = samples[:args.max_samples]
        original_indices = original_indices[:args.max_samples]

    n_samples = len(samples)

    print(f"\nCluster Transfer Optimizer")
    print(f"=" * 70)
    print(f"Samples: {n_samples}, Workers: {args.workers}")
    print(f"Fevals: full={args.max_fevals_1src}/{args.max_fevals_2src}, transfer={args.max_fevals_transfer}")
    print(f"Sigma: full={args.sigma0_1src}/{args.sigma0_2src}, transfer={args.sigma0_transfer}")
    print(f"Clusters: 1-src={args.n_clusters_1src}, 2-src={args.n_clusters_2src}")
    print(f"Fallback threshold: 1-src={args.threshold_1src}, 2-src={args.threshold_2src}")
    print(f"=" * 70)

    config = {
        'max_fevals_1src': args.max_fevals_1src,
        'max_fevals_2src': args.max_fevals_2src,
        'max_fevals_transfer': args.max_fevals_transfer,
        'sigma0_1src': args.sigma0_1src,
        'sigma0_2src': args.sigma0_2src,
        'sigma0_transfer': args.sigma0_transfer,
        'use_triangulation': True,
        'n_candidates': 3,
        'candidate_pool_size': 10,
        'refine_maxiter': 3,
        'refine_top_n': 2,
        'rmse_threshold_1src': args.threshold_1src,
        'rmse_threshold_2src': args.threshold_2src,
    }

    start_time = time.time()

    # Step 1: Cluster samples
    print("\n[Phase 1] Clustering samples...")
    cluster_assignments, representatives = cluster_samples(
        samples, args.n_clusters_1src, args.n_clusters_2src
    )

    rep_indices = set(representatives.values())
    n_reps = len(rep_indices)
    print(f"  Created {len(representatives)} clusters, {n_reps} representatives")

    # Step 2: Process representatives first
    print(f"\n[Phase 2] Optimizing {n_reps} representatives...")
    rep_results = {}

    rep_work_items = [
        (i, samples[i], meta, config, None, True)
        for i in rep_indices
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in rep_work_items}
        for future in as_completed(futures):
            result = future.result()
            rep_results[result['idx']] = result
            status = "OK" if result['success'] else "ERR"
            print(f"  Rep sample {result['idx']:3d}: {result['n_sources']}-src "
                  f"RMSE={result['best_rmse']:.4f} time={result['elapsed']:.1f}s [{status}]")

    # Step 3: Process remaining samples with transfer
    print(f"\n[Phase 3] Optimizing {n_samples - n_reps} samples with transfer...")
    all_results = list(rep_results.values())

    # Build transfer initialization map
    transfer_map = {}
    for sample_idx, cluster_id in cluster_assignments.items():
        if sample_idx not in rep_indices:
            rep_idx = representatives[cluster_id]
            if rep_idx in rep_results and rep_results[rep_idx]['best_positions'] is not None:
                transfer_map[sample_idx] = rep_results[rep_idx]['best_positions']

    # Process non-representative samples
    non_rep_work_items = [
        (i, samples[i], meta, config, transfer_map.get(i), False)
        for i in range(n_samples)
        if i not in rep_indices
    ]

    transfer_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in non_rep_work_items}
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            status = "OK" if result['success'] else "ERR"
            has_transfer = result['idx'] in transfer_map
            transfer_flag = "[T]" if has_transfer else "   "
            if has_transfer and 'transfer' in str(result['init_types']):
                transfer_count += 1
            print(f"[{len(all_results):3d}/{n_samples}] Sample {result['idx']:3d}: "
                  f"{result['n_sources']}-src RMSE={result['best_rmse']:.4f} "
                  f"sims={result['n_sims']:3d} time={result['elapsed']:.1f}s [{status}]{transfer_flag}")

    total_time = time.time() - start_time

    def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
        if n_candidates == 0:
            return 0.0
        return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in all_results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in all_results if r['success']]
    rmse_mean = np.mean(rmses)
    rmses_1src = [r['best_rmse'] for r in all_results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in all_results if r['success'] and r['n_sources'] == 2]
    projected_400 = (total_time / n_samples) * 400 / 60

    # Time breakdown
    rep_time = sum(r['elapsed'] for r in rep_results.values())
    transfer_time = sum(r['elapsed'] for r in all_results if not r.get('is_representative', False))

    print(f"\n{'='*70}")
    print(f"RESULTS: Cluster Transfer Optimizer")
    print(f"{'='*70}")
    print(f"RMSE:             {rmse_mean:.6f}")
    print(f"Submission Score: {score:.4f}")
    print(f"Total Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Projected (400):  {projected_400:.1f} min")
    print()
    if rmses_1src:
        print(f"  1-source: RMSE={np.mean(rmses_1src):.4f} (n={len(rmses_1src)})")
    if rmses_2src:
        print(f"  2-source: RMSE={np.mean(rmses_2src):.4f} (n={len(rmses_2src)})")
    print()
    print(f"Clustering stats:")
    print(f"  Representatives: {n_reps} samples (full optimization)")
    print(f"  Transfer samples: {n_samples - n_reps} samples (reduced optimization)")
    print(f"  Transfer init used: {transfer_count} samples")
    print()
    print(f"Time breakdown:")
    print(f"  Representative phase: {rep_time:.1f}s")
    print(f"  Transfer phase: {transfer_time:.1f}s")
    print()
    print(f"Target: Same score with 20%+ time reduction")
    print(f"Baseline (robust_fallback): 1.1247 @ 57.2 min")
    print(f"This cluster run:           {score:.4f} @ {projected_400:.1f} min")
    print(f"Delta:                      {score - 1.1247:+.4f} score, {projected_400 - 57.2:+.1f} min")
    print()

    baseline_time = 57.2
    time_reduction = (baseline_time - projected_400) / baseline_time * 100

    if score >= 1.1247 * 0.99 and time_reduction >= 20:
        print(f"STATUS: SUCCESS - Score maintained, {time_reduction:.1f}% time reduction!")
    elif score >= 1.1247 * 0.99 and projected_400 <= 60:
        print(f"STATUS: PARTIAL - Score OK, but only {time_reduction:.1f}% time reduction (need 20%+)")
    elif projected_400 <= 60:
        print(f"STATUS: PARTIAL - Time OK but score dropped")
    else:
        print(f"STATUS: FAILED - Over budget by {projected_400 - 60:.1f} min")

    print(f"{'='*70}\n")

    # Print outliers
    high_rmse = [r for r in all_results if r['best_rmse'] > 0.5]
    if high_rmse:
        print(f"\nHigh RMSE samples ({len(high_rmse)}):")
        for r in sorted(high_rmse, key=lambda x: -x['best_rmse'])[:10]:
            rep_flag = "[REP]" if r.get('is_representative') else "     "
            print(f"  Sample {r['idx']}: {r['n_sources']}-src RMSE={r['best_rmse']:.4f} "
                  f"sims={r['n_sims']}{rep_flag}")

    return score, projected_400


if __name__ == '__main__':
    main()
