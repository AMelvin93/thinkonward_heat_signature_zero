#!/usr/bin/env python
"""
Compare full optimization with triangulation vs hottest-sensor initialization.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import time
import numpy as np
from src.hybrid_optimizer import HybridOptimizer


def main():
    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    q_range = tuple(meta['q_range'])

    n_samples = min(20, len(samples))
    max_iter = 5  # Production uses 3, but let's use 5 for fair comparison

    print("Full optimization comparison: Triangulation vs Hottest-Sensor")
    print("=" * 70)
    print(f"Testing {n_samples} samples with max_iter={max_iter}")
    print()

    # Test with triangulation
    print("Running with triangulation...")
    optimizer_triang = HybridOptimizer(
        n_smart_inits=1,
        n_random_inits=0,
        use_triangulation=True,
    )

    triang_rmses = []
    triang_times = []
    for i, sample in enumerate(samples[:n_samples]):
        start = time.time()
        sources, rmse, _ = optimizer_triang.estimate_sources(
            sample, meta, q_range=q_range, max_iter=max_iter, verbose=False
        )
        elapsed = time.time() - start
        triang_rmses.append(rmse)
        triang_times.append(elapsed)
        print(f"  [{i+1}/{n_samples}] RMSE={rmse:.4f}, time={elapsed:.1f}s")

    print()
    print("Running without triangulation (baseline)...")
    optimizer_baseline = HybridOptimizer(
        n_smart_inits=1,
        n_random_inits=0,
        use_triangulation=False,
    )

    baseline_rmses = []
    baseline_times = []
    for i, sample in enumerate(samples[:n_samples]):
        start = time.time()
        sources, rmse, _ = optimizer_baseline.estimate_sources(
            sample, meta, q_range=q_range, max_iter=max_iter, verbose=False
        )
        elapsed = time.time() - start
        baseline_rmses.append(rmse)
        baseline_times.append(elapsed)
        print(f"  [{i+1}/{n_samples}] RMSE={rmse:.4f}, time={elapsed:.1f}s")

    # Compare results
    print()
    print("=" * 70)
    print(f"{'Sample':<12} {'Baseline RMSE':>14} {'Triang RMSE':>14} {'Winner':>10}")
    print("-" * 70)

    wins = {'baseline': 0, 'triangulation': 0, 'tie': 0}
    for i in range(n_samples):
        sample_id = samples[i]['sample_id']
        b_rmse = baseline_rmses[i]
        t_rmse = triang_rmses[i]

        if abs(b_rmse - t_rmse) < 0.01:
            winner = "tie"
            wins['tie'] += 1
        elif t_rmse < b_rmse:
            winner = "triang"
            wins['triangulation'] += 1
        else:
            winner = "baseline"
            wins['baseline'] += 1

        print(f"{sample_id:<12} {b_rmse:>14.4f} {t_rmse:>14.4f} {winner:>10}")

    print("-" * 70)
    print(f"{'AVERAGE':<12} {np.mean(baseline_rmses):>14.4f} {np.mean(triang_rmses):>14.4f}")
    print(f"{'STD':<12} {np.std(baseline_rmses):>14.4f} {np.std(triang_rmses):>14.4f}")
    print(f"{'AVG TIME':<12} {np.mean(baseline_times):>14.1f}s {np.mean(triang_times):>14.1f}s")
    print("=" * 70)

    print(f"\nWins: Baseline={wins['baseline']}, Triangulation={wins['triangulation']}, Tie={wins['tie']}")

    improvement = (np.mean(baseline_rmses) - np.mean(triang_rmses)) / np.mean(baseline_rmses) * 100
    print(f"Average RMSE improvement from triangulation: {improvement:.1f}%")

    # Calculate competition score
    def calc_score(rmses, n_candidates=1):
        scores = [1.0 / (1.0 + r) for r in rmses]
        return np.mean(scores) + 0.3 * (n_candidates / 3)

    baseline_score = calc_score(baseline_rmses)
    triang_score = calc_score(triang_rmses)
    print(f"\nCompetition scores (N_valid=1):")
    print(f"  Baseline: {baseline_score:.4f}")
    print(f"  Triangulation: {triang_score:.4f}")


if __name__ == "__main__":
    main()
