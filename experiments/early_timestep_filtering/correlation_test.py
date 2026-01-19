"""
Temporal Fidelity Feasibility Test

Goal: Check if RMSE computed with truncated timesteps correlates with full RMSE.
If correlation is high (>0.8), the approach should work.
"""

import numpy as np
import pickle
import os
import sys
from scipy.stats import pearsonr, spearmanr

# Add paths for imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))

from simulator import Heat2D

def load_test_data():
    """Load test data."""
    with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['samples'], data['meta']

def compute_rmse_at_fraction(Y_pred, Y_obs, fraction):
    """Compute RMSE using only first `fraction` of timesteps."""
    n_timesteps = int(len(Y_obs) * fraction)
    n_timesteps = max(10, n_timesteps)  # At least 10 timesteps
    Y_pred_trunc = Y_pred[:n_timesteps]
    Y_obs_trunc = Y_obs[:n_timesteps]
    return np.sqrt(np.mean((Y_pred_trunc - Y_obs_trunc) ** 2))

def run_correlation_test():
    """Test correlation between truncated and full RMSE."""
    samples, meta = load_test_data()

    print("="*60)
    print("Temporal Fidelity Correlation Test")
    print("="*60)

    # Generate random source configurations and compute RMSE at different fractions
    np.random.seed(42)

    fractions = [0.25, 0.50, 0.75, 1.0]
    results = {f: [] for f in fractions}

    # Test on first 20 samples
    n_samples_test = 20
    n_candidates_per_sample = 10

    for sample_idx in range(n_samples_test):
        sample = samples[sample_idx]
        n_sources = sample['n_sources']
        Y_obs = sample['Y_noisy']
        sensors_xy = sample['sensors_xy']
        kappa = sample['sample_metadata']['kappa']
        dt = meta['dt']
        nt = len(Y_obs) - 1
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = Heat2D(Lx=2.0, Ly=1.0, nx=100, ny=50, kappa=kappa, bc=bc)

        # Generate random candidates
        for _ in range(n_candidates_per_sample):
            sources = []
            for _ in range(n_sources):
                x = np.random.uniform(0.1, 1.9)
                y = np.random.uniform(0.1, 0.9)
                q = np.random.uniform(0.5, 2.0)
                sources.append({'x': x, 'y': y, 'q': q})

            # Run full simulation
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

            # Compute RMSE at different fractions
            for frac in fractions:
                rmse = compute_rmse_at_fraction(Y_pred, Y_obs, frac)
                results[frac].append(rmse)

        if (sample_idx + 1) % 5 == 0:
            print(f"Processed {sample_idx + 1}/{n_samples_test} samples...")

    # Analyze correlations
    print("\n" + "="*60)
    print("Correlation Analysis")
    print("="*60)

    full_rmse = np.array(results[1.0])

    print(f"\nTotal candidates tested: {len(full_rmse)}")
    print(f"Full RMSE range: [{full_rmse.min():.4f}, {full_rmse.max():.4f}]")
    print(f"Full RMSE mean: {full_rmse.mean():.4f}")

    print("\nCorrelation with full RMSE (1.0 fraction):")
    print("-"*40)
    print(f"{'Fraction':<10} | {'Pearson r':<12} | {'Spearman r':<12} | Viable?")
    print("-"*40)

    for frac in [0.25, 0.50, 0.75]:
        trunc_rmse = np.array(results[frac])
        pearson_r, _ = pearsonr(trunc_rmse, full_rmse)
        spearman_r, _ = spearmanr(trunc_rmse, full_rmse)
        viable = "YES" if spearman_r > 0.8 else ("maybe" if spearman_r > 0.7 else "NO")
        print(f"{frac:<10} | {pearson_r:>10.4f}  | {spearman_r:>10.4f}  | {viable}")

    # Check ranking preservation (most important for CMA-ES)
    print("\n" + "="*60)
    print("Ranking Preservation Analysis")
    print("="*60)
    print("\nHow often does truncated RMSE preserve rank order?")

    for frac in [0.25, 0.50, 0.75]:
        trunc_rmse = np.array(results[frac])

        # Check if best candidate (by full RMSE) is in top-3 by truncated RMSE
        # Group by sample
        n_per_sample = n_candidates_per_sample
        top3_preserved = 0
        best_preserved = 0
        total_groups = n_samples_test

        for i in range(total_groups):
            start_idx = i * n_per_sample
            end_idx = start_idx + n_per_sample

            full_slice = full_rmse[start_idx:end_idx]
            trunc_slice = trunc_rmse[start_idx:end_idx]

            # Find best by full RMSE
            best_full_idx = np.argmin(full_slice)

            # Find top 3 by truncated RMSE
            top3_trunc_idx = np.argsort(trunc_slice)[:3]

            if best_full_idx in top3_trunc_idx:
                top3_preserved += 1
            if np.argmin(trunc_slice) == best_full_idx:
                best_preserved += 1

        top3_pct = top3_preserved / total_groups * 100
        best_pct = best_preserved / total_groups * 100
        print(f"{frac*100:.0f}% timesteps: Best in top-3: {top3_pct:.0f}%, Best preserved: {best_pct:.0f}%")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    # Find best fraction
    best_frac = None
    for frac in [0.25, 0.50, 0.75]:
        trunc_rmse = np.array(results[frac])
        spearman_r, _ = spearmanr(trunc_rmse, full_rmse)
        if spearman_r > 0.8:
            if best_frac is None:
                best_frac = frac

    if best_frac:
        speedup = 1.0 / best_frac
        print(f"VIABLE: {best_frac*100:.0f}% timesteps gives good correlation")
        print(f"Expected speedup: ~{speedup:.1f}x for CMA-ES evaluations")
    else:
        print("WARNING: No fraction shows >0.8 correlation")
        print("Experiment may not work well")

if __name__ == "__main__":
    run_correlation_test()
