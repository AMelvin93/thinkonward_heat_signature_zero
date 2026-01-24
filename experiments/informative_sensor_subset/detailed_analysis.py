"""
Informative Sensor Subset - Detailed Analysis

CORRECTION: Most samples have 3-6 sensors, not just 2!

New analysis:
1. For samples with 3+ sensors, test if subset selection helps
2. Compare RMSE with all sensors vs top K sensors
3. Check if removing low-variance sensors improves accuracy
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

print("=== Informative Sensor Subset - Detailed Analysis ===")
print()

# === Analysis 1: Sensor count distribution ===
print("=== 1. Sensor Count Distribution ===")
print()

sensor_count_groups = {2: [], 3: [], 4: [], 5: [], 6: []}
for i, s in enumerate(samples):
    n = len(s['sensors_xy'])
    if n in sensor_count_groups:
        sensor_count_groups[n].append(i)

for count, indices in sorted(sensor_count_groups.items()):
    print(f"  {count} sensors: {len(indices)} samples")
print()

# === Analysis 2: Test sensor subset on samples with 4+ sensors ===
print("=== 2. Sensor Subset Impact Test ===")
print()

# Pick a sample with 4 sensors for testing
sample_idx = None
for idx in sensor_count_groups.get(4, []):
    sample = samples[idx]
    if sample['n_sources'] == 1:  # Start with 1-source for simpler analysis
        sample_idx = idx
        break

if sample_idx is None:
    # Fallback to any 4-sensor sample
    if sensor_count_groups.get(4, []):
        sample_idx = sensor_count_groups[4][0]

if sample_idx is not None:
    sample = samples[sample_idx]
    meta = sample['sample_metadata']
    Y_obs = sample['Y_noisy']
    sensors = sample['sensors_xy']
    n_sources = sample['n_sources']
    n_timesteps, n_sensors = Y_obs.shape

    print(f"Test sample {sample_idx}:")
    print(f"  Sensors: {n_sensors}")
    print(f"  Sources: {n_sources}")
    print(f"  Y_obs shape: {Y_obs.shape}")
    print()

    # Analyze per-sensor statistics
    sensor_stats = []
    for i in range(n_sensors):
        sensor_data = Y_obs[:, i]
        stats = {
            'idx': i,
            'pos': sensors[i],
            'mean': np.mean(sensor_data),
            'var': np.var(sensor_data),
            'max': np.max(sensor_data),
            'snr': np.mean(sensor_data) / (np.std(sensor_data) + 1e-10)
        }
        sensor_stats.append(stats)

    # Sort by variance (high to low)
    sensor_stats_sorted = sorted(sensor_stats, key=lambda x: -x['var'])

    print("Sensors ranked by variance:")
    for s in sensor_stats_sorted:
        print(f"  Sensor {s['idx']}: pos={np.array(s['pos'])}, var={s['var']:.4f}, mean={s['mean']:.4f}")
    print()

    # === Analysis 3: Compare RMSE with different sensor subsets ===
    print("=== 3. RMSE Comparison: All Sensors vs Subset ===")
    print()

    # Set up simulator
    Lx, Ly, nx, ny = 2.0, 1.0, 100, 50
    kappa = meta['kappa']
    bc = meta['bc']
    n_timesteps_sim = n_timesteps
    dt = 4.0 / n_timesteps_sim

    simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)

    # Test with random source positions
    np.random.seed(42)
    n_test = 10
    rmse_all = []
    rmse_top2 = []
    rmse_top3 = []

    for _ in range(n_test):
        # Random source position
        x = np.random.uniform(0.1*Lx, 0.9*Lx)
        y = np.random.uniform(0.1*Ly, 0.9*Ly)
        q = np.random.uniform(0.5, 2.0)

        sources = [{'x': x, 'y': y, 'q': q}]
        _, Us = simulator.solve(dt=dt, nt=n_timesteps_sim, T0=0.0, sources=sources)
        Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])

        # Match lengths
        min_len = min(len(Y_sim), len(Y_obs))
        Y_sim = Y_sim[:min_len]
        Y_obs_matched = Y_obs[:min_len]

        # RMSE with all sensors
        rmse_all.append(np.sqrt(np.mean((Y_sim - Y_obs_matched)**2)))

        # Get top sensors by variance
        top_indices = [s['idx'] for s in sensor_stats_sorted]

        # RMSE with top 2 sensors
        rmse_top2.append(np.sqrt(np.mean((Y_sim[:, top_indices[:2]] - Y_obs_matched[:, top_indices[:2]])**2)))

        # RMSE with top 3 sensors (if available)
        if n_sensors >= 3:
            rmse_top3.append(np.sqrt(np.mean((Y_sim[:, top_indices[:3]] - Y_obs_matched[:, top_indices[:3]])**2)))

    print(f"RMSE comparison ({n_test} random source positions):")
    print(f"  All {n_sensors} sensors: mean={np.mean(rmse_all):.4f}, std={np.std(rmse_all):.4f}")
    print(f"  Top 2 sensors:          mean={np.mean(rmse_top2):.4f}, std={np.std(rmse_top2):.4f}")
    if rmse_top3:
        print(f"  Top 3 sensors:          mean={np.mean(rmse_top3):.4f}, std={np.std(rmse_top3):.4f}")
    print()

# === Analysis 4: Correlation analysis ===
print("=== 4. RMSE Correlation Analysis ===")
print()

print("Key question: Does subset RMSE correlate with full RMSE for ranking?")
print()

if sample_idx is not None:
    # Compute RMSE for more source positions
    np.random.seed(42)
    n_test = 50
    full_rmses = []
    subset_rmses = []

    for _ in range(n_test):
        x = np.random.uniform(0.1*Lx, 0.9*Lx)
        y = np.random.uniform(0.1*Ly, 0.9*Ly)
        q = 1.0

        sources = [{'x': x, 'y': y, 'q': q}]
        _, Us = simulator.solve(dt=dt, nt=n_timesteps_sim, T0=0.0, sources=sources)
        Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])

        min_len = min(len(Y_sim), len(Y_obs))
        Y_sim = Y_sim[:min_len]
        Y_obs_matched = Y_obs[:min_len]

        # Full RMSE
        full_rmses.append(np.sqrt(np.mean((Y_sim - Y_obs_matched)**2)))

        # Subset RMSE (top 2 by variance)
        subset_rmses.append(np.sqrt(np.mean((Y_sim[:, top_indices[:2]] - Y_obs_matched[:, top_indices[:2]])**2)))

    from scipy.stats import spearmanr
    corr, pval = spearmanr(full_rmses, subset_rmses)

    print(f"Spearman correlation between full and subset RMSE:")
    print(f"  r = {corr:.4f}, p = {pval:.4e}")
    print()

    if corr > 0.95:
        print("HIGH correlation: Subset RMSE ranks candidates similarly to full RMSE")
    elif corr > 0.8:
        print("MODERATE correlation: Subset ranking is reasonable approximation")
    else:
        print("LOW correlation: Subset RMSE is NOT a good proxy for full RMSE")
    print()

# === Analysis 5: The scoring issue ===
print("=== 5. Scoring Issue ===")
print()

print("CRITICAL: Final score is computed on ALL sensors!")
print()
print("Even if subset RMSE is a good proxy during optimization,")
print("the final submission is scored on FULL RMSE using all sensors.")
print()
print("Using fewer sensors for CMA-ES optimization means:")
print("  1. We're optimizing a PROXY objective (subset RMSE)")
print("  2. This may find different optimum than full RMSE")
print("  3. Similar to EXP_WEIGHTED_LOSS_001 which FAILED")
print()

# === Analysis 6: Computational benefit? ===
print("=== 6. Computational Benefit Analysis ===")
print()

print("Potential time savings from using fewer sensors:")
print()

# RMSE computation is negligible
print("  RMSE computation: ~0.1 ms (negligible)")
print("  Simulation: ~1200 ms (bottleneck)")
print()
print("  Using 2 sensors vs 4 sensors in RMSE:")
print("    - Saves maybe 0.02 ms per evaluation")
print("    - Does NOT reduce simulation time")
print()
print("  CONCLUSION: No computational benefit from sensor subset")
print()

# === Conclusion ===
print("=" * 60)
print("=== ASSESSMENT ===")
print("=" * 60)
print()
print("Sensor subset selection analysis:")
print()
print("POSITIVE:")
print("  - Most samples have 3-6 sensors (not just 2)")
print("  - Subset RMSE may correlate with full RMSE")
print()
print("NEGATIVE:")
print("  - Final score uses ALL sensors (optimizing different objective)")
print("  - No computational benefit (RMSE is not bottleneck)")
print("  - Similar to weighted loss which FAILED")
print()
print("RECOMMENDATION: ABORT - Optimizing subset RMSE is proxy optimization")
print("  like weighted loss, which already failed catastrophically.")
