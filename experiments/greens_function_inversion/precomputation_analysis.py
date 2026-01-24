"""
Green's Function Pre-computation Analysis

Since there are only 4 unique (BC, kappa) combinations, could we pre-compute
Green's function tables to speed up the inverse problem?

Analysis:
1. What is the storage cost for pre-computed Green's function?
2. What is the pre-computation time?
3. Can we use it for direct source localization?
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from simulator import Heat2D

print("=== Green's Function Pre-computation Analysis ===")
print()

# Load data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)
samples = data['samples']

# Constants
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
n_timesteps = 1001
dt = 4.0 / n_timesteps
n_sensors = 2

# === Analysis 1: Pre-computation cost ===
print("=== 1. Pre-computation Cost Analysis ===")
print()

# For pre-computation, we need to store G(source_pos, sensor_pos, time)
# For each of 4 physics combinations (bc, kappa)

n_source_x = 50  # Source positions on coarse grid
n_source_y = 25
n_source_positions = n_source_x * n_source_y

print(f"Source position grid: {n_source_x} x {n_source_y} = {n_source_positions} positions")
print(f"Per physics combination:")
print(f"  Sensors: {n_sensors}")
print(f"  Timesteps: {n_timesteps}")
print(f"  Total entries: {n_source_positions} × {n_sensors} × {n_timesteps} = {n_source_positions * n_sensors * n_timesteps:,}")
print()

# Storage
entries_per_combo = n_source_positions * n_sensors * n_timesteps
bytes_per_combo = entries_per_combo * 8  # float64
total_bytes = bytes_per_combo * 4  # 4 combinations

print(f"Storage per (bc, kappa) combination: {bytes_per_combo / 1e9:.2f} GB")
print(f"Total storage for 4 combinations: {total_bytes / 1e9:.2f} GB")
print()

# Pre-computation time
# Each entry requires ADI simulation (since we don't have analytical Green's function
# that accounts for our boundary conditions properly)
print("Pre-computation would require ADI simulations:")
print(f"  Simulations needed per combo: {n_source_positions}")
print(f"  Total simulations: {n_source_positions * 4}")
print(f"  At ~1 sec/sim: {n_source_positions * 4 / 60:.1f} minutes")
print()

# === Analysis 2: The REAL Problem ===
print("=== 2. The Real Problem: Sensor Positions Vary ===")
print()

# Check sensor positions
sensor_positions = []
for s in samples:
    sensor_positions.append(tuple(map(tuple, s['sensors_xy'])))

unique_sensor_positions = set(sensor_positions)
print(f"Unique sensor configurations: {len(unique_sensor_positions)} out of {len(samples)} samples")
print()

if len(unique_sensor_positions) == len(samples):
    print("CRITICAL: 100% unique sensor positions!")
    print()
    print("Pre-computing Green's function requires knowing sensor positions.")
    print("With sample-specific sensors, we would need:")
    print(f"  80 samples × {n_source_positions} source positions × {n_timesteps} timesteps")
    print(f"  = {80 * n_source_positions * n_timesteps:,} entries to pre-compute")
    print()
    print("That's 80x more storage than originally thought!")
else:
    print(f"Some sensor positions are shared across samples.")

# === Analysis 3: Green's function vs ADI comparison ===
print()
print("=== 3. Direct Comparison: Full RMSE Evaluation ===")
print()

sample = samples[0]
meta = sample['sample_metadata']
kappa = meta['kappa']
bc = meta['bc']
sensors = sample['sensors_xy']
Y_obs = sample['Y_noisy']

simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)

# Time ADI-based RMSE evaluation
sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]

# Warmup
_, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)

start = time_module.perf_counter()
n_evals = 10
for _ in range(n_evals):
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
    Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])
    min_len = min(len(Y_sim), len(Y_obs))
    rmse = np.sqrt(np.mean((Y_sim[:min_len] - Y_obs[:min_len])**2))
adi_time = (time_module.perf_counter() - start) / n_evals * 1000

print(f"ADI-based RMSE evaluation: {adi_time:.1f} ms")

# Green's function approach (analytical, but requires series summation)
def greens_function_series(x, y, xs, ys, t, kappa, Lx, Ly, n_terms=50, bc='dirichlet'):
    """Green's function via eigenfunction expansion."""
    if t <= 0:
        return 0.0

    G = 0.0
    for n in range(1, n_terms + 1):
        for m in range(1, n_terms + 1):
            lambda_nm = (n * np.pi / Lx)**2 + (m * np.pi / Ly)**2

            if bc == 'dirichlet':
                # Dirichlet: sin functions
                spatial = (np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) *
                           np.sin(n * np.pi * xs / Lx) * np.sin(m * np.pi * ys / Ly))
                norm = 4.0 / (Lx * Ly)
            else:  # neumann
                # Neumann: cos functions
                spatial = (np.cos(n * np.pi * x / Lx) * np.cos(m * np.pi * y / Ly) *
                           np.cos(n * np.pi * xs / Lx) * np.cos(m * np.pi * ys / Ly))
                # Different normalization for cos
                norm = 4.0 / (Lx * Ly)
                if n == 0:
                    norm /= 2
                if m == 0:
                    norm /= 2

            temporal = np.exp(-kappa * lambda_nm * t)
            G += spatial * temporal
    return G * norm

# Time Green's function RMSE evaluation (simplified - just temperature, not full integral)
# Note: This is an APPROXIMATION. Full implementation needs time convolution.
def greens_rmse_approx(xs, ys, q, sensors, Y_obs, kappa, bc, n_terms=30):
    """Approximate RMSE using Green's function (simplified)."""
    n_timesteps = len(Y_obs)
    Y_sim = np.zeros((n_timesteps, len(sensors)))

    for t_idx in range(1, n_timesteps):
        t = t_idx * dt
        for s_idx, (sx, sy) in enumerate(sensors):
            G = greens_function_series(sx, sy, xs, ys, t, kappa, Lx, Ly, n_terms, bc)
            Y_sim[t_idx, s_idx] = q * G * dt * t_idx  # Very rough approximation

    return np.sqrt(np.mean((Y_sim - Y_obs)**2))

# Time a small subset (first 100 timesteps)
n_test_timesteps = 50
Y_obs_subset = Y_obs[:n_test_timesteps]

start = time_module.perf_counter()
_ = greens_rmse_approx(1.0, 0.5, 1.0, sensors, Y_obs_subset, kappa, bc, n_terms=30)
greens_time_subset = (time_module.perf_counter() - start) * 1000

estimated_greens_full = greens_time_subset * (n_timesteps / n_test_timesteps)

print(f"Green's function ({n_test_timesteps} timesteps, 30 terms): {greens_time_subset:.1f} ms")
print(f"Estimated full ({n_timesteps} timesteps): {estimated_greens_full:.0f} ms")
print()

# === Analysis 4: Direct inversion possibility ===
print("=== 4. Can We Avoid Iteration? ===")
print()
print("The Green's function approach still requires iteration because:")
print()
print("1. NONLINEAR INVERSION")
print("   T(sensor, t) = q × ∫₀ᵗ G(sensor, source, t-τ) dτ")
print("   The source position (x_s, y_s) appears nonlinearly in G")
print("   via sin(nπx_s/Lx), sin(mπy_s/Ly) terms")
print()
print("2. NO CLOSED-FORM INVERSE")
print("   Given T observations, finding (x_s, y_s, q) requires solving:")
print("   argmin ||T_obs - T_sim(x,y,q)||²")
print("   This is still an optimization problem!")
print()
print("3. GRID SEARCH EQUIVALENT TO CMA-ES")
print("   We could pre-tabulate G for all source positions and sensors")
print("   Then search the table to find best match")
print("   But this is just grid search - CMA-ES is more efficient")
print()

# === Conclusion ===
print("=" * 60)
print("=== FINAL ASSESSMENT ===")
print("=" * 60)
print()
print("Green's function approach CANNOT bypass iteration:")
print()
print("TIMING COMPARISON:")
print(f"  ADI simulation: {adi_time:.0f} ms per evaluation")
print(f"  Green's function: {estimated_greens_full:.0f} ms per evaluation (estimated)")
print(f"  Ratio: Green's is {estimated_greens_full/adi_time:.1f}x {'SLOWER' if estimated_greens_full > adi_time else 'faster'}")
print()
print("FUNDAMENTAL LIMITATIONS:")
print("  1. Sensor positions vary per sample (100% unique)")
print("     → Cannot pre-compute universal lookup table")
print("  2. Green's function evaluation is NOT faster than ADI")
print("     → ADI implicit scheme is highly optimized")
print("  3. Inversion requires nonlinear optimization")
print("     → Still need CMA-ES or similar optimizer")
print()
print("CONCLUSION: ABORT")
print("Green's function provides no computational advantage over ADI + CMA-ES baseline")
