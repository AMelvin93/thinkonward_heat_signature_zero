"""
Modal Identification Method - Feasibility Analysis

Hypothesis: Build low-order modal model from sensor data for fast source estimation.

Key questions:
1. Can we identify thermal modes from 2 sensors?
2. What is the computational advantage of modal space inversion?
3. Does modal truncation preserve source localization accuracy?

Analysis approach:
1. Extract dominant modes from sensor temperature time series using SVD/DMD
2. Check if modes capture sufficient thermal dynamics
3. Estimate source from modal representation
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from scipy.linalg import svd, lstsq
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']
sample = samples[0]
meta = sample['sample_metadata']

print("=== Modal Identification Method - Feasibility Analysis ===")
print()

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
kappa = meta['kappa']
bc = meta['bc']
Y_obs = sample['Y_noisy']
sensors = sample['sensors_xy']
n_timesteps, n_sensors = Y_obs.shape
dt = 4.0 / n_timesteps

print(f"Domain: {Lx} x {Ly}")
print(f"Grid: {nx} x {ny}")
print(f"Kappa: {kappa}")
print(f"Timesteps: {n_timesteps}, dt = {dt:.4f}")
print(f"Sensors: {n_sensors}")
print(f"Y_obs shape: {Y_obs.shape}")
print()

# === Analysis 1: Modal content of sensor data ===
print("=== 1. Modal Content Analysis (SVD of Y_obs) ===")
print()

# Apply SVD to the observed temperature data
# Y_obs: (n_timesteps, n_sensors) = (1001, 2)
U, s, Vh = svd(Y_obs, full_matrices=False)

print(f"SVD results:")
print(f"  U shape: {U.shape} (temporal modes)")
print(f"  s shape: {s.shape} (singular values)")
print(f"  Vh shape: {Vh.shape} (sensor weights)")
print()

# Analyze singular value distribution
total_energy = np.sum(s**2)
energy_per_mode = s**2 / total_energy * 100

print(f"Singular values: {s}")
print(f"Energy per mode: {energy_per_mode}%")
print()

# Key insight: with 2 sensors, we can only have 2 modes maximum!
print("CRITICAL LIMITATION:")
print(f"  With {n_sensors} sensors, maximum modes = {n_sensors}")
print(f"  Modal identification cannot capture more than {n_sensors} independent dynamics")
print()

# === Analysis 2: What modes can we identify? ===
print("=== 2. Modal Truncation Analysis ===")
print()

# Reconstruct signal with 1 mode vs 2 modes
Y_1mode = U[:, :1] @ np.diag(s[:1]) @ Vh[:1, :]
Y_2mode = U @ np.diag(s) @ Vh  # Full reconstruction (still only 2 modes)

rmse_1mode = np.sqrt(np.mean((Y_1mode - Y_obs)**2))
rmse_2mode = np.sqrt(np.mean((Y_2mode - Y_obs)**2))

print(f"Reconstruction RMSE:")
print(f"  1 mode: {rmse_1mode:.6f}")
print(f"  2 modes: {rmse_2mode:.6f} (full reconstruction)")
print()

# === Analysis 3: Source estimation in modal space ===
print("=== 3. Source Estimation in Modal Space ===")
print()

print("For modal identification to work, we need:")
print("  1. Modal basis that spans source-response relationship")
print("  2. Invertible mapping from modes to source position")
print()

# The fundamental problem:
# - Temperature response depends on source position: T(sensor, t) = f(x_source, y_source, q)
# - With only 2 sensors, we observe 2 time series
# - Source has 2-3 unknowns (x, y, [q])
# - In principle, 2 time series could determine 2-3 unknowns...

print("MATHEMATICAL ANALYSIS:")
print()
print("  Temperature at sensor i: T_i(t) = q × ∫ G(sensor_i, source, t-τ) dτ")
print()
print("  With 2 sensors, we have 2 equations (time series)")
print("  Unknowns: x_s, y_s, q (3 variables)")
print()
print("  Even in modal space:")
print("    - Mode 1 coefficient → function of (x_s, y_s, q)")
print("    - Mode 2 coefficient → function of (x_s, y_s, q)")
print()
print("  This is an underdetermined system!")
print("  Adding temporal structure doesn't help: all time info is already in the modes")
print()

# === Analysis 4: Comparison with current approach ===
print("=== 4. Comparison with Baseline CMA-ES ===")
print()

print("Current baseline approach:")
print("  1. CMA-ES explores (x, y) parameter space")
print("  2. For each candidate: simulate → compute RMSE")
print("  3. Select best candidates")
print()

print("Modal identification approach:")
print("  1. Extract modes from Y_obs (SVD/DMD)")
print("  2. For each candidate source position:")
print("     a. Simulate temperature response")
print("     b. Extract modes from simulation")
print("     c. Compare modes with observed modes")
print("  3. Still requires iterative optimization!")
print()

# Time the modal extraction
start = time_module.perf_counter()
for _ in range(100):
    U, s, Vh = svd(Y_obs, full_matrices=False)
svd_time = (time_module.perf_counter() - start) / 100 * 1000

print(f"Modal extraction (SVD) time: {svd_time:.4f} ms")
print()

# Compare with full RMSE computation
simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)
sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]

# Warmup
_, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)

start = time_module.perf_counter()
n_evals = 5
for _ in range(n_evals):
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
    Y_sim = np.array([simulator.sample_sensors(U_snap, sensors) for U_snap in Us])
    rmse = np.sqrt(np.mean((Y_sim[:len(Y_obs)] - Y_obs)**2))
sim_rmse_time = (time_module.perf_counter() - start) / n_evals * 1000

print(f"Full simulation + RMSE: {sim_rmse_time:.1f} ms")
print()

# === Analysis 5: The fundamental problem ===
print("=== 5. Fundamental Problem ===")
print()

print("WHY MODAL IDENTIFICATION CANNOT HELP:")
print()
print("1. SIMULATION IS THE BOTTLENECK")
print(f"   - Simulation: {sim_rmse_time:.0f} ms")
print(f"   - SVD: {svd_time:.3f} ms")
print(f"   - RMSE computation: <0.1 ms")
print("   Modal space doesn't reduce simulation cost!")
print()

print("2. MODAL BASIS STILL REQUIRES SIMULATION")
print("   To compare observed modes with candidate modes,")
print("   we still need to simulate the candidate's temperature response.")
print("   Modal extraction adds overhead without reducing simulations.")
print()

print("3. SENSOR SPARSITY LIMITS MODAL CONTENT")
print(f"   With only {n_sensors} sensors, we can observe {n_sensors} modes max.")
print("   The thermal field has infinitely many modes.")
print("   Sensor sparsity fundamentally limits what we can infer.")
print()

print("4. MODAL INVERSION IS STILL NONLINEAR")
print("   Source position appears nonlinearly in mode coefficients.")
print("   We still need CMA-ES or similar optimizer.")
print()

# === Analysis 6: What MIM papers actually do ===
print("=== 6. What MIM Literature Actually Does ===")
print()

print("The MIM literature (ResearchGate paper) assumes:")
print("  1. KNOWN modal basis (from FEM or analytical solution)")
print("  2. DENSE sensor network (many sensors, not 2)")
print("  3. LINEAR parameterization of source")
print()

print("Our problem violates all assumptions:")
print("  1. No pre-computed modal basis (physics vary per sample)")
print("  2. Only 2 sensors (extremely sparse)")
print("  3. Source position is nonlinear parameter")
print()

print("MIM reduces to standard inverse problem when assumptions don't hold.")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Modal Identification Method is NOT VIABLE")
print()
print("Reasons:")
print("  1. Simulation is bottleneck, modal space doesn't help")
print("  2. Still need to simulate for each candidate comparison")
print("  3. Only 2 sensors limits observable modal content to 2 modes")
print("  4. Source inversion remains nonlinear")
print("  5. MIM literature assumptions don't hold for our problem")
print()
print("RECOMMENDATION: ABORT - No advantage over ADI + CMA-ES")
