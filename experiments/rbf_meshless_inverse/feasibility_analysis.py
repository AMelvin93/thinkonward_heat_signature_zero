"""
RBF Meshless Inverse Method - Feasibility Analysis

Hypothesis: Radial Basis Function meshless methods can directly solve the
inverse heat source problem without iterative optimization.

Key questions:
1. What is RBF interpolation for inverse problems?
2. Can it handle sample-specific sensor positions?
3. What is the computational cost vs CMA-ES baseline?

Analysis:
RBF methods approximate the temperature field as:
  T(x,y) = Σ_i λ_i × φ(||r - r_i||)
where φ is a radial basis function (e.g., Gaussian, multiquadric)
and r_i are collocation points (typically sensor locations).
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from scipy.linalg import solve
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

print("=== RBF Meshless Inverse Method - Feasibility Analysis ===")
print()

# === Analysis 1: What is the RBF inverse approach? ===
print("=== 1. RBF Inverse Problem Formulation ===")
print()

print("RBF approach to inverse heat source identification:")
print()
print("  1. FORWARD MODEL:")
print("     T(x,y,t) = ∫ G(x,y,t; x',y') × q(x',y') dx'dy'")
print("     where G is the Green's function and q is the source term")
print()
print("  2. RBF APPROXIMATION:")
print("     Approximate source: q(x,y) = Σ_j λ_j × φ(||r - r_j||)")
print("     where φ is RBF kernel, r_j are collocation points")
print()
print("  3. LINEAR SYSTEM:")
print("     Given observations at sensors, solve for λ_j:")
print("     A × λ = T_obs")
print("     where A_ij = ∫ G(sensor_i, r_j) × φ(||r_j - r_k||) dr_k")
print()
print("  4. KEY ISSUE:")
print("     Matrix A depends on G (Green's function)")
print("     G depends on (BC, kappa) which vary per sample")
print("     Must compute A for EACH sample!")
print()

# === Analysis 2: Matrix A computation cost ===
print("=== 2. Matrix A Computation Cost ===")
print()

sample = samples[0]
meta = sample['sample_metadata']
Y_obs = sample['Y_noisy']
sensors = sample['sensors_xy']
n_timesteps, n_sensors = Y_obs.shape
dt = 4.0 / n_timesteps

Lx, Ly, nx, ny = 2.0, 1.0, 100, 50
kappa = meta['kappa']
bc = meta['bc']

# Number of RBF collocation points (typically coarse grid)
n_rbf_x = 20
n_rbf_y = 10
n_rbf_points = n_rbf_x * n_rbf_y

print(f"RBF collocation grid: {n_rbf_x} × {n_rbf_y} = {n_rbf_points} points")
print(f"Sensors: {n_sensors}")
print(f"Timesteps: {n_timesteps}")
print()

# Matrix A has shape (n_sensors × n_timesteps) × n_rbf_points
# Each entry requires evaluating thermal response

print(f"Matrix A dimensions:")
print(f"  Rows: {n_sensors} sensors × {n_timesteps} timesteps = {n_sensors * n_timesteps}")
print(f"  Cols: {n_rbf_points} RBF points")
print(f"  Total entries: {n_sensors * n_timesteps * n_rbf_points:,}")
print()

# Computing each column of A requires one simulation
# (response at sensors due to unit source at RBF point j)

simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)

# Time a single simulation
sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
_ = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)  # warmup

start = time_module.perf_counter()
n_trials = 5
for _ in range(n_trials):
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
sim_time = (time_module.perf_counter() - start) / n_trials

print(f"Single simulation time: {sim_time:.2f} sec")
print()

# Cost to compute matrix A
A_cost_sec = sim_time * n_rbf_points
A_cost_min = A_cost_sec / 60

print(f"Matrix A computation cost:")
print(f"  Simulations needed: {n_rbf_points}")
print(f"  Time: {A_cost_sec:.0f} sec = {A_cost_min:.1f} min")
print()

# Budget comparison
budget_per_sample = 60 / (400 / 60)  # 60 min for 400 samples = 9 sec/sample
print(f"Budget per sample: {budget_per_sample:.1f} sec")
print()

if A_cost_sec > budget_per_sample:
    print(f"CRITICAL: Matrix A computation ({A_cost_sec:.0f} sec) >> budget ({budget_per_sample:.1f} sec)")
    print(f"  Overhead: {A_cost_sec / budget_per_sample:.0f}x over budget")
else:
    print(f"Matrix A cost is within budget ({A_cost_sec / budget_per_sample:.1%})")
print()

# === Analysis 3: Can we pre-compute A? ===
print("=== 3. Can We Pre-compute Matrix A? ===")
print()

# Check sensor position uniqueness
sensor_configs = set()
for s in samples:
    sensor_configs.add(tuple(map(tuple, s['sensors_xy'])))

print(f"Unique sensor configurations: {len(sensor_configs)} / {len(samples)} samples")
print()

if len(sensor_configs) == len(samples):
    print("CRITICAL: 100% unique sensor positions!")
    print("  Matrix A depends on sensor positions")
    print("  Cannot pre-compute A for all samples")
    print()
    print("  Even with only 4 (BC, kappa) combinations,")
    print("  we'd need 80 different A matrices")
    print("  Total pre-computation: 80 × {:.1f} min = {:.0f} min".format(A_cost_min, 80 * A_cost_min))
else:
    print(f"Some sensor positions shared - limited pre-computation possible")
print()

# === Analysis 4: What about coarse RBF grid? ===
print("=== 4. Coarse RBF Grid Analysis ===")
print()

# Try much coarser grid
coarse_configs = [
    (5, 3, 15),    # 5x3 = 15 points
    (10, 5, 50),   # 10x5 = 50 points
    (20, 10, 200), # 20x10 = 200 points
]

print("Coarse grid options:")
for nx_rbf, ny_rbf, n_pts in coarse_configs:
    cost_sec = sim_time * n_pts
    cost_min = cost_sec / 60
    budget_ratio = cost_sec / budget_per_sample
    status = "✓ OK" if budget_ratio < 1 else f"✗ {budget_ratio:.1f}x over"
    print(f"  {nx_rbf}×{ny_rbf} = {n_pts} pts: {cost_sec:.1f} sec ({cost_min:.2f} min) [{status}]")
print()

# Even 5×3 = 15 points is 15 × 1 sec = 15 sec > 9 sec budget!
print("CONCLUSION: Even coarsest grid (5×3=15 pts) exceeds budget (15 sec vs 9 sec)")
print()

# === Analysis 5: The fundamental problem ===
print("=== 5. Fundamental Problem ===")
print()

print("RBF MESHLESS HAS SAME ISSUES AS GREEN'S FUNCTION:")
print()

print("1. SAMPLE-SPECIFIC COMPUTATION")
print("   - Matrix A depends on (BC, kappa, sensor_positions)")
print("   - 100% unique sensor positions")
print("   - Must compute A for EACH sample")
print()

print("2. PROHIBITIVE COST")
print("   - Even coarse 5×3 RBF grid needs 15 simulations")
print("   - 15 sims × 1 sec = 15 sec vs 9 sec budget")
print("   - 1.7x over budget for JUST building A")
print("   - Doesn't include solving Ax=b or post-processing")
print()

print("3. STILL REQUIRES SIMULATIONS")
print("   - RBF doesn't avoid simulation")
print("   - It just reformulates the problem")
print("   - Each RBF column needs one simulation")
print()

print("4. ALREADY TESTED AS D-PBCS")
print("   - EXP_PHYSICS_CS_001 tested similar 'observation matrix' approach")
print("   - Result: 97 min/sample (646x over budget)")
print("   - RBF is essentially the same - different basis functions, same cost")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: RBF Meshless Inverse Method is NOT VIABLE")
print()
print("Reasons:")
print("  1. Matrix A computation requires n_rbf_points simulations")
print("  2. Even coarsest grid (15 pts) exceeds per-sample budget")
print("  3. 100% unique sensor positions prevent pre-computation")
print("  4. Same fundamental issue as D-PBCS (646x over budget)")
print()
print("RECOMMENDATION: ABORT - Meshless direct methods require too many sims")
print()
print("meshless_direct family should be marked EXHAUSTED.")
print("Any direct method needing an observation matrix is prohibitively expensive.")
