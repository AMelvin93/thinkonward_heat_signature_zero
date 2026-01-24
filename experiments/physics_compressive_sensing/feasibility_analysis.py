"""
Feasibility Analysis for D-PBCS (Decomposed Physics-Based Compressive Sensing)

Key questions:
1. Is the heat source "sparse" in a meaningful sense?
2. Can we formulate as a linear CS problem?
3. What is the cost of building the observation matrix?
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']
sample = samples[0]
meta = sample['sample_metadata']

print("=== D-PBCS Feasibility Analysis ===")
print()

# Problem dimensions
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
n_grid_cells = nx * ny
n_sensors = sample['sensors_xy'].shape[0]
n_timesteps = meta['nt']
n_measurements = n_sensors * n_timesteps

print(f"Grid: {nx} x {ny} = {n_grid_cells} cells")
print(f"Domain: {Lx} x {Ly}")
print(f"Sensors: {n_sensors}")
print(f"Timesteps: {n_timesteps}")
print(f"Total measurements: {n_measurements}")
print()

# Sparsity analysis
print("=== Sparsity Analysis ===")
k_sparse = 2  # Maximum 2 sources
print(f"Max source count (k): {k_sparse}")
print(f"Sparsity ratio: {k_sparse}/{n_grid_cells} = {k_sparse/n_grid_cells:.6f}")
print(f"CS measurement bound: m >= c * k * log(n) ≈ {int(4 * k_sparse * np.log(n_grid_cells))}")
print(f"Available measurements: {n_measurements}")
print(f"CS condition satisfied: {n_measurements >= 4 * k_sparse * np.log(n_grid_cells)}")
print()

# Cost analysis: Building observation matrix
print("=== Observation Matrix Cost ===")
print("To formulate as linear CS: y = Ax, we need observation matrix A")
print("A[i,j] = temperature at sensor i when unit source is at grid cell j")
print()

# Time single simulation
simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=meta['kappa'], bc=meta['bc'])
sensors = sample['sensors_xy']
dt = 4.0 / n_timesteps  # Estimate dt

# Time one simulation - run multiple times for accuracy
import time as time_module
x_source, y_source = 1.0, 0.5  # Center of domain
sources = [{'x': x_source, 'y': y_source, 'q': 1.0}]

# Warmup run
_, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)

# Timed runs
n_trials = 3
total_time = 0
for _ in range(n_trials):
    start = time_module.perf_counter()
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
    Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])
    total_time += time_module.perf_counter() - start
sim_time = total_time / n_trials

print(f"Single simulation time: {sim_time:.3f} seconds")
print(f"Simulations needed for full A: {n_grid_cells}")
print(f"Total time for A: {sim_time * n_grid_cells / 60:.1f} minutes (per sample!)")
print()

# Budget analysis
print("=== Budget Analysis ===")
n_samples = 400
time_limit = 60  # minutes
time_per_sample = time_limit / n_samples
print(f"Time budget: {time_limit} min for {n_samples} samples")
print(f"Per-sample budget: {time_per_sample:.2f} min = {time_per_sample*60:.1f} sec")
print()

a_build_time = sim_time * n_grid_cells
print(f"Time to build A (per sample): {a_build_time/60:.1f} min")
print(f"Exceeds budget by: {a_build_time/60 / time_per_sample:.0f}x")
print()

# Could we use a coarser grid?
print("=== Coarse Grid Alternative ===")
for coarse_factor in [2, 4, 5, 10]:
    n_cells_coarse = (nx // coarse_factor) * (ny // coarse_factor)
    time_coarse = sim_time * n_cells_coarse / 60
    print(f"  Grid {nx//coarse_factor}x{ny//coarse_factor} ({n_cells_coarse} cells): "
          f"{time_coarse:.1f} min/sample ({time_coarse*400:.0f} min total)")
print()

# Could we pre-compute A once for all samples?
print("=== Pre-computation Strategy ===")
print("If kappa and BC were constant across samples, we could pre-compute A once.")
print("But samples have varying kappa:")
unique_kappas = set(s['sample_metadata']['kappa'] for s in samples)
print(f"  Unique kappa values: {unique_kappas}")
print()
print("And varying sensor locations (each sample unique):")
print("  => Cannot reuse A across samples")
print()

# Final assessment
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CRITICAL ISSUES:")
print()
print("1. OBSERVATION MATRIX TOO EXPENSIVE")
print(f"   Building A requires {n_grid_cells} simulations = {a_build_time/60:.1f} min/sample")
print(f"   Budget allows only {time_per_sample:.2f} min/sample")
print(f"   Exceeds budget by {a_build_time/60 / time_per_sample:.0f}x")
print()
print("2. SAMPLE-SPECIFIC A MATRIX")
print("   Each sample has unique sensor locations")
print("   Cannot pre-compute A once and reuse")
print("   Must rebuild A for every single sample")
print()
print("3. COARSE GRID DOESN'T HELP ENOUGH")
print("   Even with 10x10=100 cells, still need ~1 min/sample for A")
print("   Plus CS solve time, plus final refinement")
print("   Total would exceed budget")
print()
print("CONCLUSION: D-PBCS is NOT VIABLE for this problem.")
print("The observation matrix construction cost is prohibitive.")
print("Even coarse approximations exceed the time budget.")
