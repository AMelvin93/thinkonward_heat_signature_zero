"""
Analyze why JAX solver produces different results.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

from jax_heat_solver import create_jax_solver
import jax.numpy as jnp
import numpy as np
from simulator import Heat2D

# Load test data
import pickle
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['samples'][0]
meta = data['meta']

# Problem parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
kappa = sample['sample_metadata']['kappa']
dt_orig = meta['dt']
nt_orig = sample['sample_metadata']['nt']
T0 = sample['sample_metadata']['T0']
bc = sample['sample_metadata']['bc']

sensors_xy = np.array(sample['sensors_xy'])
Y_observed = sample['Y_noisy']

# Test source position
x_test, y_test, q_test = 1.0, 0.5, 1.0

print("=== Comparing simulators ===")

# Original simulator
solver_orig = Heat2D(Lx, Ly, nx, ny, kappa, bc)
sources = [{'x': x_test, 'y': y_test, 'q': q_test}]
times_orig, Us_orig = solver_orig.solve(dt_orig, nt_orig, T0=T0, sources=sources)
Y_orig = np.array([solver_orig.sample_sensors(U, sensors_xy) for U in Us_orig])
rmse_orig = np.sqrt(np.mean((Y_orig - Y_observed) ** 2))

print(f"Original simulator (implicit ADI):")
print(f"  dt={dt_orig}, nt={nt_orig}")
print(f"  Y shape: {Y_orig.shape}")
print(f"  Y mean: {Y_orig.mean():.4f}, max: {Y_orig.max():.4f}")
print(f"  RMSE vs observed: {rmse_orig:.4f}")

# JAX simulator
solver_jax = create_jax_solver(Lx, Ly, nx, ny, kappa)

# Use original dt (explicit might be unstable but let's see)
dt_jax = solver_jax['dt_max'] * 0.5
t_total = dt_orig * nt_orig
nt_jax = int(t_total / dt_jax)

print(f"\nJAX simulator (explicit Euler):")
print(f"  dt_max (stable): {solver_jax['dt_max']:.6f}")
print(f"  dt used: {dt_jax:.6f}")
print(f"  nt needed: {nt_jax}")

# The key issue: explicit Euler with small dt

# Let's verify the JAX solver produces similar temperature field
sigma = 2.5 * max(solver_jax['dx'], solver_jax['dy'])
source_params = jnp.array([x_test, y_test, q_test])

print(f"\nSimulating with JAX (nt={min(nt_jax, 500)})...")
U_history = solver_jax['solve_forward'](source_params, T0, sigma, min(nt_jax, 500), dt_jax)
print(f"  U_history shape: {U_history.shape}")
print(f"  U mean: {float(U_history[-1].mean()):.6f}, max: {float(U_history[-1].max()):.6f}")

# Sample at sensors
sensors_jax = jnp.array(sensors_xy)
Y_jax = np.array([
    solver_jax['interpolate_sensors'](U, sensors_jax) for U in U_history
])

print(f"  Y_jax shape: {Y_jax.shape}")
print(f"  Y_jax mean: {Y_jax.mean():.4f}, max: {Y_jax.max():.4f}")

# Compare temperature evolution
print("\n=== Temperature comparison ===")
print("Original (implicit):")
print(f"  Early (t=10): sensor 0 = {Y_orig[10, 0]:.4f}")
print(f"  Late (t=100): sensor 0 = {Y_orig[100, 0]:.4f}")
print(f"  Final: sensor 0 = {Y_orig[-1, 0]:.4f}")

# For JAX, need to account for different time resolution
jax_t10 = int(10 * dt_orig / dt_jax)
jax_t100 = min(int(100 * dt_orig / dt_jax), len(Y_jax)-1)
print(f"\nJAX (explicit):")
print(f"  Early (t~10): sensor 0 = {Y_jax[min(jax_t10, len(Y_jax)-1), 0]:.4f}")
print(f"  Late (t~100): sensor 0 = {Y_jax[min(jax_t100, len(Y_jax)-1), 0]:.4f}")
print(f"  Final: sensor 0 = {Y_jax[-1, 0]:.4f}")

print("\n=== ROOT CAUSE ANALYSIS ===")
print("The explicit Euler scheme:")
print(f"1. Requires dt < {solver_jax['dt_max']:.6f} for stability")
print(f"2. Original dt={dt_orig:.4f} is {dt_orig/solver_jax['dt_max']:.1f}x larger than stable limit")
print(f"3. Using smaller dt means {nt_jax} timesteps vs {nt_orig} original")
print(f"4. Running fewer timesteps (for speed) means shorter simulation time")
print()
print("Key insight: JAX explicit solver CANNOT match the implicit solver's accuracy")
print("without running many more timesteps (4x more), which defeats the speed advantage.")
print()
print("CONCLUSION: This approach likely won't work for this problem.")
print("The implicit ADI method is essential for efficiency.")
