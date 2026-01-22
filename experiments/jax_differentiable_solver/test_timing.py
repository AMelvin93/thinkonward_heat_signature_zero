"""
Test JAX solver timing with realistic parameters.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import sys
sys.path.insert(0, '/workspace')

# Import JAX solver
from jax_heat_solver import create_jax_solver
import jax.numpy as jnp
import numpy as np

# Load test data
import pickle
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['samples'][0]
meta = data['meta']

# Problem parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50  # Full grid
kappa = sample['sample_metadata']['kappa']
dt_orig = meta['dt']
nt_orig = sample['sample_metadata']['nt']
T0 = sample['sample_metadata']['T0']

print(f"Problem parameters:")
print(f"  Grid: {nx}x{ny}")
print(f"  kappa: {kappa}")
print(f"  dt (original): {dt_orig}")
print(f"  nt (original): {nt_orig}")
print(f"  T0: {T0}")

# Create JAX solver
print("\nCreating JAX solver...")
solver = create_jax_solver(Lx, Ly, nx, ny, kappa)

# Stability constraint
dt_max = solver['dt_max']
print(f"Max stable dt for explicit method: {dt_max:.6f}")

# For explicit method, we need dt < dt_max
# Original uses dt=0.1, nt=200 (total time = 20.0)
# For stability, we need much smaller dt
t_total = dt_orig * nt_orig
dt_explicit = dt_max * 0.5  # Use half for safety
nt_explicit = int(t_total / dt_explicit)

print(f"Total simulation time: {t_total}")
print(f"Explicit method needs: dt={dt_explicit:.6f}, nt={nt_explicit}")
print(f"This is {nt_explicit / nt_orig:.1f}x more timesteps than implicit method")

# Prepare data
sensors_xy = jnp.array(sample['sensors_xy'])
Y_observed = jnp.array(sample['Y_noisy'][:nt_explicit])  # Truncate to match
sigma = 2.5 * max(solver['dx'], solver['dy'])  # Match original

# Initial guess
source_params = jnp.array([1.0, 0.5, 1.0])

# Compile and time
print("\n--- Timing test with nt=100 (subset) ---")
nt_test = 100

# First call compiles
print("Compiling (first call)...")
t0 = time.time()
rmse, grads = solver['rmse_and_grad'](source_params, Y_observed[:nt_test], sensors_xy, T0, sigma, nt_test, dt_explicit)
t_compile = time.time() - t0
print(f"Compile time: {t_compile:.2f}s")

# Subsequent calls are fast
print("\nTiming subsequent calls...")
times = []
for i in range(5):
    t0 = time.time()
    rmse, grads = solver['rmse_and_grad'](source_params, Y_observed[:nt_test], sensors_xy, T0, sigma, nt_test, dt_explicit)
    times.append(time.time() - t0)
    print(f"  Run {i+1}: {times[-1]:.4f}s, RMSE={float(rmse):.6f}")

avg_time = np.mean(times)
print(f"\nAverage time per gradient eval: {avg_time:.4f}s")

# Estimate for full simulation
print("\n--- Extrapolation to full simulation ---")
# Full simulation would need nt_explicit timesteps
# Time scales roughly linearly with nt
full_time_est = avg_time * (nt_explicit / nt_test)
print(f"Estimated time for nt={nt_explicit}: {full_time_est:.2f}s per gradient")

# For L-BFGS-B optimization:
# Typically needs 10-50 function evaluations
# Each eval = forward + gradient
n_iters = 30
total_opt_time = full_time_est * n_iters
print(f"Estimated optimization time ({n_iters} iters): {total_opt_time:.1f}s per sample")
print(f"For 80 samples: {total_opt_time * 80 / 60:.1f} min")

# Compare with baseline
print("\n--- Comparison with CMA-ES baseline ---")
print("CMA-ES baseline: ~44 sec per sample (58 min / 80 samples)")
print(f"JAX L-BFGS-B estimate: {total_opt_time:.1f} sec per sample")
if total_opt_time < 44:
    print("JAX could be FASTER!")
else:
    print(f"JAX would be {total_opt_time/44:.1f}x SLOWER")
