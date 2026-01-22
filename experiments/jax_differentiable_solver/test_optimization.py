"""
Test if JAX gradients can actually optimize heat source location.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

from jax_heat_solver import create_jax_solver
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from simulator import Heat2D

# Load test data
import pickle
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['samples'][0]
meta = data['meta']

print(f"Sample 0: n_sources={sample['n_sources']}")

# Problem parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
kappa = sample['sample_metadata']['kappa']
dt_orig = meta['dt']
nt_orig = sample['sample_metadata']['nt']
T0 = sample['sample_metadata']['T0']
bc = sample['sample_metadata']['bc']

# Create original simulator for reference
print("\n=== Creating solvers ===")
solver_orig = Heat2D(Lx, Ly, nx, ny, kappa, bc)

# Create JAX solver
solver_jax = create_jax_solver(Lx, Ly, nx, ny, kappa)

# Stability constraint
dt_max = solver_jax['dt_max']
dt_explicit = dt_max * 0.5
t_total = dt_orig * nt_orig
nt_explicit = min(int(t_total / dt_explicit), 500)  # Cap at 500 for speed

print(f"Using nt_explicit={nt_explicit} for testing")

# Prepare data
sensors_xy = np.array(sample['sensors_xy'])
Y_observed = sample['Y_noisy']
sigma = 2.5 * max(solver_jax['dx'], solver_jax['dy'])
q_range = meta['q_range']

# Find hottest sensor as initial guess
avg_temps = np.mean(Y_observed, axis=0)
hot_idx = np.argmax(avg_temps)
x_init, y_init = sensors_xy[hot_idx]

print(f"\n=== Optimization test (1-source) ===")
print(f"Initial guess from hottest sensor: ({x_init:.3f}, {y_init:.3f})")

# Convert to JAX arrays
sensors_jax = jnp.array(sensors_xy)
Y_jax = jnp.array(Y_observed[:nt_explicit])

# Objective function for scipy.optimize
n_calls = [0]

def objective(params):
    """Objective for scipy.optimize - returns RMSE and gradient."""
    n_calls[0] += 1
    x, y = params
    # Fix q=1.0 for now (can compute optimal q analytically later)
    source_params = jnp.array([x, y, 1.0])

    rmse, grads = solver_jax['rmse_and_grad'](
        source_params, Y_jax, sensors_jax, T0, sigma, nt_explicit, dt_explicit
    )

    # Return only position gradients
    return float(rmse), np.array([float(grads[0]), float(grads[1])])

# Test gradient
print("\nTesting gradient at initial position...")
rmse0, grad0 = objective([x_init, y_init])
print(f"Initial RMSE: {rmse0:.6f}")
print(f"Initial gradient: [{grad0[0]:.8f}, {grad0[1]:.8f}]")

# Run L-BFGS-B optimization
print("\n--- Running L-BFGS-B optimization ---")
t0 = time.time()

bounds = [(0.1, 1.9), (0.1, 0.9)]  # Keep away from boundaries

result = minimize(
    lambda p: objective(p)[0],  # Only return value
    [x_init, y_init],
    method='L-BFGS-B',
    jac=lambda p: objective(p)[1],  # Return gradient
    bounds=bounds,
    options={'maxiter': 20, 'disp': True}
)

t_opt = time.time() - t0
print(f"\nOptimization took {t_opt:.2f}s, {n_calls[0]} function calls")
print(f"Final position: ({result.x[0]:.4f}, {result.x[1]:.4f})")
print(f"Final RMSE: {result.fun:.6f}")

# Compare with CMA-ES result
print("\n--- Comparing with baseline CMA-ES ---")
# Run a single CMA-ES optimization using the original simulator
import cma

def objective_cmaes(params):
    """Objective for CMA-ES."""
    x, y = params
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver_orig.solve(dt_orig, nt_orig, T0=T0, sources=sources)
    Y_pred = np.array([solver_orig.sample_sensors(U, sensors_xy) for U in Us])
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return rmse

t0 = time.time()
es = cma.CMAEvolutionStrategy(
    [x_init, y_init],
    0.2,
    {'bounds': [[0.05, 0.05], [1.95, 0.95]], 'maxfevals': 20, 'verbose': -9}
)
while not es.stop():
    solutions = es.ask()
    fitness = [objective_cmaes(s) for s in solutions]
    es.tell(solutions, fitness)

t_cmaes = time.time() - t0
best_cmaes = es.best.x

print(f"CMA-ES took {t_cmaes:.2f}s")
print(f"CMA-ES best position: ({best_cmaes[0]:.4f}, {best_cmaes[1]:.4f})")
print(f"CMA-ES best RMSE: {es.best.f:.6f}")

# Summary
print("\n=== SUMMARY ===")
print(f"JAX L-BFGS-B: {t_opt:.2f}s, RMSE={result.fun:.6f}")
print(f"CMA-ES:       {t_cmaes:.2f}s, RMSE={es.best.f:.6f}")

if t_opt < t_cmaes:
    print(f"JAX is {t_cmaes/t_opt:.1f}x FASTER!")
else:
    print(f"CMA-ES is {t_opt/t_cmaes:.1f}x faster")

if result.fun < es.best.f:
    print("JAX achieves BETTER accuracy!")
elif result.fun > es.best.f * 1.1:
    print("JAX achieves WORSE accuracy")
else:
    print("Similar accuracy")
