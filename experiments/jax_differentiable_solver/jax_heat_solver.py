"""
JAX-based differentiable heat equation solver.

Uses explicit time-stepping (forward Euler) for differentiability.
This trades some stability for automatic gradient computation.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial


def create_jax_solver(Lx, Ly, nx, ny, kappa):
    """
    Create a JAX-based heat equation solver.

    Returns functions that can be differentiated with jax.grad().
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = jnp.linspace(0, Lx, nx)
    y = jnp.linspace(0, Ly, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Stability constraint for explicit method: dt < dx^2 * dy^2 / (2 * kappa * (dx^2 + dy^2))
    dt_max = dx**2 * dy**2 / (2 * kappa * (dx**2 + dy**2))

    def gaussian_source(x0, y0, q, sigma):
        """Gaussian heat source at (x0, y0) with intensity q."""
        r_sq = (X - x0)**2 + (Y - y0)**2
        G = jnp.exp(-r_sq / (2.0 * sigma**2))
        integral = jnp.sum(G) * dx * dy
        return jnp.where(integral > 1e-10, G * q / integral, jnp.zeros_like(G))

    def laplacian(U):
        """Compute discrete Laplacian with zero boundary conditions."""
        # Pad with zeros for Dirichlet BC
        U_padded = jnp.pad(U, 1, mode='constant', constant_values=0)

        Uxx = (U_padded[2:, 1:-1] - 2*U_padded[1:-1, 1:-1] + U_padded[:-2, 1:-1]) / dx**2
        Uyy = (U_padded[1:-1, 2:] - 2*U_padded[1:-1, 1:-1] + U_padded[1:-1, :-2]) / dy**2

        return Uxx + Uyy

    def step_explicit(U, S, dt):
        """Single explicit Euler time step."""
        dU = kappa * laplacian(U) + S
        return U + dt * dU

    @partial(jit, static_argnums=(3, 4))
    def solve_forward(source_params, T0, sigma, nt, dt):
        """
        Solve heat equation forward in time.

        source_params: [x0, y0, q] for 1-source or [x0, y0, q0, x1, y1, q1] for 2-source
        Returns: temperature history at all timesteps
        """
        n_sources = len(source_params) // 3

        # Build source field
        S = jnp.zeros((nx, ny))
        for i in range(n_sources):
            x0 = source_params[i * 3]
            y0 = source_params[i * 3 + 1]
            q = source_params[i * 3 + 2]
            S = S + gaussian_source(x0, y0, q, sigma)

        # Initialize
        U = jnp.full((nx, ny), T0)

        # Time-step (using scan for efficiency)
        def body_fn(U, _):
            U_new = step_explicit(U, S, dt)
            return U_new, U_new

        _, U_history = jax.lax.scan(body_fn, U, None, length=nt)

        return U_history

    def interpolate_sensors(U_field, sensor_xy):
        """Bilinear interpolation at sensor locations."""
        n_sensors = sensor_xy.shape[0]

        def interp_one(sensor):
            sx, sy = sensor
            fx = sx / dx
            fy = sy / dy
            ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
            iy = jnp.clip(jnp.floor(fy).astype(int), 0, ny - 2)
            tx = fx - ix
            ty = fy - iy

            v = ((1 - tx) * (1 - ty) * U_field[ix, iy] +
                 tx * (1 - ty) * U_field[ix + 1, iy] +
                 (1 - tx) * ty * U_field[ix, iy + 1] +
                 tx * ty * U_field[ix + 1, iy + 1])
            return v

        return vmap(interp_one)(sensor_xy)

    @partial(jit, static_argnums=(5, 6))
    def compute_rmse(source_params, Y_observed, sensors_xy, T0, sigma, nt, dt):
        """
        Compute RMSE between simulated and observed sensor temperatures.

        This function is differentiable with jax.grad().
        """
        # Run simulation
        U_history = solve_forward(source_params, T0, sigma, nt, dt)

        # Sample at sensors for each timestep
        Y_simulated = vmap(lambda U: interpolate_sensors(U, sensors_xy))(U_history)

        # Compute RMSE (only using as many timesteps as observed)
        n_steps = min(len(U_history), len(Y_observed))
        diff = Y_simulated[:n_steps] - Y_observed[:n_steps]
        rmse = jnp.sqrt(jnp.mean(diff**2))

        return rmse

    # Gradient function for optimization
    @partial(jit, static_argnums=(5, 6))
    def rmse_and_grad(source_params, Y_observed, sensors_xy, T0, sigma, nt, dt):
        """Compute RMSE and its gradient w.r.t. source parameters."""
        grad_fn = grad(compute_rmse, argnums=0)
        rmse = compute_rmse(source_params, Y_observed, sensors_xy, T0, sigma, nt, dt)
        grads = grad_fn(source_params, Y_observed, sensors_xy, T0, sigma, nt, dt)
        return rmse, grads

    return {
        'solve_forward': solve_forward,
        'compute_rmse': compute_rmse,
        'rmse_and_grad': rmse_and_grad,
        'interpolate_sensors': interpolate_sensors,
        'dt_max': dt_max,
        'dx': dx,
        'dy': dy,
    }


if __name__ == '__main__':
    import numpy as np

    # Test basic functionality
    print("Testing JAX heat solver...")

    # Small grid for testing
    solver = create_jax_solver(Lx=2.0, Ly=1.0, nx=50, ny=25, kappa=0.1)

    print(f"Max stable dt: {solver['dt_max']:.6f}")

    # Test gradient computation
    source_params = jnp.array([1.0, 0.5, 1.0])  # x, y, q
    sensors_xy = jnp.array([[0.5, 0.3], [1.5, 0.7]])
    Y_observed = jnp.ones((10, 2)) * 0.1  # Dummy observed data

    dt = min(0.01, solver['dt_max'] * 0.9)
    nt = 10
    T0 = 0.0
    sigma = 0.1

    # Compute RMSE and gradient
    rmse, grads = solver['rmse_and_grad'](source_params, Y_observed, sensors_xy, T0, sigma, nt, dt)

    print(f"RMSE: {rmse:.6f}")
    print(f"Gradient: {grads}")
    print("JAX heat solver working!")
