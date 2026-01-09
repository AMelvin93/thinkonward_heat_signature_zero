"""
JAX-based Heat2D Simulator for GPU acceleration and automatic differentiation.

This module provides a JAX implementation of the heat equation solver,
enabling:
1. GPU acceleration (if CUDA available)
2. Automatic differentiation for gradient-based optimization
3. JIT compilation for speed
4. Vectorized batch simulations
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
from typing import Tuple, Dict, List, Optional
import numpy as np


@jit
def thomas_algorithm(lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm.

    The system is: diag[i]*x[i] + upper[i]*x[i+1] + lower[i]*x[i-1] = rhs[i]

    Args:
        lower: Lower diagonal (n,) - lower[0] is unused
        diag: Main diagonal (n,)
        upper: Upper diagonal (n,) - upper[n-1] is unused
        rhs: Right-hand side (n,)

    Returns:
        Solution x (n,)
    """
    n = diag.shape[0]

    # Forward elimination using scan
    def forward_step(carry, idx):
        c_prev, d_prev = carry

        # w = lower[idx] / diag[idx-1] modified
        # But we work with modified coefficients
        denom = diag[idx] - lower[idx] * c_prev
        denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)

        c_new = upper[idx] / denom
        d_new = (rhs[idx] - lower[idx] * d_prev) / denom

        return (c_new, d_new), (c_new, d_new)

    # Initialize with first row
    denom0 = jnp.where(jnp.abs(diag[0]) < 1e-12, 1e-12, diag[0])
    c0 = upper[0] / denom0
    d0 = rhs[0] / denom0

    # Forward sweep for indices 1 to n-1
    _, (c_arr, d_arr) = jax.lax.scan(
        forward_step,
        (c0, d0),
        jnp.arange(1, n)
    )

    # Prepend initial values
    c_full = jnp.concatenate([jnp.array([c0]), c_arr])
    d_full = jnp.concatenate([jnp.array([d0]), d_arr])

    # Back substitution using scan (reverse)
    def backward_step(x_next, idx):
        x_curr = d_full[idx] - c_full[idx] * x_next
        return x_curr, x_curr

    # Start from last element
    x_last = d_full[-1]

    # Backward sweep for indices n-2 down to 0
    _, x_rev = jax.lax.scan(
        backward_step,
        x_last,
        jnp.arange(n-2, -1, -1)
    )

    # Reverse and add last element
    x = jnp.concatenate([x_rev[::-1], jnp.array([x_last])])

    return x


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def gaussian_source_field(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    sources: jnp.ndarray,  # (n_sources, 3) array of [x, y, q]
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    sigma: float,
    t: float,
) -> jnp.ndarray:
    """
    Compute Gaussian source field from multiple sources.

    Args:
        X, Y: Meshgrid arrays (nx, ny)
        sources: Array of shape (n_sources, 3) with [x, y, q] per source
        sigma: Gaussian spread parameter
        t: Current time (unused, sources always on)

    Returns:
        Source field S of shape (nx, ny)
    """
    S = jnp.zeros((nx, ny))

    def add_source(S, source):
        x0, y0, q = source[0], source[1], source[2]
        r_sq = (X - x0)**2 + (Y - y0)**2
        G = jnp.exp(-r_sq / (2.0 * sigma**2))
        integral_G = jnp.sum(G) * dx * dy
        # Avoid division by zero
        integral_G = jnp.maximum(integral_G, 1e-10)
        return S + G * q / integral_G, None

    S, _ = jax.lax.scan(add_source, S, sources)
    return S


def adi_step(
    U: jnp.ndarray,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    kappa: float,
    dt: float,
    bc: str,
    S: jnp.ndarray,  # Source field
) -> jnp.ndarray:
    """
    Perform one ADI time step.

    Alternating Direction Implicit method:
    1. Implicit in x, explicit in y (half step)
    2. Implicit in y, explicit in x (half step)
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    # rx, ry for implicit matrices (diagonals)
    rx = kappa * dt / (2.0 * dx * dx)
    ry = kappa * dt / (2.0 * dy * dy)
    # r for explicit RHS (Laplacian operator already has 1/d^2)
    r = kappa * dt / 2.0

    # Half the source contribution per half-step
    S_half = S * dt / 2.0

    # === Step 1: Implicit in X, Explicit in Y ===
    # Build RHS: U + r * Lyy(U) + S_half
    # Lyy operator (second derivative in y) - includes 1/dy^2
    if bc == 'dirichlet':
        # For Dirichlet BC, use zero padding
        U_padded_y = jnp.pad(U, ((0, 0), (1, 1)), mode='constant', constant_values=0.0)
    else:
        U_padded_y = jnp.pad(U, ((0, 0), (1, 1)), mode='edge')
    Lyy_U = (U_padded_y[:, 2:] - 2*U + U_padded_y[:, :-2]) / (dy * dy)
    RHS_x = U + r * Lyy_U + S_half

    # Handle Dirichlet BC
    if bc == 'dirichlet':
        RHS_x = RHS_x.at[0, :].set(0.0)
        RHS_x = RHS_x.at[-1, :].set(0.0)

    # Build tridiagonal coefficients for x direction
    main_x = (1 + 2*rx) * jnp.ones(nx)
    lower_x = -rx * jnp.ones(nx)
    upper_x = -rx * jnp.ones(nx)

    if bc == 'dirichlet':
        # Boundary rows: identity only (no coupling to neighbors)
        main_x = main_x.at[0].set(1.0)
        main_x = main_x.at[-1].set(1.0)
        lower_x = lower_x.at[0].set(0.0)
        lower_x = lower_x.at[-1].set(0.0)  # row n-1 doesn't couple to n-2
        upper_x = upper_x.at[0].set(0.0)   # row 0 doesn't couple to 1
        upper_x = upper_x.at[-1].set(0.0)
    elif bc == 'neumann':
        main_x = main_x.at[0].set(1 + rx)
        main_x = main_x.at[-1].set(1 + rx)

    # Solve for each column using Thomas algorithm
    def solve_column(rhs_col):
        return thomas_algorithm(lower_x, main_x, upper_x, rhs_col)

    U_star = vmap(solve_column, in_axes=1, out_axes=1)(RHS_x)

    # === Step 2: Implicit in Y, Explicit in X ===
    # Build RHS: U_star + r * Lxx(U_star) + S_half
    if bc == 'dirichlet':
        U_padded_x = jnp.pad(U_star, ((1, 1), (0, 0)), mode='constant', constant_values=0.0)
    else:
        U_padded_x = jnp.pad(U_star, ((1, 1), (0, 0)), mode='edge')
    Lxx_U = (U_padded_x[2:, :] - 2*U_star + U_padded_x[:-2, :]) / (dx * dx)
    RHS_y = U_star + r * Lxx_U + S_half

    if bc == 'dirichlet':
        RHS_y = RHS_y.at[:, 0].set(0.0)
        RHS_y = RHS_y.at[:, -1].set(0.0)

    # Build tridiagonal coefficients for y direction
    main_y = (1 + 2*ry) * jnp.ones(ny)
    lower_y = -ry * jnp.ones(ny)
    upper_y = -ry * jnp.ones(ny)

    if bc == 'dirichlet':
        # Boundary rows: identity only (no coupling to neighbors)
        main_y = main_y.at[0].set(1.0)
        main_y = main_y.at[-1].set(1.0)
        lower_y = lower_y.at[0].set(0.0)
        lower_y = lower_y.at[-1].set(0.0)
        upper_y = upper_y.at[0].set(0.0)
        upper_y = upper_y.at[-1].set(0.0)
    elif bc == 'neumann':
        main_y = main_y.at[0].set(1 + ry)
        main_y = main_y.at[-1].set(1 + ry)

    # Solve for each row using Thomas algorithm
    def solve_row(rhs_row):
        return thomas_algorithm(lower_y, main_y, upper_y, rhs_row)

    U_next = vmap(solve_row, in_axes=0, out_axes=0)(RHS_y)

    return U_next


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10))
def simulate_heat(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    kappa: float,
    dt: float,
    nt: int,
    sources: jnp.ndarray,  # (n_sources, 3)
    bc: str,
    T0: float = 0.0,
    store_every: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate heat diffusion using ADI method.

    Args:
        nx, ny: Grid dimensions
        Lx, Ly: Domain size
        kappa: Thermal diffusivity
        dt: Time step
        nt: Number of time steps
        sources: Array (n_sources, 3) of [x, y, q]
        bc: Boundary condition ('dirichlet' or 'neumann')
        T0: Initial temperature
        store_every: Store solution every N steps

    Returns:
        times: Array of time points
        Us: Array of temperature fields (n_stored, nx, ny)
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Grid
    x = jnp.linspace(0, Lx, nx)
    y = jnp.linspace(0, Ly, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Gaussian sigma
    max_ds = max(dx, dy)
    sigma = 2.5 * max_ds

    # Initial condition
    U = jnp.full((nx, ny), T0)

    # Compute source field (constant in time for this problem)
    S = gaussian_source_field(X, Y, sources, nx, ny, dx, dy, sigma, 0.0)

    # Time stepping with lax.scan for efficiency
    def step_fn(U, _):
        U_next = adi_step(U, nx, ny, Lx, Ly, kappa, dt, bc, S)
        return U_next, U_next

    # Run simulation
    _, Us_all = jax.lax.scan(step_fn, U, None, length=nt)

    # Prepend initial condition
    Us_all = jnp.concatenate([U[None, :, :], Us_all], axis=0)

    # Subsample if needed
    indices = jnp.arange(0, nt + 1, store_every)
    Us = Us_all[indices]
    times = indices * dt

    return times, Us


@partial(jit, static_argnums=(2, 3))
def sample_sensors(
    U: jnp.ndarray,
    sensors_xy: jnp.ndarray,
    Lx: float,
    Ly: float,
) -> jnp.ndarray:
    """
    Sample temperature field at sensor locations using bilinear interpolation.

    Args:
        U: Temperature field (nx, ny)
        sensors_xy: Sensor positions (n_sensors, 2)
        Lx, Ly: Domain size

    Returns:
        Temperature values at sensors (n_sensors,)
    """
    nx, ny = U.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    def sample_one(sensor):
        sx, sy = sensor[0], sensor[1]
        fx = sx / dx
        fy = sy / dy

        ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
        iy = jnp.clip(jnp.floor(fy).astype(int), 0, ny - 2)

        tx = fx - ix
        ty = fy - iy

        v = (1 - tx) * (1 - ty) * U[ix, iy] + \
            tx * (1 - ty) * U[ix + 1, iy] + \
            (1 - tx) * ty * U[ix, iy + 1] + \
            tx * ty * U[ix + 1, iy + 1]

        return v

    return vmap(sample_one)(sensors_xy)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9, 10))
def simulate_and_sample(
    sources: jnp.ndarray,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    kappa: float,
    dt: float,
    nt: int,
    sensors_xy: jnp.ndarray,
    bc: str,
    T0: float = 0.0,
) -> jnp.ndarray:
    """
    Simulate and return sensor readings.

    This is the main function for optimization - it takes source parameters
    and returns sensor readings, enabling gradient computation.

    Args:
        sources: Array (n_sources, 3) of [x, y, q]
        Other args: Simulation parameters

    Returns:
        Y_pred: Predicted sensor readings (nt+1, n_sensors)
    """
    times, Us = simulate_heat(nx, ny, Lx, Ly, kappa, dt, nt, sources, bc, T0, store_every=1)

    # Sample at each time step
    def sample_timestep(U):
        return sample_sensors(U, sensors_xy, Lx, Ly)

    Y_pred = vmap(sample_timestep)(Us)
    return Y_pred


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 10, 11))
def objective_rmse(
    sources: jnp.ndarray,
    Y_observed: jnp.ndarray,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    kappa: float,
    dt: float,
    nt: int,
    sensors_xy: jnp.ndarray,
    bc: str,
    T0: float = 0.0,
) -> float:
    """
    Compute RMSE objective for optimization.

    This function is differentiable with JAX!
    """
    Y_pred = simulate_and_sample(sources, nx, ny, Lx, Ly, kappa, dt, nt, sensors_xy, bc, T0)
    return jnp.sqrt(jnp.mean((Y_pred - Y_observed) ** 2))


# Create gradient function
def create_gradient_fn(nx, ny, Lx, Ly, kappa, dt, nt, sensors_xy, bc, T0):
    """Create a JIT-compiled gradient function for the objective."""

    @jit
    def grad_fn(sources, Y_observed):
        return grad(lambda s: objective_rmse(s, Y_observed, nx, ny, Lx, Ly, kappa, dt, nt, sensors_xy, bc, T0))(sources)

    return grad_fn


class JAXHeatSimulator:
    """
    Convenience wrapper for JAX-based heat simulation.

    Provides a similar interface to the original Heat2D class.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        # Pre-compute grid
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = jnp.linspace(0, Lx, nx)
        self.y = jnp.linspace(0, Ly, ny)

    def simulate(
        self,
        sources: List[Dict],
        kappa: float,
        dt: float,
        nt: int,
        bc: str = 'dirichlet',
        T0: float = 0.0,
        store_every: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation with given sources.

        Args:
            sources: List of dicts with 'x', 'y', 'q' keys
            Other args: Simulation parameters

        Returns:
            times, Us as numpy arrays
        """
        # Convert sources to JAX array
        sources_arr = jnp.array([[s['x'], s['y'], s['q']] for s in sources])

        times, Us = simulate_heat(
            self.nx, self.ny, self.Lx, self.Ly,
            kappa, dt, nt, sources_arr, bc, T0, store_every
        )

        return np.array(times), np.array(Us)

    def simulate_and_sample(
        self,
        sources: List[Dict],
        sensors_xy: np.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str = 'dirichlet',
        T0: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate and return sensor readings.
        """
        sources_arr = jnp.array([[s['x'], s['y'], s['q']] for s in sources])
        sensors_jax = jnp.array(sensors_xy)

        Y_pred = simulate_and_sample(
            sources_arr, self.nx, self.ny, self.Lx, self.Ly,
            kappa, dt, nt, sensors_jax, bc, T0
        )

        return np.array(Y_pred)

    def compute_objective(
        self,
        sources: List[Dict],
        Y_observed: np.ndarray,
        sensors_xy: np.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str = 'dirichlet',
        T0: float = 0.0,
    ) -> float:
        """Compute RMSE objective."""
        sources_arr = jnp.array([[s['x'], s['y'], s['q']] for s in sources])

        return float(objective_rmse(
            sources_arr, jnp.array(Y_observed),
            self.nx, self.ny, self.Lx, self.Ly,
            kappa, dt, nt, jnp.array(sensors_xy), bc, T0
        ))

    def compute_gradient(
        self,
        sources: List[Dict],
        Y_observed: np.ndarray,
        sensors_xy: np.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str = 'dirichlet',
        T0: float = 0.0,
    ) -> np.ndarray:
        """
        Compute gradient of RMSE w.r.t. source parameters.

        This is the key advantage of JAX - automatic differentiation!
        """
        sources_arr = jnp.array([[s['x'], s['y'], s['q']] for s in sources])
        sensors_jax = jnp.array(sensors_xy)
        Y_obs_jax = jnp.array(Y_observed)

        grad_fn = create_gradient_fn(
            self.nx, self.ny, self.Lx, self.Ly,
            kappa, dt, nt, sensors_jax, bc, T0
        )

        grads = grad_fn(sources_arr, Y_obs_jax)
        return np.array(grads)


def check_gpu():
    """Check if JAX is using GPU."""
    try:
        devices = jax.devices()
        gpu_available = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        return {
            'devices': [str(d) for d in devices],
            'gpu_available': gpu_available,
            'default_backend': jax.default_backend(),
        }
    except Exception as e:
        return {'error': str(e)}
