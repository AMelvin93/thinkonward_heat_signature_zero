"""
Optimized JAX Heat Simulator - Memory Efficient.

Key optimization: Sample sensors DURING the simulation loop,
not after storing all temperature fields.

Memory reduction: O(nt * nx * ny) → O(nt * n_sensors)
For nt=500, nx=100, ny=50, n_sensors=20:
  Before: 2.5M floats stored
  After: 10K floats stored (250x less memory)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Tuple
import numpy as np


@jit
def thomas_algorithm(lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve tridiagonal system using Thomas algorithm."""
    n = diag.shape[0]

    def forward_step(carry, idx):
        c_prev, d_prev = carry
        denom = diag[idx] - lower[idx] * c_prev
        denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
        c_new = upper[idx] / denom
        d_new = (rhs[idx] - lower[idx] * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    denom0 = jnp.where(jnp.abs(diag[0]) < 1e-12, 1e-12, diag[0])
    c0 = upper[0] / denom0
    d0 = rhs[0] / denom0

    _, (c_arr, d_arr) = jax.lax.scan(forward_step, (c0, d0), jnp.arange(1, n))

    c_full = jnp.concatenate([jnp.array([c0]), c_arr])
    d_full = jnp.concatenate([jnp.array([d0]), d_arr])

    def backward_step(x_next, idx):
        x_curr = d_full[idx] - c_full[idx] * x_next
        return x_curr, x_curr

    x_last = d_full[-1]
    _, x_rev = jax.lax.scan(backward_step, x_last, jnp.arange(n-2, -1, -1))
    x = jnp.concatenate([x_rev[::-1], jnp.array([x_last])])

    return x


def adi_step_fast(
    U: jnp.ndarray,
    S: jnp.ndarray,
    nx: int,
    ny: int,
    rx: float,
    ry: float,
    r: float,
    dt: float,
    bc: str,
    main_x: jnp.ndarray,
    lower_x: jnp.ndarray,
    upper_x: jnp.ndarray,
    main_y: jnp.ndarray,
    lower_y: jnp.ndarray,
    upper_y: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """ADI step with precomputed coefficients."""
    S_half = S * dt / 2.0

    # Step 1: Implicit in X, Explicit in Y
    if bc == 'dirichlet':
        U_padded_y = jnp.pad(U, ((0, 0), (1, 1)), mode='constant', constant_values=0.0)
    else:
        U_padded_y = jnp.pad(U, ((0, 0), (1, 1)), mode='edge')
    Lyy_U = (U_padded_y[:, 2:] - 2*U + U_padded_y[:, :-2]) / (dy * dy)
    RHS_x = U + r * Lyy_U + S_half

    if bc == 'dirichlet':
        RHS_x = RHS_x.at[0, :].set(0.0)
        RHS_x = RHS_x.at[-1, :].set(0.0)

    def solve_column(rhs_col):
        return thomas_algorithm(lower_x, main_x, upper_x, rhs_col)

    U_star = vmap(solve_column, in_axes=1, out_axes=1)(RHS_x)

    # Step 2: Implicit in Y, Explicit in X
    if bc == 'dirichlet':
        U_padded_x = jnp.pad(U_star, ((1, 1), (0, 0)), mode='constant', constant_values=0.0)
    else:
        U_padded_x = jnp.pad(U_star, ((1, 1), (0, 0)), mode='edge')
    Lxx_U = (U_padded_x[2:, :] - 2*U_star + U_padded_x[:-2, :]) / (dx * dx)
    RHS_y = U_star + r * Lxx_U + S_half

    if bc == 'dirichlet':
        RHS_y = RHS_y.at[:, 0].set(0.0)
        RHS_y = RHS_y.at[:, -1].set(0.0)

    def solve_row(rhs_row):
        return thomas_algorithm(lower_y, main_y, upper_y, rhs_row)

    U_next = vmap(solve_row, in_axes=0, out_axes=0)(RHS_y)

    return U_next


@partial(jit, static_argnums=(2, 3))
def sample_sensors_fast(
    U: jnp.ndarray,
    sensors_xy: jnp.ndarray,
    Lx: float,
    Ly: float,
) -> jnp.ndarray:
    """Sample temperature at sensor locations."""
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


def create_simulate_and_sample_for_nt(nt: int, nx: int, ny: int, Lx: float, Ly: float, bc: str):
    """
    Create a JIT-compiled simulate_and_sample function for a specific nt.

    This avoids recompilation by creating one function per unique nt value.
    Cache these functions for each nt encountered.
    """

    @jit
    def simulate_and_sample_fixed_nt(
        sources: jnp.ndarray,
        kappa: float,
        dt: float,
        sensors_xy: jnp.ndarray,
        T0: float,
    ) -> jnp.ndarray:
        """
        Simulate and sample in one pass - no full field storage.

        Returns: Y_pred (nt+1, n_sensors)
        """
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)

        # Precompute ADI coefficients
        rx = kappa * dt / (2.0 * dx * dx)
        ry = kappa * dt / (2.0 * dy * dy)
        r = kappa * dt / 2.0

        # X-direction coefficients
        main_x = (1 + 2*rx) * jnp.ones(nx)
        lower_x = -rx * jnp.ones(nx)
        upper_x = -rx * jnp.ones(nx)

        if bc == 'dirichlet':
            main_x = main_x.at[0].set(1.0)
            main_x = main_x.at[-1].set(1.0)
            lower_x = lower_x.at[0].set(0.0)
            lower_x = lower_x.at[-1].set(0.0)
            upper_x = upper_x.at[0].set(0.0)
            upper_x = upper_x.at[-1].set(0.0)
        elif bc == 'neumann':
            main_x = main_x.at[0].set(1 + rx)
            main_x = main_x.at[-1].set(1 + rx)

        # Y-direction coefficients
        main_y = (1 + 2*ry) * jnp.ones(ny)
        lower_y = -ry * jnp.ones(ny)
        upper_y = -ry * jnp.ones(ny)

        if bc == 'dirichlet':
            main_y = main_y.at[0].set(1.0)
            main_y = main_y.at[-1].set(1.0)
            lower_y = lower_y.at[0].set(0.0)
            lower_y = lower_y.at[-1].set(0.0)
            upper_y = upper_y.at[0].set(0.0)
            upper_y = upper_y.at[-1].set(0.0)
        elif bc == 'neumann':
            main_y = main_y.at[0].set(1 + ry)
            main_y = main_y.at[-1].set(1 + ry)

        # Grid and source field
        x = jnp.linspace(0, Lx, nx)
        y = jnp.linspace(0, Ly, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        max_ds = max(dx, dy)
        sigma = 2.5 * max_ds

        # Compute source field
        S = jnp.zeros((nx, ny))
        def add_source(S, source):
            x0, y0, q = source[0], source[1], source[2]
            r_sq = (X - x0)**2 + (Y - y0)**2
            G = jnp.exp(-r_sq / (2.0 * sigma**2))
            integral_G = jnp.maximum(jnp.sum(G) * dx * dy, 1e-10)
            return S + G * q / integral_G, None
        S, _ = jax.lax.scan(add_source, S, sources)

        # Initial state
        U = jnp.full((nx, ny), T0)
        Y0 = sample_sensors_fast(U, sensors_xy, Lx, Ly)

        # Time stepping with sensor sampling
        def step_and_sample(U, _):
            U_next = adi_step_fast(
                U, S, nx, ny, rx, ry, r, dt, bc,
                main_x, lower_x, upper_x,
                main_y, lower_y, upper_y,
                dx, dy
            )
            Y_t = sample_sensors_fast(U_next, sensors_xy, Lx, Ly)
            return U_next, Y_t

        _, Y_rest = jax.lax.scan(step_and_sample, U, None, length=nt)

        # Combine initial + all timesteps
        Y_pred = jnp.concatenate([Y0[None, :], Y_rest], axis=0)

        return Y_pred

    return simulate_and_sample_fixed_nt


# Cache of compiled functions per (nt, bc) combination
_COMPILED_CACHE = {}


def get_simulate_fn(nt: int, nx: int, ny: int, Lx: float, Ly: float, bc: str):
    """Get or create compiled simulation function for given nt."""
    key = (nt, nx, ny, Lx, Ly, bc)
    if key not in _COMPILED_CACHE:
        _COMPILED_CACHE[key] = create_simulate_and_sample_for_nt(nt, nx, ny, Lx, Ly, bc)
    return _COMPILED_CACHE[key]


def simulate_and_sample_fast(
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
    Fast simulate and sample - caches compiled functions per nt.

    Returns: Y_pred (nt+1, n_sensors)
    """
    fn = get_simulate_fn(nt, nx, ny, Lx, Ly, bc)
    return fn(sources, kappa, dt, sensors_xy, T0)


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
