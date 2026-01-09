"""
Pure JAX Optimizer - No scipy, fully JIT-compiled.

Eliminates Python callback overhead by keeping entire optimization in JAX.
Uses Adam optimizer with fixed number of steps.

Key insight: scipy.optimize calls back to Python for each evaluation,
which adds significant overhead. Pure JAX stays on GPU.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from functools import partial

try:
    from .jax_simulator_fast import check_gpu
except ImportError:
    from jax_simulator_fast import check_gpu


@dataclass
class CandidateResult:
    params: np.ndarray
    rmse: float
    init_type: str


@jit
def thomas_solve(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal systems."""
    n = diag.shape[0]

    def forward_step(carry, idx):
        c_prev, d_prev = carry
        denom = jnp.maximum(jnp.abs(diag[idx] - lower[idx] * c_prev), 1e-12)
        denom = jnp.sign(diag[idx] - lower[idx] * c_prev + 1e-12) * denom
        c_new = upper[idx] / denom
        d_new = (rhs[idx] - lower[idx] * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    denom0 = jnp.where(jnp.abs(diag[0]) < 1e-12, 1e-12, diag[0])
    c0, d0 = upper[0] / denom0, rhs[0] / denom0

    _, (c_arr, d_arr) = jax.lax.scan(forward_step, (c0, d0), jnp.arange(1, n))
    c_full = jnp.concatenate([jnp.array([c0]), c_arr])
    d_full = jnp.concatenate([jnp.array([d0]), d_arr])

    def backward_step(x_next, idx):
        x_curr = d_full[idx] - c_full[idx] * x_next
        return x_curr, x_curr

    _, x_rev = jax.lax.scan(backward_step, d_full[-1], jnp.arange(n-2, -1, -1))
    return jnp.concatenate([x_rev[::-1], jnp.array([d_full[-1]])])


def create_simulator_and_optimizer(nt: int, bc: str, nx: int = 100, ny: int = 50):
    """
    Create a fully JIT-compiled simulator + optimizer for a specific nt.

    Returns functions that run entirely on GPU without Python callbacks.
    """
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    @partial(jit, static_argnums=())
    def simulate(sources, kappa, dt, sensors_xy, T0):
        """Forward simulation with sensor sampling."""
        rx = kappa * dt / (2.0 * dx * dx)
        ry = kappa * dt / (2.0 * dy * dy)
        r = kappa * dt / 2.0

        # Precompute tridiagonal coefficients
        main_x = (1 + 2*rx) * jnp.ones(nx)
        lower_x = -rx * jnp.ones(nx)
        upper_x = -rx * jnp.ones(nx)
        main_y = (1 + 2*ry) * jnp.ones(ny)
        lower_y = -ry * jnp.ones(ny)
        upper_y = -ry * jnp.ones(ny)

        if bc == 'dirichlet':
            main_x = main_x.at[0].set(1.0).at[-1].set(1.0)
            lower_x = lower_x.at[0].set(0.0).at[-1].set(0.0)
            upper_x = upper_x.at[0].set(0.0).at[-1].set(0.0)
            main_y = main_y.at[0].set(1.0).at[-1].set(1.0)
            lower_y = lower_y.at[0].set(0.0).at[-1].set(0.0)
            upper_y = upper_y.at[0].set(0.0).at[-1].set(0.0)

        x = jnp.linspace(0, Lx, nx)
        y = jnp.linspace(0, Ly, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        sigma = 2.5 * max(dx, dy)
        S = jnp.zeros((nx, ny))

        def add_source(S, source):
            x0, y0, q = source
            r_sq = (X - x0)**2 + (Y - y0)**2
            G = jnp.exp(-r_sq / (2.0 * sigma**2))
            integral_G = jnp.maximum(jnp.sum(G) * dx * dy, 1e-10)
            return S + G * q / integral_G, None
        S, _ = jax.lax.scan(add_source, S, sources)

        def sample_sensors(U):
            def sample_one(sensor):
                sx, sy = sensor
                fx, fy = sx / dx, sy / dy
                ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
                iy = jnp.clip(jnp.floor(fy).astype(int), 0, ny - 2)
                tx, ty = fx - ix, fy - iy
                return ((1-tx)*(1-ty)*U[ix,iy] + tx*(1-ty)*U[ix+1,iy] +
                        (1-tx)*ty*U[ix,iy+1] + tx*ty*U[ix+1,iy+1])
            return vmap(sample_one)(sensors_xy)

        U = jnp.full((nx, ny), T0)

        def adi_step(U, _):
            S_half = S * dt / 2.0

            # Step 1: Implicit X
            if bc == 'dirichlet':
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), constant_values=0.0)
            else:
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), mode='edge')
            Lyy = (U_pad_y[:,2:] - 2*U + U_pad_y[:,:-2]) / (dy**2)
            RHS_x = U + r * Lyy + S_half
            if bc == 'dirichlet':
                RHS_x = RHS_x.at[0,:].set(0.0).at[-1,:].set(0.0)
            U_star = vmap(lambda col: thomas_solve(lower_x, main_x, upper_x, col),
                         in_axes=1, out_axes=1)(RHS_x)

            # Step 2: Implicit Y
            if bc == 'dirichlet':
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), constant_values=0.0)
            else:
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), mode='edge')
            Lxx = (U_pad_x[2:,:] - 2*U_star + U_pad_x[:-2,:]) / (dx**2)
            RHS_y = U_star + r * Lxx + S_half
            if bc == 'dirichlet':
                RHS_y = RHS_y.at[:,0].set(0.0).at[:,-1].set(0.0)
            U_next = vmap(lambda row: thomas_solve(lower_y, main_y, upper_y, row),
                         in_axes=0, out_axes=0)(RHS_y)

            return U_next, sample_sensors(U_next)

        Y0 = sample_sensors(U)
        _, Y_rest = jax.lax.scan(adi_step, U, None, length=nt)
        return jnp.concatenate([Y0[None,:], Y_rest], axis=0)

    @jit
    def loss_fn(params, n_sources, kappa, dt, sensors_xy, T0, Y_obs):
        """MSE loss function."""
        sources = params.reshape(n_sources, 3)
        Y_pred = simulate(sources, kappa, dt, sensors_xy, T0)
        return jnp.mean((Y_pred - Y_obs) ** 2)

    @jit
    def adam_optimize(params_init, n_sources, kappa, dt, sensors_xy, T0, Y_obs,
                      bounds_low, bounds_high, n_steps=50, lr=0.05):
        """
        Pure JAX Adam optimizer - runs entirely on GPU.

        No Python callbacks, no scipy overhead.
        """
        # Adam state
        m = jnp.zeros_like(params_init)
        v = jnp.zeros_like(params_init)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def adam_step(carry, t):
            params, m, v = carry

            # Compute gradient
            loss, grads = value_and_grad(loss_fn)(params, n_sources, kappa, dt, sensors_xy, T0, Y_obs)

            # Adam update
            m_new = beta1 * m + (1 - beta1) * grads
            v_new = beta2 * v + (1 - beta2) * (grads ** 2)

            m_hat = m_new / (1 - beta1 ** (t + 1))
            v_hat = v_new / (1 - beta2 ** (t + 1))

            params_new = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)

            # Clip to bounds
            params_new = jnp.clip(params_new, bounds_low, bounds_high)

            return (params_new, m_new, v_new), loss

        # Run optimization
        (final_params, _, _), losses = jax.lax.scan(
            adam_step, (params_init, m, v), jnp.arange(n_steps)
        )

        final_loss = loss_fn(final_params, n_sources, kappa, dt, sensors_xy, T0, Y_obs)
        return final_params, final_loss

    return simulate, adam_optimize


class PureJAXOptimizer:
    """
    Optimizer using pure JAX - no scipy, no Python callbacks.

    All optimization runs on GPU without leaving JAX.
    """

    def __init__(self, Lx: float = 2.0, Ly: float = 1.0, nx: int = 100, ny: int = 50):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.device_info = check_gpu()
        self._cache = {}

    def _get_bounds(self, n_sources, q_range, margin=0.05):
        bounds_low = []
        bounds_high = []
        for _ in range(n_sources):
            bounds_low.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            bounds_high.extend([(1-margin) * self.Lx, (1-margin) * self.Ly, q_range[1]])
        return jnp.array(bounds_low), jnp.array(bounds_high)

    def _smart_init(self, sample, n_sources, q_range):
        """Smart initialization from hottest sensors."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[p]) >= 0.2 for p in selected):
                selected.append(idx)
        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        params = []
        max_temp = np.max(avg_temps) + 1e-8
        for idx in selected:
            x, y = sensors[idx]
            q = 0.5 + (avg_temps[idx] / max_temp) * 1.5
            params.extend([x, y, np.clip(q, q_range[0], q_range[1])])
        return jnp.array(params)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        n_steps: int = 50,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate sources using pure JAX Adam optimizer.
        """
        n_sources = sample['n_sources']
        sensors_xy = jnp.array(sample['sensors_xy'])
        Y_obs = jnp.array(sample['Y_noisy'])

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        bounds_low, bounds_high = self._get_bounds(n_sources, q_range)

        # Get or create optimizer for this nt
        cache_key = (nt, bc)
        if cache_key not in self._cache:
            if verbose:
                print(f"  Creating optimizer for nt={nt}, bc={bc}...")
            self._cache[cache_key] = create_simulator_and_optimizer(nt, bc, self.nx, self.ny)

        simulate, adam_optimize = self._cache[cache_key]

        # Smart initialization
        params_init = self._smart_init(sample, n_sources, q_range)

        if verbose:
            print(f"  Running {n_steps} Adam steps...")

        # Run pure JAX optimization
        final_params, final_mse = adam_optimize(
            params_init, n_sources, kappa, dt, sensors_xy, T0, Y_obs,
            bounds_low, bounds_high, n_steps=n_steps, lr=lr
        )

        final_rmse = float(jnp.sqrt(final_mse))

        if verbose:
            print(f"  Final RMSE: {final_rmse:.4f}")

        # Convert to output format
        sources = []
        final_params = np.array(final_params)
        for i in range(n_sources):
            x, y, q = final_params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        return sources, final_rmse, [CandidateResult(final_params, final_rmse, 'adam')]
