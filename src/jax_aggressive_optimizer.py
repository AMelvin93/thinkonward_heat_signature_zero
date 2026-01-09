"""
Aggressive Coarse-to-Fine Optimizer.

ULTRA-aggressive settings to hit 9s/sample target:
- Coarse: 25x12 grid, 1/8 timesteps (~64x faster)
- Fine: Minimal refinement (3 iterations)
- Single smart initialization

Target: 400 samples in <60 min = 9s per sample
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, vmap

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
    """Fast Thomas algorithm for tridiagonal systems."""
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


def create_ultracoarse_simulator(nt_full: int, bc: str, time_subsample: int = 8):
    """
    Ultra-coarse simulator: 25x12 grid, 1/8 timesteps.
    ~64x faster than full resolution.
    """
    nx_c, ny_c = 25, 12
    Lx, Ly = 2.0, 1.0
    dx_c = Lx / (nx_c - 1)
    dy_c = Ly / (ny_c - 1)
    nt_coarse = max(1, nt_full // time_subsample)

    @jit
    def simulate(sources, kappa, dt, sensors_xy, T0):
        dt_eff = dt * time_subsample
        rx = kappa * dt_eff / (2.0 * dx_c * dx_c)
        ry = kappa * dt_eff / (2.0 * dy_c * dy_c)
        r = kappa * dt_eff / 2.0

        # Tridiagonal coefficients
        main_x = (1 + 2*rx) * jnp.ones(nx_c)
        lower_x = -rx * jnp.ones(nx_c)
        upper_x = -rx * jnp.ones(nx_c)
        main_y = (1 + 2*ry) * jnp.ones(ny_c)
        lower_y = -ry * jnp.ones(ny_c)
        upper_y = -ry * jnp.ones(ny_c)

        if bc == 'dirichlet':
            main_x = main_x.at[0].set(1.0).at[-1].set(1.0)
            lower_x = lower_x.at[0].set(0.0).at[-1].set(0.0)
            upper_x = upper_x.at[0].set(0.0).at[-1].set(0.0)
            main_y = main_y.at[0].set(1.0).at[-1].set(1.0)
            lower_y = lower_y.at[0].set(0.0).at[-1].set(0.0)
            upper_y = upper_y.at[0].set(0.0).at[-1].set(0.0)
        else:
            main_x = main_x.at[0].set(1 + rx).at[-1].set(1 + rx)
            main_y = main_y.at[0].set(1 + ry).at[-1].set(1 + ry)

        x = jnp.linspace(0, Lx, nx_c)
        y = jnp.linspace(0, Ly, ny_c)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        sigma = 2.5 * max(dx_c, dy_c)
        S = jnp.zeros((nx_c, ny_c))
        def add_source(S, source):
            x0, y0, q = source
            r_sq = (X - x0)**2 + (Y - y0)**2
            G = jnp.exp(-r_sq / (2.0 * sigma**2))
            integral_G = jnp.maximum(jnp.sum(G) * dx_c * dy_c, 1e-10)
            return S + G * q / integral_G, None
        S, _ = jax.lax.scan(add_source, S, sources)

        def sample_sensors(U):
            def sample_one(sensor):
                sx, sy = sensor
                fx, fy = sx / dx_c, sy / dy_c
                ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx_c - 2)
                iy = jnp.clip(jnp.floor(fy).astype(int), 0, ny_c - 2)
                tx, ty = fx - ix, fy - iy
                return ((1-tx)*(1-ty)*U[ix,iy] + tx*(1-ty)*U[ix+1,iy] +
                        (1-tx)*ty*U[ix,iy+1] + tx*ty*U[ix+1,iy+1])
            return vmap(sample_one)(sensors_xy)

        U = jnp.full((nx_c, ny_c), T0)

        def adi_step(U, _):
            S_half = S * dt_eff / 2.0
            if bc == 'dirichlet':
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), constant_values=0.0)
            else:
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), mode='edge')
            Lyy = (U_pad_y[:,2:] - 2*U + U_pad_y[:,:-2]) / (dy_c**2)
            RHS_x = U + r * Lyy + S_half
            if bc == 'dirichlet':
                RHS_x = RHS_x.at[0,:].set(0.0).at[-1,:].set(0.0)
            U_star = vmap(lambda col: thomas_solve(lower_x, main_x, upper_x, col),
                         in_axes=1, out_axes=1)(RHS_x)

            if bc == 'dirichlet':
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), constant_values=0.0)
            else:
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), mode='edge')
            Lxx = (U_pad_x[2:,:] - 2*U_star + U_pad_x[:-2,:]) / (dx_c**2)
            RHS_y = U_star + r * Lxx + S_half
            if bc == 'dirichlet':
                RHS_y = RHS_y.at[:,0].set(0.0).at[:,-1].set(0.0)
            U_next = vmap(lambda row: thomas_solve(lower_y, main_y, upper_y, row),
                         in_axes=0, out_axes=0)(RHS_y)

            return U_next, sample_sensors(U_next)

        Y0 = sample_sensors(U)
        _, Y_rest = jax.lax.scan(adi_step, U, None, length=nt_coarse)
        return jnp.concatenate([Y0[None,:], Y_rest], axis=0)

    return simulate, nt_coarse


def create_fine_simulator(nt: int, bc: str):
    """Full resolution simulator for final evaluation."""
    nx, ny = 100, 50
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    @jit
    def simulate(sources, kappa, dt, sensors_xy, T0):
        rx = kappa * dt / (2.0 * dx * dx)
        ry = kappa * dt / (2.0 * dy * dy)
        r = kappa * dt / 2.0

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
        else:
            main_x = main_x.at[0].set(1 + rx).at[-1].set(1 + rx)
            main_y = main_y.at[0].set(1 + ry).at[-1].set(1 + ry)

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

    return simulate


class AggressiveOptimizer:
    """
    Ultra-aggressive optimizer for speed.

    Strategy:
    1. Single smart init
    2. Ultra-coarse exploration (25x12, 1/8 time)
    3. Minimal fine refinement (3 iters)
    """

    def __init__(self, Lx: float = 2.0, Ly: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.device_info = check_gpu()
        self._coarse_cache = {}
        self._fine_cache = {}

    def _get_bounds(self, n_sources, q_range, margin=0.05):
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (margin * self.Lx, (1 - margin) * self.Lx),
                (margin * self.Ly, (1 - margin) * self.Ly),
                q_range
            ])
        return bounds

    def _smart_init(self, sample, n_sources, q_range):
        """Single smart initialization from hottest sensors."""
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
        return np.array(params)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        coarse_iters: int = 12,
        fine_iters: int = 3,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """Ultra-fast estimation."""
        n_sources = sample['n_sources']
        sensors_xy = jnp.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        bounds = self._get_bounds(n_sources, q_range)
        n_params = n_sources * 3

        # === COARSE PHASE ===
        if verbose:
            print("  Coarse (25x12, 1/8 time)...")

        coarse_key = (nt, bc)
        if coarse_key not in self._coarse_cache:
            self._coarse_cache[coarse_key] = create_ultracoarse_simulator(nt, bc, time_subsample=8)
        sim_coarse, nt_coarse = self._coarse_cache[coarse_key]

        # Subsample observations
        time_indices = np.arange(0, nt + 1, 8)[:nt_coarse + 1]
        Y_obs_coarse = jnp.array(Y_observed[time_indices])

        @jit
        def coarse_obj(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = sim_coarse(sources, kappa, dt, sensors_xy, T0)
            min_len = min(len(Y_pred), len(Y_obs_coarse))
            return jnp.mean((Y_pred[:min_len] - Y_obs_coarse[:min_len]) ** 2)

        # Warmup
        x0 = jnp.array(self._smart_init(sample, n_sources, q_range))
        _ = coarse_obj(x0)

        # Gradient via finite diff (forward only for speed)
        def coarse_scipy(params_np):
            params = jnp.array(params_np)
            loss = float(coarse_obj(params))
            eps = 1e-4
            grad = np.zeros(n_params)
            for i in range(n_params):
                params_p = params.at[i].add(eps)
                loss_p = float(coarse_obj(params_p))
                grad[i] = (loss_p - loss) / eps
            return loss, grad

        # Single optimization
        result = minimize(coarse_scipy, x0=np.array(x0), method='L-BFGS-B', jac=True,
                         bounds=bounds, options={'maxiter': coarse_iters})
        coarse_params = result.x

        if verbose:
            print(f"    Coarse MSE: {result.fun:.6f}")

        # === FINE PHASE (minimal) ===
        if verbose:
            print("  Fine (3 iters)...")

        fine_key = (nt, bc)
        if fine_key not in self._fine_cache:
            self._fine_cache[fine_key] = create_fine_simulator(nt, bc)
        sim_fine = self._fine_cache[fine_key]

        Y_obs_fine = jnp.array(Y_observed)

        @jit
        def fine_obj(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = sim_fine(sources, kappa, dt, sensors_xy, T0)
            return jnp.mean((Y_pred - Y_obs_fine) ** 2)

        # Warmup fine
        _ = fine_obj(jnp.array(coarse_params))

        def fine_scipy(params_np):
            params = jnp.array(params_np)
            loss = float(fine_obj(params))
            eps = 1e-4
            grad = np.zeros(n_params)
            for i in range(n_params):
                params_p = params.at[i].add(eps)
                loss_p = float(fine_obj(params_p))
                grad[i] = (loss_p - loss) / eps
            return loss, grad

        result = minimize(fine_scipy, x0=coarse_params, method='L-BFGS-B', jac=True,
                         bounds=bounds, options={'maxiter': fine_iters})

        final_rmse = np.sqrt(result.fun)
        final_params = result.x

        if verbose:
            print(f"    Final RMSE: {final_rmse:.4f}")

        sources = []
        for i in range(n_sources):
            x, y, q = final_params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        return sources, final_rmse, [CandidateResult(final_params, final_rmse, 'aggressive')]
