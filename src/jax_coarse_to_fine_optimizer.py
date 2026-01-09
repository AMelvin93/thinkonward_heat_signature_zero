"""
Coarse-to-Fine JAX Optimizer.

Strategy:
1. COARSE PHASE: Use 50x25 grid, subsampled timesteps for fast exploration
2. FINE PHASE: Refine top candidates with full 100x50 grid

This gives ~16x speedup for exploration while maintaining accuracy.
Still uses simulator at inference (meets competition requirements).

Expected speedup: 10-20x over full-resolution-only approach
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

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


# ============================================================
# COARSE SIMULATOR (50x25 grid, subsampled time)
# ============================================================

@jit
def thomas_solve(lower, diag, upper, rhs):
    """Fast Thomas algorithm."""
    n = diag.shape[0]

    def forward_step(carry, idx):
        c_prev, d_prev = carry
        denom = jnp.maximum(jnp.abs(diag[idx] - lower[idx] * c_prev), 1e-12)
        denom = jnp.sign(diag[idx] - lower[idx] * c_prev + 1e-12) * denom
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
    return jnp.concatenate([x_rev[::-1], jnp.array([x_last])])


def create_coarse_simulator(nt_full: int, bc: str, time_subsample: int = 4):
    """
    Create a coarse simulator function.

    Uses 50x25 grid (vs 100x50) and subsampled timesteps.
    """
    nx_c, ny_c = 50, 25  # Coarse grid
    Lx, Ly = 2.0, 1.0
    dx_c = Lx / (nx_c - 1)
    dy_c = Ly / (ny_c - 1)

    # Effective number of coarse timesteps
    nt_coarse = nt_full // time_subsample

    @jit
    def simulate_coarse(sources, kappa, dt, sensors_xy, T0):
        """Coarse simulation with sensor sampling during loop."""
        # Effective dt for coarser time stepping
        dt_eff = dt * time_subsample

        rx = kappa * dt_eff / (2.0 * dx_c * dx_c)
        ry = kappa * dt_eff / (2.0 * dy_c * dy_c)
        r = kappa * dt_eff / 2.0

        # Precompute tridiagonal coefficients
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

        # Grid
        x = jnp.linspace(0, Lx, nx_c)
        y = jnp.linspace(0, Ly, ny_c)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        # Source field
        sigma = 2.5 * max(dx_c, dy_c)
        S = jnp.zeros((nx_c, ny_c))
        def add_source(S, source):
            x0, y0, q = source
            r_sq = (X - x0)**2 + (Y - y0)**2
            G = jnp.exp(-r_sq / (2.0 * sigma**2))
            integral_G = jnp.maximum(jnp.sum(G) * dx_c * dy_c, 1e-10)
            return S + G * q / integral_G, None
        S, _ = jax.lax.scan(add_source, S, sources)

        # Sensor sampling function
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

        # Initial state
        U = jnp.full((nx_c, ny_c), T0)

        # ADI step function
        def adi_step(U, _):
            S_half = S * dt_eff / 2.0

            # Step 1: Implicit X
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

            # Step 2: Implicit Y
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

            Y_t = sample_sensors(U_next)
            return U_next, Y_t

        # Run simulation
        Y0 = sample_sensors(U)
        _, Y_rest = jax.lax.scan(adi_step, U, None, length=nt_coarse)

        # We need to interpolate back to full time resolution for comparison
        # For now, just return coarse time samples (compare at coarse resolution)
        return jnp.concatenate([Y0[None,:], Y_rest], axis=0)

    return simulate_coarse, nt_coarse


def create_fine_simulator(nt: int, bc: str):
    """Create full-resolution simulator."""
    nx, ny = 100, 50
    Lx, Ly = 2.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    @jit
    def simulate_fine(sources, kappa, dt, sensors_xy, T0):
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

            Y_t = sample_sensors(U_next)
            return U_next, Y_t

        Y0 = sample_sensors(U)
        _, Y_rest = jax.lax.scan(adi_step, U, None, length=nt)
        return jnp.concatenate([Y0[None,:], Y_rest], axis=0)

    return simulate_fine


class CoarseToFineOptimizer:
    """
    Two-phase optimizer:
    1. Coarse phase: Fast exploration with reduced resolution
    2. Fine phase: Accurate refinement with full resolution
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        time_subsample: int = 4,  # Use every 4th timestep for coarse
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.time_subsample = time_subsample
        self.device_info = check_gpu()

        # Caches
        self._coarse_cache = {}
        self._fine_cache = {}

    def _get_bounds(self, n_sources, q_range, margin=0.05):
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
            bounds.append(q_range)
        return bounds

    def _smart_init(self, sample, n_sources, q_range):
        """Quick smart initialization from hottest sensors."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[p]) >= 0.25 for p in selected):
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
            q = np.clip(q, q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

    def _random_init(self, n_sources, q_range, margin=0.1):
        params = []
        for _ in range(n_sources):
            params.extend([
                np.random.uniform(margin * self.Lx, (1-margin) * self.Lx),
                np.random.uniform(margin * self.Ly, (1-margin) * self.Ly),
                np.random.uniform(q_range[0], q_range[1])
            ])
        return np.array(params)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        coarse_iters: int = 15,
        fine_iters: int = 10,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Two-phase optimization:
        1. Coarse: 2 inits, 15 iters each (fast)
        2. Fine: 1 best candidate, 10 iters (accurate)
        """
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

        # ========================================
        # COARSE PHASE
        # ========================================
        if verbose:
            print(f"  Coarse phase (50x25 grid, 1/{self.time_subsample} timesteps)...")

        # Get/create coarse simulator
        coarse_key = (nt, bc)
        if coarse_key not in self._coarse_cache:
            self._coarse_cache[coarse_key] = create_coarse_simulator(nt, bc, self.time_subsample)
        sim_coarse, nt_coarse = self._coarse_cache[coarse_key]

        # Subsample observed data to match coarse time resolution
        time_indices = np.arange(0, nt + 1, self.time_subsample)
        Y_obs_coarse = jnp.array(Y_observed[time_indices[:nt_coarse+1]])

        # Coarse objective
        @jit
        def coarse_objective(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = sim_coarse(sources, kappa, dt, sensors_xy, T0)
            # Match lengths
            min_len = min(len(Y_pred), len(Y_obs_coarse))
            return jnp.mean((Y_pred[:min_len] - Y_obs_coarse[:min_len]) ** 2)

        @jit
        def coarse_batched(params_batch):
            return vmap(coarse_objective)(params_batch)

        # Warmup
        dummy = jnp.array(self._random_init(n_sources, q_range))
        dummy_batch = jnp.stack([dummy] * (1 + 2*n_params))
        _ = coarse_batched(dummy_batch)

        def coarse_scipy_obj(params_np):
            params = jnp.array(params_np)
            # Batched finite differences
            perturbs = [params]
            eps = 1e-4
            for i in range(n_params):
                perturbs.append(params.at[i].add(eps))
                perturbs.append(params.at[i].add(-eps))
            losses = coarse_batched(jnp.stack(perturbs))
            grad = jnp.array([(losses[1+2*i] - losses[2+2*i]) / (2*eps) for i in range(n_params)])
            return float(losses[0]), np.array(grad)

        # Run coarse optimization (2 inits)
        coarse_candidates = []
        for init_type, x0 in [('smart', self._smart_init(sample, n_sources, q_range)),
                               ('random', self._random_init(n_sources, q_range))]:
            try:
                result = minimize(coarse_scipy_obj, x0=x0, method='L-BFGS-B', jac=True,
                                bounds=bounds, options={'maxiter': coarse_iters})
                coarse_candidates.append((result.x, result.fun, init_type))
                if verbose:
                    print(f"    {init_type}: coarse_mse={result.fun:.6f}")
            except Exception as e:
                if verbose:
                    print(f"    {init_type}: failed - {e}")

        if not coarse_candidates:
            return [], float('inf'), []

        # ========================================
        # FINE PHASE
        # ========================================
        if verbose:
            print(f"  Fine phase (100x50 grid, all timesteps)...")

        # Get/create fine simulator
        fine_key = (nt, bc)
        if fine_key not in self._fine_cache:
            self._fine_cache[fine_key] = create_fine_simulator(nt, bc)
        sim_fine = self._fine_cache[fine_key]

        Y_obs_fine = jnp.array(Y_observed)

        @jit
        def fine_objective(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = sim_fine(sources, kappa, dt, sensors_xy, T0)
            return jnp.mean((Y_pred - Y_obs_fine) ** 2)

        @jit
        def fine_batched(params_batch):
            return vmap(fine_objective)(params_batch)

        # Warmup fine
        _ = fine_batched(dummy_batch)

        def fine_scipy_obj(params_np):
            params = jnp.array(params_np)
            perturbs = [params]
            eps = 1e-4
            for i in range(n_params):
                perturbs.append(params.at[i].add(eps))
                perturbs.append(params.at[i].add(-eps))
            losses = fine_batched(jnp.stack(perturbs))
            grad = jnp.array([(losses[1+2*i] - losses[2+2*i]) / (2*eps) for i in range(n_params)])
            return float(losses[0]), np.array(grad)

        # Refine best coarse candidate
        coarse_candidates.sort(key=lambda x: x[1])
        best_coarse = coarse_candidates[0][0]

        try:
            result = minimize(fine_scipy_obj, x0=best_coarse, method='L-BFGS-B', jac=True,
                            bounds=bounds, options={'maxiter': fine_iters})
            final_rmse = np.sqrt(result.fun)
            final_params = result.x
            if verbose:
                print(f"    Fine refinement: RMSE={final_rmse:.4f}")
        except Exception as e:
            if verbose:
                print(f"    Fine refinement failed: {e}")
            final_rmse = np.sqrt(coarse_candidates[0][1])
            final_params = best_coarse

        # Build result
        sources = []
        for i in range(n_sources):
            x, y, q = final_params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        return sources, final_rmse, [CandidateResult(final_params, final_rmse, 'coarse_to_fine')]
