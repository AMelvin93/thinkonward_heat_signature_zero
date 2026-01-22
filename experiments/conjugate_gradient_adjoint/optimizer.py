"""
Conjugate Gradient with Adjoint Gradient Computation

The adjoint method computes gradients in O(1) simulations (forward + adjoint)
instead of O(2n) with finite differences. This enables efficient L-BFGS-B optimization.

Key insight: For n=4 parameters (2-source), adjoint requires 2 simulations per gradient
vs 9 simulations with finite differences. That's a 4.5x speedup per iteration.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu
from scipy.optimize import minimize

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


@dataclass
class CandidateResult:
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources):
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1, sources2):
    norm1 = normalize_sources(sources1)
    norm2 = normalize_sources(sources2)
    n = len(sources1)
    if n != len(sources2):
        return float('inf')
    if n == 1:
        return np.linalg.norm(norm1[0] - norm2[0])
    min_total = float('inf')
    for perm in permutations(range(n)):
        total = sum(np.linalg.norm(norm1[i] - norm2[j])**2 for i, j in enumerate(perm))
        min_total = min(min_total, np.sqrt(total / n))
    return min_total


def filter_dissimilar(candidates, tau=TAU, n_max=N_MAX):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[1])
    kept = [candidates[0]]
    for cand in candidates[1:]:
        if all(candidate_distance(cand[0], k[0]) >= tau for k in kept):
            kept.append(cand)
            if len(kept) >= n_max:
                break
    return kept


class AdjointGradientSolver:
    """
    Solver that computes RMSE and its gradient using the adjoint method.

    The adjoint method computes gradients in O(1) simulations:
    1. Forward solve: T(x,y,t) given sources
    2. Compute residual at sensors
    3. Adjoint solve: λ(x,y,t) backward in time with sensor residuals as source
    4. Gradient: ∂J/∂params from inner product of λ with source sensitivity
    """

    def __init__(self, Lx, Ly, nx, ny, kappa, bc, dt, nt, T0, sensors_xy, Y_observed, q_range):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa
        self.bc = bc
        self.dt = dt
        self.nt = nt
        self.T0 = T0
        self.sensors_xy = np.array(sensors_xy)
        self.Y_observed = Y_observed
        self.q_range = q_range

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Build operators
        self.solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

        # Pre-compute source spread
        max_ds = max(self.dx, self.dy)
        self.source_sigma = 2.5 * max_ds

        # Counter for function evaluations
        self.n_evals = 0

    def _gaussian_field(self, x0, y0, q=1.0):
        """Create Gaussian source field."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        return G * q / integral_G

    def _gaussian_gradient_x(self, x0, y0, q=1.0):
        """Gradient of Gaussian field w.r.t. x0."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        # dG/dx0 = G * (x - x0) / sigma^2
        dG_dx0 = G * (self.X - x0) / (self.source_sigma**2)
        return dG_dx0 * q / integral_G

    def _gaussian_gradient_y(self, x0, y0, q=1.0):
        """Gradient of Gaussian field w.r.t. y0."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        # dG/dy0 = G * (y - y0) / sigma^2
        dG_dy0 = G * (self.Y - y0) / (self.source_sigma**2)
        return dG_dy0 * q / integral_G

    def _interpolate_at_sensors(self, U_field):
        """Interpolate temperature at sensor locations."""
        return self.solver.sample_sensors(U_field, self.sensors_xy)

    def _sensor_to_field(self, sensor_values):
        """Distribute sensor values to grid using bilinear interpolation weights."""
        S = np.zeros((self.nx, self.ny))
        for i, (sx, sy) in enumerate(self.sensors_xy):
            fx = sx / self.dx
            fy = sy / self.dy
            ix = np.clip(int(np.floor(fx)), 0, self.nx - 2)
            iy = np.clip(int(np.floor(fy)), 0, self.ny - 2)
            tx = fx - ix
            ty = fy - iy

            val = sensor_values[i]
            S[ix, iy] += (1 - tx) * (1 - ty) * val
            S[ix + 1, iy] += tx * (1 - ty) * val
            S[ix, iy + 1] += (1 - tx) * ty * val
            S[ix + 1, iy + 1] += tx * ty * val
        return S

    def forward_solve_1src(self, x, y, q):
        """Forward solve for 1 source, return temperature at sensors."""
        sources = [{'x': x, 'y': y, 'q': q}]
        times, Us = self.solver.solve(dt=self.dt, nt=self.nt, T0=self.T0, sources=sources)
        Y_sim = np.array([self._interpolate_at_sensors(U) for U in Us])
        return Y_sim, Us

    def forward_solve_2src(self, x1, y1, q1, x2, y2, q2):
        """Forward solve for 2 sources, return temperature at sensors."""
        sources = [{'x': x1, 'y': y1, 'q': q1}, {'x': x2, 'y': y2, 'q': q2}]
        times, Us = self.solver.solve(dt=self.dt, nt=self.nt, T0=self.T0, sources=sources)
        Y_sim = np.array([self._interpolate_at_sensors(U) for U in Us])
        return Y_sim, Us

    def compute_optimal_intensity_1src(self, x, y):
        """Compute optimal intensity for 1-source using least squares."""
        # Simulate with unit intensity
        Y_unit, Us_unit = self.forward_solve_1src(x, y, 1.0)
        self.n_evals += 1

        Y_unit_flat = Y_unit.flatten()
        Y_obs_flat = self.Y_observed.flatten()
        denominator = np.dot(Y_unit_flat, Y_unit_flat)
        if denominator < 1e-10:
            q_optimal = 1.0
        else:
            q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
        q_optimal = np.clip(q_optimal, self.q_range[0], self.q_range[1])

        Y_pred = q_optimal * Y_unit
        rmse = np.sqrt(np.mean((Y_pred - self.Y_observed) ** 2))

        # Store for gradient computation
        self._last_Us = [U * q_optimal for U in Us_unit]
        self._last_q = q_optimal

        return q_optimal, rmse

    def compute_optimal_intensity_2src(self, x1, y1, x2, y2):
        """Compute optimal intensities for 2-source using least squares."""
        # Simulate each source with unit intensity
        Y1, Us1 = self.forward_solve_1src(x1, y1, 1.0)
        Y2, Us2 = self.forward_solve_1src(x2, y2, 1.0)
        self.n_evals += 2

        Y1_flat = Y1.flatten()
        Y2_flat = Y2.flatten()
        Y_obs_flat = self.Y_observed.flatten()

        A = np.array([
            [np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
            [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]
        ])
        b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])

        try:
            q1, q2 = np.linalg.solve(A + 1e-6 * np.eye(2), b)
        except:
            q1, q2 = 1.0, 1.0

        q1 = np.clip(q1, self.q_range[0], self.q_range[1])
        q2 = np.clip(q2, self.q_range[0], self.q_range[1])

        Y_pred = q1 * Y1 + q2 * Y2
        rmse = np.sqrt(np.mean((Y_pred - self.Y_observed) ** 2))

        # Store for gradient computation
        self._last_Us = [Us1[t] * q1 + Us2[t] * q2 for t in range(len(Us1))]
        self._last_Us1 = Us1
        self._last_Us2 = Us2
        self._last_q1 = q1
        self._last_q2 = q2

        return (q1, q2), rmse

    def adjoint_solve(self, Us, residual_at_sensors):
        """
        Solve adjoint equation backward in time.

        -∂λ/∂t - κ∇²λ = sensor_residuals (distributed to grid)
        λ(T) = 0 (final time condition)

        Returns λ at all timesteps.
        """
        # Initialize adjoint state at final time
        lambda_field = np.zeros((self.nx, self.ny))
        lambdas = [lambda_field.copy()]

        # ADI setup (same as forward but negative kappa for backward)
        r = self.kappa * self.dt / 2.0

        Ax = (self.solver.Ix_1D - r * self.solver.Lx_1D).tocsc()
        Ay = (self.solver.Iy_1D - r * self.solver.Ly_1D).tocsc()
        Ax_lu = splu(Ax)
        Ay_lu = splu(Ay)

        # Solve backward in time
        for t_idx in range(len(residual_at_sensors) - 1, -1, -1):
            # Source term: distribute sensor residuals to grid
            S_adj = self._sensor_to_field(residual_at_sensors[t_idx]) * self.dt / 2.0

            # ADI Step 1: Implicit in X, Explicit in Y
            RHS_x = lambda_field.copy()
            for i in range(self.nx):
                RHS_x[i, :] += r * (self.solver.Ly_1D @ lambda_field[i, :])
            RHS_x += S_adj

            lambda_star = np.zeros_like(lambda_field)
            for j in range(self.ny):
                lambda_star[:, j] = Ax_lu.solve(RHS_x[:, j])

            # ADI Step 2: Implicit in Y, Explicit in X
            RHS_y = lambda_star.copy()
            for j in range(self.ny):
                RHS_y[:, j] += r * (self.solver.Lx_1D @ lambda_star[:, j])
            RHS_y += S_adj

            lambda_next = np.zeros_like(lambda_field)
            for i in range(self.nx):
                lambda_next[i, :] = Ay_lu.solve(RHS_y[i, :])

            lambda_field = lambda_next
            lambdas.insert(0, lambda_field.copy())

        return lambdas

    def compute_gradient_1src(self, x, y):
        """
        Compute gradient of RMSE w.r.t. (x, y) for 1-source using adjoint.

        Must be called after compute_optimal_intensity_1src.
        """
        # Compute residuals at sensors
        Y_sim = np.array([self._interpolate_at_sensors(U) for U in self._last_Us])
        residuals = Y_sim - self.Y_observed  # (nt+1, n_sensors)

        # Solve adjoint equation
        lambdas = self.adjoint_solve(self._last_Us, residuals)

        # Compute gradient: ∂J/∂x0 = ∫∫∫ λ * q * ∂G/∂x0 dt dx dy
        q = self._last_q
        grad_x = 0.0
        grad_y = 0.0

        dG_dx = self._gaussian_gradient_x(x, y, q)
        dG_dy = self._gaussian_gradient_y(x, y, q)

        for t_idx, lam in enumerate(lambdas):
            grad_x += np.sum(lam * dG_dx) * self.dx * self.dy * self.dt
            grad_y += np.sum(lam * dG_dy) * self.dx * self.dy * self.dt

        # Scale by 1/n for RMSE gradient
        n = self.Y_observed.size
        grad_x *= 1.0 / n
        grad_y *= 1.0 / n

        return np.array([grad_x, grad_y])

    def compute_gradient_2src(self, x1, y1, x2, y2):
        """
        Compute gradient of RMSE w.r.t. (x1, y1, x2, y2) for 2-source using adjoint.

        Must be called after compute_optimal_intensity_2src.
        """
        # Compute residuals at sensors
        Y_sim = np.array([self._interpolate_at_sensors(U) for U in self._last_Us])
        residuals = Y_sim - self.Y_observed

        # Solve adjoint equation
        lambdas = self.adjoint_solve(self._last_Us, residuals)

        q1, q2 = self._last_q1, self._last_q2

        # Compute gradients for source 1
        dG1_dx = self._gaussian_gradient_x(x1, y1, q1)
        dG1_dy = self._gaussian_gradient_y(x1, y1, q1)

        # Compute gradients for source 2
        dG2_dx = self._gaussian_gradient_x(x2, y2, q2)
        dG2_dy = self._gaussian_gradient_y(x2, y2, q2)

        grad_x1, grad_y1, grad_x2, grad_y2 = 0.0, 0.0, 0.0, 0.0

        for t_idx, lam in enumerate(lambdas):
            grad_x1 += np.sum(lam * dG1_dx) * self.dx * self.dy * self.dt
            grad_y1 += np.sum(lam * dG1_dy) * self.dx * self.dy * self.dt
            grad_x2 += np.sum(lam * dG2_dx) * self.dx * self.dy * self.dt
            grad_y2 += np.sum(lam * dG2_dy) * self.dx * self.dy * self.dt

        # Scale by 1/n for RMSE gradient
        n = self.Y_observed.size
        grad_x1 *= 1.0 / n
        grad_y1 *= 1.0 / n
        grad_x2 *= 1.0 / n
        grad_y2 *= 1.0 / n

        return np.array([grad_x1, grad_y1, grad_x2, grad_y2])


class AdjointOptimizer:
    """
    Optimizer using L-BFGS-B with adjoint gradients.

    Key advantages:
    - O(1) gradient computation (2 sims) vs O(2n) with finite differences
    - For 2-source: 2 sims vs 9 sims per gradient = 4.5x faster per iteration
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_restarts_1src: int = 3,
        n_restarts_2src: int = 5,
        max_iter_per_restart: int = 20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        timestep_fraction: float = 0.4,  # Use reduced timesteps during optimization
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_restarts_1src = n_restarts_1src
        self.n_restarts_2src = n_restarts_2src
        self.max_iter_per_restart = max_iter_per_restart
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.timestep_fraction = timestep_fraction

    def _get_bounds(self, n_sources, margin=0.05):
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
        return bounds

    def _smart_init(self, sample, n_sources):
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
        for idx in selected:
            x, y = sensors[idx]
            params.extend([x, y])
        return np.array(params)

    def _triangulation_init(self, sample, meta, n_sources, q_range):
        if not self.use_triangulation:
            return None
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except:
            return None

    def _random_init(self, n_sources, margin=0.1):
        params = []
        for _ in range(n_sources):
            x = margin * self.Lx + np.random.random() * (1 - 2*margin) * self.Lx
            y = margin * self.Ly + np.random.random() * (1 - 2*margin) * self.Ly
            params.extend([x, y])
        return np.array(params)

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']
        dt = meta['dt']

        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        # Use reduced timesteps during optimization
        nt_opt = max(10, int(nt_full * self.timestep_fraction))
        Y_observed_trunc = Y_observed[:nt_opt + 1]

        # Create adjoint solver for optimization
        adj_solver = AdjointGradientSolver(
            self.Lx, self.Ly, self.nx, self.ny, kappa, bc,
            dt, nt_opt, T0, sensors_xy, Y_observed_trunc, q_range
        )

        # Create full solver for final evaluation
        full_solver = Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

        bounds = self._get_bounds(n_sources)
        n_restarts = self.n_restarts_1src if n_sources == 1 else self.n_restarts_2src

        # Generate starting points
        init_points = []
        tri_init = self._triangulation_init(sample, meta, n_sources, q_range)
        if tri_init is not None:
            init_points.append((tri_init, 'triangulation'))
        init_points.append((self._smart_init(sample, n_sources), 'smart'))

        while len(init_points) < n_restarts:
            init_points.append((self._random_init(n_sources), f'random_{len(init_points)}'))

        all_results = []

        if n_sources == 1:
            def objective_and_gradient(xy_params):
                x, y = xy_params
                q, rmse = adj_solver.compute_optimal_intensity_1src(x, y)
                grad = adj_solver.compute_gradient_1src(x, y)
                return rmse, grad
        else:
            def objective_and_gradient(xy_params):
                x1, y1, x2, y2 = xy_params
                (q1, q2), rmse = adj_solver.compute_optimal_intensity_2src(x1, y1, x2, y2)
                grad = adj_solver.compute_gradient_2src(x1, y1, x2, y2)
                return rmse, grad

        for init_params, init_type in init_points[:n_restarts]:
            try:
                result = minimize(
                    objective_and_gradient,
                    init_params,
                    method='L-BFGS-B',
                    jac=True,  # Gradient is included in return
                    bounds=bounds,
                    options={
                        'maxiter': self.max_iter_per_restart,
                        'disp': False,
                    }
                )
                all_results.append((result.x, result.fun, init_type))
                if verbose:
                    print(f"  {init_type}: RMSE={result.fun:.4f} after {result.nit} iterations")
            except Exception as e:
                if verbose:
                    print(f"  L-BFGS-B failed from {init_type}: {e}")
                continue

        if not all_results:
            smart_init = self._smart_init(sample, n_sources)
            try:
                result = minimize(
                    objective_and_gradient,
                    smart_init,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': self.max_iter_per_restart}
                )
                all_results.append((result.x, result.fun, 'fallback'))
            except:
                all_results.append((smart_init, float('inf'), 'fallback'))

        n_sims = adj_solver.n_evals

        # Evaluate on full timesteps
        candidates_raw = []
        for pos_params, rmse_opt, init_type in all_results:
            if n_sources == 1:
                x, y = pos_params
                # Full simulation for final RMSE
                sources = [{'x': x, 'y': y, 'q': 1.0}]
                _, Us = full_solver.solve(dt=dt, nt=nt_full, T0=T0, sources=sources)
                Y_unit = np.array([full_solver.sample_sensors(U, sensors_xy) for U in Us])
                n_sims += 1

                # Compute optimal q on full data
                Y_unit_flat = Y_unit.flatten()
                Y_obs_flat = Y_observed.flatten()
                denom = np.dot(Y_unit_flat, Y_unit_flat)
                q = np.dot(Y_unit_flat, Y_obs_flat) / denom if denom > 1e-10 else 1.0
                q = np.clip(q, q_range[0], q_range[1])

                Y_pred = q * Y_unit
                final_rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

                full_params = np.array([x, y, q])
                sources_list = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                # Full simulation for each source
                sources1 = [{'x': x1, 'y': y1, 'q': 1.0}]
                sources2 = [{'x': x2, 'y': y2, 'q': 1.0}]
                _, Us1 = full_solver.solve(dt=dt, nt=nt_full, T0=T0, sources=sources1)
                _, Us2 = full_solver.solve(dt=dt, nt=nt_full, T0=T0, sources=sources2)
                Y1 = np.array([full_solver.sample_sensors(U, sensors_xy) for U in Us1])
                Y2 = np.array([full_solver.sample_sensors(U, sensors_xy) for U in Us2])
                n_sims += 2

                # Compute optimal q1, q2 on full data
                Y1_flat, Y2_flat = Y1.flatten(), Y2.flatten()
                Y_obs_flat = Y_observed.flatten()
                A = np.array([
                    [np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
                    [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]
                ])
                b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])
                try:
                    q1, q2 = np.linalg.solve(A + 1e-6 * np.eye(2), b)
                except:
                    q1, q2 = 1.0, 1.0
                q1 = np.clip(q1, q_range[0], q_range[1])
                q2 = np.clip(q2, q_range[0], q_range[1])

                Y_pred = q1 * Y1 + q2 * Y2
                final_rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources_list = [(float(x1), float(y1), float(q1)),
                               (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources_list, full_params, final_rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[1], rmse=c[2], init_type=c[3],
                n_evals=n_sims // len(final_candidates) if final_candidates else n_sims
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims
