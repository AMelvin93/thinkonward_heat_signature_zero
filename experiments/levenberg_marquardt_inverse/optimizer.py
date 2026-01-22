"""
Levenberg-Marquardt Optimizer for Inverse Heat Source Problem

Uses scipy.optimize.least_squares with method='lm' (Levenberg-Marquardt) which is
specifically designed for nonlinear least squares problems.

Key advantages:
- Quadratic convergence near optimum (faster than NM)
- Uses Jacobian-based descent (more informed than evolutionary)
- Classic method for inverse heat transfer problems

Key challenges:
- Local optimizer: needs good initialization
- No native bound handling (uses projection)
- Finite differences for Jacobian can be expensive
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
from scipy.optimize import least_squares, minimize

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


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    """Run simulation and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensity for 1-source."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)

    n_steps = len(Y_unit)
    Y_obs_trunc = Y_observed[:n_steps]

    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])

    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    residuals = (Y_pred - Y_obs_trunc).flatten()
    return q_optimal, Y_pred, rmse, residuals


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensities for 2-source."""
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)

    n_steps = len(Y1)
    Y_obs_trunc = Y_observed[:n_steps]

    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
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
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    residuals = (Y_pred - Y_obs_trunc).flatten()
    return (q1, q2), Y_pred, rmse, residuals


class LevenbergMarquardtOptimizer:
    """
    Optimizer using Levenberg-Marquardt algorithm for inverse heat source identification.

    LM is specifically designed for nonlinear least squares problems, which is exactly
    what our RMSE minimization is. It typically converges faster than derivative-free
    methods when initialized near the optimum.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_multi_starts: int = 5,  # Number of different starting points
        max_nfev: int = 50,  # Max function evaluations per start
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        timestep_fraction: float = 0.40,  # Use 40% timesteps like best baseline
        nm_polish_iters: int = 8,  # Nelder-Mead polish iterations
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_multi_starts = n_multi_starts
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.timestep_fraction = timestep_fraction
        self.nm_polish_iters = nm_polish_iters

    def _create_solver(self, kappa, bc):
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources, margin=0.05):
        """Get bounds for position optimization."""
        lb = []
        ub = []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return np.array(lb), np.array(ub)

    def _clip_to_bounds(self, params, lb, ub):
        """Project parameters to feasible region."""
        return np.clip(params, lb, ub)

    def _smart_init_positions(self, sample, n_sources):
        """Initialize at hottest sensors."""
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

    def _weighted_centroid_init(self, sample, n_sources):
        """Initialize using weighted centroid of sensors."""
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])
        max_temps = np.max(readings, axis=0)
        weights = max_temps / (max_temps.sum() + 1e-8)
        centroid = np.average(sensors, axis=0, weights=weights)

        if n_sources == 1:
            return np.array([centroid[0], centroid[1]])
        else:
            spread = np.sqrt(np.average(
                (sensors[:, 0] - centroid[0])**2 + (sensors[:, 1] - centroid[1])**2,
                weights=weights
            ))
            offset = max(0.1, spread * 0.3)
            return np.array([
                centroid[0] - offset, centroid[1],
                centroid[0] + offset, centroid[1]
            ])

    def _perturbed_init(self, base_init, sigma=0.1):
        """Create a perturbed version of an initialization."""
        perturbation = np.random.randn(len(base_init)) * sigma
        return base_init + perturbation

    def _triangulation_init_positions(self, sample, meta, n_sources, q_range):
        """Initialize using triangulation."""
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

    def _run_lm_optimization(self, init_params, residual_fn, lb, ub, init_type):
        """
        Run Levenberg-Marquardt optimization from a single starting point.

        Note: scipy's LM doesn't support bounds, so we use a wrapper that
        projects to the feasible region.
        """
        n_evals = [0]

        def residual_with_projection(params):
            n_evals[0] += 1
            # Project to bounds
            clipped = self._clip_to_bounds(params, lb, ub)
            return residual_fn(clipped)

        try:
            # Use 'lm' (Levenberg-Marquardt) - note it doesn't support bounds directly
            result = least_squares(
                residual_with_projection,
                init_params,
                method='lm',  # Levenberg-Marquardt
                ftol=self.ftol,
                xtol=self.xtol,
                max_nfev=self.max_nfev,
                verbose=0,
            )

            # Project final result to bounds
            final_params = self._clip_to_bounds(result.x, lb, ub)
            final_rmse = np.sqrt(np.mean(residual_fn(final_params)**2))

            return {
                'params': final_params,
                'rmse': final_rmse,
                'n_evals': n_evals[0],
                'success': result.success,
                'init_type': init_type,
            }
        except Exception as e:
            return {
                'params': init_params,
                'rmse': float('inf'),
                'n_evals': n_evals[0],
                'success': False,
                'init_type': init_type,
                'error': str(e),
            }

    def _nm_polish(self, params, objective_fn, lb, ub):
        """Apply Nelder-Mead polish to refine solution."""
        def bounded_objective(p):
            clipped = self._clip_to_bounds(p, lb, ub)
            return objective_fn(clipped)

        result = minimize(
            bounded_objective,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': self.nm_polish_iters,
                'xatol': 0.01,
                'fatol': 0.001,
            }
        )
        return self._clip_to_bounds(result.x, lb, ub), result.fun

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        """
        Estimate heat source parameters using Levenberg-Marquardt optimization.
        """
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        solver = self._create_solver(kappa, bc)

        # Compute reduced timesteps for optimization phase
        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        lb, ub = self._get_bounds(n_sources)
        n_total_evals = [0]

        # Define residual function for LM
        if n_sources == 1:
            def residual_fn(xy_params):
                x, y = xy_params
                n_total_evals[0] += 1
                _, _, _, residuals = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt_reduced, T0, sensors_xy, q_range)
                return residuals

            def objective_fn(xy_params):
                x, y = xy_params
                _, _, rmse, _ = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse
        else:
            def residual_fn(xy_params):
                x1, y1, x2, y2 = xy_params
                n_total_evals[0] += 2
                _, _, _, residuals = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt_reduced, T0, sensors_xy, q_range)
                return residuals

            def objective_fn(xy_params):
                x1, y1, x2, y2 = xy_params
                _, _, rmse, _ = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

        # Generate multiple starting points
        initializations = []

        # 1. Smart init (hottest sensor)
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # 2. Triangulation init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # 3. Centroid init
        centroid_init = self._weighted_centroid_init(sample, n_sources)
        initializations.append((centroid_init, 'centroid'))

        # 4. Perturbed versions of best inits
        while len(initializations) < self.n_multi_starts:
            base = initializations[len(initializations) % 2][0]  # Alternate between smart and centroid
            perturbed = self._perturbed_init(base, sigma=0.15)
            perturbed = self._clip_to_bounds(perturbed, lb, ub)
            initializations.append((perturbed, f'perturbed_{len(initializations)}'))

        # Run LM from each starting point
        all_results = []
        for init_params, init_type in initializations[:self.n_multi_starts]:
            result = self._run_lm_optimization(init_params, residual_fn, lb, ub, init_type)
            all_results.append(result)

        # Sort by RMSE
        all_results.sort(key=lambda x: x['rmse'])

        # Apply NM polish to top results
        polished_results = []
        for i, result in enumerate(all_results[:3]):  # Polish top 3
            if self.nm_polish_iters > 0:
                polished_params, polished_rmse = self._nm_polish(
                    result['params'], objective_fn, lb, ub)
                if polished_rmse < result['rmse']:
                    polished_results.append({
                        'params': polished_params,
                        'rmse': polished_rmse,
                        'init_type': f"{result['init_type']}_polished",
                        'n_evals': result['n_evals'] + self.nm_polish_iters,
                    })
                else:
                    polished_results.append(result)
            else:
                polished_results.append(result)

        # Add remaining unpolished results
        for result in all_results[3:]:
            polished_results.append(result)

        # Evaluate on FULL timesteps for final candidates
        candidates_raw = []
        for result in polished_results:
            pos_params = result['params']

            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse, _ = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt_full, T0, sensors_xy, q_range)
                n_total_evals[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse, _ = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt_full, T0, sensors_xy, q_range)
                n_total_evals[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, result['init_type']))

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
                n_evals=n_total_evals[0] // len(final_candidates) if final_candidates else n_total_evals[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_total_evals[0]
