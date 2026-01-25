"""
Weighted Centroid Nelder-Mead Optimizer

Based on 2023 paper: "Weighted Centroids in Adaptive Nelder–Mead Simplex:
With heat source locator and multiple myeloma predictor applications"
(Applied Soft Computing, Vol. 151, January 2024)

Key modification: Instead of simple centroid, use weighted mean where weights
are inversely proportional to function values. Better vertices get higher weights,
biasing search towards promising regions.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from itertools import permutations

import numpy as np
import cma

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


def weighted_nelder_mead(
    func: Callable,
    x0: np.ndarray,
    maxiter: int = 100,
    xatol: float = 0.005,
    fatol: float = 0.0005,
    alpha: float = 1.0,    # Reflection coefficient
    gamma: float = 2.0,    # Expansion coefficient
    rho: float = 0.5,      # Contraction coefficient
    sigma: float = 0.5,    # Shrinkage coefficient
    weight_power: float = 2.0,  # Power for weight calculation
) -> dict:
    """
    Weighted Centroid Nelder-Mead optimization.

    The key modification from standard NM: centroid is computed as a weighted
    mean of the n best vertices (excluding worst). Weights are inversely
    proportional to function values raised to weight_power.

    For minimization, lower function values → higher weights.
    """
    n = len(x0)

    # Initialize simplex using standard approach
    # Create initial simplex by adding unit vectors
    simplex = [x0.copy()]
    step_size = 0.05  # Small initial step for local refinement
    for i in range(n):
        point = x0.copy()
        point[i] += step_size
        simplex.append(point)
    simplex = np.array(simplex)

    # Evaluate all vertices
    values = np.array([func(v) for v in simplex])
    n_evals = n + 1

    for iteration in range(maxiter):
        # Sort vertices by function value (ascending)
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        # Check convergence
        f_range = values[-1] - values[0]
        x_range = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1)])

        if f_range < fatol and x_range < xatol:
            break

        # WEIGHTED CENTROID CALCULATION
        # Exclude worst vertex, weight remaining by inverse function value
        best_n = simplex[:-1]  # Exclude worst (last after sort)
        best_vals = values[:-1]

        # Shift values to be positive (since we're minimizing)
        shifted_vals = best_vals - best_vals.min() + 1e-10

        # Weights inversely proportional to function value
        # Better (lower) values → higher weights
        weights = 1.0 / (shifted_vals ** weight_power)
        weights = weights / weights.sum()

        # Weighted centroid
        centroid = np.zeros(n)
        for i, (v, w) in enumerate(zip(best_n, weights)):
            centroid += w * v

        # Standard NM operations using weighted centroid
        worst = simplex[-1]
        worst_val = values[-1]

        # Reflection
        reflected = centroid + alpha * (centroid - worst)
        reflected_val = func(reflected)
        n_evals += 1

        if values[0] <= reflected_val < values[-2]:
            # Accept reflection
            simplex[-1] = reflected
            values[-1] = reflected_val
            continue

        if reflected_val < values[0]:
            # Try expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded_val = func(expanded)
            n_evals += 1

            if expanded_val < reflected_val:
                simplex[-1] = expanded
                values[-1] = expanded_val
            else:
                simplex[-1] = reflected
                values[-1] = reflected_val
            continue

        # Contraction
        if reflected_val < worst_val:
            # Outside contraction
            contracted = centroid + rho * (reflected - centroid)
        else:
            # Inside contraction
            contracted = centroid - rho * (centroid - worst)

        contracted_val = func(contracted)
        n_evals += 1

        if contracted_val < min(reflected_val, worst_val):
            simplex[-1] = contracted
            values[-1] = contracted_val
            continue

        # Shrinkage
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            values[i] = func(simplex[i])
            n_evals += 1

    # Return best
    best_idx = np.argmin(values)
    return {
        'x': simplex[best_idx].copy(),
        'fun': values[best_idx],
        'nfev': n_evals,
        'nit': iteration + 1,
        'success': True
    }


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
    return q_optimal, Y_pred, rmse


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
    return (q1, q2), Y_pred, rmse


class WeightedCentroidNMOptimizer:
    """
    Same as baseline optimizer but uses Weighted Centroid NM for final polish.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        max_fevals_1src: int = 20,
        max_fevals_2src: int = 36,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        refine_maxiter: int = 3,
        refine_top_n: int = 2,
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
        timestep_fraction: float = 0.40,
        final_polish_maxiter: int = 8,
        # Weighted NM parameters
        weight_power: float = 2.0,
        use_weighted_nm: bool = True,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.timestep_fraction = timestep_fraction
        self.final_polish_maxiter = final_polish_maxiter
        self.weight_power = weight_power
        self.use_weighted_nm = use_weighted_nm

    def _create_solver(self, kappa, bc, coarse=False):
        if coarse:
            return Heat2D(Lx=self.Lx, Ly=self.Ly, nx=self.nx_coarse, ny=self.ny_coarse,
                         kappa=kappa, bc=bc)
        return Heat2D(Lx=self.Lx, Ly=self.Ly, nx=self.nx_fine, ny=self.ny_fine,
                     kappa=kappa, bc=bc)

    def _smart_init_positions(self, sample, n_sources):
        """Initialize from hottest sensor locations."""
        Y_obs = sample['Y_noisy']
        sensors_xy = sample['sensors_xy']
        last_temps = Y_obs[-1]
        sorted_idx = np.argsort(last_temps)[::-1]
        positions = []
        for i in range(n_sources):
            x, y = sensors_xy[sorted_idx[i % len(sorted_idx)]]
            positions.extend([x, y])
        return np.array(positions)

    def _triangulation_init_positions(self, sample, meta, n_sources, q_range):
        """Initialize using triangulation."""
        if not self.use_triangulation:
            return None
        try:
            inits = triangulation_init(sample, meta, n_sources, q_range, n_inits=1, grid_res=15)
            if inits:
                pos = []
                for src in inits[0]['sources']:
                    pos.extend([src['x'], src['y']])
                return np.array(pos)
        except:
            pass
        return None

    def _run_single_optimization(
        self, init_pos, n_sources, Y_observed, solver_coarse, solver_fine,
        dt, nt_coarse, nt_full, T0, sensors_xy, q_range,
    ):
        """Run CMA-ES optimization with truncated timesteps, return candidates."""
        n_sims = [0]

        def objective_coarse(pos_params):
            """Coarse grid, truncated timesteps."""
            if n_sources == 1:
                x, y = pos_params
                if not (0.05 <= x <= 1.95 and 0.05 <= y <= 0.95):
                    return 10.0
                _, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_coarse, T0, sensors_xy, q_range)
                n_sims[0] += 1
            else:
                x1, y1, x2, y2 = pos_params
                if not (0.05 <= x1 <= 1.95 and 0.05 <= y1 <= 0.95 and
                       0.05 <= x2 <= 1.95 and 0.05 <= y2 <= 0.95):
                    return 10.0
                (_, _), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_coarse, T0, sensors_xy, q_range)
                n_sims[0] += 2
            return rmse

        # CMA-ES parameters
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        bounds = [[0.05, 1.95], [0.05, 0.95]] * n_sources

        opts = {
            'bounds': list(zip(*bounds)),
            'maxfevals': max_fevals,
            'verbose': -9,
            'seed': 42,
        }

        es = cma.CMAEvolutionStrategy(init_pos, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            values = [objective_coarse(s) for s in solutions]
            es.tell(solutions, values)
        es.result_pretty()

        best_pos = es.result.xbest
        best_val = es.result.fbest

        # Evaluate on fine grid with full timesteps
        def objective_fine_full(pos_params):
            """Fine grid with full timesteps."""
            if n_sources == 1:
                x, y = pos_params
                _, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1
            else:
                x1, y1, x2, y2 = pos_params
                (_, _), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2
            return rmse

        # Get final RMSE
        final_rmse = objective_fine_full(best_pos)

        # Compute final parameters
        if n_sources == 1:
            x, y = best_pos
            q, _, _ = compute_optimal_intensity_1src(
                x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
            n_sims[0] += 1
            full_params = np.array([x, y, q])
            sources = [(float(x), float(y), float(q))]
        else:
            x1, y1, x2, y2 = best_pos
            (q1, q2), _, _ = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
            n_sims[0] += 2
            full_params = np.array([x1, y1, q1, x2, y2, q2])
            sources = [(float(x1), float(y1), float(q1)),
                      (float(x2), float(y2), float(q2))]

        return (sources, full_params, final_rmse, best_pos), n_sims[0], objective_fine_full

    def _run_all_optimizations(
        self, n_sources, Y_observed, solver_coarse, solver_fine,
        dt, nt_coarse, nt_full, T0, sensors_xy, q_range,
        primary_inits,
    ):
        """Run optimizations from all initializations."""
        candidates_raw = []
        n_sims = [0]

        for init_pos, init_type in primary_inits:
            result, n, objective_fine_full = self._run_single_optimization(
                init_pos, n_sources, Y_observed, solver_coarse, solver_fine,
                dt, nt_coarse, nt_full, T0, sensors_xy, q_range,
            )
            sources, full_params, final_rmse, best_pos = result
            candidates_raw.append((sources, final_rmse, full_params, init_type, best_pos))
            n_sims[0] += n

        # FINAL POLISH: Weighted Centroid NM on best candidate
        if self.final_polish_maxiter > 0 and candidates_raw:
            best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][1])
            best_pos_params = candidates_raw[best_idx][4]
            best_rmse = candidates_raw[best_idx][1]

            # Define objective for polish (fine grid, full timesteps)
            def objective_fine_full(pos_params):
                if n_sources == 1:
                    x, y = pos_params
                    if not (0.05 <= x <= 1.95 and 0.05 <= y <= 0.95):
                        return 10.0
                    _, _, rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 1
                else:
                    x1, y1, x2, y2 = pos_params
                    if not (0.05 <= x1 <= 1.95 and 0.05 <= y1 <= 0.95 and
                           0.05 <= x2 <= 1.95 and 0.05 <= y2 <= 0.95):
                        return 10.0
                    (_, _), _, rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 2
                return rmse

            # Use weighted NM or standard NM
            if self.use_weighted_nm:
                result = weighted_nelder_mead(
                    objective_fine_full,
                    best_pos_params,
                    maxiter=self.final_polish_maxiter,
                    xatol=0.005,
                    fatol=0.0005,
                    weight_power=self.weight_power,
                )
            else:
                from scipy.optimize import minimize
                result = minimize(
                    objective_fine_full,
                    best_pos_params,
                    method='Nelder-Mead',
                    options={'maxiter': self.final_polish_maxiter, 'xatol': 0.005, 'fatol': 0.0005}
                )

            if result['fun'] < best_rmse:
                # Update best candidate with polished result
                if n_sources == 1:
                    x, y = result['x']
                    q, _, final_rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 1
                    full_params = np.array([x, y, q])
                    sources = [(float(x), float(y), float(q))]
                else:
                    x1, y1, x2, y2 = result['x']
                    (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 2
                    full_params = np.array([x1, y1, q1, x2, y2, q2])
                    sources = [(float(x1), float(y1), float(q1)),
                              (float(x2), float(y2), float(q2))]

                candidates_raw[best_idx] = (sources, final_rmse, full_params, 'polished', result['x'])

        # Remove pos_params from output
        candidates_raw = [(c[0], c[1], c[2], c[3]) for c in candidates_raw]

        return candidates_raw, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        # Primary initializations
        primary_inits = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            primary_inits.append((tri_init, 'triangulation'))
        smart_init = self._smart_init_positions(sample, n_sources)
        primary_inits.append((smart_init, 'smart'))

        if not primary_inits:
            smart_init = self._smart_init_positions(sample, n_sources)
            primary_inits.append((smart_init, 'smart'))

        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']
        Y_observed = sample['Y_noisy']
        sensors_xy = sample['sensors_xy']

        # Run optimization with final polish
        candidates_raw, total_sims = self._run_all_optimizations(
            n_sources, Y_observed, solver_coarse, solver_fine,
            dt, nt_reduced, nt_full, T0, sensors_xy, q_range,
            primary_inits,
        )

        # Filter candidates for diversity
        candidates_filtered = filter_dissimilar(
            [(c[0], c[1]) for c in candidates_raw],
            tau=TAU, n_max=N_MAX
        )

        # Build results
        results = []
        for sources, rmse in candidates_filtered:
            full_params = None
            init_type = 'unknown'
            for c in candidates_raw:
                if c[0] == sources:
                    full_params = c[2]
                    init_type = c[3]
                    break
            results.append(CandidateResult(
                params=full_params,
                rmse=rmse,
                init_type=init_type,
                n_evals=0
            ))

        if not results:
            return [], float('inf'), [], total_sims

        best_rmse = min(r.rmse for r in results)
        candidates_out = [r.params.tolist() for r in results]

        return candidates_out, best_rmse, results, total_sims
