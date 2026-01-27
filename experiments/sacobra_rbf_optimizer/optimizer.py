"""
SACOBRA-style RBF Surrogate Optimizer

Sequential model-based optimization using RBF surrogates.
Inspired by SACOBRA (Self-Adjusting COBRA) algorithm.

Key idea: Instead of population-based CMA-ES, use sequential sampling
guided by an RBF surrogate model. This can be more sample-efficient
for very low evaluation budgets (<50 evals).

Algorithm:
1. Initial sampling (Latin Hypercube or random)
2. Fit RBF surrogate
3. Find minimum of surrogate (exploration/exploitation balance)
4. Evaluate actual objective at minimum
5. Update surrogate and repeat
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import permutations

import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc  # Latin Hypercube

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


class SACOBRAOptimizer:
    """
    SACOBRA-style RBF Surrogate Optimizer.

    Uses sequential RBF surrogate optimization instead of CMA-ES.
    Hypothesis: More sample-efficient for low evaluation budgets.
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
        initial_samples: int = 8,  # Initial LHC samples before surrogate
        rbf_function: str = 'multiquadric',  # RBF kernel type
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        refine_maxiter: int = 3,
        refine_top_n: int = 2,
        timestep_fraction: float = 0.40,
        final_polish_maxiter: int = 5,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.initial_samples = initial_samples
        self.rbf_function = rbf_function
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.timestep_fraction = timestep_fraction
        self.final_polish_maxiter = final_polish_maxiter

    def _create_solver(self, kappa, bc, coarse=False):
        if coarse:
            return Heat2D(self.Lx, self.Ly, self.nx_coarse, self.ny_coarse, kappa, bc=bc)
        return Heat2D(self.Lx, self.Ly, self.nx_fine, self.ny_fine, kappa, bc=bc)

    def _get_position_bounds(self, n_sources, margin=0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return np.array(lb), np.array(ub)

    def _smart_init_positions(self, sample, n_sources):
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

    def _triangulation_init_positions(self, sample, meta, n_sources, q_range):
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

    def _generate_lhc_samples(self, n_samples, n_sources, lb, ub):
        """Generate Latin Hypercube samples."""
        dim = 2 * n_sources
        sampler = qmc.LatinHypercube(d=dim)
        samples_unit = sampler.random(n=n_samples)
        # Scale to bounds
        samples = qmc.scale(samples_unit, lb, ub)
        return samples

    def _fit_rbf_surrogate(self, X, y):
        """Fit RBF surrogate to evaluated points."""
        if len(X) < 3:
            return None
        try:
            # Transpose for Rbf (expects each dimension as a separate argument)
            args = [X[:, i] for i in range(X.shape[1])] + [y]
            rbf = Rbf(*args, function=self.rbf_function, smooth=0.1)
            return rbf
        except Exception as e:
            return None

    def _optimize_surrogate(self, rbf, lb, ub, n_restarts=5):
        """Find minimum of RBF surrogate."""
        dim = len(lb)
        best_x = None
        best_f = float('inf')

        bounds = [(lb[i], ub[i]) for i in range(dim)]

        def surrogate_func(x):
            try:
                return rbf(*x)
            except:
                return 1e10

        # Multiple restarts from random points
        for _ in range(n_restarts):
            x0 = lb + np.random.rand(dim) * (ub - lb)
            try:
                result = minimize(
                    surrogate_func, x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except:
                continue

        return best_x, best_f

    def _run_single_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                                  n_sources, nt_reduced, nt_full):
        """Run SACOBRA-style sequential RBF optimization."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]

        # Create objective functions
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

            def objective_fine_full(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

            def objective_fine_full(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        lb, ub = self._get_position_bounds(n_sources)
        dim = 2 * n_sources

        # Track all evaluated points
        X_evaluated = []
        y_evaluated = []

        # Phase 1: Initial sampling including smart initializations
        initial_points = []

        # Add triangulation init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initial_points.append(('triangulation', tri_init))

        # Add smart init
        smart_init = self._smart_init_positions(sample, n_sources)
        initial_points.append(('smart', smart_init))

        # Fill rest with LHC samples
        n_lhc = max(1, self.initial_samples - len(initial_points))
        lhc_samples = self._generate_lhc_samples(n_lhc, n_sources, lb, ub)
        for sample_point in lhc_samples:
            initial_points.append(('lhc', sample_point))

        # Evaluate initial points
        for init_type, point in initial_points:
            if n_sims[0] >= max_fevals:
                break
            # Clip to bounds
            point = np.clip(point, lb, ub)
            rmse = objective_coarse(point)
            X_evaluated.append(point)
            y_evaluated.append(rmse)

        X_evaluated = np.array(X_evaluated)
        y_evaluated = np.array(y_evaluated)

        # Phase 2: Sequential surrogate optimization
        while n_sims[0] < max_fevals:
            # Fit RBF surrogate
            rbf = self._fit_rbf_surrogate(X_evaluated, y_evaluated)
            if rbf is None:
                # Fall back to random sampling
                new_point = lb + np.random.rand(dim) * (ub - lb)
            else:
                # Find minimum of surrogate
                new_point, _ = self._optimize_surrogate(rbf, lb, ub)
                if new_point is None:
                    new_point = lb + np.random.rand(dim) * (ub - lb)

            # Clip to bounds
            new_point = np.clip(new_point, lb, ub)

            # Evaluate actual objective
            rmse = objective_coarse(new_point)
            X_evaluated = np.vstack([X_evaluated, new_point])
            y_evaluated = np.append(y_evaluated, rmse)

        # Get top candidates
        sorted_idx = np.argsort(y_evaluated)
        top_candidates = []
        for idx in sorted_idx[:self.candidate_pool_size]:
            top_candidates.append((X_evaluated[idx], y_evaluated[idx], 'surrogate'))

        # Phase 3: Refine top candidates on coarse grid
        refined_solutions = []
        for pos_params, rmse_coarse, init_type in top_candidates[:self.refine_top_n]:
            if self.refine_maxiter > 0:
                bounds = [(lb[i], ub[i]) for i in range(dim)]
                result = minimize(
                    objective_coarse,
                    pos_params,
                    method='Nelder-Mead',
                    options={'maxiter': self.refine_maxiter, 'xatol': 0.01, 'fatol': 0.001}
                )
                if result.fun < rmse_coarse:
                    refined_solutions.append((result.x, result.fun, 'refined'))
                else:
                    refined_solutions.append((pos_params, rmse_coarse, init_type))
            else:
                refined_solutions.append((pos_params, rmse_coarse, init_type))

        # Add remaining top candidates
        for pos_params, rmse_coarse, init_type in top_candidates[self.refine_top_n:]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

        # Phase 4: Evaluate on fine grid with full timesteps
        candidates_raw = []
        for pos_params, rmse_coarse, init_type in refined_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, Y_pred, rmse_fine = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1
                sources_list = [[x, y, q]]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), Y_pred, rmse_fine = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2
                sources_list = [[x1, y1, q1], [x2, y2, q2]]
            candidates_raw.append((sources_list, rmse_fine))

        # Filter dissimilar candidates
        candidates = filter_dissimilar(candidates_raw, tau=TAU, n_max=self.n_candidates)

        # Phase 5: Final polish on best candidate
        if candidates and self.final_polish_maxiter > 0:
            best_sources, best_rmse = candidates[0]

            if n_sources == 1:
                x0 = [best_sources[0][0], best_sources[0][1]]
                bounds = [(lb[0], ub[0]), (lb[1], ub[1])]
                result = minimize(
                    objective_fine_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': self.final_polish_maxiter, 'xatol': 0.01, 'fatol': 0.001}
                )
                x_opt, y_opt = result.x
                q_opt, _, rmse_polished = compute_optimal_intensity_1src(
                    x_opt, y_opt, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1

                if rmse_polished < best_rmse:
                    candidates[0] = ([[x_opt, y_opt, q_opt]], rmse_polished)
            else:
                x0 = [best_sources[0][0], best_sources[0][1],
                      best_sources[1][0], best_sources[1][1]]
                bounds = [(lb[0], ub[0]), (lb[1], ub[1]),
                          (lb[2], ub[2]), (lb[3], ub[3])]
                result = minimize(
                    objective_fine_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': self.final_polish_maxiter, 'xatol': 0.01, 'fatol': 0.001}
                )
                x1, y1, x2, y2 = result.x
                (q1, q2), _, rmse_polished = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2

                if rmse_polished < best_rmse:
                    candidates[0] = ([[x1, y1, q1], [x2, y2, q2]], rmse_polished)

        # Format results
        results_out = []
        for sources_list, rmse in candidates:
            params = np.array([coord for src in sources_list for coord in src])
            results_out.append(CandidateResult(params, rmse, 'sacobra', n_sims[0]))

        best_rmse = candidates[0][1] if candidates else float('inf')
        return candidates, best_rmse, results_out, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        """Main entry point - estimate heat sources for a sample."""
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        n_sources = sample['n_sources']
        dt = meta['dt']
        nt_full = sample['sample_metadata']['nt']  # Use actual nt, not Y_noisy.shape[0]
        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        candidates, best_rmse, results, n_sims = self._run_single_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            n_sources, nt_reduced, nt_full
        )

        if verbose:
            print(f"  Sources: {n_sources}, Best RMSE: {best_rmse:.4f}, Sims: {n_sims}")

        return candidates, best_rmse, results, n_sims
