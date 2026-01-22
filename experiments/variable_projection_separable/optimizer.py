"""
Variable Projection (VP) Optimizer for Inverse Heat Source Problem.

Our problem is naturally separable:
- Nonlinear parameters: positions (x, y) for each source
- Linear parameters: intensities (q) for each source

VP (Golub-Pereyra, 1973) eliminates linear parameters analytically:
1. For given positions, solve for optimal q via least squares
2. Optimize positions using Gauss-Newton on the "reduced residual"

Key insight: The baseline ALREADY uses VP implicitly (computes q analytically).
This experiment tests whether explicit Gauss-Newton on the reduced residual
converges faster than CMA-ES.

Expected outcome: LIKELY TO FAIL
- Previous gradient-based approaches (LM, L-BFGS-B) failed due to local minima
- Gauss-Newton is a local optimizer, unsuitable for multi-modal RMSE landscape
- But worth testing to confirm
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
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """VP step: solve for optimal q given (x, y)."""
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
    residual = (Y_pred - Y_obs_trunc).flatten()
    rmse = np.sqrt(np.mean(residual ** 2))
    return q_optimal, residual, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """VP step: solve for optimal (q1, q2) given (x1, y1, x2, y2)."""
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
    residual = (Y_pred - Y_obs_trunc).flatten()
    rmse = np.sqrt(np.mean(residual ** 2))
    return (q1, q2), residual, rmse


class VariableProjectionOptimizer:
    """
    Variable Projection optimizer using Gauss-Newton on the reduced residual.

    The VP approach:
    1. For each position (x, y), compute optimal q analytically (least squares)
    2. Compute reduced residual r(x, y) = A(x,y)*q_opt - b
    3. Minimize ||r(x, y)||^2 using Gauss-Newton (via scipy.least_squares)

    Note: scipy.least_squares uses Levenberg-Marquardt or Trust-Region Reflective,
    both are local methods that may get stuck in local minima.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        n_multi_starts: int = 5,
        max_nfev: int = 30,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        refine_maxiter: int = 8,
        rmse_threshold_1src: float = 0.40,
        rmse_threshold_2src: float = 0.50,
        timestep_fraction: float = 0.40,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.n_multi_starts = n_multi_starts
        self.max_nfev = max_nfev
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.refine_maxiter = refine_maxiter
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.timestep_fraction = timestep_fraction

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

    def _weighted_centroid_init(self, sample, n_sources):
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

    def _random_init_positions(self, n_sources, margin=0.1):
        params = []
        for _ in range(n_sources):
            x = margin * self.Lx + np.random.random() * (1 - 2*margin) * self.Lx
            y = margin * self.Ly + np.random.random() * (1 - 2*margin) * self.Ly
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

    def _run_vp_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                              initializations, n_sources, nt_reduced):
        """Run VP optimization using least_squares (Gauss-Newton/LM)."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt_full = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]
        lb, ub = self._get_position_bounds(n_sources)

        # Define the residual function for VP
        if n_sources == 1:
            def residual_fn(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, residual, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return residual
        else:
            def residual_fn(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), residual, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return residual

        # Define RMSE objective for NM polish
        if n_sources == 1:
            def objective_fn(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_fn(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

        all_solutions = []

        # Multi-start: run VP from each initialization
        for init_params, init_type in initializations:
            try:
                # Use Trust-Region Reflective for bounded optimization
                result = least_squares(
                    residual_fn,
                    init_params,
                    bounds=(lb, ub),
                    method='trf',  # Trust-Region Reflective
                    ftol=1e-4,
                    xtol=1e-4,
                    max_nfev=self.max_nfev,
                    verbose=0
                )
                rmse = np.sqrt(np.mean(result.fun ** 2))
                all_solutions.append((result.x, rmse, f'vp_{init_type}'))
            except Exception as e:
                # Fallback to Nelder-Mead
                try:
                    result = minimize(
                        objective_fn,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 20}
                    )
                    all_solutions.append((result.x, result.fun, f'nm_{init_type}'))
                except:
                    pass

        if not all_solutions:
            # Emergency fallback
            init_params = self._smart_init_positions(sample, n_sources)
            rmse = objective_fn(init_params)
            all_solutions.append((init_params, rmse, 'fallback'))

        # Sort by RMSE
        all_solutions.sort(key=lambda x: x[1])

        # NM polish on top solutions
        refined_solutions = []
        for pos_params, rmse_coarse, init_type in all_solutions[:2]:
            if self.refine_maxiter > 0:
                result = minimize(
                    objective_fn,
                    pos_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': self.refine_maxiter,
                        'xatol': 0.01,
                        'fatol': 0.001,
                    }
                )
                if result.fun < rmse_coarse:
                    refined_solutions.append((result.x, result.fun, 'refined'))
                else:
                    refined_solutions.append((pos_params, rmse_coarse, init_type))
            else:
                refined_solutions.append((pos_params, rmse_coarse, init_type))

        # Add remaining solutions
        for sol in all_solutions[2:5]:
            refined_solutions.append(sol)

        # Evaluate on fine grid with full timesteps
        candidates_raw = []
        for pos_params, rmse_coarse, init_type in refined_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        return candidates_raw, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        if verbose:
            print(f"  Using {nt_reduced}/{nt_full} timesteps for VP")

        # Prepare initializations
        initializations = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))
        centroid_init = self._weighted_centroid_init(sample, n_sources)
        initializations.append((centroid_init, 'centroid'))

        # Add random inits
        for i in range(self.n_multi_starts - 3):
            random_init = self._random_init_positions(n_sources)
            initializations.append((random_init, f'random_{i}'))

        # Run VP optimization
        candidates_raw, n_sims = self._run_vp_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            initializations, n_sources, nt_reduced
        )

        # Check if result is bad - add more inits
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            extra_inits = []
            for i in range(3):
                extra_inits.append((self._random_init_positions(n_sources), f'extra_{i}'))

            extra_candidates, extra_sims = self._run_vp_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                extra_inits, n_sources, nt_reduced
            )
            n_sims += extra_sims
            candidates_raw.extend(extra_candidates)

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
