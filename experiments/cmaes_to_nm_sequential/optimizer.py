"""
CMA-ES to Nelder-Mead Sequential Handoff Optimizer.

Key Innovation: Use CMA-ES for global exploration with reduced budget, then
hand off the best result to Nelder-Mead for local refinement. This captures
the benefits of both optimizers without doubling simulation count.

Hypothesis: CMA-ES is great for exploration but NM is better at local refinement.
Ensemble experiment showed NM won 74% but running parallel doubled simulations.
Sequential handoff avoids this overhead.

Informed by: EXP_ENSEMBLE_001 findings - NM should be used for REFINEMENT ONLY.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma
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


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
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
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return (q1, q2), Y_pred, rmse


class SequentialHandoffOptimizer:
    """
    CMA-ES global exploration -> Nelder-Mead local refinement.

    Key differences from baseline:
    1. REDUCED CMA-ES budget (global exploration only)
    2. INCREASED NM budget (local refinement is what matters)
    3. Sequential (not parallel) to avoid doubling simulations
    4. NM starts from CMA-ES best, not random inits
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        # REDUCED CMA-ES budget for exploration
        cmaes_fevals_1src: int = 15,   # Reduced from 20
        cmaes_fevals_2src: int = 28,   # Reduced from 36
        sigma0_1src: float = 0.18,      # Slightly larger for exploration
        sigma0_2src: float = 0.22,      # Slightly larger for exploration
        # MODEST NM budget for refinement (NM on fine grid is expensive!)
        nm_maxiter_1src: int = 8,       # Reduced - NM on fine is expensive
        nm_maxiter_2src: int = 12,      # Reduced - NM on fine is expensive
        nm_top_n: int = 2,              # Refine only top 2 candidates
        nm_use_coarse_grid: bool = False,  # Use coarse grid for NM (faster but less accurate)
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 8,
        rmse_threshold_1src: float = 0.35,
        rmse_threshold_2src: float = 0.45,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.cmaes_fevals_1src = cmaes_fevals_1src
        self.cmaes_fevals_2src = cmaes_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.nm_maxiter_1src = nm_maxiter_1src
        self.nm_maxiter_2src = nm_maxiter_2src
        self.nm_top_n = nm_top_n
        self.nm_use_coarse_grid = nm_use_coarse_grid
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src

    def _create_solver(self, kappa, bc, coarse=False):
        if coarse:
            return Heat2D(self.Lx, self.Ly, self.nx_coarse, self.ny_coarse, kappa, bc=bc)
        return Heat2D(self.Lx, self.Ly, self.nx_fine, self.ny_fine, kappa, bc=bc)

    def _get_position_bounds(self, n_sources, margin=0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

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

    def _weighted_centroid_init(self, sample, n_sources):
        """Alternative init using weighted centroid of all sensors."""
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

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        n_sims = [0]

        # Objective on COARSE grid for CMA-ES exploration
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Objective on FINE grid for NM refinement
        if n_sources == 1:
            def objective_fine(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_fine(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Parameters based on n_sources
        cmaes_fevals = self.cmaes_fevals_1src if n_sources == 1 else self.cmaes_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        nm_maxiter = self.nm_maxiter_1src if n_sources == 1 else self.nm_maxiter_2src
        lb, ub = self._get_position_bounds(n_sources)

        # Build initializations
        initializations = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # Allocate CMA-ES budget across inits
        fevals_per_init = max(5, cmaes_fevals // len(initializations))

        # ================================================================
        # PHASE 1: CMA-ES Global Exploration (reduced budget)
        # ================================================================
        all_cmaes_solutions = []

        for init_params, init_type in initializations:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective_coarse(s) for s in solutions]
                es.tell(solutions, fitness)
                for sol, fit in zip(solutions, fitness):
                    all_cmaes_solutions.append((np.array(sol), fit, init_type))

        # Sort by coarse fitness
        all_cmaes_solutions.sort(key=lambda x: x[1])

        # ================================================================
        # PHASE 2: Nelder-Mead Local Refinement
        # ================================================================
        nm_refined_solutions = []

        # Choose objective for NM: coarse (fast) or fine (accurate)
        nm_objective = objective_coarse if self.nm_use_coarse_grid else objective_fine
        nm_grid_label = "coarse" if self.nm_use_coarse_grid else "fine"

        for pos_params, cmaes_rmse, init_type in all_cmaes_solutions[:self.nm_top_n]:
            # Run Nelder-Mead from CMA-ES best
            result = minimize(
                nm_objective,
                pos_params,
                method='Nelder-Mead',
                options={
                    'maxiter': nm_maxiter,
                    'xatol': 0.005,
                    'fatol': 0.0005,
                }
            )
            refined_pos = np.clip(result.x, lb, ub)
            # Always evaluate final RMSE on fine grid for comparison
            final_rmse = objective_fine(refined_pos) if self.nm_use_coarse_grid else result.fun
            nm_refined_solutions.append((refined_pos, final_rmse, f'{init_type}_nm'))

        # Also keep CMA-ES-only solutions for diversity
        for pos_params, cmaes_rmse, init_type in all_cmaes_solutions[self.nm_top_n:self.candidate_pool_size]:
            # Evaluate on fine grid without NM refinement
            fine_rmse = objective_fine(pos_params)
            nm_refined_solutions.append((pos_params, fine_rmse, init_type))

        # Sort by fine-grid RMSE
        nm_refined_solutions.sort(key=lambda x: x[1])

        # Check if result is bad - trigger fallback
        best_rmse_after_nm = nm_refined_solutions[0][1] if nm_refined_solutions else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_after_nm > threshold:
            # Fallback: try alternative init with full NM budget
            centroid_init = self._weighted_centroid_init(sample, n_sources)

            # Run CMA-ES with centroid init
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9

            es = cma.CMAEvolutionStrategy(centroid_init.tolist(), sigma0 * 1.5, opts)
            while not es.stop():
                solutions = es.ask()
                fitness = [objective_coarse(s) for s in solutions]
                es.tell(solutions, fitness)

            # NM refinement on fallback
            result = minimize(
                nm_objective,
                es.result.xbest,
                method='Nelder-Mead',
                options={'maxiter': nm_maxiter * 2, 'xatol': 0.005, 'fatol': 0.0005}
            )
            fallback_pos = np.clip(result.x, lb, ub)
            fallback_rmse = objective_fine(fallback_pos) if self.nm_use_coarse_grid else result.fun
            nm_refined_solutions.append((fallback_pos, fallback_rmse, 'fallback_nm'))
            nm_refined_solutions.sort(key=lambda x: x[1])

        # Convert to full params with intensities
        candidates_raw = []
        for pos_params, rmse, init_type in nm_refined_solutions[:self.candidate_pool_size]:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        candidate_sources = [c[0] for c in final_candidates]
        best_rmse = min(c[2] for c in final_candidates) if final_candidates else float('inf')

        results = [
            CandidateResult(
                params=c[1], rmse=c[2], init_type=c[3],
                n_evals=n_sims[0] // len(final_candidates) if final_candidates else n_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims[0]
