"""
Multi-Fidelity Optimization with 3-Level Pyramid Grid.

Key hypothesis: Using literature-recommended 4:1 cell ratios between levels
can give significant speedup while maintaining accuracy.

Grid pyramid:
- Level 1 (coarse): 25x12 = 300 cells (~17x faster than fine)
- Level 2 (medium): 50x25 = 1250 cells (~4x faster than fine)
- Level 3 (fine): 100x50 = 5000 cells (baseline)

Strategy:
1. Run CMA-ES on COARSE grid to explore broad parameter space
2. Transfer top candidates to MEDIUM grid for refinement
3. Final evaluation on FINE grid only for top N candidates
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
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
                                    q_range=(0.5, 2.0), early_fraction=1.0):
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
    n_early = max(1, int(len(Y_unit) * early_fraction))
    rmse_early = np.sqrt(np.mean((q_optimal * Y_unit[:n_early] - Y_observed[:n_early]) ** 2))
    rmse_full = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return q_optimal, Y_pred, rmse_early, rmse_full


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0), early_fraction=1.0):
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
    n_early = max(1, int(len(Y1) * early_fraction))
    rmse_early = np.sqrt(np.mean((q1 * Y1[:n_early] + q2 * Y2[:n_early] - Y_observed[:n_early]) ** 2))
    rmse_full = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return (q1, q2), Y_pred, rmse_early, rmse_full


class MultiFidelityPyramidOptimizer:
    """
    Multi-fidelity optimizer with 3-level grid pyramid.

    Grid levels (4:1 ratio between levels):
    - Coarse: 25x12 = 300 cells (~17x faster than fine)
    - Medium: 50x25 = 1250 cells (~4x faster than fine)
    - Fine: 100x50 = 5000 cells (baseline)

    Strategy:
    1. Explore on coarse grid with CMA-ES
    2. Refine top candidates on medium grid
    3. Final evaluation on fine grid
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        # Grid sizes for 3-level pyramid
        nx_coarse: int = 25,
        ny_coarse: int = 12,
        nx_medium: int = 50,
        ny_medium: int = 25,
        nx_fine: int = 100,
        ny_fine: int = 50,
        # CMA-ES budget per level
        coarse_fevals_1src: int = 10,
        coarse_fevals_2src: int = 16,
        medium_fevals_1src: int = 8,
        medium_fevals_2src: int = 12,
        # Transfer settings
        transfer_top_n: int = 5,  # Transfer top N from coarse to medium
        fine_top_n: int = 3,       # Evaluate top N on fine grid
        # CMA-ES settings
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        early_fraction: float = 0.3,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.nx_medium = nx_medium
        self.ny_medium = ny_medium
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.coarse_fevals_1src = coarse_fevals_1src
        self.coarse_fevals_2src = coarse_fevals_2src
        self.medium_fevals_1src = medium_fevals_1src
        self.medium_fevals_2src = medium_fevals_2src
        self.transfer_top_n = transfer_top_n
        self.fine_top_n = fine_top_n
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.early_fraction = early_fraction

    def _create_solver(self, kappa, bc, level='fine'):
        if level == 'coarse':
            return Heat2D(self.Lx, self.Ly, self.nx_coarse, self.ny_coarse, kappa, bc=bc)
        elif level == 'medium':
            return Heat2D(self.Lx, self.Ly, self.nx_medium, self.ny_medium, kappa, bc=bc)
        else:  # fine
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

    def _run_cmaes_on_level(self, objective_func, init_params, n_sources, max_fevals, sigma0, lb, ub):
        """Run CMA-ES on a single fidelity level."""
        all_solutions = []

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

        while not es.stop():
            solutions = es.ask()
            fitness = [objective_func(s) for s in solutions]
            es.tell(solutions, fitness)
            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit))

        # Sort by fitness
        all_solutions.sort(key=lambda x: x[1])
        return all_solutions

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create solvers for each level
        solver_coarse = self._create_solver(kappa, bc, 'coarse')
        solver_medium = self._create_solver(kappa, bc, 'medium')
        solver_fine = self._create_solver(kappa, bc, 'fine')

        n_sims = [0]
        early_frac = self.early_fraction

        # Define objective functions for each level
        def make_objective(solver, n_sources):
            if n_sources == 1:
                def objective(xy_params):
                    x, y = xy_params
                    n_sims[0] += 1
                    q, Y_pred, rmse_early, rmse_full = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                        early_fraction=early_frac)
                    return rmse_early
            else:
                def objective(xy_params):
                    x1, y1, x2, y2 = xy_params
                    n_sims[0] += 2
                    (q1, q2), Y_pred, rmse_early, rmse_full = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                        early_fraction=early_frac)
                    return rmse_early
            return objective

        objective_coarse = make_objective(solver_coarse, n_sources)
        objective_medium = make_objective(solver_medium, n_sources)

        # Get initializations
        initializations = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        lb, ub = self._get_position_bounds(n_sources)

        # Get parameters for this problem type
        coarse_fevals = self.coarse_fevals_1src if n_sources == 1 else self.coarse_fevals_2src
        medium_fevals = self.medium_fevals_1src if n_sources == 1 else self.medium_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        # ===== LEVEL 1: COARSE GRID =====
        # Run CMA-ES on coarse grid for each initialization
        coarse_solutions = []
        fevals_per_init = max(5, coarse_fevals // len(initializations))

        for init_params, init_type in initializations:
            solutions = self._run_cmaes_on_level(
                objective_coarse, init_params, n_sources,
                fevals_per_init, sigma0, lb, ub
            )
            for sol, fit in solutions:
                coarse_solutions.append((sol, fit, init_type))

        # Sort and take top N for transfer
        coarse_solutions.sort(key=lambda x: x[1])
        transfer_candidates = coarse_solutions[:self.transfer_top_n]

        if verbose:
            print(f"  [Coarse] Best RMSE: {transfer_candidates[0][1]:.4f}")

        # ===== LEVEL 2: MEDIUM GRID =====
        # Refine top candidates on medium grid
        medium_solutions = []
        fevals_per_candidate = max(3, medium_fevals // len(transfer_candidates))

        for pos_params, rmse_coarse, init_type in transfer_candidates:
            # Use smaller sigma since we're refining
            solutions = self._run_cmaes_on_level(
                objective_medium, pos_params, n_sources,
                fevals_per_candidate, sigma0 * 0.5, lb, ub  # Reduced sigma for refinement
            )
            for sol, fit in solutions:
                medium_solutions.append((sol, fit, 'medium_refined'))

        # Sort and take top N for fine evaluation
        medium_solutions.sort(key=lambda x: x[1])
        fine_candidates = medium_solutions[:self.fine_top_n]

        if verbose:
            print(f"  [Medium] Best RMSE: {fine_candidates[0][1]:.4f}")

        # ===== LEVEL 3: FINE GRID (evaluation only) =====
        # Evaluate top candidates on fine grid
        candidates_raw = []

        for pos_params, rmse_medium, init_type in fine_candidates:
            if n_sources == 1:
                x, y = pos_params
                q, _, rmse_early, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=1.0)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, rmse_early, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=1.0)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        if verbose:
            print(f"  [Fine] Best RMSE: {min(c[2] for c in candidates_raw):.4f}")

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
                n_evals=n_sims[0] // len(final_candidates) if final_candidates else n_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims[0]
