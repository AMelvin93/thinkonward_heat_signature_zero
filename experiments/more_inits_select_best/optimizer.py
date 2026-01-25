"""
More Initializations with Best Selection Optimizer

Key idea: Run more CMA-ES initializations (6-8 instead of 3) to increase
chance of finding global optimum. Select best 3 by RMSE before dissimilarity
filtering to improve accuracy component of the score.

Trade-off: More inits means fewer fevals per init, but parallel execution
keeps total time similar.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
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


class MoreInitsSelectBestOptimizer:
    """
    Optimizer that runs more CMA-ES initializations and selects best candidates.

    Key difference from baseline:
    - Runs 6-8 initializations (vs 3)
    - Fewer fevals per init (to stay in budget)
    - Selects top 3 by RMSE before dissimilarity filtering
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        n_inits: int = 6,  # NEW: number of initializations
        max_fevals_1src: int = 20,  # Total fevals (distributed across inits)
        max_fevals_2src: int = 36,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        keep_top_n: int = 5,  # Keep top N by RMSE before dissimilarity filter
        refine_maxiter: int = 3,
        refine_top_n: int = 2,
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
        timestep_fraction: float = 0.40,
        final_polish_maxiter: int = 8,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.n_inits = n_inits
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.keep_top_n = keep_top_n
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
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
        return lb, ub

    def _generate_initializations(self, sample, meta, n_sources, q_range):
        """Generate diverse initializations for CMA-ES."""
        inits = []
        
        # 1. Triangulation-based init (if available)
        if self.use_triangulation:
            try:
                full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
                positions = []
                for i in range(n_sources):
                    positions.extend([full_init[i*3], full_init[i*3 + 1]])
                inits.append((np.array(positions), 'triangulation'))
            except:
                pass
        
        # 2. Smart init (hottest sensors)
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
        inits.append((np.array(params), 'smart'))
        
        # 3. Weighted centroid
        max_temps = np.max(readings, axis=0)
        weights = max_temps / (max_temps.sum() + 1e-8)
        centroid = np.average(np.array(sensors), axis=0, weights=weights)
        
        if n_sources == 1:
            inits.append((np.array([centroid[0], centroid[1]]), 'centroid'))
        else:
            spread = np.sqrt(np.average(
                (np.array(sensors)[:, 0] - centroid[0])**2 + 
                (np.array(sensors)[:, 1] - centroid[1])**2,
                weights=weights
            ))
            offset = max(0.1, spread * 0.3)
            inits.append((np.array([
                centroid[0] - offset, centroid[1],
                centroid[0] + offset, centroid[1]
            ]), 'centroid'))
        
        # 4-N. Random initializations to reach n_inits
        margin = 0.1
        while len(inits) < self.n_inits:
            params = []
            for _ in range(n_sources):
                x = margin * self.Lx + np.random.random() * (1 - 2*margin) * self.Lx
                y = margin * self.Ly + np.random.random() * (1 - 2*margin) * self.Ly
                params.extend([x, y])
            inits.append((np.array(params), f'random_{len(inits)-3}'))
        
        return inits[:self.n_inits]

    def _run_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                          initializations, n_sources, nt_reduced, nt_full):
        """Run CMA-ES from all initializations and select best candidates."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]

        # Objective using COARSE grid with REDUCED timesteps
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

        # Distribute fevals across initializations
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        fevals_per_init = max(3, max_fevals // len(initializations))
        
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_position_bounds(n_sources)

        all_solutions = []

        for init_params, init_type in initializations:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            best_pos = init_params
            best_rmse = float('inf')
            
            while not es.stop():
                solutions = es.ask()
                fitness = [objective_coarse(s) for s in solutions]
                es.tell(solutions, fitness)
                
                # Track best from this init
                for sol, fit in zip(solutions, fitness):
                    if fit < best_rmse:
                        best_rmse = fit
                        best_pos = np.array(sol)
            
            all_solutions.append((best_pos, best_rmse, init_type))

        # Sort by coarse fitness and keep top N
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.keep_top_n]

        # Quick NM refine on top solutions (coarse, reduced timesteps)
        refined_solutions = []
        for pos_params, rmse_coarse, init_type in top_solutions:
            if self.refine_maxiter > 0:
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

        # Evaluate on FINE grid with FULL timesteps
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

            candidates_raw.append((sources, full_params, final_rmse, init_type, pos_params))

        # Final polish on best candidate with full timesteps
        if self.final_polish_maxiter > 0 and candidates_raw:
            best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][2])
            best_pos_params = candidates_raw[best_idx][4]
            best_rmse = candidates_raw[best_idx][2]

            result = minimize(
                objective_fine_full,
                best_pos_params,
                method='Nelder-Mead',
                options={'maxiter': self.final_polish_maxiter, 'xatol': 0.005, 'fatol': 0.0005}
            )

            if result.fun < best_rmse:
                if n_sources == 1:
                    x, y = result.x
                    q, _, final_rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 1
                    full_params = np.array([x, y, q])
                    sources = [(float(x), float(y), float(q))]
                else:
                    x1, y1, x2, y2 = result.x
                    (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 2
                    full_params = np.array([x1, y1, q1, x2, y2, q2])
                    sources = [(float(x1), float(y1), float(q1)),
                              (float(x2), float(y2), float(q2))]

                candidates_raw[best_idx] = (sources, full_params, final_rmse, 'polished', result.x)

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

        # Generate all initializations
        initializations = self._generate_initializations(sample, meta, n_sources, q_range)

        if verbose:
            print(f"  Running {len(initializations)} initializations with "
                  f"{self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src} total fevals")

        # Run optimization
        candidates_raw, n_sims = self._run_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            initializations, n_sources, nt_reduced, nt_full
        )

        # Check for fallback (if result is bad)
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            # Generate more random inits as fallback
            fallback_inits = []
            margin = 0.1
            for i in range(3):
                params = []
                for _ in range(n_sources):
                    x = margin * self.Lx + np.random.random() * (1 - 2*margin) * self.Lx
                    y = margin * self.Ly + np.random.random() * (1 - 2*margin) * self.Ly
                    params.extend([x, y])
                fallback_inits.append((np.array(params), f'fallback_{i}'))

            fallback_candidates, fallback_sims = self._run_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                fallback_inits, n_sources, nt_reduced, nt_full
            )
            n_sims += fallback_sims
            candidates_raw.extend(fallback_candidates)

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
