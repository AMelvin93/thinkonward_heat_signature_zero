"""
Adaptive NM Iterations Optimizer

Same as TemporalFidelityWithPolishOptimizer but with adaptive NM polish:
- Start with minimum iterations (e.g., 4)
- Continue if RMSE still improving significantly
- Stop early if converged (saves time on easy samples)
- Allow up to max iterations for hard samples

Hypothesis: Fixed 8 NM iterations is wasteful for easy samples, insufficient for hard ones.
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
    polish_iters: int = 0  # Track actual polish iterations used


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


class AdaptiveNMPolishOptimizer:
    """
    Optimizer with adaptive Nelder-Mead polish iterations.

    Key difference from baseline:
    - Instead of fixed 8 NM polish iterations, use adaptive logic:
      1. Start with min_polish_iters (e.g., 4)
      2. Check convergence after each batch
      3. Continue if RMSE still improving significantly
      4. Stop early if converged (delta < threshold)
      5. Allow up to max_polish_iters for hard samples
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
        # Adaptive NM polish parameters
        min_polish_iters: int = 4,       # Start with this many iterations
        max_polish_iters: int = 12,      # Allow up to this many
        polish_batch_size: int = 2,      # Check convergence every N iterations
        convergence_threshold: float = 0.001,  # Stop if RMSE improvement < this
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
        # Adaptive polish
        self.min_polish_iters = min_polish_iters
        self.max_polish_iters = max_polish_iters
        self.polish_batch_size = polish_batch_size
        self.convergence_threshold = convergence_threshold

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

    def _adaptive_nm_polish(self, objective_func, initial_params, initial_rmse):
        """
        Adaptive NM polish: run iterations in batches, check convergence.

        Returns: (polished_params, final_rmse, total_iters_used)
        """
        current_params = initial_params.copy()
        current_rmse = initial_rmse
        total_iters = 0

        # Run minimum iterations first
        result = minimize(
            objective_func,
            current_params,
            method='Nelder-Mead',
            options={
                'maxiter': self.min_polish_iters,
                'xatol': 0.005,
                'fatol': 0.0005
            }
        )
        total_iters = self.min_polish_iters

        if result.fun < current_rmse:
            current_params = result.x
            current_rmse = result.fun

        # Continue with batches if not yet at max and still improving
        while total_iters < self.max_polish_iters:
            prev_rmse = current_rmse

            # Run a batch of iterations
            result = minimize(
                objective_func,
                current_params,
                method='Nelder-Mead',
                options={
                    'maxiter': self.polish_batch_size,
                    'xatol': 0.005,
                    'fatol': 0.0005
                }
            )
            total_iters += self.polish_batch_size

            if result.fun < current_rmse:
                current_params = result.x
                current_rmse = result.fun

            # Check convergence: stop if improvement is below threshold
            improvement = prev_rmse - current_rmse
            if improvement < self.convergence_threshold:
                break

        return current_params, current_rmse, total_iters

    def _run_single_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                                  initializations, n_sources, nt_reduced, nt_full):
        """Run optimization with adaptive NM polish."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]
        polish_iters_used = [0]

        # Objective using COARSE grid with REDUCED timesteps
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

            def objective_fine_full(xy_params):
                """Full timesteps on fine grid for final polish."""
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
                """Full timesteps on fine grid for final polish."""
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_position_bounds(n_sources)
        fevals_per_init = max(5, max_fevals // len(initializations))

        all_solutions = []

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
                    all_solutions.append((np.array(sol), fit, init_type))

        # Sort by coarse fitness
        all_solutions.sort(key=lambda x: x[1])

        # Refine top N solutions on COARSE grid with reduced timesteps (quick)
        refined_solutions = []
        for i, (pos_params, rmse_coarse, init_type) in enumerate(all_solutions[:self.refine_top_n]):
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

        # Add remaining top solutions
        for pos_params, rmse_coarse, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

        # Evaluate candidates on FINE grid with FULL timesteps
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

        # ADAPTIVE NM POLISH: on best candidate using full timesteps
        if self.max_polish_iters > 0 and candidates_raw:
            # Find best candidate
            best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][2])
            best_pos_params = candidates_raw[best_idx][4]
            best_rmse = candidates_raw[best_idx][2]

            # Adaptive polish with full timesteps
            polished_params, polished_rmse, iters_used = self._adaptive_nm_polish(
                objective_fine_full, best_pos_params, best_rmse
            )
            polish_iters_used[0] = iters_used

            if polished_rmse < best_rmse:
                # Update best candidate with polished result
                if n_sources == 1:
                    x, y = polished_params
                    q, _, final_rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 1
                    full_params = np.array([x, y, q])
                    sources = [(float(x), float(y), float(q))]
                else:
                    x1, y1, x2, y2 = polished_params
                    (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 2
                    full_params = np.array([x1, y1, q1, x2, y2, q2])
                    sources = [(float(x1), float(y1), float(q1)),
                              (float(x2), float(y2), float(q2))]

                candidates_raw[best_idx] = (sources, full_params, final_rmse, 'polished', polished_params)

        # Remove pos_params from output
        candidates_raw = [(c[0], c[1], c[2], c[3]) for c in candidates_raw]

        return candidates_raw, n_sims[0], polish_iters_used[0]

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

        # Run optimization with adaptive polish
        candidates_raw, n_sims, polish_iters = self._run_single_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            primary_inits, n_sources, nt_reduced, nt_full
        )

        # Check if result is bad
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            # Fallback
            fallback_inits = []
            centroid_init = self._weighted_centroid_init(sample, n_sources)
            fallback_inits.append((centroid_init, 'centroid'))
            random_init = self._random_init_positions(n_sources)
            fallback_inits.append((random_init, 'random'))

            fallback_candidates, fallback_sims, fallback_polish = self._run_single_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                fallback_inits, n_sources, nt_reduced, nt_full
            )
            n_sims += fallback_sims
            polish_iters = max(polish_iters, fallback_polish)
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
                n_evals=n_sims // len(final_candidates) if final_candidates else n_sims,
                polish_iters=polish_iters
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims, polish_iters
