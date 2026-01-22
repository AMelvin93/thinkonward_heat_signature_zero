"""
Adaptive Population Size CMA-ES Optimizer

Hypothesis: Start with larger popsize (12) for exploration in first 60% of budget,
then reduce to default (~6-8) for exploitation in remaining 40%.

Different from IPOP (which increases). This starts large then reduces.
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


class AdaptivePopsizeOptimizer:
    """
    CMA-ES with adaptive population size: start large for exploration, reduce for exploitation.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        fevals_1src: int = 20,
        fevals_2src: int = 36,
        sigma_1src: float = 0.18,
        sigma_2src: float = 0.22,
        candidate_pool_size: int = 10,
        refine_maxiter: int = 8,
        refine_top_n: int = 3,
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
        timestep_fraction: float = 0.40,
        popsize_phase1: int = 12,  # Large popsize for exploration
        popsize_phase2: int = 6,   # Default popsize for exploitation
        phase1_fraction: float = 0.60,  # 60% of budget for exploration
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.fevals_1src = fevals_1src
        self.fevals_2src = fevals_2src
        self.sigma_1src = sigma_1src
        self.sigma_2src = sigma_2src
        self.candidate_pool_size = candidate_pool_size
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.timestep_fraction = timestep_fraction
        self.popsize_phase1 = popsize_phase1
        self.popsize_phase2 = popsize_phase2
        self.phase1_fraction = phase1_fraction

    def _get_smart_inits(self, readings, sensors, n_sources):
        inits = []
        avg_temps = np.mean(readings, axis=0)
        hottest_idx = np.argmax(avg_temps)
        hot_sensor = sensors[hottest_idx]

        if n_sources == 1:
            inits.append((hot_sensor.copy(), 'hottest'))
        else:
            # For 2-source, need 4 params
            second_hot = sensors[np.argsort(avg_temps)[::-1][1]]
            inits.append((np.concatenate([hot_sensor, second_hot]), 'hottest'))

        if n_sources == 2:
            try:
                tri_init = triangulation_init(readings, sensors, n_sources=2)
                if tri_init is not None:
                    inits.append((tri_init.flatten()[:4], 'triangulation'))
            except:
                pass

            sorted_idx = np.argsort(avg_temps)[::-1]
            two_hot = sensors[sorted_idx[:2]].flatten()
            inits.append((two_hot, 'two_hot'))

            weights = avg_temps - avg_temps.min() + 0.1
            centroid = np.average(sensors, axis=0, weights=weights)
            combined = np.concatenate([hot_sensor, centroid])
            inits.append((combined, 'centroid_hot'))

        return inits

    def _run_adaptive_cmaes(self, objective, init_params, sigma, total_fevals, bounds):
        """
        Run CMA-ES with two phases: large popsize then small popsize.
        """
        # Calculate feval budget for each phase
        phase1_fevals = int(total_fevals * self.phase1_fraction)
        phase2_fevals = total_fevals - phase1_fevals

        all_evaluated = []

        # Phase 1: Large popsize for exploration
        opts1 = {
            'bounds': list(zip(*bounds)),
            'maxfevals': phase1_fevals,
            'popsize': self.popsize_phase1,
            'verbose': -9,
            'seed': 42
        }

        es1 = cma.CMAEvolutionStrategy(init_params.tolist(), sigma, opts1)

        while not es1.stop():
            X = es1.ask()
            fitnesses = [objective(x) for x in X]
            for x, f in zip(X, fitnesses):
                all_evaluated.append((np.array(x), f))
            es1.tell(X, fitnesses)

        result1 = es1.result
        best_x_phase1 = result1.xbest
        best_f_phase1 = result1.fbest
        sigma_phase1 = result1.stds.mean()  # Use average std from phase 1

        # Phase 2: Small popsize for exploitation (continue from best)
        if phase2_fevals > 0:
            opts2 = {
                'bounds': list(zip(*bounds)),
                'maxfevals': phase2_fevals,
                'popsize': self.popsize_phase2,
                'verbose': -9,
                'seed': 43
            }

            # Use reduced sigma for exploitation
            sigma_phase2 = min(sigma_phase1, sigma * 0.5)

            es2 = cma.CMAEvolutionStrategy(best_x_phase1.tolist(), sigma_phase2, opts2)

            while not es2.stop():
                X = es2.ask()
                fitnesses = [objective(x) for x in X]
                for x, f in zip(X, fitnesses):
                    all_evaluated.append((np.array(x), f))
                es2.tell(X, fitnesses)

            result2 = es2.result
            final_best_x = result2.xbest
            final_best_f = result2.fbest
        else:
            final_best_x = best_x_phase1
            final_best_f = best_f_phase1

        return final_best_x, final_best_f, all_evaluated

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=True):
        n_sources = sample['n_sources']
        Y_observed = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])

        dt = meta['dt']
        nt_full = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']

        solver_coarse = Heat2D(
            Lx=self.Lx, Ly=self.Ly, nx=self.nx_coarse, ny=self.ny_coarse,
            kappa=kappa, bc=bc
        )
        solver_fine = Heat2D(
            Lx=self.Lx, Ly=self.Ly, nx=100, ny=50, kappa=kappa, bc=bc
        )

        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        inits = self._get_smart_inits(Y_observed, sensors_xy, n_sources)

        if n_sources == 1:
            bounds = [[0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05]]
            sigma = self.sigma_1src
            fevals = self.fevals_1src
        else:
            bounds = [[0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05],
                      [0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05]]
            sigma = self.sigma_2src
            fevals = self.fevals_2src

        n_sims = 0

        if n_sources == 1:
            def objective(xy_params):
                nonlocal n_sims
                n_sims += 1
                x, y = xy_params
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                nonlocal n_sims
                n_sims += 2
                x1, y1, x2, y2 = xy_params
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

        # Run adaptive CMA-ES for each init
        all_solutions = []
        for init_params, init_type in inits:
            best_x, best_f, _ = self._run_adaptive_cmaes(
                objective, init_params, sigma, fevals, bounds
            )
            all_solutions.append((best_x, best_f, init_type))

        # Sort by RMSE
        all_solutions.sort(key=lambda x: x[1])

        # NM refinement (same as baseline)
        if n_sources == 1:
            def objective_nm(xy_params):
                nonlocal n_sims
                n_sims += 1
                x, y = xy_params
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_nm(xy_params):
                nonlocal n_sims
                n_sims += 2
                x1, y1, x2, y2 = xy_params
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse

        refined_solutions = []
        for i, (pos_params, rmse_coarse, init_type) in enumerate(all_solutions[:self.refine_top_n]):
            bounds_nm = [(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
            try:
                result = minimize(
                    objective_nm, pos_params, method='Nelder-Mead',
                    options={'maxiter': self.refine_maxiter, 'xatol': 0.01, 'fatol': 0.001}
                )
                if result.fun < rmse_coarse:
                    refined_solutions.append((result.x, result.fun, init_type))
                else:
                    refined_solutions.append((pos_params, rmse_coarse, init_type))
            except:
                refined_solutions.append((pos_params, rmse_coarse, init_type))

        for pos_params, rmse_coarse, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

        # Final evaluation with full simulation
        candidates_raw = []
        for pos_params, _, init_type in refined_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                sources = [(x, y, q)]
                full_params = np.array([x, y, q])
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                sources = [(x1, y1, q1), (x2, y2, q2)]
                full_params = np.array([x1, y1, q1, x2, y2, q2])

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        # Filter for diversity
        candidates_for_filter = [(c[0], c[2]) for c in candidates_raw]
        filtered = filter_dissimilar(candidates_for_filter, tau=TAU, n_max=N_MAX)

        # Fallback
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            fallback_sigma = sigma * 1.5
            fallback_fevals = fevals * 2

            for init_params, init_type in inits:
                try:
                    best_x, best_f, _ = self._run_adaptive_cmaes(
                        objective, init_params, fallback_sigma, fallback_fevals, bounds
                    )

                    if n_sources == 1:
                        x, y = best_x
                        q, _, final_rmse = compute_optimal_intensity_1src(
                            x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                        sources = [(x, y, q)]
                        full_params = np.array([x, y, q])
                    else:
                        x1, y1, x2, y2 = best_x
                        (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                            x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                        sources = [(x1, y1, q1), (x2, y2, q2)]
                        full_params = np.array([x1, y1, q1, x2, y2, q2])

                    candidates_raw.append((sources, full_params, final_rmse, init_type + '_fallback'))
                except:
                    pass

            candidates_for_filter = [(c[0], c[2]) for c in candidates_raw]
            filtered = filter_dissimilar(candidates_for_filter, tau=TAU, n_max=N_MAX)

        # Build final results
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
                n_evals=fevals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims
