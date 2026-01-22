"""
Adaptive Simulated Annealing Optimizer for Heat Source Inverse Problem

Uses scipy.optimize.dual_annealing (generalized SA with local search) as the
global optimizer instead of CMA-ES. Based on 2024 Nature paper showing ASA
successfully reconstructs internal heat sources.

Key differences from CMA-ES:
- SA uses probabilistic uphill moves to escape local minima
- dual_annealing combines SA with local search (L-BFGS-B)
- Different exploration mechanism may find different optima
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
from scipy.optimize import dual_annealing, minimize

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
    """Compute optimal intensity for 1-source using reduced timesteps."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)

    # Use only as many timesteps as we simulated
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
    """Compute optimal intensities for 2-source using reduced timesteps."""
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)

    # Use only as many timesteps as we simulated
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


class DualAnnealingOptimizer:
    """
    Optimizer that uses scipy's dual_annealing for global optimization.

    dual_annealing combines simulated annealing with local search (L-BFGS-B).
    It explores parameter space using probabilistic uphill moves, which may
    find different optima than CMA-ES's covariance-adaptive approach.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 100,
        max_fevals_2src: int = 200,
        n_restarts: int = 3,
        refine_maxiter: int = 8,
        refine_top_n: int = 2,
        timestep_fraction: float = 0.4,
        initial_temp: float = 5000,
        restart_temp_ratio: float = 0.00001,
        visit: float = 2.62,
        accept: float = -5.0,
        no_local_search: bool = False,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.n_restarts = n_restarts
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.timestep_fraction = timestep_fraction

        # Simulated annealing parameters
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit  # Controls step length distribution (2.62 = Cauchy-like)
        self.accept = accept  # Controls acceptance probability
        self.no_local_search = no_local_search

    def _get_bounds(self, n_sources, margin=0.05):
        """Get bounds for source positions."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
        return bounds

    def _smart_init(self, sample, n_sources):
        """Initialize from hottest sensors."""
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
        """Initialize using triangulation method."""
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except:
            return None

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        """
        Estimate heat source parameters using dual_annealing.

        Returns: (candidate_sources, best_rmse, results, n_sims)
        """
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']
        dt = meta['dt']

        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        # Create solvers for different fidelities
        solver_full = Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

        nt_opt = max(10, int(nt_full * self.timestep_fraction))

        bounds = self._get_bounds(n_sources)
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src

        n_sims = 0
        all_candidates = []

        # Create objective function
        if n_sources == 1:
            def objective(xy_params):
                nonlocal n_sims
                x, y = xy_params
                n_sims += 1
                _, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_full, dt, nt_opt, T0, sensors_xy, q_range
                )
                return rmse
        else:
            def objective(xy_params):
                nonlocal n_sims
                x1, y1, x2, y2 = xy_params
                n_sims += 2
                _, _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_full, dt, nt_opt, T0, sensors_xy, q_range
                )
                return rmse

        # Get initial points
        tri_init = self._triangulation_init(sample, meta, n_sources, q_range)
        smart_init = self._smart_init(sample, n_sources)

        # Run dual_annealing with different seeds
        for restart_idx in range(self.n_restarts):
            try:
                # Use different initial guess for seeding
                if restart_idx == 0 and tri_init is not None:
                    x0 = tri_init
                    init_type = 'triangulation'
                elif restart_idx == 1:
                    x0 = smart_init
                    init_type = 'smart'
                else:
                    # Random initialization
                    x0 = np.array([
                        np.random.uniform(b[0], b[1]) for b in bounds
                    ])
                    init_type = f'random_{restart_idx}'

                result = dual_annealing(
                    objective,
                    bounds=bounds,
                    maxfun=max_fevals // self.n_restarts,
                    x0=x0,
                    initial_temp=self.initial_temp,
                    restart_temp_ratio=self.restart_temp_ratio,
                    visit=self.visit,
                    accept=self.accept,
                    no_local_search=self.no_local_search,
                    seed=restart_idx,
                )

                all_candidates.append((result.x, result.fun, init_type, restart_idx))

                if verbose:
                    print(f"  {init_type}: RMSE={result.fun:.4f} after {result.nfev} fevals")

            except Exception as e:
                if verbose:
                    print(f"  {init_type}: ERROR - {e}")

        if not all_candidates:
            # Fallback
            x0 = smart_init
            result = dual_annealing(
                objective,
                bounds=bounds,
                maxfun=max_fevals,
                x0=x0,
                seed=42,
            )
            all_candidates.append((result.x, result.fun, 'fallback', -1))

        # Nelder-Mead refinement on top candidates (full timesteps)
        all_candidates.sort(key=lambda x: x[1])
        refined_candidates = []

        for idx, (pos_params, opt_rmse, init_type, restart_idx) in enumerate(all_candidates[:self.refine_top_n]):
            if n_sources == 1:
                def refine_obj(xy_params):
                    nonlocal n_sims
                    x, y = xy_params
                    n_sims += 1
                    _, _, rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_full, dt, nt_full, T0, sensors_xy, q_range
                    )
                    return rmse
            else:
                def refine_obj(xy_params):
                    nonlocal n_sims
                    x1, y1, x2, y2 = xy_params
                    n_sims += 2
                    _, _, rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_full, dt, nt_full, T0, sensors_xy, q_range
                    )
                    return rmse

            try:
                refined_result = minimize(
                    refine_obj,
                    pos_params,
                    method='Nelder-Mead',
                    options={'maxiter': self.refine_maxiter, 'adaptive': True}
                )
                refined_candidates.append((refined_result.x, refined_result.fun, init_type))
            except:
                # If refinement fails, use original
                full_rmse = refine_obj(pos_params)
                refined_candidates.append((pos_params, full_rmse, init_type))

        # Build final candidates with full parameters
        candidates_raw = []
        for pos_params, rmse, init_type in refined_candidates:
            if n_sources == 1:
                x, y = pos_params
                q, _, _ = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_full, dt, nt_full, T0, sensors_xy, q_range
                )
                n_sims += 1
                full_params = np.array([x, y, q])
                sources_list = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, _ = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_full, dt, nt_full, T0, sensors_xy, q_range
                )
                n_sims += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources_list = [(float(x1), float(y1), float(q1)),
                               (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources_list, full_params, rmse, init_type))

        # Filter for dissimilar candidates
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
