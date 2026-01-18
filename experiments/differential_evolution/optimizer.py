"""
Differential Evolution Optimizer.

Uses scipy.optimize.differential_evolution as an alternative to CMA-ES.
DE is simpler and may converge faster for low-dimensional problems.

Key differences from CMA-ES:
- Uses mutation and crossover instead of covariance adaptation
- Population-based like CMA-ES, so can evaluate in parallel
- May converge faster on simpler landscapes
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
from scipy.optimize import differential_evolution, minimize

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


class DifferentialEvolutionOptimizer:
    """
    Differential Evolution optimizer using scipy.optimize.differential_evolution.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        strategy: str = 'best1bin',
        maxiter_1src: int = 8,
        maxiter_2src: int = 12,
        popsize: int = 3,
        mutation: tuple = (0.5, 1.0),
        recombination: float = 0.7,
        polish: bool = False,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        rmse_threshold_1src: float = 0.40,
        rmse_threshold_2src: float = 0.50,
        tol: float = 0.01,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.strategy = strategy
        self.maxiter_1src = maxiter_1src
        self.maxiter_2src = maxiter_2src
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.polish = polish
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.tol = tol

    def _create_solver(self, kappa, bc):
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources, margin=0.05):
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
        return bounds

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

    def _create_initial_population(self, bounds, init_points, ndim, popsize_actual):
        """Create initial population with good starting points."""
        n_init = len(init_points)
        pop = []

        # Add known good starting points
        for init_params in init_points:
            if len(pop) < popsize_actual:
                # Normalize to [0, 1] range for DE
                normalized = []
                for i, (lo, hi) in enumerate(bounds):
                    val = (init_params[i] - lo) / (hi - lo)
                    normalized.append(np.clip(val, 0, 1))
                pop.append(normalized)

        # Fill rest with random points
        while len(pop) < popsize_actual:
            pop.append(np.random.random(ndim))

        return np.array(pop)

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']

        solver = self._create_solver(kappa, bc)

        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]
        bounds = self._get_position_bounds(n_sources)
        ndim = n_sources * 2
        maxiter = self.maxiter_1src if n_sources == 1 else self.maxiter_2src

        # Objective function
        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Get initial points
        init_points = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            init_points.append(tri_init)
        smart_init = self._smart_init_positions(sample, n_sources)
        init_points.append(smart_init)

        # Calculate actual population size
        popsize_actual = max(self.popsize * (ndim + 1), len(init_points) + 2)

        # Create initial population with good starting points
        init_pop = self._create_initial_population(bounds, init_points, ndim, popsize_actual)

        all_results = []

        # Run DE
        try:
            result = differential_evolution(
                objective,
                bounds,
                strategy=self.strategy,
                maxiter=maxiter,
                popsize=self.popsize,
                mutation=self.mutation,
                recombination=self.recombination,
                polish=self.polish,
                init=init_pop,
                tol=self.tol,
                seed=np.random.randint(0, 10000),
                workers=1,  # Don't parallelize DE internally
                updating='deferred',  # More stable
            )
            all_results.append((result.x, result.fun, 'de_primary'))
        except Exception as e:
            if verbose:
                print(f"DE failed: {e}")

        # Check if result is bad - try with different init
        if all_results:
            best_rmse = min(r[1] for r in all_results)
            threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

            if best_rmse > threshold:
                # Try with random init
                try:
                    result = differential_evolution(
                        objective,
                        bounds,
                        strategy=self.strategy,
                        maxiter=maxiter,
                        popsize=self.popsize,
                        mutation=self.mutation,
                        recombination=self.recombination,
                        polish=self.polish,
                        init='random',
                        tol=self.tol,
                        seed=np.random.randint(0, 10000) + 1,
                        workers=1,
                    )
                    all_results.append((result.x, result.fun, 'de_random'))
                except:
                    pass

        if not all_results:
            # Fallback: simple Nelder-Mead from smart init
            smart_init = self._smart_init_positions(sample, n_sources)
            result = minimize(objective, smart_init, method='Nelder-Mead',
                            options={'maxiter': 50})
            all_results.append((result.x, result.fun, 'fallback'))

        # Build candidate list with full parameters
        candidates_raw = []
        for pos_params, rmse, init_type in all_results:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
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
