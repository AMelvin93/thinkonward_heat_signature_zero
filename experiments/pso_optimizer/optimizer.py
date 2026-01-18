"""
Particle Swarm Optimization for Heat Source Identification.

PSO is an alternative to CMA-ES that uses different dynamics:
- Particles move through search space
- Each particle is attracted to personal best and global best
- May converge faster for low-dimensional (2D/4D) problems

Based on: experiments/robust_fallback/optimizer.py structure
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
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


class SimplePSO:
    """
    Simple Particle Swarm Optimization implementation.

    Parameters:
    - n_particles: number of particles in swarm
    - max_iter: maximum iterations
    - w: inertia weight
    - c1: cognitive coefficient (personal best attraction)
    - c2: social coefficient (global best attraction)
    """

    def __init__(self, n_particles=10, max_iter=10, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, objective, bounds, x0=None, max_fevals=None):
        """
        Minimize objective within bounds.

        Args:
            objective: function to minimize
            bounds: (lower_bounds, upper_bounds) arrays
            x0: optional initial position for one particle
            max_fevals: optional max function evaluations

        Returns:
            best_position, best_fitness, n_evals, all_solutions
        """
        lb, ub = np.array(bounds[0]), np.array(bounds[1])
        dim = len(lb)

        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.n_particles, dim))
        if x0 is not None:
            positions[0] = np.clip(x0, lb, ub)

        velocities = np.random.uniform(-(ub - lb) * 0.1, (ub - lb) * 0.1, (self.n_particles, dim))

        # Evaluate initial positions
        fitness = np.array([objective(p) for p in positions])
        n_evals = self.n_particles

        # Personal best
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()

        # Global best
        gbest_idx = np.argmin(fitness)
        gbest_position = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]

        # Track all solutions
        all_solutions = [(positions[i].copy(), fitness[i]) for i in range(self.n_particles)]

        # Main loop
        for iteration in range(self.max_iter):
            if max_fevals and n_evals >= max_fevals:
                break

            # Update velocities
            r1 = np.random.random((self.n_particles, dim))
            r2 = np.random.random((self.n_particles, dim))

            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (gbest_position - positions)
            velocities = self.w * velocities + cognitive + social

            # Clamp velocities to avoid explosion
            max_vel = (ub - lb) * 0.2
            velocities = np.clip(velocities, -max_vel, max_vel)

            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            # Evaluate new positions
            fitness = np.array([objective(p) for p in positions])
            n_evals += self.n_particles

            # Update personal bests
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            # Update global best
            if np.min(fitness) < gbest_fitness:
                gbest_idx = np.argmin(fitness)
                gbest_position = positions[gbest_idx].copy()
                gbest_fitness = fitness[gbest_idx]

            # Track solutions
            for i in range(self.n_particles):
                all_solutions.append((positions[i].copy(), fitness[i]))

            if max_fevals and n_evals >= max_fevals:
                break

        return gbest_position, gbest_fitness, n_evals, all_solutions


class PSOOptimizer:
    """
    PSO-based optimizer for heat source identification.
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
        n_particles: int = 8,
        pso_iterations: int = 8,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        early_fraction: float = 0.3,
        refine_maxiter: int = 3,
        refine_top_n: int = 2,
        rmse_threshold_1src: float = 0.35,
        rmse_threshold_2src: float = 0.45,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.n_particles = n_particles
        self.pso_iterations = pso_iterations
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.early_fraction = early_fraction
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
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

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]
        early_frac = self.early_fraction

        # Objective using COARSE grid
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse_early, rmse_full = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=early_frac)
                return rmse_early
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse_early, rmse_full = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=early_frac)
                return rmse_early

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        lb, ub = self._get_position_bounds(n_sources)

        # Get initial position from triangulation or smart init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        x0 = tri_init if tri_init is not None else self._smart_init_positions(sample, n_sources)

        # Run PSO
        pso = SimplePSO(
            n_particles=self.n_particles,
            max_iter=self.pso_iterations,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

        best_pos, best_fit, n_evals_pso, all_solutions = pso.optimize(
            objective_coarse,
            bounds=(lb, ub),
            x0=x0,
            max_fevals=max_fevals
        )

        # Sort all solutions by fitness
        all_solutions.sort(key=lambda x: x[1])

        # Refine top N solutions on COARSE grid
        refined_solutions = []
        for i, (pos_params, rmse_coarse) in enumerate(all_solutions[:self.refine_top_n]):
            if self.refine_maxiter > 0:
                result = minimize(
                    objective_coarse,
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
                    refined_solutions.append((pos_params, rmse_coarse, 'pso'))
            else:
                refined_solutions.append((pos_params, rmse_coarse, 'pso'))

        # Add remaining top solutions
        for pos_params, rmse_coarse in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, 'pso'))

        # Evaluate candidates on FINE grid
        candidates_raw = []
        for pos_params, rmse_coarse, init_type in refined_solutions:
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

        # Check if result is bad - run fallback
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            # Fallback: try with different initial positions
            centroid_init = self._weighted_centroid_init(sample, n_sources)
            random_init = self._random_init_positions(n_sources)

            for x0_fallback in [centroid_init, random_init]:
                _, _, _, fallback_solutions = pso.optimize(
                    objective_coarse,
                    bounds=(lb, ub),
                    x0=x0_fallback,
                    max_fevals=max_fevals // 2  # Use half budget for fallback
                )

                # Evaluate top fallback solutions on fine grid
                fallback_solutions.sort(key=lambda x: x[1])
                for pos_params, rmse_coarse in fallback_solutions[:3]:
                    if n_sources == 1:
                        x, y = pos_params
                        q, _, _, final_rmse = compute_optimal_intensity_1src(
                            x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                        n_sims[0] += 1
                        sources = [(float(x), float(y), float(q))]
                        full_params = np.array([x, y, q])
                    else:
                        x1, y1, x2, y2 = pos_params
                        (q1, q2), _, _, final_rmse = compute_optimal_intensity_2src(
                            x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                        n_sims[0] += 2
                        sources = [(float(x1), float(y1), float(q1)),
                                  (float(x2), float(y2), float(q2))]
                        full_params = np.array([x1, y1, q1, x2, y2, q2])

                    candidates_raw.append((sources, full_params, final_rmse, 'fallback'))

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
