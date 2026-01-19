"""
Physics-Informed Initialization Optimizer

Uses temperature gradient analysis to provide better initial guesses for
heat source locations instead of random or hottest-sensor initialization.

Key insight: Temperature gradients point toward heat sources. By analyzing
gradient directions at multiple sensors, we can triangulate source locations.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
import cma
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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


def compute_gradient_at_sensor(sensors_xy, temperatures, sensor_idx, n_neighbors=4):
    """
    Compute approximate temperature gradient at a sensor location.

    Uses nearby sensors to estimate gradient direction.
    Returns unit vector pointing toward increasing temperature (toward source).
    """
    x0, y0 = sensors_xy[sensor_idx]
    T0 = temperatures[sensor_idx]

    # Find nearest neighbors
    distances = np.linalg.norm(sensors_xy - np.array([x0, y0]), axis=1)
    distances[sensor_idx] = np.inf  # Exclude self
    neighbor_indices = np.argsort(distances)[:n_neighbors]

    # Compute weighted gradient using finite differences
    grad_x = 0.0
    grad_y = 0.0
    total_weight = 0.0

    for idx in neighbor_indices:
        dx = sensors_xy[idx, 0] - x0
        dy = sensors_xy[idx, 1] - y0
        dT = temperatures[idx] - T0
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 1e-6:
            # Weight by inverse distance and temperature difference
            weight = 1.0 / dist
            grad_x += weight * dT * dx / dist
            grad_y += weight * dT * dy / dist
            total_weight += weight

    if total_weight > 0:
        grad_x /= total_weight
        grad_y /= total_weight

    # Normalize to unit vector
    mag = np.sqrt(grad_x**2 + grad_y**2)
    if mag > 1e-6:
        grad_x /= mag
        grad_y /= mag
    else:
        # No clear gradient, return zero
        return np.array([0.0, 0.0]), 0.0

    return np.array([grad_x, grad_y]), mag


def estimate_source_from_gradients(sensors_xy, temperatures, Lx, Ly, n_sources=1):
    """
    Estimate heat source location(s) using gradient analysis.

    Key insight: Temperature gradients point toward heat sources.
    We find where gradient rays converge to estimate source location.
    """
    n_sensors = len(sensors_xy)

    # Compute gradients at all sensors
    gradients = []
    gradient_mags = []
    for i in range(n_sensors):
        grad, mag = compute_gradient_at_sensor(sensors_xy, temperatures, i)
        gradients.append(grad)
        gradient_mags.append(mag)

    gradients = np.array(gradients)
    gradient_mags = np.array(gradient_mags)

    # Weight sensors by gradient magnitude and temperature
    weights = gradient_mags * temperatures
    weights = weights / (weights.sum() + 1e-8)

    if n_sources == 1:
        # Find point that best fits gradient directions
        # Use weighted centroid shifted by gradient
        centroid = np.average(sensors_xy, axis=0, weights=weights)

        # Average gradient direction from hot sensors
        hot_mask = temperatures > np.percentile(temperatures, 75)
        if hot_mask.sum() > 0:
            avg_gradient = np.average(gradients[hot_mask], axis=0, weights=weights[hot_mask])
            avg_gradient_mag = np.linalg.norm(avg_gradient)
            if avg_gradient_mag > 0.1:
                # Move from centroid in gradient direction
                step_size = 0.2  # Move 20% of domain in gradient direction
                estimated_x = centroid[0] + step_size * Lx * avg_gradient[0]
                estimated_y = centroid[1] + step_size * Ly * avg_gradient[1]
            else:
                estimated_x, estimated_y = centroid
        else:
            estimated_x, estimated_y = centroid

        # Clamp to domain
        margin = 0.05
        estimated_x = np.clip(estimated_x, margin * Lx, (1 - margin) * Lx)
        estimated_y = np.clip(estimated_y, margin * Ly, (1 - margin) * Ly)

        return np.array([estimated_x, estimated_y])

    else:  # 2 sources
        # Cluster sensors into two groups and find source for each
        # Use k-means style clustering on weighted positions
        from scipy.cluster.vq import kmeans2

        # Create weighted sensor positions for clustering
        # Weight by temperature (hotter sensors more important)
        try:
            # Repeat sensors based on weight (simple way to weight kmeans)
            centers, _ = kmeans2(sensors_xy, 2, minit='++')

            # Assign sensors to clusters
            dists_to_0 = np.linalg.norm(sensors_xy - centers[0], axis=1)
            dists_to_1 = np.linalg.norm(sensors_xy - centers[1], axis=1)
            cluster_0_mask = dists_to_0 < dists_to_1
            cluster_1_mask = ~cluster_0_mask

            estimates = []
            for mask in [cluster_0_mask, cluster_1_mask]:
                if mask.sum() == 0:
                    # Fallback to centroid
                    estimates.append(np.mean(sensors_xy, axis=0))
                    continue

                cluster_sensors = sensors_xy[mask]
                cluster_temps = temperatures[mask]
                cluster_grads = gradients[mask]
                cluster_weights = weights[mask]

                if cluster_weights.sum() > 0:
                    cluster_weights = cluster_weights / cluster_weights.sum()
                else:
                    cluster_weights = np.ones(len(cluster_sensors)) / len(cluster_sensors)

                # Weighted centroid
                centroid = np.average(cluster_sensors, axis=0, weights=cluster_weights)

                # Shift by gradient
                avg_gradient = np.average(cluster_grads, axis=0, weights=cluster_weights)
                avg_gradient_mag = np.linalg.norm(avg_gradient)
                if avg_gradient_mag > 0.1:
                    step_size = 0.15
                    est_x = centroid[0] + step_size * Lx * avg_gradient[0]
                    est_y = centroid[1] + step_size * Ly * avg_gradient[1]
                else:
                    est_x, est_y = centroid

                # Clamp
                margin = 0.05
                est_x = np.clip(est_x, margin * Lx, (1 - margin) * Lx)
                est_y = np.clip(est_y, margin * Ly, (1 - margin) * Ly)
                estimates.append(np.array([est_x, est_y]))

            return np.array([estimates[0][0], estimates[0][1],
                           estimates[1][0], estimates[1][1]])
        except:
            # Fallback to simple split
            centroid = np.average(sensors_xy, axis=0, weights=weights)
            offset = 0.3 * Lx
            return np.array([
                centroid[0] - offset/2, centroid[1],
                centroid[0] + offset/2, centroid[1]
            ])


class PhysicsInformedInitOptimizer:
    """
    Optimizer that uses physics-informed initialization based on temperature gradients.

    Instead of using sensor positions directly (hottest sensor approach),
    this uses gradient analysis to estimate where heat sources are located.
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
        sigma0_1src: float = 0.18,
        sigma0_2src: float = 0.22,
        use_triangulation: bool = True,
        use_gradient_init: bool = True,  # NEW: Use gradient-based initialization
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
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
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.use_gradient_init = use_gradient_init
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
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

    def _smart_init_positions(self, sample, n_sources):
        """Original hottest-sensor initialization."""
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

    def _gradient_init_positions(self, sample, n_sources):
        """NEW: Physics-informed initialization using temperature gradients."""
        sensors_xy = np.array(sample['sensors_xy'])
        readings = sample['Y_noisy']

        # Use final timestep temperatures (most informative)
        final_temps = readings[-1]

        # Estimate source position from gradient analysis
        return estimate_source_from_gradients(
            sensors_xy, final_temps, self.Lx, self.Ly, n_sources
        )

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

    def _run_single_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                                  initializations, n_sources, nt_reduced, nt_full):
        """Run optimization with final polish using full timesteps."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]

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

        all_solutions.sort(key=lambda x: x[1])

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

        for pos_params, rmse_coarse, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

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

        # FINAL POLISH
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

        # Primary initializations - SAME NUMBER AS BASELINE (2 inits)
        primary_inits = []

        # 1. Triangulation initialization (always used)
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            primary_inits.append((tri_init, 'triangulation'))

        # 2. Either gradient init OR smart init (not both)
        if self.use_gradient_init:
            # Use gradient-based physics-informed initialization
            gradient_init = self._gradient_init_positions(sample, n_sources)
            primary_inits.append((gradient_init, 'gradient'))
        else:
            # Use smart (hottest sensor) initialization (W2 baseline)
            smart_init = self._smart_init_positions(sample, n_sources)
            primary_inits.append((smart_init, 'smart'))

        # Run optimization with final polish
        candidates_raw, n_sims = self._run_single_optimization(
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

            fallback_candidates, fallback_sims = self._run_single_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                fallback_inits, n_sources, nt_reduced, nt_full
            )
            n_sims += fallback_sims
            candidates_raw.extend(fallback_candidates)

        # Filter dissimilar and select best
        candidates_for_filter = [(c[0], c[2]) for c in candidates_raw]
        final_candidates = filter_dissimilar(candidates_for_filter)

        # Build results
        results = []
        final_params = []
        for sources, rmse in final_candidates:
            for cand in candidates_raw:
                if cand[0] == sources and cand[2] == rmse:
                    final_params.append(cand[1])
                    results.append(CandidateResult(
                        params=cand[1],
                        rmse=rmse,
                        init_type=cand[3],
                        n_evals=n_sims
                    ))
                    break

        if not results:
            final_params = []
            results = []
            for c in candidates_raw[:self.n_candidates]:
                final_params.append(c[1])
                results.append(CandidateResult(
                    params=c[1],
                    rmse=c[2],
                    init_type=c[3],
                    n_evals=n_sims
                ))

        best_rmse = min(r.rmse for r in results) if results else float('inf')

        return final_params, best_rmse, results, n_sims
