"""
Onset + Gradient Hybrid Initialization Optimizer.

Key idea: Combine two physics-based signals for better source localization:
1. Onset time - tells us how FAR the source is (closer = earlier onset)
2. Temperature gradient - tells us which DIRECTION the heat is coming from

For each sensor:
- Earlier onset means closer to source
- Gradient pointing toward source (heat flows FROM source)

Combine these for better initial estimates, especially for 2-source problems.
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


class OnsetGradientOptimizer:
    """
    Optimizer with hybrid onset + gradient initialization.
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
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
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

    def _onset_gradient_init_1src(self, sample, kappa):
        """
        Hybrid onset + gradient init for single source.

        1. Find sensors with earliest onset times (they're closest to source)
        2. Compute temperature gradient to estimate direction
        3. Move from hottest sensor toward gradient direction
        """
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])
        n_sensors = len(sensors)

        # Find onset times (when temperature starts rising)
        onset_times = []
        max_temps = np.max(readings, axis=0)

        for i in range(n_sensors):
            signal = readings[:, i]
            # Threshold: 10% of max temperature at this sensor
            threshold = 0.1 * signal.max()
            if signal.max() > threshold:
                onset_idx = np.argmax(signal > threshold)
                onset_times.append(onset_idx)
            else:
                onset_times.append(999)  # No clear signal

        onset_times = np.array(onset_times)

        # Weight sensors by inverse onset time (earlier = higher weight)
        weights = 1.0 / (onset_times + 1)
        weights[onset_times >= 999] = 0
        weights = weights / (weights.sum() + 1e-8)

        # Weighted centroid based on onset times
        onset_centroid = np.average(sensors, axis=0, weights=weights)

        # Now compute gradient direction
        # Use spatial gradient of max temperatures
        # For each hot sensor, compute local gradient pointing toward source
        hot_mask = max_temps > np.percentile(max_temps, 60)
        hot_sensors = sensors[hot_mask]
        hot_temps = max_temps[hot_mask]

        if len(hot_sensors) >= 3:
            # Fit a plane to estimate gradient direction
            # T = a*x + b*y + c
            A = np.column_stack([hot_sensors[:, 0], hot_sensors[:, 1], np.ones(len(hot_sensors))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, hot_temps, rcond=None)
                grad_direction = np.array([coeffs[0], coeffs[1]])
                grad_mag = np.linalg.norm(grad_direction)
                if grad_mag > 1e-6:
                    grad_direction = grad_direction / grad_mag
                    # Move from hottest sensor in gradient direction
                    hottest_idx = np.argmax(max_temps)
                    hottest_sensor = sensors[hottest_idx]
                    # Source is likely where gradient points (higher temp)
                    step_size = 0.1  # Small step toward gradient
                    source_estimate = hottest_sensor + step_size * grad_direction
                else:
                    source_estimate = onset_centroid
            except:
                source_estimate = onset_centroid
        else:
            source_estimate = onset_centroid

        # Blend onset centroid with gradient-based estimate
        final_estimate = 0.6 * onset_centroid + 0.4 * source_estimate

        # Clip to domain
        final_estimate[0] = np.clip(final_estimate[0], 0.05 * self.Lx, 0.95 * self.Lx)
        final_estimate[1] = np.clip(final_estimate[1], 0.05 * self.Ly, 0.95 * self.Ly)

        return final_estimate

    def _onset_gradient_init_2src(self, sample, kappa):
        """
        Hybrid onset + gradient init for 2-source problems.

        Strategy:
        1. Partition sensors into 2 groups based on onset timing patterns
        2. For each group, apply single-source onset+gradient logic
        """
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])
        n_sensors = len(sensors)

        # Compute onset times and max temps
        onset_times = []
        max_temps = np.max(readings, axis=0)

        for i in range(n_sensors):
            signal = readings[:, i]
            threshold = 0.1 * signal.max()
            if signal.max() > threshold:
                onset_idx = np.argmax(signal > threshold)
                onset_times.append(onset_idx)
            else:
                onset_times.append(999)

        onset_times = np.array(onset_times)

        # Find the two earliest-responding sensors that are far apart
        valid_mask = onset_times < 999
        valid_sensors = sensors[valid_mask]
        valid_onsets = onset_times[valid_mask]
        valid_temps = max_temps[valid_mask]

        if len(valid_sensors) < 2:
            # Fallback to smart init
            return self._smart_init_positions(sample, 2)

        # Sort by onset time
        sorted_idx = np.argsort(valid_onsets)

        # Find two seed sensors: earliest and earliest that's far from first
        seed1_local = sorted_idx[0]
        seed2_local = None

        for idx in sorted_idx[1:]:
            dist = np.linalg.norm(valid_sensors[idx] - valid_sensors[seed1_local])
            if dist >= 0.3:  # Minimum separation
                seed2_local = idx
                break

        if seed2_local is None:
            # Use the second earliest even if close
            seed2_local = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]

        # Partition sensors: assign each to nearest seed
        cluster1 = []
        cluster2 = []

        for i in range(len(valid_sensors)):
            d1 = np.linalg.norm(valid_sensors[i] - valid_sensors[seed1_local])
            d2 = np.linalg.norm(valid_sensors[i] - valid_sensors[seed2_local])
            if d1 <= d2:
                cluster1.append(i)
            else:
                cluster2.append(i)

        # For each cluster, compute onset+gradient-based estimate
        positions = []

        for cluster_indices in [cluster1, cluster2]:
            if len(cluster_indices) == 0:
                # Empty cluster - use seed position
                positions.append(valid_sensors[seed1_local if len(positions) == 0 else seed2_local])
                continue

            cluster_sensors = valid_sensors[cluster_indices]
            cluster_onsets = valid_onsets[cluster_indices]
            cluster_temps = valid_temps[cluster_indices]

            # Onset-weighted centroid
            weights = 1.0 / (cluster_onsets + 1)
            weights = weights / (weights.sum() + 1e-8)
            onset_centroid = np.average(cluster_sensors, axis=0, weights=weights)

            # Temperature gradient within cluster
            if len(cluster_sensors) >= 3:
                A = np.column_stack([cluster_sensors[:, 0], cluster_sensors[:, 1], np.ones(len(cluster_sensors))])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, cluster_temps, rcond=None)
                    grad_direction = np.array([coeffs[0], coeffs[1]])
                    grad_mag = np.linalg.norm(grad_direction)
                    if grad_mag > 1e-6:
                        grad_direction = grad_direction / grad_mag
                        hottest_local = np.argmax(cluster_temps)
                        hottest_sensor = cluster_sensors[hottest_local]
                        source_estimate = hottest_sensor + 0.1 * grad_direction
                        final = 0.6 * onset_centroid + 0.4 * source_estimate
                    else:
                        final = onset_centroid
                except:
                    final = onset_centroid
            else:
                final = onset_centroid

            positions.append(final)

        # Ensure minimum separation
        if np.linalg.norm(positions[0] - positions[1]) < 0.2:
            # Push apart
            center = (positions[0] + positions[1]) / 2
            direction = positions[1] - positions[0]
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0.2, 0])
            else:
                direction = direction / np.linalg.norm(direction) * 0.15
            positions[0] = center - direction
            positions[1] = center + direction

        # Clip to domain
        for i in range(2):
            positions[i][0] = np.clip(positions[i][0], 0.05 * self.Lx, 0.95 * self.Lx)
            positions[i][1] = np.clip(positions[i][1], 0.05 * self.Ly, 0.95 * self.Ly)

        return np.array([positions[0][0], positions[0][1], positions[1][0], positions[1][1]])

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

    def _run_single_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                                  initializations, n_sources, early_frac):
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]

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

        for pos_params, rmse_coarse, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

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

        return candidates_raw, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        early_frac = self.early_fraction

        # Build initializations: triangulation, smart, and our new onset+gradient
        primary_inits = []

        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            primary_inits.append((tri_init, 'triangulation'))

        smart_init = self._smart_init_positions(sample, n_sources)
        primary_inits.append((smart_init, 'smart'))

        # Add our onset+gradient hybrid init
        if n_sources == 1:
            og_init = self._onset_gradient_init_1src(sample, kappa)
        else:
            og_init = self._onset_gradient_init_2src(sample, kappa)
        primary_inits.append((og_init, 'onset_gradient'))

        candidates_raw, n_sims = self._run_single_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            primary_inits, n_sources, early_frac
        )

        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            fallback_inits = []
            centroid_init = self._weighted_centroid_init(sample, n_sources)
            fallback_inits.append((centroid_init, 'centroid'))
            random_init = self._random_init_positions(n_sources)
            fallback_inits.append((random_init, 'random'))

            fallback_candidates, fallback_sims = self._run_single_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                fallback_inits, n_sources, early_frac
            )
            n_sims += fallback_sims
            candidates_raw.extend(fallback_candidates)

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
