"""
Cluster-based Initialization for 2-Source Problems.

Key idea: Use K-means clustering on sensor temperatures to identify
two distinct "hot zones" and use their weighted centroids as initial
source positions.

This should give better initialization especially when sources are
well-separated.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
import cma
from scipy.optimize import minimize
from sklearn.cluster import KMeans

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


class ClusterInit2srcOptimizer:
    """
    Multi-fidelity optimizer with cluster-based initialization for 2-source problems.

    For 2-source problems, uses K-means clustering on sensor temperatures to
    identify two distinct hot zones and initializes sources at weighted centroids.
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
        """Original smart init based on hottest sensors."""
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

    def _cluster_init_2src(self, sample):
        """
        Cluster-based initialization for 2-source problems.

        Uses K-means on sensor positions weighted by temperatures
        to find two "hot zone" centroids.
        """
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # Use max temperature per sensor as weight
        max_temps = np.max(readings, axis=0)

        # Normalize weights to [0, 1]
        weights = max_temps / (max_temps.max() + 1e-8)

        # Only use sensors with significant temperature
        threshold = np.percentile(weights, 50)  # Top 50%
        hot_mask = weights >= threshold
        hot_sensors = sensors[hot_mask]
        hot_weights = weights[hot_mask]

        if len(hot_sensors) < 4:
            # Fall back to all sensors
            hot_sensors = sensors
            hot_weights = weights

        # K-means clustering with 2 clusters
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(hot_sensors)

            # Compute weighted centroid for each cluster
            centroids = []
            for c in range(2):
                cluster_mask = labels == c
                if cluster_mask.sum() > 0:
                    cluster_sensors = hot_sensors[cluster_mask]
                    cluster_weights = hot_weights[cluster_mask]
                    cluster_weights = cluster_weights / (cluster_weights.sum() + 1e-8)
                    centroid = np.average(cluster_sensors, axis=0, weights=cluster_weights)
                    centroids.append(centroid)
                else:
                    # Fallback if cluster is empty
                    centroids.append(sensors[np.argmax(weights)])

            # Clip to bounds
            centroids = np.array(centroids)
            centroids[:, 0] = np.clip(centroids[:, 0], 0.05 * self.Lx, 0.95 * self.Lx)
            centroids[:, 1] = np.clip(centroids[:, 1], 0.05 * self.Ly, 0.95 * self.Ly)

            return np.array([centroids[0, 0], centroids[0, 1],
                           centroids[1, 0], centroids[1, 1]])

        except Exception:
            # Fallback to smart init
            return self._smart_init_positions(sample, 2)

    def _onset_init_2src(self, sample):
        """
        Onset-based initialization for 2-source problems.

        Uses onset times to estimate source distances from sensors,
        then triangulates positions.
        """
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # Compute onset times (when temp exceeds 10% of max)
        onset_times = []
        for i in range(readings.shape[1]):
            signal = readings[:, i]
            threshold = 0.1 * signal.max()
            onset_idx = np.argmax(signal > threshold)
            onset_times.append(onset_idx if signal.max() > threshold else 999)

        onset_times = np.array(onset_times)

        # Find 2 sensors with earliest onsets that are far apart
        earliest_idx = np.argsort(onset_times)[:10]  # Top 10 earliest

        if len(earliest_idx) < 2:
            return self._smart_init_positions(sample, 2)

        # Pick two earliest that are far apart
        s1_idx = earliest_idx[0]
        s2_idx = None
        for idx in earliest_idx[1:]:
            if np.linalg.norm(sensors[idx] - sensors[s1_idx]) >= 0.3:
                s2_idx = idx
                break

        if s2_idx is None:
            s2_idx = earliest_idx[1]

        # Use sensor positions as initial estimates (shifted slightly toward center)
        s1 = sensors[s1_idx]
        s2 = sensors[s2_idx]

        # Shift slightly toward domain center
        center = np.array([self.Lx / 2, self.Ly / 2])
        s1 = 0.9 * s1 + 0.1 * center
        s2 = 0.9 * s2 + 0.1 * center

        return np.array([s1[0], s1[1], s2[0], s2[1]])

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

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

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

        # Build initializations
        initializations = []

        # Triangulation init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # Smart init
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # For 2-source, add cluster-based and onset-based inits
        if n_sources == 2:
            cluster_init = self._cluster_init_2src(sample)
            initializations.append((cluster_init, 'cluster'))

            onset_init = self._onset_init_2src(sample)
            initializations.append((onset_init, 'onset'))

        # CMA-ES on coarse grid
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

        # Refine top N solutions on COARSE grid
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

        # Add remaining top solutions
        for pos_params, rmse_coarse, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_coarse, init_type))

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
