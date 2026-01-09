"""
Better 2-Source Initialization Optimizer.

Key Innovation: Improved position estimation for 2-source problems using:
1. K-means clustering of hot sensors
2. Time-delay triangulation
3. Temperature gradient analysis
4. NMF-based signal decomposition

The goal is to provide better starting points for CMA-ES specifically for
2-source problems where the 4D parameter space is hard to search.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add simulator path
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


# Competition parameters
N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


@dataclass
class CandidateResult:
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
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


def filter_dissimilar(candidates: List[Tuple], tau: float = TAU, n_max: int = N_MAX) -> List[Tuple]:
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


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    basic = [
        np.max(Y) / 10.0,
        np.mean(Y) / 5.0,
        np.std(Y) / 2.0,
        kappa * 10,
        len(sensors) / 10.0,
    ]

    max_temps_per_sensor = Y.max(axis=0)
    weights = max_temps_per_sensor / (max_temps_per_sensor.sum() + 1e-8)
    centroid_x = np.average(sensors[:, 0], weights=weights) / 2.0
    centroid_y = np.average(sensors[:, 1], weights=weights)
    spatial_spread = np.sqrt(
        np.average((sensors[:, 0] / 2.0 - centroid_x)**2 +
                   (sensors[:, 1] - centroid_y)**2, weights=weights)
    )
    spatial = [centroid_x, centroid_y, spatial_spread]

    onset_times = []
    for i in range(Y.shape[1]):
        signal = Y[:, i]
        threshold = 0.1 * (signal.max() + 1e-8)
        onset_idx = np.argmax(signal > threshold)
        onset_times.append(onset_idx)
    onset_times = np.array(onset_times)
    onset_mean = np.mean(onset_times) / 100.0
    onset_std = np.std(onset_times) / 50.0
    temporal = [onset_mean, onset_std]

    if Y.shape[1] > 1:
        try:
            corr_matrix = np.corrcoef(Y.T)
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix[triu_indices]
            correlations = correlations[~np.isnan(correlations)]
            avg_corr = np.mean(correlations) if len(correlations) > 0 else 0.5
        except:
            avg_corr = 0.5
    else:
        avg_corr = 1.0
    correlation = [avg_corr]

    return np.array(basic + spatial + temporal + correlation)


def find_similar_solutions(
    features: np.ndarray,
    history: List[Tuple[np.ndarray, np.ndarray]],
    k: int = 1
) -> List[np.ndarray]:
    if not history or k == 0:
        return []
    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


def simulate_unit_source(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                         T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        numerator = np.dot(Y_unit_flat, Y_obs_flat)
        q_optimal = numerator / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], np.ndarray, float]:
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
        A_reg = A + 1e-6 * np.eye(2)
        q1, q2 = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        q1, q2 = 1.0, 1.0
    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])
    Y_pred = q1 * Y1 + q2 * Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return (q1, q2), Y_pred, rmse


def kmeans_init_2src(sample: Dict, Lx: float = 2.0, Ly: float = 1.0) -> Optional[np.ndarray]:
    """
    Use K-means clustering on weighted sensor positions to find 2 source locations.
    Weight sensors by their maximum temperature.
    """
    try:
        Y = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # Weight by max temperature
        max_temps = Y.max(axis=0)
        weights = max_temps / (max_temps.sum() + 1e-8)

        # Only use sensors with significant temperature
        threshold = 0.1 * max_temps.max()
        significant_mask = max_temps > threshold
        if significant_mask.sum() < 2:
            return None

        significant_sensors = sensors[significant_mask]
        significant_weights = weights[significant_mask]

        # Weighted K-means initialization
        # Create weighted dataset by repeating samples
        n_repeats = (significant_weights * 100).astype(int)
        n_repeats = np.maximum(n_repeats, 1)

        weighted_sensors = np.repeat(significant_sensors, n_repeats, axis=0)

        if len(weighted_sensors) < 2:
            return None

        kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
        kmeans.fit(weighted_sensors)

        centers = kmeans.cluster_centers_
        x1, y1 = centers[0]
        x2, y2 = centers[1]

        # Ensure within bounds
        margin = 0.05
        x1 = np.clip(x1, margin * Lx, (1 - margin) * Lx)
        y1 = np.clip(y1, margin * Ly, (1 - margin) * Ly)
        x2 = np.clip(x2, margin * Lx, (1 - margin) * Lx)
        y2 = np.clip(y2, margin * Ly, (1 - margin) * Ly)

        return np.array([x1, y1, x2, y2])

    except Exception:
        return None


def nmf_init_2src(sample: Dict, Lx: float = 2.0, Ly: float = 1.0) -> Optional[np.ndarray]:
    """
    Use NMF to decompose sensor readings into 2 components,
    then estimate source positions from component weights.
    """
    try:
        Y = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # NMF expects non-negative data
        Y_pos = np.maximum(Y, 0)

        if Y_pos.max() < 1e-10:
            return None

        # Decompose: Y ≈ W × H (time × components) × (components × sensors)
        nmf = NMF(n_components=2, init='random', random_state=42, max_iter=200)
        W = nmf.fit_transform(Y_pos)  # shape: (time, 2)
        H = nmf.components_           # shape: (2, sensors)

        # Each component's H row gives weights for each sensor
        # Use weighted centroid of sensor positions as source estimate
        positions = []
        for i in range(2):
            weights = H[i, :]
            weights = weights / (weights.sum() + 1e-8)

            x = np.average(sensors[:, 0], weights=weights)
            y = np.average(sensors[:, 1], weights=weights)

            margin = 0.05
            x = np.clip(x, margin * Lx, (1 - margin) * Lx)
            y = np.clip(y, margin * Ly, (1 - margin) * Ly)
            positions.extend([x, y])

        return np.array(positions)

    except Exception:
        return None


def onset_time_init_2src(sample: Dict, kappa: float, Lx: float = 2.0, Ly: float = 1.0) -> Optional[np.ndarray]:
    """
    Use temperature onset times at sensors to triangulate source positions.
    Earlier onset = closer to source.
    """
    try:
        Y = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # Compute onset time for each sensor
        onset_times = []
        for i in range(Y.shape[1]):
            signal = Y[:, i]
            threshold = 0.1 * (signal.max() + 1e-8)
            onset_idx = np.argmax(signal > threshold)
            if signal[onset_idx] > threshold:
                onset_times.append(onset_idx)
            else:
                onset_times.append(Y.shape[0])  # Never reached threshold

        onset_times = np.array(onset_times)

        # Find sensors with earliest onset (likely closest to sources)
        sorted_idx = np.argsort(onset_times)

        # Pick 2 sensors with earliest onset that are well-separated
        selected = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if len(selected) >= 2:
                break
            if np.linalg.norm(sensors[idx] - sensors[selected[0]]) >= 0.3:
                selected.append(idx)

        if len(selected) < 2:
            selected = sorted_idx[:2].tolist()

        # Use selected sensor positions as initial source positions
        positions = []
        for idx in selected:
            x, y = sensors[idx]
            positions.extend([x, y])

        return np.array(positions)

    except Exception:
        return None


def gradient_init_2src(sample: Dict, Lx: float = 2.0, Ly: float = 1.0) -> Optional[np.ndarray]:
    """
    Find positions where temperature gradient points towards.
    Multiple hotspots indicate multiple sources.
    """
    try:
        Y = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])

        # Use maximum temperatures for spatial analysis
        max_temps = Y.max(axis=0)

        # Find top k hot sensors
        k = min(8, len(sensors) // 2)
        hot_idx = np.argsort(max_temps)[-k:]
        hot_sensors = sensors[hot_idx]
        hot_temps = max_temps[hot_idx]

        # Cluster hot sensors into 2 groups
        if len(hot_sensors) < 2:
            return None

        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
        labels = kmeans.fit_predict(hot_sensors)

        # For each cluster, compute temperature-weighted centroid
        positions = []
        for cluster in [0, 1]:
            mask = labels == cluster
            if mask.sum() == 0:
                # Use overall center
                x = Lx / 2
                y = Ly / 2
            else:
                cluster_sensors = hot_sensors[mask]
                cluster_temps = hot_temps[mask]
                weights = cluster_temps / (cluster_temps.sum() + 1e-8)
                x = np.average(cluster_sensors[:, 0], weights=weights)
                y = np.average(cluster_sensors[:, 1], weights=weights)

            margin = 0.05
            x = np.clip(x, margin * Lx, (1 - margin) * Lx)
            y = np.clip(y, margin * Ly, (1 - margin) * Ly)
            positions.extend([x, y])

        return np.array(positions)

    except Exception:
        return None


class Better2SrcOptimizer:
    """
    CMA-ES optimizer with improved 2-source initialization.

    For 1-source: Uses standard SmartInit approach.
    For 2-source: Uses multiple advanced initialization strategies:
    - K-means clustering of hot sensors
    - NMF-based signal decomposition
    - Onset time triangulation
    - Temperature gradient analysis
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 12,
        max_fevals_2src: int = 23,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        use_advanced_init: bool = True,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.use_advanced_init = use_advanced_init

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, sample: Dict, n_sources: int) -> np.ndarray:
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

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        if not self.use_triangulation:
            return None
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _get_2src_initializations(self, sample: Dict, meta: Dict, q_range: Tuple[float, float]) -> List[Tuple[np.ndarray, str]]:
        """Get all 2-source initializations."""
        initializations = []

        # Standard triangulation
        tri_init = self._triangulation_init_positions(sample, meta, 2, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # Standard hottest sensors
        smart_init = self._smart_init_positions(sample, 2)
        initializations.append((smart_init, 'smart'))

        if self.use_advanced_init:
            # K-means clustering
            kmeans_init = kmeans_init_2src(sample, self.Lx, self.Ly)
            if kmeans_init is not None:
                initializations.append((kmeans_init, 'kmeans'))

            # NMF decomposition - disabled due to slow convergence
            # nmf_init = nmf_init_2src(sample, self.Lx, self.Ly)
            # if nmf_init is not None:
            #     initializations.append((nmf_init, 'nmf'))

            # Onset time triangulation
            kappa = sample['sample_metadata']['kappa']
            onset_init = onset_time_init_2src(sample, kappa, self.Lx, self.Ly)
            if onset_init is not None:
                initializations.append((onset_init, 'onset'))

            # Temperature gradient clustering
            gradient_init = gradient_init_2src(sample, self.Lx, self.Ly)
            if gradient_init is not None:
                initializations.append((gradient_init, 'gradient'))

        return initializations

    def _evaluate_init(self, init_params: np.ndarray, n_sources: int,
                       Y_observed: np.ndarray, solver: Heat2D, dt: float,
                       nt: int, T0: float, sensors_xy: np.ndarray,
                       q_range: Tuple[float, float]) -> float:
        if n_sources == 1:
            x, y = init_params
            _, _, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        else:
            x1, y1, x2, y2 = init_params
            _, _, rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        return rmse

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """Estimate sources with improved 2-source initialization."""
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)
        features = extract_enhanced_features(sample, meta)

        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        # === BUILD INITIALIZATIONS ===
        if n_sources == 1:
            initializations = []
            tri_init = self._triangulation_init_positions(sample, meta, 1, q_range)
            if tri_init is not None:
                initializations.append((tri_init, 'triangulation'))
            smart_init = self._smart_init_positions(sample, 1)
            initializations.append((smart_init, 'smart'))
        else:
            initializations = self._get_2src_initializations(sample, meta, q_range)

        # Add transfer learning
        lb, ub = self._get_position_bounds(n_sources)
        for i, sol in enumerate(similar_solutions):
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        # === EVALUATE ALL INITS ===
        init_evaluations = []
        n_selection_sims = 0
        for init_params, init_type in initializations:
            rmse = self._evaluate_init(
                init_params, n_sources, Y_observed, solver, dt, nt, T0, sensors_xy, q_range
            )
            init_evaluations.append((init_params, init_type, rmse))
            n_selection_sims += (1 if n_sources == 1 else 2)

        init_evaluations.sort(key=lambda x: x[2])
        best_init_params, best_init_type, best_init_rmse = init_evaluations[0]

        if verbose:
            print(f"  Init selection ({len(initializations)} inits):")
            for params, itype, rmse in init_evaluations[:5]:
                marker = " <-- BEST" if itype == best_init_type else ""
                print(f"    {itype}: RMSE={rmse:.4f}{marker}")

        # === RUN CMA-ES ON BEST INIT ===
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        n_opt_sims = [0]

        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_opt_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_opt_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(best_init_params.tolist(), sigma0, opts)

        all_solutions = []
        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit, best_init_type))

        # === BUILD CANDIDATES ===
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        candidates_raw = []
        for pos_params, rmse, init_type in top_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

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

        if final_candidates:
            best_idx = np.argmin([c[2] for c in final_candidates])
            best_full_params = final_candidates[best_idx][1]
            if n_sources == 1:
                best_positions = best_full_params[:2]
            else:
                best_positions = np.array([best_full_params[0], best_full_params[1],
                                           best_full_params[3], best_full_params[4]])
        else:
            best_positions = best_init_params

        total_sims = n_selection_sims + n_opt_sims[0]

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=total_sims // len(final_candidates) if final_candidates else total_sims
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, features, best_positions, n_transferred
