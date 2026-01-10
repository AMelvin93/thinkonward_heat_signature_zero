"""
A14: Gradient-Based Triangulation for Heat Source Localization.

Key Innovation: Use temperature GRADIENT DIRECTION to triangulate source position.

Physics Insight:
    Heat flows from hot to cold, so the temperature gradient ∇T points AWAY from
    the heat source. By tracing rays from sensors in the -∇T direction (toward
    the source), we can find where the rays converge.

    This is Angle-of-Arrival (AOA) localization applied to thermal fields!

Method:
    1. Interpolate temperature field from sensor readings (RBF or similar)
    2. Compute temperature gradient at each sensor location
    3. Draw rays from each sensor in the -∇T direction (toward source)
    4. Find intersection point using least-squares
    5. Use as initialization for CMA-ES with analytical intensity

Advantages over onset-time triangulation:
    - Uses spatial gradient (more information per time point)
    - More robust to noise in timing detection
    - Works even when onset is hard to detect
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize, least_squares
import cma

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
    """Result from optimization."""
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    """Normalize source parameters using scale factors."""
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
    """Compute minimum distance between two candidate sets."""
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
    """Filter to keep only dissimilar candidates."""
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


def compute_temperature_gradient(
    sensors_xy: np.ndarray,
    temperatures: np.ndarray,
    query_points: np.ndarray = None,
    epsilon: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute temperature gradient at sensor locations using RBF interpolation.

    Args:
        sensors_xy: Sensor positions (n_sensors, 2)
        temperatures: Temperature at each sensor
        query_points: Points where to compute gradient (default: sensor locations)
        epsilon: Step size for numerical gradient

    Returns:
        grad_x, grad_y: Gradient components at each query point
    """
    if query_points is None:
        query_points = sensors_xy

    # Create RBF interpolator
    try:
        rbf = RBFInterpolator(sensors_xy, temperatures, kernel='thin_plate_spline')
    except Exception:
        # Fallback: return zeros
        return np.zeros(len(query_points)), np.zeros(len(query_points))

    # Compute gradient numerically
    grad_x = np.zeros(len(query_points))
    grad_y = np.zeros(len(query_points))

    for i, (x, y) in enumerate(query_points):
        # Central differences
        T_plus_x = rbf([[x + epsilon, y]])[0]
        T_minus_x = rbf([[x - epsilon, y]])[0]
        T_plus_y = rbf([[x, y + epsilon]])[0]
        T_minus_y = rbf([[x, y - epsilon]])[0]

        grad_x[i] = (T_plus_x - T_minus_x) / (2 * epsilon)
        grad_y[i] = (T_plus_y - T_minus_y) / (2 * epsilon)

    return grad_x, grad_y


def find_ray_intersection(
    sensors_xy: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Find the point where rays from sensors converge.

    Each sensor i has position (xi, yi) and direction (dx_i, dy_i).
    The ray is: P_i(t) = (xi, yi) + t * (dx_i, dy_i) for t >= 0

    We want to find point (x, y) that minimizes the sum of squared
    distances to all rays (weighted by temperature).

    Args:
        sensors_xy: Sensor positions (n, 2)
        directions: Ray directions (n, 2), pointing toward source
        weights: Optional weights for each sensor
        bounds: ((x_min, x_max), (y_min, y_max))

    Returns:
        (x, y) estimated source position
    """
    if weights is None:
        weights = np.ones(len(sensors_xy))

    weights = weights / (weights.sum() + 1e-8)

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions = directions / norms

    # Filter out sensors with very small gradients
    valid = norms.flatten() > 1e-6
    if np.sum(valid) < 2:
        # Not enough valid rays, use weighted centroid
        return tuple(np.average(sensors_xy, weights=weights, axis=0))

    sensors_valid = sensors_xy[valid]
    directions_valid = directions[valid]
    weights_valid = weights[valid]
    weights_valid = weights_valid / (weights_valid.sum() + 1e-8)

    def objective(p):
        """Sum of squared distances from point p to all rays."""
        total = 0.0
        for i in range(len(sensors_valid)):
            s = sensors_valid[i]
            d = directions_valid[i]
            w = weights_valid[i]

            # Vector from sensor to point
            v = p - s

            # Project onto ray direction
            proj = np.dot(v, d)

            # Point on ray closest to p
            # Only consider forward direction (t >= 0)
            t = max(0, proj)
            closest = s + t * d

            # Squared distance
            dist_sq = np.sum((p - closest) ** 2)
            total += w * dist_sq

        return total

    # Initial guess: weighted centroid
    x0 = np.average(sensors_valid, weights=weights_valid, axis=0)

    if bounds is not None:
        scipy_bounds = [bounds[0], bounds[1]]
    else:
        scipy_bounds = None

    result = minimize(objective, x0, method='L-BFGS-B', bounds=scipy_bounds)

    return tuple(result.x)


def gradient_triangulate_1src(
    sample: Dict,
    meta: Dict,
    Lx: float = 2.0,
    Ly: float = 1.0
) -> Tuple[float, float]:
    """
    Triangulate single source position using temperature gradients.

    Physics: Temperature gradient points AWAY from source.
    Trace rays from sensors in -gradient direction to find source.
    """
    Y_noisy = sample['Y_noisy']
    sensors_xy = np.array(sample['sensors_xy'])

    # Use late-time temperatures (quasi-steady state)
    n_late = max(1, Y_noisy.shape[0] // 5)
    T_late = Y_noisy[-n_late:].mean(axis=0)

    # Compute gradients
    grad_x, grad_y = compute_temperature_gradient(sensors_xy, T_late)

    # Ray directions: OPPOSITE to gradient (toward source)
    directions = np.column_stack([-grad_x, -grad_y])

    # Weight by temperature (hotter sensors are closer/more reliable)
    weights = np.maximum(T_late, 0)

    # Find intersection
    bounds = ((0.05 * Lx, 0.95 * Lx), (0.05 * Ly, 0.95 * Ly))
    x, y = find_ray_intersection(sensors_xy, directions, weights, bounds)

    # Clip to domain
    x = np.clip(x, 0.1, Lx - 0.1)
    y = np.clip(y, 0.1, Ly - 0.1)

    return x, y


def gradient_triangulate_2src(
    sample: Dict,
    meta: Dict,
    Lx: float = 2.0,
    Ly: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Triangulate two source positions using temperature gradients.

    Strategy:
    1. Cluster sensors by proximity to each source (using gradient directions)
    2. Triangulate within each cluster
    """
    Y_noisy = sample['Y_noisy']
    sensors_xy = np.array(sample['sensors_xy'])
    n_sensors = len(sensors_xy)

    # Use late-time temperatures
    n_late = max(1, Y_noisy.shape[0] // 5)
    T_late = Y_noisy[-n_late:].mean(axis=0)

    # First pass: find approximate source region using temperature peaks
    # Use K-means style clustering based on sensor temperatures
    from sklearn.cluster import KMeans

    # Weight sensors by temperature for clustering
    features = np.column_stack([
        sensors_xy[:, 0],
        sensors_xy[:, 1],
        T_late * 2  # Emphasize temperature in clustering
    ])

    try:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
    except Exception:
        # Fallback: split by hottest half
        sorted_idx = np.argsort(T_late)[::-1]
        labels = np.zeros(n_sensors, dtype=int)
        labels[sorted_idx[:n_sensors//2]] = 0
        labels[sorted_idx[n_sensors//2:]] = 1

    sources = []

    for cluster_id in range(2):
        cluster_mask = labels == cluster_id

        if np.sum(cluster_mask) < 2:
            # Not enough sensors, use hottest sensor in cluster
            hot_idx = np.argmax(T_late * cluster_mask)
            x, y = sensors_xy[hot_idx]
            sources.append((x, y))
            continue

        cluster_sensors = sensors_xy[cluster_mask]
        cluster_temps = T_late[cluster_mask]

        # Compute gradients for cluster sensors
        grad_x, grad_y = compute_temperature_gradient(sensors_xy, T_late, cluster_sensors)

        # Ray directions: toward source
        directions = np.column_stack([-grad_x, -grad_y])

        # Find intersection
        bounds = ((0.05 * Lx, 0.95 * Lx), (0.05 * Ly, 0.95 * Ly))
        x, y = find_ray_intersection(cluster_sensors, directions, cluster_temps, bounds)

        x = np.clip(x, 0.1, Lx - 0.1)
        y = np.clip(y, 0.1, Ly - 0.1)

        sources.append((x, y))

    # Ensure sources are separated
    if len(sources) == 2:
        dist = np.sqrt((sources[0][0] - sources[1][0])**2 +
                       (sources[0][1] - sources[1][1])**2)
        if dist < 0.2:
            # Sources too close, spread them apart
            cx, cy = (sources[0][0] + sources[1][0]) / 2, (sources[0][1] + sources[1][1]) / 2
            sources = [
                (np.clip(cx - 0.15, 0.1, Lx - 0.1), cy),
                (np.clip(cx + 0.15, 0.1, Lx - 0.1), cy)
            ]

    return sources


def simulate_unit_source(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                         T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    """Simulate a source with q=1.0 and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    """Compute optimal intensity for 1-source analytically."""
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
    """Compute optimal intensities for 2-source analytically."""
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


class GradientTriangulationOptimizer:
    """
    CMA-ES optimizer with gradient-based triangulation initialization.

    Key Innovation: Uses temperature gradient direction to triangulate
    source positions, providing better initial estimates than onset-time
    triangulation for some cases.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 25,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_onset_triangulation: bool = True,
        use_gradient_triangulation: bool = True,
        n_candidates: int = N_MAX,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_onset_triangulation = use_onset_triangulation
        self.use_gradient_triangulation = use_gradient_triangulation
        self.n_candidates = min(n_candidates, N_MAX)

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, sample: Dict, n_sources: int) -> np.ndarray:
        """Get position-only initialization from hottest sensors."""
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

    def _gradient_init_positions(self, sample: Dict, meta: Dict, n_sources: int) -> Optional[np.ndarray]:
        """Get position-only initialization from gradient triangulation."""
        if not self.use_gradient_triangulation:
            return None

        try:
            if n_sources == 1:
                x, y = gradient_triangulate_1src(sample, meta, self.Lx, self.Ly)
                return np.array([x, y])
            else:
                sources = gradient_triangulate_2src(sample, meta, self.Lx, self.Ly)
                return np.array([sources[0][0], sources[0][1],
                                sources[1][0], sources[1][1]])
        except Exception:
            return None

    def _onset_init_positions(self, sample: Dict, meta: Dict,
                              n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get position-only initialization from onset triangulation."""
        if not self.use_onset_triangulation:
            return None

        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources with gradient-based initialization.
        """
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        n_sims = [0]

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

        # Build initialization pool
        initializations = []

        # 1. Gradient triangulation (NEW)
        grad_init = self._gradient_init_positions(sample, meta, n_sources)
        if grad_init is not None:
            initializations.append((grad_init, 'gradient'))

        # 2. Onset triangulation (existing)
        onset_init = self._onset_init_positions(sample, meta, n_sources, q_range)
        if onset_init is not None:
            initializations.append((onset_init, 'onset'))

        # 3. Hottest sensor
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        if verbose:
            print(f"  Initializations: {[i[1] for i in initializations]}")

        # CMA-ES optimization
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
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit, init_type))

        # Convert to full params
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:10]

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

        n_evals = n_sims[0]

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_evals // len(final_candidates) if final_candidates else n_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_evals
