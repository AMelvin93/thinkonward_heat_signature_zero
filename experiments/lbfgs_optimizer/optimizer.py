"""
A9: L-BFGS-B Gradient-Based Optimizer for Heat Source Identification.

Key Hypothesis: L-BFGS-B gradient-based optimization should converge faster
than CMA-ES for this smooth objective function, requiring fewer function
evaluations for the same or better accuracy.

Advantages over CMA-ES:
- Uses gradient information (numerical) for efficient search
- Well-suited for bounded, low-dimensional optimization
- Should converge in fewer iterations

Implementation:
- Uses scipy.optimize.minimize with L-BFGS-B method
- Analytical intensity computation (same as baseline)
- Multiple restarts from different initializations
- Smart init selection (same as baseline)
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
from scipy.optimize import minimize

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


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    """Enhanced feature extraction for similarity matching."""
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    # Basic features
    basic = [
        np.max(Y) / 10.0,
        np.mean(Y) / 5.0,
        np.std(Y) / 2.0,
        kappa * 10,
        len(sensors) / 10.0,
    ]

    # Spatial features
    max_temps_per_sensor = Y.max(axis=0)
    weights = max_temps_per_sensor / (max_temps_per_sensor.sum() + 1e-8)
    centroid_x = np.average(sensors[:, 0], weights=weights) / 2.0
    centroid_y = np.average(sensors[:, 1], weights=weights)
    spatial_spread = np.sqrt(
        np.average((sensors[:, 0] / 2.0 - centroid_x)**2 +
                   (sensors[:, 1] - centroid_y)**2, weights=weights)
    )
    spatial = [centroid_x, centroid_y, spatial_spread]

    # Temporal features
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

    # Correlation feature
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
    """Find k most similar solutions from history."""
    if not history or k == 0:
        return []

    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


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


class LBFGSOptimizer:
    """
    L-BFGS-B Gradient-Based Optimizer with SMART initialization selection.

    Key Hypothesis: Gradient-based optimization (L-BFGS-B) should converge
    faster than CMA-ES for this smooth objective function.

    Uses same smart init selection as baseline, but replaces CMA-ES with
    scipy.optimize.minimize using L-BFGS-B method.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 24,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        use_enhanced_features: bool = True,
        n_restarts: int = 3,  # Number of restarts from different inits
        ftol: float = 1e-6,   # Function tolerance for L-BFGS-B
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.use_enhanced_features = use_enhanced_features
        self.n_restarts = n_restarts
        self.ftol = ftol

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        """Get bounds for L-BFGS-B."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
        return bounds

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

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get position-only initialization from triangulation."""
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

    def _evaluate_init(self, init_params: np.ndarray, n_sources: int,
                       Y_observed: np.ndarray, solver: Heat2D, dt: float,
                       nt: int, T0: float, sensors_xy: np.ndarray,
                       q_range: Tuple[float, float]) -> float:
        """Quickly evaluate an init by computing its RMSE."""
        if n_sources == 1:
            x, y = init_params
            _, _, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        else:
            x1, y1, x2, y2 = init_params
            _, _, rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        return rmse

    def _optimize_scipy(
        self,
        init_params: np.ndarray,
        n_sources: int,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        max_fevals: int,
        method: str = 'Powell',
    ) -> Tuple[np.ndarray, float, int]:
        """Run scipy optimization from init_params."""

        n_evals = [0]  # Use list to allow modification in closure
        bounds = self._get_position_bounds(n_sources)

        if n_sources == 1:
            def objective(xy_params):
                n_evals[0] += 1
                # Clip to bounds
                x = np.clip(xy_params[0], bounds[0][0], bounds[0][1])
                y = np.clip(xy_params[1], bounds[1][0], bounds[1][1])
                _, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                n_evals[0] += 1
                # Clip to bounds
                x1 = np.clip(xy_params[0], bounds[0][0], bounds[0][1])
                y1 = np.clip(xy_params[1], bounds[1][0], bounds[1][1])
                x2 = np.clip(xy_params[2], bounds[2][0], bounds[2][1])
                y2 = np.clip(xy_params[3], bounds[3][0], bounds[3][1])
                _, _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Method-specific options
        if method == 'Powell':
            options = {
                'maxfev': max_fevals,
                'maxiter': max_fevals,
                'ftol': self.ftol,
                'disp': False,
            }
            result = minimize(objective, init_params, method='Powell', options=options)
        elif method == 'Nelder-Mead':
            options = {
                'maxfev': max_fevals,
                'maxiter': max_fevals,
                'fatol': self.ftol,
                'xatol': 1e-4,
                'disp': False,
            }
            result = minimize(objective, init_params, method='Nelder-Mead', options=options)
        else:  # L-BFGS-B
            options = {
                'maxfun': max_fevals,
                'maxiter': max_fevals,
                'ftol': self.ftol,
                'gtol': 1e-5,
                'disp': False,
            }
            result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, options=options)

        return result.x, result.fun, n_evals[0]

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """
        Estimate sources using L-BFGS-B optimization.

        Phase 1: Evaluate all inits (1-2 sims each)
        Phase 2: Run L-BFGS-B from best init(s)
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

        # Feature extraction
        if self.use_enhanced_features:
            features = extract_enhanced_features(sample, meta)
        else:
            features = extract_enhanced_features(sample, meta)[:5]

        # Transfer learning
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        # === PHASE 1: BUILD AND EVALUATE ALL INITS ===
        initializations = []

        # 1. Triangulation
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # 2. Hottest sensor
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # 3. Transfer learning
        for i, sol in enumerate(similar_solutions):
            positions = []
            for j in range(n_sources):
                positions.extend([sol[j*3], sol[j*3 + 1]])
            initializations.append((np.array(positions), f'transfer_{i}'))

        # Evaluate all inits
        init_results = []
        for init_params, init_type in initializations:
            rmse = self._evaluate_init(
                init_params, n_sources, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            init_results.append((init_params, init_type, rmse))

        # Sort by RMSE
        init_results.sort(key=lambda x: x[2])

        # === PHASE 2: L-BFGS-B OPTIMIZATION ===
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src

        # Run L-BFGS-B from best init(s)
        # Use fewer restarts than available inits to save budget
        n_to_optimize = min(self.n_restarts, len(init_results))

        # Distribute fevals across restarts
        fevals_per_restart = max(max_fevals // n_to_optimize, 5)

        all_results = []
        total_evals = 0

        for init_params, init_type, init_rmse in init_results[:n_to_optimize]:
            best_params, best_rmse, n_evals = self._optimize_scipy(
                init_params, n_sources, Y_observed, solver, dt, nt, T0,
                sensors_xy, q_range, fevals_per_restart, method='Powell'
            )
            total_evals += n_evals

            # Get final sources with intensities
            if n_sources == 1:
                x, y = best_params
                q, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                sources = [(x, y, q)]
            else:
                x1, y1, x2, y2 = best_params
                (q1, q2), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                sources = [(x1, y1, q1), (x2, y2, q2)]

            # Store full params for history
            full_params = np.array([p for src in sources for p in src])

            all_results.append(CandidateResult(
                params=full_params,
                rmse=rmse,
                init_type=init_type,
                n_evals=n_evals,
            ))

        # Sort by RMSE
        all_results.sort(key=lambda x: x.rmse)

        # Build candidates list for filtering
        candidate_list = []
        for result in all_results:
            if n_sources == 1:
                sources = [(result.params[0], result.params[1], result.params[2])]
            else:
                sources = [
                    (result.params[0], result.params[1], result.params[2]),
                    (result.params[3], result.params[4], result.params[5])
                ]
            candidate_list.append((sources, result.rmse, result))

        # Filter for diversity
        filtered = filter_dissimilar(candidate_list, TAU, self.n_candidates)

        # Build final output
        candidates = [f[0] for f in filtered]
        best_rmse = filtered[0][1] if filtered else float('inf')
        results = [f[2] for f in filtered]

        # Best positions for history
        best_result = all_results[0] if all_results else None
        best_positions = best_result.params if best_result else np.zeros(n_sources * 3)

        return candidates, best_rmse, results, features, best_positions, n_transferred
