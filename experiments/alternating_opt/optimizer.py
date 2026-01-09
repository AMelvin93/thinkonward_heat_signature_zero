"""
A10: Alternating Optimization for Heat Source Identification.

Key Hypothesis: For 2-source problems, decompose the 4D optimization into
alternating 2D subproblems. This allows CMA-ES to converge faster and more
accurately on each subproblem.

Approach:
- 1-source: Standard CMA-ES (2D)
- 2-source: Alternating optimization
  1. Fix source 2, optimize source 1 (2D CMA-ES)
  2. Fix source 1, optimize source 2 (2D CMA-ES)
  3. Repeat for n_rounds

Benefits:
- 2D CMA-ES converges faster than 4D
- Each subproblem has a cleaner landscape
- Exploits separability of the objective
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
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


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    """Enhanced feature extraction for similarity matching."""
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


class AlternatingOptimizer:
    """
    Alternating Optimization for Heat Source Identification.

    For 2-source problems, instead of 4D CMA-ES, we use alternating 2D CMA-ES:
    - Round 1: Fix source 2, optimize source 1
    - Round 2: Fix source 1, optimize source 2
    - Repeat for n_rounds

    This decomposes the hard 4D problem into easier 2D subproblems.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 12,
        max_fevals_2src: int = 24,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.15,  # Smaller sigma for 2D subproblems
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        n_rounds: int = 2,  # Number of alternating rounds for 2-source
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
        self.n_rounds = n_rounds

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds_1src(self, margin: float = 0.05):
        """Get bounds for 1-source (2D)."""
        lb = [margin * self.Lx, margin * self.Ly]
        ub = [(1 - margin) * self.Lx, (1 - margin) * self.Ly]
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

    def _optimize_1src(
        self,
        init_params: np.ndarray,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        max_fevals: int,
    ) -> Tuple[np.ndarray, float, int]:
        """Run CMA-ES for 1-source (2D optimization)."""

        def objective(xy_params):
            x, y = xy_params
            _, _, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            return rmse

        lb, ub = self._get_position_bounds_1src()

        opts = {
            'maxfevals': max_fevals,
            'bounds': [lb, ub],
            'verbose': -9,
            'tolfun': 1e-6,
            'tolx': 1e-6,
        }

        es = cma.CMAEvolutionStrategy(init_params.tolist(), self.sigma0_1src, opts)
        n_evals = 0

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            n_evals += len(fitness)
            es.tell(solutions, fitness)

        best_params = np.array(es.result.xbest)
        best_rmse = es.result.fbest

        return best_params, best_rmse, n_evals

    def _optimize_single_source_2d(
        self,
        init_xy: np.ndarray,
        fixed_xy: np.ndarray,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        max_fevals: int,
        source_idx: int,  # 0 or 1 (which source to optimize)
    ) -> Tuple[np.ndarray, float, int]:
        """
        Optimize one source position while keeping the other fixed.

        Args:
            init_xy: Initial (x, y) for the source to optimize
            fixed_xy: Fixed (x, y) for the other source
            source_idx: 0 means optimizing first source, 1 means second
        """

        def objective(xy_params):
            x, y = xy_params
            if source_idx == 0:
                # Optimizing source 1, source 2 is fixed
                _, _, rmse = compute_optimal_intensity_2src(
                    x, y, fixed_xy[0], fixed_xy[1],
                    Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            else:
                # Optimizing source 2, source 1 is fixed
                _, _, rmse = compute_optimal_intensity_2src(
                    fixed_xy[0], fixed_xy[1], x, y,
                    Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            return rmse

        lb, ub = self._get_position_bounds_1src()

        opts = {
            'maxfevals': max_fevals,
            'bounds': [lb, ub],
            'verbose': -9,
            'tolfun': 1e-6,
            'tolx': 1e-6,
        }

        es = cma.CMAEvolutionStrategy(init_xy.tolist(), self.sigma0_2src, opts)
        n_evals = 0

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            n_evals += len(fitness)
            es.tell(solutions, fitness)

        best_params = np.array(es.result.xbest)
        best_rmse = es.result.fbest

        return best_params, best_rmse, n_evals

    def _optimize_2src_alternating(
        self,
        init_params: np.ndarray,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        max_fevals: int,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Run alternating optimization for 2-source problem.

        Alternates between:
        - Optimizing source 1 while source 2 is fixed
        - Optimizing source 2 while source 1 is fixed
        """
        x1, y1, x2, y2 = init_params
        source1 = np.array([x1, y1])
        source2 = np.array([x2, y2])

        # Distribute fevals across rounds
        fevals_per_subproblem = max(max_fevals // (2 * self.n_rounds), 4)

        total_evals = 0
        best_rmse = float('inf')

        for round_idx in range(self.n_rounds):
            # Optimize source 1, keep source 2 fixed
            source1, rmse1, n_evals = self._optimize_single_source_2d(
                source1, source2, Y_observed, solver, dt, nt, T0,
                sensors_xy, q_range, fevals_per_subproblem, source_idx=0
            )
            total_evals += n_evals

            # Optimize source 2, keep source 1 fixed
            source2, rmse2, n_evals = self._optimize_single_source_2d(
                source2, source1, Y_observed, solver, dt, nt, T0,
                sensors_xy, q_range, fevals_per_subproblem, source_idx=1
            )
            total_evals += n_evals

            best_rmse = rmse2  # Last computed RMSE

        best_params = np.array([source1[0], source1[1], source2[0], source2[1]])
        return best_params, best_rmse, total_evals

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
        Estimate sources using alternating optimization.
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
        features = extract_enhanced_features(sample, meta)

        # Transfer learning
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        # Build initializations
        initializations = []

        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

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

        # Optimize from best init only (full fevals)
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        best_init, best_init_type, _ = init_results[0]

        all_results = []

        if n_sources == 1:
            opt_params, opt_rmse, n_evals = self._optimize_1src(
                best_init, Y_observed, solver, dt, nt, T0, sensors_xy, q_range, max_fevals
            )
            q, _, rmse = compute_optimal_intensity_1src(
                opt_params[0], opt_params[1], Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            sources = [(opt_params[0], opt_params[1], q)]
        else:
            opt_params, opt_rmse, n_evals = self._optimize_2src_alternating(
                best_init, Y_observed, solver, dt, nt, T0, sensors_xy, q_range, max_fevals
            )
            (q1, q2), _, rmse = compute_optimal_intensity_2src(
                opt_params[0], opt_params[1], opt_params[2], opt_params[3],
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            sources = [(opt_params[0], opt_params[1], q1), (opt_params[2], opt_params[3], q2)]

        full_params = np.array([p for src in sources for p in src])
        all_results.append(CandidateResult(
            params=full_params,
            rmse=rmse,
            init_type=best_init_type,
            n_evals=n_evals,
        ))

        # Add other inits as additional candidates (not fully optimized)
        for init_params, init_type, init_rmse in init_results[1:self.n_candidates]:
            if n_sources == 1:
                x, y = init_params
                q, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                sources_extra = [(x, y, q)]
            else:
                x1, y1, x2, y2 = init_params
                (q1, q2), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                sources_extra = [(x1, y1, q1), (x2, y2, q2)]

            extra_params = np.array([p for src in sources_extra for p in src])
            all_results.append(CandidateResult(
                params=extra_params,
                rmse=rmse,
                init_type=init_type,
                n_evals=0,  # Already evaluated during init
            ))

        # Sort by RMSE
        all_results.sort(key=lambda x: x.rmse)

        # Build candidates list for filtering
        candidate_list = []
        for result in all_results:
            if n_sources == 1:
                sources_cand = [(result.params[0], result.params[1], result.params[2])]
            else:
                sources_cand = [
                    (result.params[0], result.params[1], result.params[2]),
                    (result.params[3], result.params[4], result.params[5])
                ]
            candidate_list.append((sources_cand, result.rmse, result))

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
