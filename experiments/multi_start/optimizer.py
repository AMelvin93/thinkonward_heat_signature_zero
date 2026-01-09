"""
Multi-Start CMA-ES Optimizer for Heat Source Identification.

Key Innovation: Run CMA-ES multiple times from different starting points
to escape local minima, especially for 2-source problems.

Approach:
1. Generate diverse starting points (triangulation, hottest sensors, random, transfer)
2. Run short CMA-ES from each starting point
3. Keep best solution(s) from each run
4. Optionally refine top candidates with more fevals
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


class MultiStartOptimizer:
    """
    Multi-Start CMA-ES optimizer for better global search.

    Key Innovation: Run CMA-ES from multiple diverse starting points
    to escape local minima, especially for 2-source problems.

    Strategy:
    1. Generate diverse inits (triangulation, hottest sensors, random, transfer)
    2. Run short CMA-ES from each starting point (exploration phase)
    3. Keep best solutions from each run
    4. Optionally refine top candidates (exploitation phase)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        # Number of restarts
        n_restarts_1src: int = 2,
        n_restarts_2src: int = 3,
        # Fevals per restart
        fevals_per_restart_1src: int = 6,
        fevals_per_restart_2src: int = 8,
        # Refinement fevals for top candidate
        refine_fevals_1src: int = 4,
        refine_fevals_2src: int = 6,
        sigma0_1src: float = 0.20,  # Larger sigma for exploration
        sigma0_2src: float = 0.25,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        n_random_inits: int = 2,  # Number of random starting points
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_restarts_1src = n_restarts_1src
        self.n_restarts_2src = n_restarts_2src
        self.fevals_per_restart_1src = fevals_per_restart_1src
        self.fevals_per_restart_2src = fevals_per_restart_2src
        self.refine_fevals_1src = refine_fevals_1src
        self.refine_fevals_2src = refine_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.n_random_inits = n_random_inits

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _generate_random_init(self, n_sources: int, margin: float = 0.1) -> np.ndarray:
        """Generate a random starting point within bounds."""
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            params.extend([x, y])
        return np.array(params)

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

    def _run_cmaes(self, init_params: np.ndarray, n_sources: int, max_fevals: int,
                   Y_observed: np.ndarray, solver: Heat2D, dt: float, nt: int,
                   T0: float, sensors_xy: np.ndarray, q_range: Tuple[float, float],
                   sigma0: float) -> Tuple[np.ndarray, float, int]:
        """Run CMA-ES from given starting point."""
        lb, ub = self._get_position_bounds(n_sources)
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

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

        best_sol = init_params.copy()
        best_rmse = float('inf')

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                if fit < best_rmse:
                    best_rmse = fit
                    best_sol = np.array(sol)

        return best_sol, best_rmse, n_sims[0]

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
        Multi-start optimization with restart diversity.
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

        features = extract_enhanced_features(sample, meta)

        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        # === PHASE 1: GENERATE DIVERSE STARTING POINTS ===
        initializations = []

        # Triangulation init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # Hottest sensor init
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # Transfer learning inits
        lb, ub = self._get_position_bounds(n_sources)
        for i, sol in enumerate(similar_solutions):
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        # Random inits for diversity
        for i in range(self.n_random_inits):
            rand_init = self._generate_random_init(n_sources)
            initializations.append((rand_init, f'random_{i}'))

        # === PHASE 2: RUN SHORT CMA-ES FROM EACH STARTING POINT ===
        n_restarts = self.n_restarts_1src if n_sources == 1 else self.n_restarts_2src
        fevals_per_restart = self.fevals_per_restart_1src if n_sources == 1 else self.fevals_per_restart_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        # Select which inits to use (up to n_restarts)
        selected_inits = initializations[:n_restarts]

        # Pad with random if needed
        while len(selected_inits) < n_restarts:
            rand_init = self._generate_random_init(n_sources)
            selected_inits.append((rand_init, f'random_extra_{len(selected_inits)}'))

        restart_results = []
        total_sims = 0

        for init_params, init_type in selected_inits:
            best_sol, best_rmse, n_sims = self._run_cmaes(
                init_params, n_sources, fevals_per_restart,
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range, sigma0
            )
            total_sims += n_sims
            restart_results.append((best_sol, best_rmse, init_type))

            if verbose:
                print(f"  Restart {init_type}: RMSE={best_rmse:.4f}")

        # === PHASE 3: REFINE TOP CANDIDATE ===
        restart_results.sort(key=lambda x: x[1])
        top_sol, top_rmse, top_type = restart_results[0]

        refine_fevals = self.refine_fevals_1src if n_sources == 1 else self.refine_fevals_2src

        if refine_fevals > 0:
            refined_sol, refined_rmse, refine_sims = self._run_cmaes(
                top_sol, n_sources, refine_fevals,
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range, sigma0 * 0.5  # Smaller sigma for refinement
            )
            total_sims += refine_sims

            if refined_rmse < top_rmse:
                top_sol = refined_sol
                top_rmse = refined_rmse
                restart_results[0] = (top_sol, top_rmse, top_type + '_refined')

            if verbose:
                print(f"  Refined: RMSE={top_rmse:.4f}")

        # === BUILD CANDIDATES ===
        candidates_raw = []

        for pos_params, rmse, init_type in restart_results[:self.candidate_pool_size]:
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
            best_positions = top_sol

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
