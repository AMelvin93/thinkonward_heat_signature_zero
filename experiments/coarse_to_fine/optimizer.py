"""
Coarse-to-Fine Optimizer for Heat Source Identification.

Key Innovation: Use coarse grid (50x25) for CMA-ES exploration which is
~4x faster than fine grid (100x50). Only use fine grid for final polish.

Combined with Smart Init Selection:
1. Evaluate all inits quickly on coarse grid
2. Run CMA-ES with many fevals on coarse grid (fast exploration)
3. Polish top solutions on fine grid (accurate refinement)

Expected Benefits:
    - 4x speedup on CMA-ES exploration
    - Enable 30+ fevals for 2-source within budget
    - Potential score: 1.05+ if accuracy is maintained
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


# ============================================================
# COARSE GRID SIMULATION (50x25)
# ============================================================

def create_coarse_solver(kappa: float, bc: str, Lx: float = 2.0, Ly: float = 1.0) -> Heat2D:
    """Create a coarse-resolution solver (50x25 grid)."""
    return Heat2D(Lx, Ly, 50, 25, kappa, bc=bc)


def simulate_unit_source_coarse(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                                T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    """Simulate a source with q=1.0 on coarse grid."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src_coarse(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    """Compute optimal intensity for 1-source on coarse grid."""
    Y_unit = simulate_unit_source_coarse(x, y, solver, dt, nt, T0, sensors_xy)

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


def compute_optimal_intensity_2src_coarse(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], np.ndarray, float]:
    """Compute optimal intensities for 2-source on coarse grid."""
    Y1 = simulate_unit_source_coarse(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source_coarse(x2, y2, solver, dt, nt, T0, sensors_xy)

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


# ============================================================
# FINE GRID SIMULATION (100x50) - for polishing
# ============================================================

def simulate_unit_source_fine(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                              T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    """Simulate a source with q=1.0 on fine grid."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src_fine(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    """Compute optimal intensity for 1-source on fine grid."""
    Y_unit = simulate_unit_source_fine(x, y, solver, dt, nt, T0, sensors_xy)

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


def compute_optimal_intensity_2src_fine(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], np.ndarray, float]:
    """Compute optimal intensities for 2-source on fine grid."""
    Y1 = simulate_unit_source_fine(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source_fine(x2, y2, solver, dt, nt, T0, sensors_xy)

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


class CoarseToFineOptimizer:
    """
    Coarse-to-Fine optimizer with smart init selection.

    Strategy:
        1. Init selection: Evaluate all inits on COARSE grid (fast)
        2. CMA-ES exploration: Run on COARSE grid with many fevals (fast)
        3. Fine polish: Refine top solutions on FINE grid (accurate)

    This achieves ~4x speedup on exploration, enabling 30+ 2-source fevals
    while staying within the 60-minute budget.

    Polish modes:
        - 'eval_only': Just evaluate top coarse solutions on fine grid (fastest)
        - 'light_cmaes': Run a few CMA-ES iterations on fine grid (default)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        max_fevals_1src: int = 12,
        max_fevals_2src: int = 30,  # Higher than SmartInit due to coarse speedup
        polish_fevals_1src: int = 3,
        polish_fevals_2src: int = 5,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        use_enhanced_features: bool = True,
        polish_mode: str = 'eval_only',  # 'eval_only' or 'light_cmaes'
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.polish_fevals_1src = polish_fevals_1src
        self.polish_fevals_2src = polish_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.use_enhanced_features = use_enhanced_features
        self.polish_mode = polish_mode

    def _create_fine_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx_fine, self.ny_fine, kappa, bc=bc)

    def _create_coarse_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx_coarse, self.ny_coarse, kappa, bc=bc)

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

    def _evaluate_init_coarse(self, init_params: np.ndarray, n_sources: int,
                              Y_observed: np.ndarray, solver: Heat2D, dt: float,
                              nt: int, T0: float, sensors_xy: np.ndarray,
                              q_range: Tuple[float, float]) -> float:
        """Quickly evaluate an init on COARSE grid."""
        if n_sources == 1:
            x, y = init_params
            _, _, rmse = compute_optimal_intensity_1src_coarse(
                x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        else:
            x1, y1, x2, y2 = init_params
            _, _, rmse = compute_optimal_intensity_2src_coarse(
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
        """
        Estimate sources using coarse-to-fine strategy.

        Phase 1: Init selection on COARSE grid (fast)
        Phase 2: CMA-ES exploration on COARSE grid (fast)
        Phase 3: Polish top solutions on FINE grid (accurate)
        """
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create both solvers
        solver_coarse = self._create_coarse_solver(kappa, bc)
        solver_fine = self._create_fine_solver(kappa, bc)

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

        # === PHASE 1: INIT SELECTION ON COARSE GRID ===
        if verbose:
            print(f"  Phase 1: Init selection (coarse {self.nx_coarse}x{self.ny_coarse})")

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
            lb, ub = self._get_position_bounds(n_sources)
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        # Evaluate each init on COARSE grid (fast!)
        init_evaluations = []
        n_coarse_sims = 0
        for init_params, init_type in initializations:
            rmse = self._evaluate_init_coarse(
                init_params, n_sources, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range
            )
            init_evaluations.append((init_params, init_type, rmse))
            n_coarse_sims += (1 if n_sources == 1 else 2)

        # Sort by RMSE and pick the best
        init_evaluations.sort(key=lambda x: x[2])
        best_init_params, best_init_type, best_init_rmse = init_evaluations[0]

        if verbose:
            print(f"    Evaluated {len(initializations)} inits ({n_coarse_sims} coarse sims)")
            for params, itype, rmse in init_evaluations:
                marker = " <-- BEST" if itype == best_init_type else ""
                print(f"      {itype}: RMSE={rmse:.4f}{marker}")

        # === PHASE 2: CMA-ES ON COARSE GRID (FAST EXPLORATION) ===
        if verbose:
            print(f"  Phase 2: CMA-ES exploration (coarse {self.nx_coarse}x{self.ny_coarse})")

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_position_bounds(n_sources)

        # Track simulations
        n_cmaes_sims = [0]

        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_cmaes_sims[0] += 1
                _, _, rmse = compute_optimal_intensity_1src_coarse(
                    x, y, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_cmaes_sims[0] += 2
                _, _, rmse = compute_optimal_intensity_2src_coarse(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Run CMA-ES with FULL budget on coarse grid
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
            fitness = [objective_coarse(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit, best_init_type))

        if verbose:
            print(f"    CMA-ES completed: {n_cmaes_sims[0]} coarse sims, best coarse RMSE={es.result.fbest:.4f}")

        # === PHASE 3: POLISH TOP SOLUTIONS ON FINE GRID ===
        if verbose:
            print(f"  Phase 3: Fine polish (fine {self.nx_fine}x{self.ny_fine}, mode={self.polish_mode})")

        polish_fevals = self.polish_fevals_1src if n_sources == 1 else self.polish_fevals_2src

        # Sort by coarse RMSE and take top solutions for polishing
        all_solutions.sort(key=lambda x: x[1])
        top_coarse_solutions = all_solutions[:self.candidate_pool_size]

        # Track fine sims
        n_fine_sims = [0]

        polished_solutions = []

        if self.polish_mode == 'eval_only':
            # FAST MODE: Just evaluate top coarse solutions on fine grid (no CMA-ES)
            for pos_params, coarse_rmse, init_type in top_coarse_solutions:
                if n_sources == 1:
                    x, y = pos_params
                    _, _, fine_rmse = compute_optimal_intensity_1src_fine(
                        x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    n_fine_sims[0] += 1
                else:
                    x1, y1, x2, y2 = pos_params
                    _, _, fine_rmse = compute_optimal_intensity_2src_fine(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    n_fine_sims[0] += 2

                polished_solutions.append((pos_params, fine_rmse, init_type))

        else:  # light_cmaes mode
            # SLOWER MODE: Run a few CMA-ES iterations on fine grid
            if n_sources == 1:
                def objective_fine(xy_params):
                    x, y = xy_params
                    n_fine_sims[0] += 1
                    _, _, rmse = compute_optimal_intensity_1src_fine(
                        x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    return rmse
            else:
                def objective_fine(xy_params):
                    x1, y1, x2, y2 = xy_params
                    n_fine_sims[0] += 2
                    _, _, rmse = compute_optimal_intensity_2src_fine(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    return rmse

            for pos_params, coarse_rmse, init_type in top_coarse_solutions:
                # Quick polish with fine-grid CMA-ES
                opts_polish = cma.CMAOptions()
                opts_polish['maxfevals'] = polish_fevals
                opts_polish['bounds'] = [lb, ub]
                opts_polish['verbose'] = -9
                opts_polish['tolfun'] = 1e-7
                opts_polish['tolx'] = 1e-7

                # Use smaller sigma for polishing (already near optimum)
                sigma_polish = sigma0 * 0.3

                es_polish = cma.CMAEvolutionStrategy(pos_params.tolist(), sigma_polish, opts_polish)

                while not es_polish.stop():
                    solutions = es_polish.ask()
                    fitness = [objective_fine(s) for s in solutions]
                    es_polish.tell(solutions, fitness)

                polished_params = np.array(es_polish.result.xbest)
                polished_rmse = es_polish.result.fbest
                polished_solutions.append((polished_params, polished_rmse, init_type))

        if verbose:
            print(f"    Polish completed: {n_fine_sims[0]} fine sims")

        # === BUILD CANDIDATES FROM POLISHED SOLUTIONS ===
        polished_solutions.sort(key=lambda x: x[1])

        candidates_raw = []
        for pos_params, rmse, init_type in polished_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src_fine(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src_fine(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        # Match filtered back to full candidates
        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # === BUILD RESULTS ===
        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        # Best params for history
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

        total_sims = n_coarse_sims + n_cmaes_sims[0] + n_fine_sims[0]

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=total_sims // len(final_candidates) if final_candidates else total_sims
            )
            for c in final_candidates
        ]

        if verbose:
            print(f"  Total: {total_sims} sims ({n_coarse_sims} init + {n_cmaes_sims[0]} CMA-ES + {n_fine_sims[0]} polish)")
            print(f"  Best RMSE (fine): {best_rmse:.4f}")

        return candidate_sources, best_rmse, results, features, best_positions, n_transferred
