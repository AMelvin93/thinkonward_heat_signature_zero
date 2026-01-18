"""
Cluster Transfer Optimizer for Heat Source Identification.

Key Innovation: Cluster similar samples based on sensor reading features,
solve one representative sample per cluster fully, then use that solution
to warm-start optimization for other samples in the same cluster.

Hypothesis: Similar sensor patterns have similar optimal solutions.
This can save optimization time by transferring solutions.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cma
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

    from itertools import permutations
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


def extract_features(sample: Dict) -> np.ndarray:
    """
    Extract features from sensor readings for clustering.

    Features capture the spatial and temporal characteristics
    that determine optimal source positions.

    Returns fixed-length feature vector (14 features).
    """
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    features = []

    # 1. Basic statistics (5 features)
    features.append(np.max(Y))
    features.append(np.mean(Y))
    features.append(np.std(Y))
    features.append(kappa)
    features.append(float(sample['n_sources']))

    # 2. Spatial centroid (weighted by max temperature) (2 features)
    max_temps = Y.max(axis=0)
    weights = max_temps / (max_temps.sum() + 1e-8)
    centroid_x = np.average(sensors[:, 0], weights=weights)
    centroid_y = np.average(sensors[:, 1], weights=weights)
    features.extend([centroid_x, centroid_y])

    # 3. Spatial spread (2 features)
    spread_x = np.sqrt(np.average((sensors[:, 0] - centroid_x)**2, weights=weights))
    spread_y = np.sqrt(np.average((sensors[:, 1] - centroid_y)**2, weights=weights))
    features.extend([spread_x, spread_y])

    # 4. Temporal features - onset times (2 features)
    onset_times = []
    for i in range(Y.shape[1]):
        signal = Y[:, i]
        threshold = 0.1 * (signal.max() + 1e-8)
        onset_idx = np.argmax(signal > threshold)
        onset_times.append(onset_idx)
    onset_times = np.array(onset_times)
    features.append(np.mean(onset_times))
    features.append(np.std(onset_times))

    # 5. Peak temperatures - top 3 sensors (3 features, pad with zeros if fewer sensors)
    sorted_temps = np.sort(max_temps)[::-1]  # Descending order
    top_3_temps = np.zeros(3)
    for i in range(min(3, len(sorted_temps))):
        top_3_temps[i] = sorted_temps[i]
    features.extend(top_3_temps.tolist())

    return np.array(features, dtype=np.float64)


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
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator

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
        q1, q2 = np.linalg.solve(A + 1e-6 * np.eye(2), b)
    except:
        q1, q2 = 1.0, 1.0

    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])

    Y_pred = q1 * Y1 + q2 * Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    return (q1, q2), Y_pred, rmse


class ClusterTransferOptimizer:
    """
    CMA-ES optimizer with cluster-based transfer learning.

    Key Innovation: Cluster similar samples and transfer solutions
    from fully-solved representatives to warm-start others.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        max_fevals_1src: int = 20,
        max_fevals_2src: int = 36,
        max_fevals_transfer: int = 10,  # Reduced fevals when using transfer
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        sigma0_transfer: float = 0.08,  # Smaller sigma for transfer warmstart
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        refine_maxiter: int = 3,
        refine_top_n: int = 2,
        rmse_threshold_1src: float = 0.35,
        rmse_threshold_2src: float = 0.45,
    ):
        """Initialize the cluster transfer optimizer."""
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.max_fevals_transfer = max_fevals_transfer
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.sigma0_transfer = sigma0_transfer
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src

    def _create_solver(self, kappa: float, bc: str, nx: int = None, ny: int = None) -> Heat2D:
        """Create a Heat2D solver instance."""
        nx = nx or self.nx
        ny = ny or self.ny
        return Heat2D(self.Lx, self.Ly, nx, ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        """Get bounds for POSITION parameters only."""
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
        except:
            return None

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        transfer_init: Optional[np.ndarray] = None,  # Transfer from cluster representative
        is_representative: bool = False,  # If True, do full optimization
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources with optional transfer initialization.

        Args:
            transfer_init: Position params from cluster representative
            is_representative: If True, use full fevals (solving cluster rep)
        """
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver_coarse = self._create_solver(kappa, bc, self.nx_coarse, self.ny_coarse)
        solver_fine = self._create_solver(kappa, bc, self.nx, self.ny)

        n_sims = [0]

        # Objective function on coarse grid
        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Determine fevals and sigma based on transfer mode
        if transfer_init is not None and not is_representative:
            # Transfer mode: use reduced fevals and smaller sigma
            max_fevals = self.max_fevals_transfer
            sigma0 = self.sigma0_transfer
            init_type = 'transfer'
        else:
            # Full optimization mode
            max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
            sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
            init_type = 'full'

        lb, ub = self._get_position_bounds(n_sources)

        # Build initialization pool
        initializations = []

        # 1. Transfer initialization (if provided)
        if transfer_init is not None:
            initializations.append((transfer_init, 'transfer'))

        # 2. Triangulation
        if self.use_triangulation:
            tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
            if tri_init is not None:
                initializations.append((tri_init, 'triangulation'))

        # 3. Hottest sensor
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        if verbose:
            print(f"  Mode: {'representative' if is_representative else 'transfer'}, "
                  f"fevals={max_fevals}, sigma={sigma0:.3f}")

        # Allocate fevals per init
        fevals_per_init = max(5, max_fevals // len(initializations))

        all_solutions = []

        for init_params, init_name in initializations:
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
                    all_solutions.append((np.array(sol), fit, init_name))

        # Get top solutions
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        # Light Nelder-Mead refinement on coarse grid
        refined_solutions = []
        for pos_params, rmse, init_name in top_solutions[:self.refine_top_n]:
            try:
                result = minimize(objective, pos_params, method='Nelder-Mead',
                                  options={'maxiter': self.refine_maxiter, 'maxfev': 10})
                refined_pos = np.clip(result.x, lb, ub)
                refined_rmse = objective(refined_pos)
                refined_solutions.append((refined_pos, refined_rmse, init_name + '_refined'))
            except:
                refined_solutions.append((pos_params, rmse, init_name))

        # Add unrefined solutions
        for pos_params, rmse, init_name in top_solutions[self.refine_top_n:]:
            refined_solutions.append((pos_params, rmse, init_name))

        # Evaluate on fine grid and convert to full params
        candidates_raw = []
        for pos_params, coarse_rmse, init_name in refined_solutions[:self.candidate_pool_size]:
            if n_sources == 1:
                x, y = pos_params
                q, _, fine_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, fine_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, fine_rmse, init_name))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        # Match filtered back to full candidates
        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # Check if fallback needed
        best_rmse = min(c[2] for c in final_candidates) if final_candidates else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse > threshold:
            # Fallback with increased fevals
            fallback_sigma = (self.sigma0_1src if n_sources == 1 else self.sigma0_2src) * 1.5
            fallback_fevals = (self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src) * 2

            if verbose:
                print(f"  Fallback triggered: RMSE={best_rmse:.4f} > {threshold}")

            best_init = final_candidates[0][1][:n_sources*2] if final_candidates else smart_init

            opts = cma.CMAOptions()
            opts['maxfevals'] = fallback_fevals
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9

            es = cma.CMAEvolutionStrategy(best_init.tolist(), fallback_sigma, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

            best_pos = np.array(es.result.xbest)

            if n_sources == 1:
                x, y = best_pos
                q, _, fb_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 1
                fb_params = np.array([x, y, q])
                fb_sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = best_pos
                (q1, q2), _, fb_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                n_sims[0] += 2
                fb_params = np.array([x1, y1, q1, x2, y2, q2])
                fb_sources = [(float(x1), float(y1), float(q1)),
                              (float(x2), float(y2), float(q2))]

            if fb_rmse < best_rmse:
                final_candidates = [(fb_sources, fb_params, fb_rmse, 'fallback')]
                best_rmse = fb_rmse

        # Build results
        candidate_sources = [c[0] for c in final_candidates]

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_sims[0] // len(final_candidates) if final_candidates else n_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims[0]
