"""
Fast ICA Decomposition Optimizer for Heat Source Identification.

Accelerated version of ICA decomposition with speedup strategies:
1. FastICA with max_iter=50 instead of 500 (10x faster)
2. Reduced CMA-ES fevals (10/15 instead of 15/20)
3. Simplified without batched transfer learning
4. Coarse grid (50x25) for initial optimization
5. Only run ICA for 2-source problems

Physics Insight:
    The heat equation is LINEAR - temperature fields from multiple sources ADD:
    T_total = T_source1 + T_source2

    ICA can decompose the mixed signals to extract position information
    from the mixing matrix A in Y = A @ S.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.decomposition import FastICA
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
    """Compute optimal intensities for 2-source analytically using linearity."""
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


def fast_ica_decompose_2source(
    Y_obs: np.ndarray,
    sensors_xy: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 1.0,
    margin: float = 0.05,
    max_iter: int = 50,  # Reduced from 500!
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Fast ICA decomposition for 2-source position estimation.

    Key speedup: max_iter=50 instead of 500 (10x faster).
    """
    info = {
        'success': False,
        'method': 'fast_ica',
        'max_iter': max_iter,
    }

    if Y_obs.shape[1] < 2:
        info['error'] = 'Not enough sensors for ICA'
        return None, info

    try:
        # Fast ICA with reduced iterations
        ica = FastICA(n_components=2, random_state=42, max_iter=max_iter, tol=1e-3)

        S = ica.fit_transform(Y_obs)  # (time, 2) - estimated source signals
        A = ica.mixing_  # (n_sensors, 2) - mixing matrix

        info['success'] = True
        info['n_iter'] = getattr(ica, 'n_iter_', max_iter)

        # Extract positions from mixing matrix using weighted centroids
        positions = []
        for i in range(2):
            mixing_coeffs = np.abs(A[:, i])
            weights = mixing_coeffs / (mixing_coeffs.sum() + 1e-8)

            x_est = np.average(sensors_xy[:, 0], weights=weights)
            y_est = np.average(sensors_xy[:, 1], weights=weights)

            x_est = np.clip(x_est, margin * Lx, (1 - margin) * Lx)
            y_est = np.clip(y_est, margin * Ly, (1 - margin) * Ly)

            positions.extend([x_est, y_est])

        return np.array(positions), info

    except Exception as e:
        info['error'] = str(e)
        return None, info


class FastICAOptimizer:
    """
    Accelerated CMA-ES optimizer with Fast ICA initialization.

    Speedup strategies:
    1. FastICA with max_iter=50 (10x faster than default 500)
    2. Reduced CMA-ES fevals (10/15 vs 15/20)
    3. Coarse grid (50x25) for initial search
    4. No batched transfer learning (simpler, faster)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        max_fevals_1src: int = 10,
        max_fevals_2src: int = 15,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        use_ica: bool = True,
        ica_max_iter: int = 50,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 8,
        use_coarse_search: bool = True,
        refine_maxiter: int = 3,
        rmse_threshold_1src: float = 0.35,
        rmse_threshold_2src: float = 0.45,
    ):
        """Initialize the Fast ICA optimizer."""
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.use_ica = use_ica
        self.ica_max_iter = ica_max_iter
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.use_coarse_search = use_coarse_search
        self.refine_maxiter = refine_maxiter
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
        except Exception:
            return None

    def _ica_init_positions(self, sample: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        """Get position initialization from Fast ICA (2-source only)."""
        if not self.use_ica or sample['n_sources'] != 2:
            return None, {'success': False, 'reason': 'ICA only for 2-source'}

        Y_obs = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])

        return fast_ica_decompose_2source(
            Y_obs, sensors_xy, self.Lx, self.Ly,
            max_iter=self.ica_max_iter
        )

    def _refine_with_nelder_mead(self, pos_params: np.ndarray, objective_fn, n_sources: int) -> np.ndarray:
        """Quick Nelder-Mead refinement (3 iterations max)."""
        from scipy.optimize import minimize

        lb, ub = self._get_position_bounds(n_sources)
        bounds = [(l, u) for l, u in zip(lb, ub)]

        try:
            result = minimize(
                objective_fn,
                pos_params,
                method='Nelder-Mead',
                options={'maxiter': self.refine_maxiter, 'maxfev': 10}
            )
            refined = np.clip(result.x, lb, ub)
            return refined
        except Exception:
            return pos_params

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """Estimate sources with Fast ICA initialization."""
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Use coarse grid for initial search
        if self.use_coarse_search:
            solver_coarse = self._create_solver(kappa, bc, self.nx_coarse, self.ny_coarse)
        solver_fine = self._create_solver(kappa, bc, self.nx, self.ny)

        ica_info = {'used': False, 'success': False}
        n_sims = [0]

        # Objective function for coarse search
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                solver = solver_coarse if self.use_coarse_search else solver_fine
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

            def objective_fine(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                solver = solver_coarse if self.use_coarse_search else solver_fine
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

            def objective_fine(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # Build initialization pool
        initializations = []

        # 1. Fast ICA (2-source only) - THE KEY INNOVATION
        if n_sources == 2 and self.use_ica:
            ica_init, ica_info = self._ica_init_positions(sample)
            if ica_init is not None:
                initializations.append((ica_init, 'ica'))
                if verbose:
                    print(f"  ICA init: success={ica_info.get('success')}")

        # 2. Triangulation
        if self.use_triangulation:
            tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
            if tri_init is not None:
                initializations.append((tri_init, 'triangulation'))

        # 3. Hottest sensor
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        if verbose:
            print(f"  Initializations: {[init[1] for init in initializations]}")

        # CMA-ES optimization on coarse grid
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

        # Get top solutions and refine on fine grid
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        # Refine top solutions with Nelder-Mead on fine grid
        refined_solutions = []
        for pos_params, coarse_rmse, init_type in top_solutions:
            # Quick Nelder-Mead refinement
            refined_pos = self._refine_with_nelder_mead(pos_params, objective_fine, n_sources)
            fine_rmse = objective_fine(refined_pos)
            refined_solutions.append((refined_pos, fine_rmse, init_type + '_refined'))

        # Convert to full params with analytical intensity
        candidates_raw = []
        for pos_params, rmse, init_type in refined_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
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

        # Check if fallback needed
        best_rmse = min(c[2] for c in final_candidates) if final_candidates else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse > threshold:
            # Fallback: run more fevals with larger sigma
            fallback_sigma = sigma0 * 1.5
            fallback_fevals = max_fevals * 2

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
                fitness = [objective_fine(s) for s in solutions]
                es.tell(solutions, fitness)

            best_pos = np.array(es.result.xbest)

            if n_sources == 1:
                x, y = best_pos
                q, _, fb_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                fb_params = np.array([x, y, q])
                fb_sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = best_pos
                (q1, q2), _, fb_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
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
