"""
Multi-Fidelity GP Surrogate Optimizer for Heat Source Identification.

APPROACH A2: Use coarse simulations to guide expensive fine simulations.

Key Innovation:
- Phase 1: Explore with coarse grid (50x25) - ~2x faster per sim
- Phase 2: Build GP surrogate from coarse evaluations
- Phase 3: Use Expected Improvement to find best candidates
- Phase 4: Evaluate top candidates on fine grid (100x50)

Expected Benefits:
- More exploration with same time budget
- Better initial positions for CMA-ES
- Smarter allocation of expensive simulations
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
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

    return np.array(basic + [centroid_x, centroid_y, spatial_spread])


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
) -> Tuple[float, float]:
    """Compute optimal intensity for 1-source analytically. Returns (q, rmse)."""
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

    return q_optimal, rmse


def compute_optimal_intensity_2src(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], float]:
    """Compute optimal intensities for 2-source analytically. Returns ((q1, q2), rmse)."""
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

    return (q1, q2), rmse


def expected_improvement(X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
    """Compute Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-8)

    z = (y_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))

    return ei


class MultiFidelityGPOptimizer:
    """
    Multi-Fidelity GP Surrogate Optimizer.

    Key Innovation:
    - Use coarse grid for exploration (faster per sim)
    - Build GP surrogate from coarse evaluations
    - Use Expected Improvement to find promising candidates
    - Only refine top candidates on fine grid
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        n_coarse_samples: int = 15,
        n_bo_iterations: int = 5,
        n_fine_candidates: int = 5,
        max_fevals_fine: int = 8,
        sigma0_fine: float = 0.10,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.n_coarse_samples = n_coarse_samples
        self.n_bo_iterations = n_bo_iterations
        self.n_fine_candidates = n_fine_candidates
        self.max_fevals_fine = max_fevals_fine
        self.sigma0_fine = sigma0_fine
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar

    def _create_solver(self, kappa: float, bc: str, nx: int, ny: int) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, nx, ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return np.array(lb), np.array(ub)

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

    def _evaluate_coarse(self, positions: np.ndarray, n_sources: int, Y_observed_coarse: np.ndarray,
                         solver_coarse: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
                         q_range: Tuple[float, float]) -> float:
        """Evaluate positions on coarse grid."""
        if n_sources == 1:
            x, y = positions
            _, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed_coarse, solver_coarse, dt, nt, T0, sensors_xy, q_range)
        else:
            x1, y1, x2, y2 = positions
            _, rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed_coarse, solver_coarse, dt, nt, T0, sensors_xy, q_range)
        return rmse

    def _evaluate_fine(self, positions: np.ndarray, n_sources: int, Y_observed: np.ndarray,
                       solver_fine: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
                       q_range: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """Evaluate positions on fine grid. Returns (full_params, rmse)."""
        if n_sources == 1:
            x, y = positions
            q, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
            full_params = np.array([x, y, q])
        else:
            x1, y1, x2, y2 = positions
            (q1, q2), rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
            full_params = np.array([x1, y1, q1, x2, y2, q2])
        return full_params, rmse

    def _generate_coarse_observation(self, sample: Dict, meta: Dict, solver_coarse: Heat2D) -> np.ndarray:
        """Generate observation on coarse grid from fine observation."""
        # For simplicity, we'll use the original observation directly
        # In practice, we might want to interpolate or re-simulate
        return sample['Y_noisy']

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """Estimate sources using multi-fidelity GP approach."""
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create solvers
        solver_coarse = self._create_solver(kappa, bc, self.nx_coarse, self.ny_coarse)
        solver_fine = self._create_solver(kappa, bc, self.nx_fine, self.ny_fine)

        # Feature extraction
        features = extract_enhanced_features(sample, meta)

        # Transfer learning
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        lb, ub = self._get_position_bounds(n_sources)
        n_dims = 2 * n_sources

        n_sims = [0]

        # === PHASE 1: Initial sampling on coarse grid ===
        X_coarse = []
        y_coarse = []

        # Add known good initializations
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            X_coarse.append(tri_init)

        smart_init = self._smart_init_positions(sample, n_sources)
        X_coarse.append(smart_init)

        for sol in similar_solutions:
            X_coarse.append(np.clip(sol, lb, ub))

        # Add random samples to fill coarse sample budget
        n_random = max(0, self.n_coarse_samples - len(X_coarse))
        for _ in range(n_random):
            random_sample = lb + np.random.rand(n_dims) * (ub - lb)
            X_coarse.append(random_sample)

        # Evaluate all coarse samples
        for x in X_coarse:
            rmse = self._evaluate_coarse(
                x, n_sources, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range
            )
            y_coarse.append(rmse)
            n_sims[0] += (1 if n_sources == 1 else 2)

        X_coarse = np.array(X_coarse)
        y_coarse = np.array(y_coarse)

        if verbose:
            print(f"  Phase 1: {len(X_coarse)} coarse samples, best RMSE={np.min(y_coarse):.4f}")

        # === PHASE 2: Build GP surrogate ===
        kernel = Matern(nu=2.5, length_scale=0.2, length_scale_bounds=(0.05, 1.0)) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
        gp.fit(X_coarse, y_coarse)

        # === PHASE 3: Bayesian optimization iterations ===
        for i in range(self.n_bo_iterations):
            # Find next point via Expected Improvement
            n_candidates = 500
            X_candidates = lb + np.random.rand(n_candidates, n_dims) * (ub - lb)

            ei = expected_improvement(X_candidates, gp, np.min(y_coarse))
            best_idx = np.argmax(ei)
            x_next = X_candidates[best_idx]

            # Evaluate on coarse grid
            y_next = self._evaluate_coarse(
                x_next, n_sources, Y_observed, solver_coarse, dt, nt, T0, sensors_xy, q_range
            )
            n_sims[0] += (1 if n_sources == 1 else 2)

            # Update GP
            X_coarse = np.vstack([X_coarse, x_next])
            y_coarse = np.append(y_coarse, y_next)
            gp.fit(X_coarse, y_coarse)

        if verbose:
            print(f"  Phase 2-3: +{self.n_bo_iterations} BO iterations, best RMSE={np.min(y_coarse):.4f}")

        # === PHASE 4: Evaluate top candidates on fine grid ===
        # Sort by coarse RMSE and take top candidates
        sorted_indices = np.argsort(y_coarse)[:self.n_fine_candidates]

        fine_results = []
        for idx in sorted_indices:
            positions = X_coarse[idx]

            # Short CMA-ES refinement on fine grid
            if n_sources == 1:
                def objective(xy):
                    x, y = xy
                    n_sims[0] += 1
                    _, rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    return rmse
            else:
                def objective(xy):
                    x1, y1, x2, y2 = xy
                    n_sims[0] += 2
                    _, rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range)
                    return rmse

            opts = cma.CMAOptions()
            opts['maxfevals'] = self.max_fevals_fine
            opts['bounds'] = [lb.tolist(), ub.tolist()]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6

            es = cma.CMAEvolutionStrategy(positions.tolist(), self.sigma0_fine, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

            best_pos = np.array(es.result.xbest)
            full_params, final_rmse = self._evaluate_fine(
                best_pos, n_sources, Y_observed, solver_fine, dt, nt, T0, sensors_xy, q_range
            )
            n_sims[0] += (1 if n_sources == 1 else 2)

            fine_results.append((full_params, final_rmse, 'multi_fidelity'))

        if verbose:
            print(f"  Phase 4: {len(fine_results)} fine evaluations, best RMSE={min(r[1] for r in fine_results):.4f}")

        # === BUILD CANDIDATES ===
        # Sort by RMSE and take top candidates
        fine_results.sort(key=lambda x: x[1])

        candidates_raw = []
        for full_params, rmse, init_type in fine_results[:self.candidate_pool_size]:
            if n_sources == 1:
                sources = [(float(full_params[0]), float(full_params[1]), float(full_params[2]))]
            else:
                sources = [
                    (float(full_params[0]), float(full_params[1]), float(full_params[2])),
                    (float(full_params[3]), float(full_params[4]), float(full_params[5]))
                ]
            candidates_raw.append((sources, full_params, rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        # Match filtered back to full candidates
        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # Build results
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
            best_positions = smart_init

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_sims[0] // len(final_candidates) if final_candidates else n_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, features, best_positions, n_transferred
