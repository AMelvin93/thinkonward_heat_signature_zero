"""
Bayesian Optimization for Heat Source Identification.

Key Innovation: Use Gaussian Process surrogate for more sample-efficient optimization.

CMA-ES uses ~22 fevals for 2-source. BO could potentially achieve similar accuracy
with 30-50% fewer fevals by being more strategic about where to sample.

Strategy:
1. Latin Hypercube sampling for initial points (5-8 points)
2. Fit GP surrogate to observed (x, y) -> RMSE mapping
3. Use Expected Improvement to select next point
4. Repeat until budget exhausted
5. Return best solution found
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add simulator path
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
except ImportError:
    raise ImportError("sklearn is required for Bayesian Optimization. Install with: pip install scikit-learn")


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


def latin_hypercube_sample(bounds: np.ndarray, n_samples: int, seed: int = None) -> np.ndarray:
    """Generate Latin Hypercube samples within bounds."""
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_dims):
        # Create equally spaced intervals
        cut = np.linspace(0, 1, n_samples + 1)
        # Sample uniformly within each interval
        u = np.random.uniform(low=cut[:-1], high=cut[1:])
        # Shuffle to get latin hypercube property
        np.random.shuffle(u)
        # Scale to bounds
        samples[:, i] = bounds[i, 0] + u * (bounds[i, 1] - bounds[i, 0])

    return samples


def expected_improvement(X: np.ndarray, gp: GaussianProcessRegressor,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Compute Expected Improvement at points X."""
    mu, sigma = gp.predict(X, return_std=True)

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-8)

    # Standardized improvement
    Z = (y_best - mu - xi) / sigma

    # EI formula
    ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))

    # Handle edge case where sigma is very small
    ei[sigma < 1e-8] = 0.0

    return ei


class BayesianOptimizer:
    """
    Bayesian Optimization for heat source identification.

    Uses Gaussian Process surrogate with Expected Improvement acquisition
    to efficiently explore the position space.

    Key differences from CMA-ES:
    - Model-based: builds a surrogate of the objective function
    - Sample-efficient: strategically samples where improvement is expected
    - Could match CMA-ES accuracy with fewer fevals
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_initial_1src: int = 5,
        n_initial_2src: int = 8,
        n_bo_iter_1src: int = 7,
        n_bo_iter_2src: int = 14,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        use_enhanced_features: bool = True,
        # GP parameters
        gp_kernel: str = 'matern',  # 'rbf' or 'matern'
        xi: float = 0.01,  # exploration-exploitation trade-off
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_initial_1src = n_initial_1src
        self.n_initial_2src = n_initial_2src
        self.n_bo_iter_1src = n_bo_iter_1src
        self.n_bo_iter_2src = n_bo_iter_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.use_enhanced_features = use_enhanced_features
        self.gp_kernel = gp_kernel
        self.xi = xi

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05) -> np.ndarray:
        """Get bounds as numpy array for BO."""
        bounds = []
        for _ in range(n_sources):
            bounds.append([margin * self.Lx, (1 - margin) * self.Lx])
            bounds.append([margin * self.Ly, (1 - margin) * self.Ly])
        return np.array(bounds)

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

    def _create_gp(self) -> GaussianProcessRegressor:
        """Create Gaussian Process with appropriate kernel."""
        if self.gp_kernel == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.01)
        else:  # matern
            kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.01)

        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        )

    def _optimize_acquisition(self, gp: GaussianProcessRegressor, bounds: np.ndarray,
                               y_best: float, n_restarts: int = 10) -> np.ndarray:
        """Find the point that maximizes expected improvement."""
        n_dims = len(bounds)

        def neg_ei(x):
            return -expected_improvement(x.reshape(1, -1), gp, y_best, self.xi)[0]

        best_x = None
        best_ei = float('inf')

        # Multi-start optimization
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

            try:
                result = minimize(
                    neg_ei,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )

                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x
            except Exception:
                continue

        if best_x is None:
            # Fallback: random sample
            best_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

        return best_x

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
        Estimate sources using Bayesian Optimization.

        Phase 1: Initial sampling (Latin Hypercube + good inits)
        Phase 2: BO iterations with GP surrogate and EI acquisition
        Phase 3: Build candidates from best solutions
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
        bounds = self._get_position_bounds(n_sources)

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

        # Simulation counter
        n_sims = [0]

        # === OBJECTIVE FUNCTION ===
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

        # === PHASE 1: INITIAL SAMPLING ===
        n_initial = self.n_initial_1src if n_sources == 1 else self.n_initial_2src

        # Start with known good initializations
        init_points = []

        # Triangulation
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            init_points.append(('triangulation', tri_init))

        # Smart (hottest sensor)
        smart_init = self._smart_init_positions(sample, n_sources)
        init_points.append(('smart', smart_init))

        # Transfer
        for i, sol in enumerate(similar_solutions):
            sol_clipped = np.clip(sol, bounds[:, 0], bounds[:, 1])
            init_points.append((f'transfer_{i}', sol_clipped))

        # Fill remaining with LHS
        n_lhs = max(0, n_initial - len(init_points))
        if n_lhs > 0:
            lhs_samples = latin_hypercube_sample(bounds, n_lhs, seed=42)
            for i, sample_pt in enumerate(lhs_samples):
                init_points.append((f'lhs_{i}', sample_pt))

        # Evaluate initial points
        X_train = []
        y_train = []
        all_solutions = []
        best_init_type = 'unknown'
        best_rmse_so_far = float('inf')

        for init_type, init_params in init_points[:n_initial]:
            rmse = objective(init_params)
            X_train.append(init_params)
            y_train.append(rmse)
            all_solutions.append((init_params.copy(), rmse, init_type))

            if rmse < best_rmse_so_far:
                best_rmse_so_far = rmse
                best_init_type = init_type

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if verbose:
            print(f"  Initial sampling ({len(X_train)} points): best RMSE = {best_rmse_so_far:.4f} ({best_init_type})")

        # === PHASE 2: BAYESIAN OPTIMIZATION ===
        n_bo_iter = self.n_bo_iter_1src if n_sources == 1 else self.n_bo_iter_2src
        gp = self._create_gp()

        for i in range(n_bo_iter):
            # Fit GP
            try:
                gp.fit(X_train, y_train)
            except Exception as e:
                if verbose:
                    print(f"  GP fit failed at iter {i}: {e}")
                break

            # Find next point via EI
            y_best = np.min(y_train)
            next_x = self._optimize_acquisition(gp, bounds, y_best)

            # Evaluate
            next_y = objective(next_x)

            # Update training data
            X_train = np.vstack([X_train, next_x])
            y_train = np.append(y_train, next_y)
            all_solutions.append((next_x.copy(), next_y, f'bo_{i}'))

            if next_y < best_rmse_so_far:
                best_rmse_so_far = next_y
                best_init_type = f'bo_{i}'

            if verbose and i % 3 == 0:
                print(f"  BO iter {i+1}/{n_bo_iter}: RMSE = {next_y:.4f}, best = {best_rmse_so_far:.4f}")

        # === PHASE 3: BUILD CANDIDATES FROM TOP SOLUTIONS ===
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
            best_positions = np.zeros(n_sources * 2)

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
