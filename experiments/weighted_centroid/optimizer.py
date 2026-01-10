"""
A13: Temperature-Weighted Centroid for Heat Source Localization.

Physics-based approach that DIRECTLY estimates source position from sensor temperatures.

Key Insight: Heat sources create temperature gradients. The weighted centroid of
sensor positions (weighted by temperature) estimates source position because:
1. Hotter sensors are closer to the source
2. Weighting by temperature "pulls" the centroid toward the source

For 2-source: Use clustering to assign sensors to sources, then apply
weighted centroid to each cluster.

This approach:
- Requires NO optimization for position (direct computation)
- Uses analytical intensity (closed-form q solution)
- Only needs minimal CMA-ES polish for final refinement
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma
from sklearn.cluster import KMeans
from scipy.interpolate import RBFInterpolator

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


class WeightedCentroidOptimizer:
    """
    Optimizer using temperature-weighted centroid for direct position estimation.

    Key Innovation: Physics-based position estimation that requires NO optimization:
    1. Compute temperature-weighted centroid of sensor positions
    2. Use analytical intensity for q estimation
    3. Minimal CMA-ES polish for final refinement
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
        use_gradient_refinement: bool = True,
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
        self.use_triangulation = use_triangulation
        self.use_gradient_refinement = use_gradient_refinement
        self.n_candidates = min(n_candidates, N_MAX)

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05):
        """Get parameter bounds."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return lb, ub

    def _weighted_centroid_1src(self, sample: Dict) -> Tuple[float, float]:
        """
        Compute temperature-weighted centroid for 1-source.

        Physics: Heat sources create temperature gradients. Hotter sensors are
        closer to the source. Weighted centroid estimates source position.
        """
        sensors = np.array(sample['sensors_xy'])
        Y = sample['Y_noisy']

        # Use late-time temperatures (quasi-steady, more reliable)
        # Take average of last 20% of timesteps
        n_late = max(1, Y.shape[0] // 5)
        T_late = Y[-n_late:].mean(axis=0)

        # Subtract baseline (ambient) temperature
        T_baseline = np.min(T_late)
        T_elevated = T_late - T_baseline

        # Use power weighting (emphasize hot sensors more)
        # Higher power = more weight to hottest sensors
        power = 2.0
        weights = np.power(T_elevated + 1e-8, power)
        weights = weights / weights.sum()

        # Weighted centroid
        x_est = np.average(sensors[:, 0], weights=weights)
        y_est = np.average(sensors[:, 1], weights=weights)

        # Clip to valid domain
        x_est = np.clip(x_est, 0.1, self.Lx - 0.1)
        y_est = np.clip(y_est, 0.1, self.Ly - 0.1)

        return x_est, y_est

    def _weighted_centroid_2src(self, sample: Dict) -> List[Tuple[float, float]]:
        """
        Compute temperature-weighted centroids for 2-source using clustering.

        Approach:
        1. Cluster sensors based on temperature patterns
        2. Apply weighted centroid to each cluster
        """
        sensors = np.array(sample['sensors_xy'])
        Y = sample['Y_noisy']
        n_sensors = len(sensors)

        # Use late-time temperatures
        n_late = max(1, Y.shape[0] // 5)
        T_late = Y[-n_late:].mean(axis=0)

        # Build feature matrix for clustering: [x, y, T_normalized]
        T_norm = (T_late - T_late.min()) / (T_late.max() - T_late.min() + 1e-8)
        features = np.column_stack([
            sensors[:, 0] / self.Lx,  # Normalized x
            sensors[:, 1] / self.Ly,  # Normalized y
            T_norm * 2.0,  # Temperature (weighted more)
        ])

        # K-means clustering
        if n_sensors >= 4:  # Need at least 2 sensors per cluster
            try:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = kmeans.fit_predict(features)
            except:
                labels = np.array([0 if i < n_sensors // 2 else 1 for i in range(n_sensors)])
        else:
            labels = np.array([0 if i < n_sensors // 2 else 1 for i in range(n_sensors)])

        # Compute weighted centroid for each cluster
        sources = []
        for cluster_id in range(2):
            mask = labels == cluster_id
            if mask.sum() == 0:
                # Empty cluster - use overall hottest sensor
                hot_idx = np.argmax(T_late)
                sources.append((sensors[hot_idx, 0], sensors[hot_idx, 1]))
                continue

            cluster_sensors = sensors[mask]
            cluster_temps = T_late[mask]

            # Subtract baseline
            T_baseline = np.min(cluster_temps)
            T_elevated = cluster_temps - T_baseline

            # Power weighting
            power = 2.0
            weights = np.power(T_elevated + 1e-8, power)
            weights = weights / weights.sum()

            x_est = np.average(cluster_sensors[:, 0], weights=weights)
            y_est = np.average(cluster_sensors[:, 1], weights=weights)

            x_est = np.clip(x_est, 0.1, self.Lx - 0.1)
            y_est = np.clip(y_est, 0.1, self.Ly - 0.1)

            sources.append((x_est, y_est))

        # Ensure sources are sufficiently separated
        if len(sources) == 2:
            dist = np.sqrt((sources[0][0] - sources[1][0])**2 +
                          (sources[0][1] - sources[1][1])**2)
            if dist < 0.3:  # Too close - spread them
                # Move apart along the line connecting them
                dx = sources[1][0] - sources[0][0]
                dy = sources[1][1] - sources[0][1]
                norm = np.sqrt(dx**2 + dy**2) + 1e-8
                dx, dy = dx / norm, dy / norm

                sources[0] = (
                    np.clip(sources[0][0] - 0.2 * dx, 0.1, self.Lx - 0.1),
                    np.clip(sources[0][1] - 0.2 * dy, 0.1, self.Ly - 0.1)
                )
                sources[1] = (
                    np.clip(sources[1][0] + 0.2 * dx, 0.1, self.Lx - 0.1),
                    np.clip(sources[1][1] + 0.2 * dy, 0.1, self.Ly - 0.1)
                )

        return sources

    def _gradient_refinement(
        self,
        x: float,
        y: float,
        sample: Dict
    ) -> Tuple[float, float]:
        """
        Refine position estimate using temperature gradient direction.

        Physics: Temperature gradient points AWAY from heat source (toward cooler regions).
        So we move the estimate in the OPPOSITE direction of the gradient.
        """
        sensors = np.array(sample['sensors_xy'])
        Y = sample['Y_noisy']

        # Use late-time temperatures
        n_late = max(1, Y.shape[0] // 5)
        T_late = Y[-n_late:].mean(axis=0)

        # Fit RBF interpolator to estimate temperature field
        try:
            rbf = RBFInterpolator(sensors, T_late, kernel='thin_plate_spline')
        except:
            return x, y  # Fallback

        # Estimate gradient at current position using finite differences
        eps = 0.02
        T_center = rbf([[x, y]])[0]

        # Handle boundary cases
        x_left = max(0.05, x - eps)
        x_right = min(self.Lx - 0.05, x + eps)
        y_bottom = max(0.05, y - eps)
        y_top = min(self.Ly - 0.05, y + eps)

        dT_dx = (rbf([[x_right, y]])[0] - rbf([[x_left, y]])[0]) / (x_right - x_left)
        dT_dy = (rbf([[x, y_top]])[0] - rbf([[x, y_bottom]])[0]) / (y_top - y_bottom)

        # Move in OPPOSITE direction of gradient (toward source)
        grad_norm = np.sqrt(dT_dx**2 + dT_dy**2) + 1e-8
        step_size = 0.1  # Step size

        x_new = x - step_size * dT_dx / grad_norm
        y_new = y - step_size * dT_dy / grad_norm

        # Clip to valid domain
        x_new = np.clip(x_new, 0.1, self.Lx - 0.1)
        y_new = np.clip(y_new, 0.1, self.Ly - 0.1)

        return x_new, y_new

    def _peak_finding_init(self, sample: Dict) -> Tuple[float, float]:
        """
        Find source position by locating the peak of interpolated temperature field.

        This is more sophisticated than weighted centroid because it:
        1. Interpolates the full temperature field
        2. Searches for the maximum (where source should be)
        """
        from scipy.optimize import minimize

        sensors = np.array(sample['sensors_xy'])
        Y = sample['Y_noisy']

        # Use late-time temperatures (quasi-steady)
        n_late = max(1, Y.shape[0] // 5)
        T_late = Y[-n_late:].mean(axis=0)

        # Fit RBF interpolator
        try:
            rbf = RBFInterpolator(sensors, T_late, kernel='thin_plate_spline')
        except:
            # Fallback to weighted centroid
            return self._weighted_centroid_1src(sample)

        # Find maximum by optimization
        # Start from hottest sensor
        hot_idx = np.argmax(T_late)
        x0, y0 = sensors[hot_idx]

        def neg_temp(xy):
            return -rbf([[xy[0], xy[1]]])[0]

        try:
            result = minimize(
                neg_temp,
                x0=[x0, y0],
                bounds=[(0.1, self.Lx - 0.1), (0.1, self.Ly - 0.1)],
                method='L-BFGS-B',
                options={'maxiter': 20}
            )
            x_peak, y_peak = result.x
        except:
            x_peak, y_peak = x0, y0

        return float(x_peak), float(y_peak)

    def _analytical_intensity(
        self,
        positions: List[Tuple[float, float]],
        sample: Dict,
        meta: Dict,
        solver: Heat2D,
        q_range: Tuple[float, float]
    ) -> List[float]:
        """
        Compute optimal intensities analytically using linearity of heat equation.

        For n sources: Solve n×n linear system for optimal q values.
        """
        Y_obs = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        n_sources = len(positions)

        # Compute unit responses
        Y_units = []
        for x, y in positions:
            sources = [{'x': x, 'y': y, 'q': 1.0}]
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            Y_units.append(Y_unit.flatten())

        # Build linear system: A @ q = b
        # A[i,j] = Y_i · Y_j
        # b[i] = Y_i · Y_obs
        A = np.zeros((n_sources, n_sources))
        b = np.zeros(n_sources)
        Y_obs_flat = Y_obs.flatten()

        for i in range(n_sources):
            b[i] = np.dot(Y_units[i], Y_obs_flat)
            for j in range(n_sources):
                A[i, j] = np.dot(Y_units[i], Y_units[j])

        # Solve with regularization for stability
        try:
            # Add small regularization
            A += 1e-6 * np.eye(n_sources)
            q_opt = np.linalg.solve(A, b)
        except:
            # Fallback to mean
            q_opt = np.array([np.mean(q_range)] * n_sources)

        # Clip to valid range
        q_opt = np.clip(q_opt, q_range[0], q_range[1])

        return list(q_opt)

    def _compute_rmse(
        self,
        params: np.ndarray,
        n_sources: int,
        Y_obs: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray
    ) -> float:
        """Compute RMSE for given parameters."""
        sources = []
        for i in range(n_sources):
            sources.append({
                'x': params[i * 3],
                'y': params[i * 3 + 1],
                'q': params[i * 3 + 2]
            })
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
        return np.sqrt(np.mean((Y_pred - Y_obs) ** 2))

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """Estimate sources using temperature-weighted centroid approach."""
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_obs = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        # Objective function for CMA-ES
        def objective(params):
            return self._compute_rmse(params, n_sources, Y_obs, solver, dt, nt, T0, sensors_xy)

        # Generate multiple initialization methods
        inits = []

        # 1. Weighted centroid initialization (NEW - physics-based)
        if n_sources == 1:
            x_wc, y_wc = self._weighted_centroid_1src(sample)
            if self.use_gradient_refinement:
                x_wc, y_wc = self._gradient_refinement(x_wc, y_wc, sample)
            positions = [(x_wc, y_wc)]
        else:
            positions = self._weighted_centroid_2src(sample)
            if self.use_gradient_refinement:
                positions = [self._gradient_refinement(x, y, sample) for x, y in positions]

        # Get analytical intensities
        q_values = self._analytical_intensity(positions, sample, meta, solver, q_range)

        wc_params = []
        for (x, y), q in zip(positions, q_values):
            wc_params.extend([x, y, q])
        wc_params = np.array(wc_params)
        inits.append((wc_params, 'weighted_centroid'))

        # 2. Triangulation initialization (existing method)
        if self.use_triangulation:
            try:
                tri_params = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
                inits.append((tri_params, 'triangulation'))
            except:
                pass

        # 3. Hottest sensor initialization
        readings = sample['Y_noisy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors_xy[idx] - sensors_xy[p]) >= 0.25 for p in selected):
                selected.append(idx)
        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        smart_params = []
        q_mean = np.mean(q_range)
        for idx in selected:
            smart_params.extend([sensors_xy[idx, 0], sensors_xy[idx, 1], q_mean])
        smart_params = np.array(smart_params)
        inits.append((smart_params, 'smart'))

        # Evaluate all inits
        lb, ub = self._get_bounds(n_sources, q_range)
        init_scores = []
        for params, init_type in inits:
            params_clipped = np.clip(params, lb, ub)
            rmse = objective(params_clipped)
            init_scores.append((params_clipped, rmse, init_type))

        # Sort by RMSE and pick best
        init_scores.sort(key=lambda x: x[1])
        best_init, best_init_rmse, best_init_type = init_scores[0]

        if verbose:
            print(f"  Best init: {best_init_type} (RMSE={best_init_rmse:.4f})")

        # Run CMA-ES from best init
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6

        es = cma.CMAEvolutionStrategy(best_init.tolist(), sigma0, opts)
        total_evals = len(inits)  # Include init evaluations

        # Collect all solutions for candidate generation
        all_solutions = []

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
            total_evals += len(solutions)

            for sol, fit in zip(solutions, fitness):
                sources = []
                for i in range(n_sources):
                    sources.append((float(sol[i*3]), float(sol[i*3+1]), float(sol[i*3+2])))
                all_solutions.append((sources, fit, np.array(sol)))

        # Best from CMA-ES
        best_params = np.array(es.result.xbest)
        best_rmse = es.result.fbest

        # Add best solution
        best_sources = []
        for i in range(n_sources):
            best_sources.append((float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])))
        all_solutions.append((best_sources, best_rmse, best_params))

        # Filter for dissimilar candidates
        filtered = filter_dissimilar([(s[0], s[1]) for s in all_solutions], tau=TAU, n_max=self.n_candidates)

        # Build final results
        final_candidates = []
        for sources, rmse in filtered:
            for s in all_solutions:
                if s[0] == sources and abs(s[1] - rmse) < 1e-10:
                    final_candidates.append((sources, rmse, s[2]))
                    break

        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[1] for c in final_candidates]
        best_rmse_final = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[2],
                rmse=c[1],
                init_type=best_init_type,
                n_evals=total_evals // len(final_candidates) if final_candidates else total_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse_final, results, total_evals
