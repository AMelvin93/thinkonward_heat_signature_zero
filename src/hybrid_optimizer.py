"""
Hybrid Multi-Start L-BFGS-B Optimizer for Heat Source Identification.

This optimizer combines:
1. Smart initialization from sensor temperature patterns
2. Multi-start L-BFGS-B for accuracy
3. Diversity-enforced candidate generation (for competition scoring)

Goal: Achieve low RMSE (like baseline_lbfgs) while generating 3+ distinct
candidates to maximize the competition score.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

from src.triangulation import triangulation_init

# Default to n-1 workers to leave one core free for system
DEFAULT_N_JOBS = max(1, os.cpu_count() - 1)

# Add the starter notebook path to import the simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


@dataclass
class CandidateResult:
    """Result from a single optimization run."""
    params: np.ndarray
    rmse: float
    init_type: str  # 'smart' or 'random'


def _run_lbfgsb_optimization(
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    objective_args: Tuple,
    max_iter: int,
) -> Tuple[np.ndarray, float]:
    """
    Run L-BFGS-B optimization from given initial point.
    Module-level function for joblib pickling.
    """
    Lx, Ly, nx, ny, kappa, bc, n_sources, dt, nt, T0, sensors_xy, Y_observed = objective_args

    # Create solver (fresh for each worker)
    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    def objective(params):
        sources = []
        for i in range(n_sources):
            sources.append({
                'x': params[i * 3],
                'y': params[i * 3 + 1],
                'q': params[i * 3 + 2]
            })

        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
        return np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    try:
        result = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-7},
        )
        return result.x, result.fun
    except Exception:
        return x0, float('inf')


class HybridOptimizer:
    """
    Hybrid Multi-Start L-BFGS-B Optimizer.

    Strategy:
    1. Generate smart initial points from sensor temperature analysis
    2. Generate diverse random initial points (well-separated)
    3. Run L-BFGS-B from all initial points in parallel
    4. Filter to keep top N distinct candidates
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_smart_inits: int = 4,
        n_random_inits: int = 8,
        min_candidate_distance: float = 0.15,
        n_max_candidates: int = 3,
        use_triangulation: bool = True,
    ):
        """
        Initialize the hybrid optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            n_smart_inits: Number of smart initializations from sensor data
            n_random_inits: Number of random diverse initializations
            min_candidate_distance: Minimum normalized distance between candidates
            n_max_candidates: Maximum candidates to return (for competition N_max)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_smart_inits = n_smart_inits
        self.n_random_inits = n_random_inits
        self.min_candidate_distance = min_candidate_distance
        self.n_max_candidates = n_max_candidates
        self.use_triangulation = use_triangulation

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))  # x
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))  # y
            bounds.append(q_range)  # q
        return bounds

    # =========================================================================
    # Smart Initialization from Sensor Observations
    # =========================================================================

    def _analyze_sensor_temperatures(
        self,
        sample: Dict,
        n_sources: int,
        min_separation: float = 0.3,
    ) -> List[Tuple[float, float, float]]:
        """
        Analyze sensor temperatures to infer likely source locations.

        For single source: Start near the hottest sensor.
        For multiple sources: Find well-separated hot regions.
        """
        readings = sample['Y_noisy']  # Shape: (timesteps, n_sensors)
        sensors = sample['sensors_xy']

        # Average temperature per sensor
        avg_temps = np.mean(readings, axis=0)

        # Sort sensors by temperature (hottest first)
        hot_indices = np.argsort(avg_temps)[::-1]

        # Select well-separated hot sensors
        selected = []
        for idx in hot_indices:
            if len(selected) >= n_sources:
                break

            # Check separation from already selected
            is_separated = True
            for prev_idx in selected:
                dist = np.linalg.norm(sensors[idx] - sensors[prev_idx])
                if dist < min_separation:
                    is_separated = False
                    break

            if is_separated:
                selected.append(idx)

        # Fill with remaining hot ones if needed
        for idx in hot_indices:
            if len(selected) >= n_sources:
                break
            if idx not in selected:
                selected.append(idx)

        # Build source estimates
        sources = []
        max_temp = np.max(avg_temps) + 1e-8

        for s_idx in selected:
            loc = sensors[s_idx]
            temp_ratio = avg_temps[s_idx] / max_temp

            # Estimate intensity from relative temperature
            q = 0.5 + temp_ratio * 1.2  # Map to roughly [0.5, 1.7]
            q = np.clip(q, 0.5, 2.0)

            sources.append((loc[0], loc[1], q))

        return sources

    def _triangulation_init(
        self,
        sample: Dict,
        meta: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
        perturbation: float = 0.0,
    ) -> np.ndarray:
        """
        Generate initial point using triangulation from sensor onset times.

        Uses heat diffusion physics to estimate source positions more accurately
        than simply using hottest sensor locations.
        """
        try:
            params = triangulation_init(
                sample, meta, n_sources, q_range, self.Lx, self.Ly
            )

            if perturbation > 0:
                bounds = self._get_bounds(n_sources, q_range)
                for i in range(n_sources):
                    params[i*3] += np.random.uniform(-perturbation, perturbation) * self.Lx
                    params[i*3+1] += np.random.uniform(-perturbation, perturbation) * self.Ly
                    params[i*3+2] += np.random.uniform(-0.2, 0.2)

                    # Clip to bounds
                    params[i*3] = np.clip(params[i*3], bounds[i*3][0], bounds[i*3][1])
                    params[i*3+1] = np.clip(params[i*3+1], bounds[i*3+1][0], bounds[i*3+1][1])
                    params[i*3+2] = np.clip(params[i*3+2], q_range[0], q_range[1])

            return params

        except Exception:
            # Fall back to smart init if triangulation fails
            return self._smart_init(sample, n_sources, q_range, perturbation)

    def _smart_init(
        self,
        sample: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
        perturbation: float = 0.1,
    ) -> np.ndarray:
        """
        Generate smart initial point from sensor analysis with perturbation.
        """
        base_sources = self._analyze_sensor_temperatures(sample, n_sources)
        bounds = self._get_bounds(n_sources, q_range)

        params = []
        for i, (x, y, q) in enumerate(base_sources):
            # Add random perturbation
            x_new = x + np.random.uniform(-perturbation, perturbation) * self.Lx
            y_new = y + np.random.uniform(-perturbation, perturbation) * self.Ly
            q_new = q + np.random.uniform(-0.2, 0.2)

            # Clip to bounds
            x_new = np.clip(x_new, bounds[i*3][0], bounds[i*3][1])
            y_new = np.clip(y_new, bounds[i*3+1][0], bounds[i*3+1][1])
            q_new = np.clip(q_new, q_range[0], q_range[1])

            params.extend([x_new, y_new, q_new])

        return np.array(params)

    def _random_init(
        self,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05,
    ) -> np.ndarray:
        """Generate random initial parameters."""
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

    def _diverse_random_inits(
        self,
        n_inits: int,
        n_sources: int,
        q_range: Tuple[float, float],
        min_distance: float = 0.2,
    ) -> List[np.ndarray]:
        """
        Generate diverse random initial points that are well-separated.
        """
        inits = []
        max_attempts = n_inits * 10

        for _ in range(max_attempts):
            if len(inits) >= n_inits:
                break

            candidate = self._random_init(n_sources, q_range)

            # Check if sufficiently different from existing
            is_diverse = True
            for existing in inits:
                if self._normalized_distance(candidate, existing, n_sources, q_range) < min_distance:
                    is_diverse = False
                    break

            if is_diverse:
                inits.append(candidate)

        # Fill with random if needed
        while len(inits) < n_inits:
            inits.append(self._random_init(n_sources, q_range))

        return inits

    # =========================================================================
    # Candidate Filtering
    # =========================================================================

    def _normalized_distance(
        self,
        params1: np.ndarray,
        params2: np.ndarray,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> float:
        """
        Compute normalized distance between two parameter sets.
        """
        total_dist = 0
        q_span = q_range[1] - q_range[0]

        for i in range(n_sources):
            dx = (params1[i*3] - params2[i*3]) / self.Lx
            dy = (params1[i*3+1] - params2[i*3+1]) / self.Ly
            dq = (params1[i*3+2] - params2[i*3+2]) / q_span
            total_dist += np.sqrt(dx**2 + dy**2 + dq**2)

        return total_dist / n_sources

    def _filter_distinct_candidates(
        self,
        results: List[CandidateResult],
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> List[CandidateResult]:
        """
        Filter results to keep only distinct candidates.
        """
        # Sort by RMSE (best first)
        sorted_results = sorted(results, key=lambda r: r.rmse)

        distinct = []
        for result in sorted_results:
            if len(distinct) >= self.n_max_candidates:
                break

            is_distinct = True
            for existing in distinct:
                dist = self._normalized_distance(
                    result.params, existing.params, n_sources, q_range
                )
                if dist < self.min_candidate_distance:
                    is_distinct = False
                    break

            if is_distinct and result.rmse < float('inf'):
                distinct.append(result)

        return distinct

    # =========================================================================
    # Main Optimization
    # =========================================================================

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 100,
        parallel: bool = True,
        n_jobs: int = None,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate source parameters with multiple distinct candidates.

        Args:
            sample: Sample dict
            meta: Meta dict
            q_range: Valid range for source intensity
            max_iter: Maximum L-BFGS-B iterations
            parallel: Use parallel processing
            n_jobs: Number of parallel jobs (None = cpu_count - 1)
            verbose: Print progress

        Returns:
            (best_sources, best_rmse, all_candidates)
        """
        if n_jobs is None:
            n_jobs = DEFAULT_N_JOBS

        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        bounds = self._get_bounds(n_sources, q_range)

        # Generate initial points
        initial_points = []

        # Smart initializations (use triangulation for 1-source only)
        for i in range(self.n_smart_inits):
            perturbation = 0.05 + i * 0.03  # Increasing perturbation
            # Triangulation works well for 1-source but worse for 2-source
            if self.use_triangulation and n_sources == 1:
                x0 = self._triangulation_init(sample, meta, n_sources, q_range, perturbation)
                initial_points.append(('triang', x0))
            else:
                x0 = self._smart_init(sample, n_sources, q_range, perturbation)
                initial_points.append(('smart', x0))

        # Diverse random initializations
        random_inits = self._diverse_random_inits(
            self.n_random_inits, n_sources, q_range, min_distance=0.25
        )
        for x0 in random_inits:
            initial_points.append(('random', x0))

        if verbose:
            print(f"  Running {len(initial_points)} L-BFGS-B optimizations...")

        # Pack arguments for parallel function
        objective_args = (
            self.Lx, self.Ly, self.nx, self.ny,
            kappa, bc, n_sources, dt, nt, T0,
            sensors_xy, Y_observed
        )

        if parallel:
            # Run all optimizations in parallel
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_lbfgsb_optimization)(x0, bounds, objective_args, max_iter)
                for init_type, x0 in initial_points
            )

            # Convert to CandidateResult
            candidates = [
                CandidateResult(params=params, rmse=rmse, init_type=init_type)
                for (init_type, _), (params, rmse) in zip(initial_points, results)
            ]
        else:
            # Sequential execution
            solver = self._create_solver(kappa, bc)
            candidates = []

            for init_type, x0 in initial_points:
                def objective(params):
                    sources = [{'x': params[i*3], 'y': params[i*3+1], 'q': params[i*3+2]}
                               for i in range(n_sources)]
                    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
                    Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
                    return np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

                try:
                    result = minimize(
                        objective, x0=x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-7},
                    )
                    candidates.append(CandidateResult(result.x, result.fun, init_type))
                except Exception:
                    pass

        # Filter to distinct candidates
        distinct_candidates = self._filter_distinct_candidates(candidates, n_sources, q_range)

        if verbose:
            print(f"  Found {len(distinct_candidates)} distinct candidates")
            for i, c in enumerate(distinct_candidates):
                print(f"    #{i+1}: RMSE={c.rmse:.6f} ({c.init_type})")

        # Best result
        if distinct_candidates:
            best = distinct_candidates[0]
            best_sources = []
            for i in range(n_sources):
                x, y, q = best.params[i*3:(i+1)*3]
                best_sources.append((x, y, q))
            return best_sources, best.rmse, distinct_candidates
        else:
            # Fallback
            return [], float('inf'), []

    def estimate_sources_simple(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 100,
        parallel: bool = True,
        n_jobs: int = None,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Simple interface returning just best sources and RMSE.

        Args:
            n_jobs: Number of parallel jobs (None = cpu_count - 1)
        """
        sources, rmse, _ = self.estimate_sources(
            sample, meta, q_range, max_iter, parallel, n_jobs
        )
        return sources, rmse
