"""
Heat Source Optimizer - Option 1: Gradient-Based Optimization

This module implements a simple gradient-free optimization approach to estimate
heat source parameters (x, y, q) from sensor temperature readings.

Supports parallel execution via joblib for multi-core speedup.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
from joblib import Parallel, delayed
import sys
import os

# Add the starter notebook path to import the simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


def _run_single_optimization(
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    objective_args: Tuple,
    method: str,
    max_iter: int,
) -> Tuple[np.ndarray, float]:
    """
    Run a single optimization from given initial point.
    This is a module-level function for pickling with joblib.
    """
    Lx, Ly, nx, ny, kappa, bc, n_sources, dt, nt, T0, sensors_xy, Y_observed = objective_args

    # Create solver (must create fresh for each worker)
    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    def objective(params):
        # Unpack and simulate
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
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter},
        )
        return result.x, result.fun
    except Exception as e:
        return x0, float('inf')


class HeatSourceOptimizer:
    """
    Optimizer for estimating heat source parameters from sensor readings.

    Uses scipy.optimize to minimize the difference between simulated and
    observed sensor temperatures.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
    ):
        """
        Initialize the optimizer with domain parameters.

        Args:
            Lx: Domain length in x direction
            Ly: Domain length in y direction
            nx: Number of grid points in x
            ny: Number of grid points in y
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        # Track optimization history for visualization
        self.history: List[Dict] = []

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _simulate(
        self,
        params: np.ndarray,
        n_sources: int,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Run simulation with given source parameters.

        Args:
            params: Flattened array of [x1, y1, q1, x2, y2, q2, ...]
            n_sources: Number of sources
            solver: Heat2D solver instance
            dt: Time step
            nt: Number of time steps
            T0: Initial temperature
            sensors_xy: Sensor positions (n_sensors, 2)

        Returns:
            Simulated sensor readings (nt+1, n_sensors)
        """
        # Unpack parameters
        sources = []
        for i in range(n_sources):
            x = params[i * 3]
            y = params[i * 3 + 1]
            q = params[i * 3 + 2]
            sources.append({'x': x, 'y': y, 'q': q})

        # Run simulation
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)

        # Sample at sensor locations
        Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
        return Y_pred

    def _objective(
        self,
        params: np.ndarray,
        n_sources: int,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        Y_observed: np.ndarray,
        track_history: bool = False,
    ) -> float:
        """
        Compute RMSE between simulated and observed sensor readings.
        """
        Y_pred = self._simulate(params, n_sources, solver, dt, nt, T0, sensors_xy)

        # Compute RMSE
        rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

        # Track history if requested
        if track_history:
            self.history.append({
                'params': params.copy(),
                'rmse': rmse,
            })

        return rmse

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))  # x
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))  # y
            bounds.append(q_range)  # q
        return bounds

    def _random_init(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05) -> np.ndarray:
        """Generate random initial parameters."""
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        method: str = 'L-BFGS-B',
        n_restarts: int = 5,
        max_iter: int = 100,
        track_history: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Estimate source parameters for a single sample.

        Args:
            sample: Sample dict with keys: n_sources, sensors_xy, Y_noisy, sample_metadata
            meta: Meta dict with dt
            q_range: Valid range for source intensity
            method: Optimization method ('L-BFGS-B', 'Nelder-Mead', 'differential_evolution')
            n_restarts: Number of random restarts
            max_iter: Maximum iterations per restart
            track_history: Whether to track optimization history
            parallel: Whether to run restarts in parallel (uses all CPU cores)
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            Tuple of (estimated_sources, best_rmse)
            estimated_sources is a list of (x, y, q) tuples
        """
        # Clear history
        if track_history:
            self.history = []

        # Extract sample info
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Get bounds
        bounds = self._get_bounds(n_sources, q_range)

        best_params = None
        best_rmse = float('inf')

        if method == 'differential_evolution':
            # Create solver for DE (not parallelizable this way)
            solver = self._create_solver(kappa, bc)
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                args=(n_sources, solver, dt, nt, T0, sensors_xy, Y_observed, track_history),
                maxiter=max_iter,
                seed=42,
                polish=True,
            )
            best_params = result.x
            best_rmse = result.fun

        elif parallel and n_restarts > 1:
            # Parallel multi-start optimization
            # Generate all initial points
            x0_list = [self._random_init(n_sources, q_range) for _ in range(n_restarts)]

            # Pack arguments for parallel function
            objective_args = (
                self.Lx, self.Ly, self.nx, self.ny,
                kappa, bc, n_sources, dt, nt, T0,
                sensors_xy, Y_observed
            )

            # Run in parallel
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_single_optimization)(
                    x0, bounds, objective_args, method, max_iter
                )
                for x0 in x0_list
            )

            # Find best result
            for params, rmse in results:
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params

        else:
            # Sequential multi-start local optimization
            solver = self._create_solver(kappa, bc)

            for restart in range(n_restarts):
                x0 = self._random_init(n_sources, q_range)

                try:
                    result = minimize(
                        self._objective,
                        x0=x0,
                        args=(n_sources, solver, dt, nt, T0, sensors_xy, Y_observed, track_history),
                        method=method,
                        bounds=bounds,
                        options={'maxiter': max_iter},
                    )

                    if result.fun < best_rmse:
                        best_rmse = result.fun
                        best_params = result.x
                except Exception as e:
                    print(f"Restart {restart} failed: {e}")
                    continue

        # Convert to list of tuples
        estimated_sources = []
        for i in range(n_sources):
            x = best_params[i * 3]
            y = best_params[i * 3 + 1]
            q = best_params[i * 3 + 2]
            estimated_sources.append((x, y, q))

        return estimated_sources, best_rmse

    def estimate_multiple_candidates(
        self,
        sample: Dict,
        meta: Dict,
        n_candidates: int = 3,
        q_range: Tuple[float, float] = (0.5, 2.0),
        method: str = 'L-BFGS-B',
        n_restarts: int = 10,
        max_iter: int = 100,
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Generate multiple candidate solutions for diversity bonus.

        Runs optimization multiple times with different initializations
        and keeps the best distinct solutions.
        """
        all_results = []

        for _ in range(n_restarts * n_candidates):
            sources, rmse = self.estimate_sources(
                sample, meta, q_range, method, n_restarts=1, max_iter=max_iter
            )
            all_results.append((sources, rmse))

        # Sort by RMSE and take top candidates
        all_results.sort(key=lambda x: x[1])

        # Filter for distinct candidates (simple distance check)
        candidates = []
        for sources, rmse in all_results:
            is_distinct = True
            for existing in candidates:
                # Check if too similar
                dist = self._candidate_distance(sources, existing)
                if dist < 0.1:  # Threshold for distinctness
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(sources)
            if len(candidates) >= n_candidates:
                break

        return candidates

    def _candidate_distance(
        self,
        c1: List[Tuple[float, float, float]],
        c2: List[Tuple[float, float, float]],
    ) -> float:
        """Compute normalized distance between two candidates."""
        # Sort both by x coordinate for consistent comparison
        c1_sorted = sorted(c1)
        c2_sorted = sorted(c2)

        total_dist = 0
        for s1, s2 in zip(c1_sorted, c2_sorted):
            # Normalize by domain size and q_range
            dx = (s1[0] - s2[0]) / self.Lx
            dy = (s1[1] - s2[1]) / self.Ly
            dq = (s1[2] - s2[2]) / 2.0  # Assuming q_range max is 2
            total_dist += np.sqrt(dx**2 + dy**2 + dq**2)

        return total_dist / len(c1)


def estimate_all_samples(
    test_dataset: Dict,
    n_candidates: int = 3,
    method: str = 'L-BFGS-B',
    n_restarts: int = 5,
    max_iter: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """
    Estimate sources for all samples in the test dataset.

    Returns submission-ready format.
    """
    optimizer = HeatSourceOptimizer()
    meta = test_dataset['meta']
    q_range = meta['q_range']

    submission = []

    for i, sample in enumerate(test_dataset['samples']):
        if verbose:
            print(f"Processing sample {i+1}/{len(test_dataset['samples'])}: {sample['sample_id']}")

        candidates = optimizer.estimate_multiple_candidates(
            sample, meta, n_candidates, q_range, method, n_restarts, max_iter
        )

        submission.append({
            'sample_id': sample['sample_id'],
            'estimated_sources': candidates,
        })

    return submission
