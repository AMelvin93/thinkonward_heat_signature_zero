"""
Sparse L1 Optimization for Heat Source Identification.

Based on research:
- "Heat source identification based on l1 constrained minimization" (UCLA)
- "Decomposed physics-based compressive sensing for inverse heat source detection" (2024)

Key idea: Discretize the domain into a grid and formulate as sparse optimization.
The L1 norm promotes sparsity (1-2 active sources).

Algorithm:
1. Create coarse grid of potential source locations (e.g., 20x10)
2. For each grid point, compute its contribution to sensors (forward model matrix A)
3. Solve: minimize ||Ax - b||^2 + lambda * ||x||_1
   where x = source intensities at grid points, b = sensor readings
4. Extract active (non-zero) sources from solution
5. Refine positions using local CMA-ES
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.sparse import csr_matrix
import cma

# Add simulator path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))

from simulator import Heat2D

# Competition constants
N_MAX = 3


@dataclass
class CandidateResult:
    """Result for a single candidate solution."""
    sources: np.ndarray  # Shape: (n_sources, 3) - (x, y, q) for each source
    rmse: float
    init_type: str


class SparseL1Optimizer:
    """
    Optimizer using sparse L1 optimization for heat source identification.

    Uses compressed sensing principles to find sparse heat sources on a discretized grid.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 10),
        lambda_l1: float = 0.1,
        max_fevals_refine_1src: int = 10,
        max_fevals_refine_2src: int = 18,
        n_candidates: int = 3,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
    ):
        """
        Args:
            grid_size: (nx_grid, ny_grid) for sparse grid
            lambda_l1: L1 regularization strength
            max_fevals_refine_1src: Max fevals for 1-source CMA-ES refinement
            max_fevals_refine_2src: Max fevals for 2-source CMA-ES refinement
            n_candidates: Number of diverse candidates to return
            Lx, Ly, nx, ny: Domain and grid parameters
        """
        self.grid_size = grid_size
        self.lambda_l1 = lambda_l1
        self.max_fevals_refine_1src = max_fevals_refine_1src
        self.max_fevals_refine_2src = max_fevals_refine_2src
        self.n_candidates = n_candidates
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

    def _create_forward_matrix(
        self,
        sample: dict,
        meta: dict,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """
        Create forward model matrix A where A @ intensities = sensor_readings.

        A[i, j] = contribution of grid point j to sensor i.
        """
        sensor_locs = sample['sensors_xy']
        n_sensors = len(sensor_locs)
        n_grid = len(grid_x)

        # Get thermal parameters from sample metadata
        kappa = sample.get('sample_metadata', {}).get('kappa', 1.0)
        nt = sample.get('sample_metadata', {}).get('nt', 100)
        dt = meta.get('dt', 0.004)

        # Create simulator
        simulator = Heat2D(
            Lx=self.Lx,
            Ly=self.Ly,
            nx=self.nx,
            ny=self.ny,
            kappa=kappa,
        )

        # Forward matrix: A[sensor_idx, grid_idx]
        A = np.zeros((n_sensors, n_grid))

        for j in range(n_grid):
            # Simulate with unit intensity at grid point j
            sources = np.array([[grid_x[j], grid_y[j], 1.0]])

            u_full = simulator.solve(dt, nt, sources=sources)  # Shape: (nt, ny, nx)
            u_sensors = simulator.get_sensor_readings(u_full, sensor_locs)  # Shape: (n_sensors, nt)

            # Use final timestep as feature (steady-state)
            A[:, j] = u_sensors[:, -1]

        return A

    def _sparse_solve_l1(
        self,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve: minimize ||Ax - b||^2 + lambda * ||x||_1

        This is LASSO regression, solved using scipy.optimize.
        """
        n_grid = A.shape[1]

        def objective(x):
            """Objective: ||Ax - b||^2 + lambda * ||x||_1"""
            residual = A @ x - b
            data_fit = np.sum(residual ** 2)
            l1_penalty = self.lambda_l1 * np.sum(np.abs(x))
            return data_fit + l1_penalty

        def gradient(x):
            """Gradient of objective"""
            residual = A @ x - b
            grad_data = 2 * A.T @ residual
            grad_l1 = self.lambda_l1 * np.sign(x)
            return grad_data + grad_l1

        # Initialize with non-negative constraint (intensities > 0)
        x0 = np.zeros(n_grid)
        bounds = [(0, None)] * n_grid

        # Solve with L-BFGS-B
        result = fmin_l_bfgs_b(
            objective,
            x0,
            fprime=gradient,
            bounds=bounds,
            maxiter=100,
            disp=0,
        )

        x_opt = result[0]
        return x_opt

    def _extract_sources(
        self,
        x_sparse: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        q_range: Tuple[float, float],
        threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Extract sources from sparse solution.

        Returns:
            sources: (n_sources, 3) array of (x, y, q)
        """
        # Find non-zero (active) grid points
        active_mask = x_sparse > threshold
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            # No sources found, use highest intensity point
            active_indices = [np.argmax(x_sparse)]

        # Extract positions and intensities
        sources = []
        for idx in active_indices:
            x = grid_x[idx]
            y = grid_y[idx]
            q = x_sparse[idx]

            # Clip intensity to valid range
            q = np.clip(q, q_range[0], q_range[1])

            sources.append([x, y, q])

        return np.array(sources)

    def _refine_with_cmaes(
        self,
        sources_init: np.ndarray,
        sample: dict,
        meta: dict,
        q_range: Tuple[float, float],
        max_fevals: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Refine source parameters using CMA-ES.
        """
        n_sources = len(sources_init)
        sensor_locs = sample['sensors_xy']
        sensor_readings = sample['Y_noisy']

        # Get thermal parameters from sample metadata
        kappa = sample.get('sample_metadata', {}).get('kappa', 1.0)
        nt = sample.get('sample_metadata', {}).get('nt', 100)
        dt = meta.get('dt', 0.004)

        # Create simulator
        simulator = Heat2D(
            Lx=self.Lx,
            Ly=self.Ly,
            nx=self.nx,
            ny=self.ny,
            kappa=kappa,
        )

        def objective(params):
            """Objective: RMSE between predicted and actual sensor readings"""
            sources = params.reshape(n_sources, 3)

            # Clip to valid bounds
            sources[:, 0] = np.clip(sources[:, 0], 0, self.Lx)
            sources[:, 1] = np.clip(sources[:, 1], 0, self.Ly)
            sources[:, 2] = np.clip(sources[:, 2], q_range[0], q_range[1])

            try:
                u_full = simulator.solve(dt, nt, sources=sources)
                u_sensors = simulator.get_sensor_readings(u_full, sensor_locs)
                rmse = np.sqrt(np.mean((u_sensors - sensor_readings) ** 2))
                return rmse
            except:
                return 1e6

        # Flatten sources for CMA-ES
        x0 = sources_init.flatten()

        # Bounds
        bounds_lower = np.array([[0, 0, q_range[0]]] * n_sources).flatten()
        bounds_upper = np.array([[self.Lx, self.Ly, q_range[1]]] * n_sources).flatten()

        # CMA-ES options
        sigma0 = 0.2
        popsize = 4 + int(3 * np.log(len(x0)))

        opts = {
            'bounds': [bounds_lower, bounds_upper],
            'popsize': popsize,
            'maxfevals': max_fevals,
            'verb_disp': 0,
            'verbose': -9,
        }

        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            es.optimize(objective)
            x_best = es.result.xbest
            rmse_best = es.result.fbest
        except:
            x_best = x0
            rmse_best = objective(x0)

        sources_best = x_best.reshape(n_sources, 3)
        return sources_best, rmse_best

    def estimate_sources(
        self,
        sample: dict,
        meta: dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], float, List[CandidateResult], int]:
        """
        Estimate heat sources using sparse L1 optimization.

        Returns:
            candidates: List of candidate source arrays
            best_rmse: RMSE of best candidate
            results: List of CandidateResult objects
            n_evals: Number of simulator evaluations
        """
        sensor_locs = sample['sensors_xy']
        sensor_readings = sample['Y_noisy']
        n_sensors = len(sensor_locs)

        # Create sparse grid
        nx_grid, ny_grid = self.grid_size
        x_grid = np.linspace(0.1, self.Lx - 0.1, nx_grid)
        y_grid = np.linspace(0.1, self.Ly - 0.1, ny_grid)

        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        n_grid = len(grid_x)

        # Build forward matrix A
        if verbose:
            print(f"Building forward matrix ({n_sensors} sensors x {n_grid} grid points)...")

        A = self._create_forward_matrix(sample, meta, grid_x, grid_y)
        n_evals = n_grid  # One simulation per grid point

        # Use final timestep of sensor readings
        b = sensor_readings[:, -1]

        # Solve sparse L1 optimization
        if verbose:
            print("Solving sparse L1 optimization...")

        x_sparse = self._sparse_solve_l1(A, b)

        # Extract sources
        sources_sparse = self._extract_sources(x_sparse, grid_x, grid_y, q_range)
        n_sources = len(sources_sparse)

        if verbose:
            print(f"Found {n_sources} sources from sparse solution")

        # Determine max fevals for refinement
        max_fevals = (
            self.max_fevals_refine_1src if n_sources == 1
            else self.max_fevals_refine_2src
        )

        # Refine with CMA-ES
        if verbose:
            print(f"Refining with CMA-ES (max_fevals={max_fevals})...")

        sources_refined, rmse_refined = self._refine_with_cmaes(
            sources_sparse, sample, meta, q_range, max_fevals
        )
        n_evals += max_fevals

        # For now, return single best candidate
        # TODO: Generate diverse candidates
        results = [
            CandidateResult(
                sources=sources_refined,
                rmse=rmse_refined,
                init_type='sparse_l1',
            )
        ]

        candidates = [sources_refined]
        best_rmse = rmse_refined

        return candidates, best_rmse, results, n_evals
