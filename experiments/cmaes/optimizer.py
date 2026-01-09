"""
CMA-ES Optimizer for Heat Source Identification.

Uses Covariance Matrix Adaptation Evolution Strategy for optimization,
which is particularly effective for multi-modal problems like 2-source
heat identification where L-BFGS-B can get stuck in local minima.

Key advantages over L-BFGS-B:
- Gradient-free (no finite difference overhead)
- Learns covariance structure of the landscape
- Better at escaping local minima
- Handles permutation symmetry in 2-source problems

References:
- https://cma-es.github.io/
- Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cma

# Add project root to path for imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add the starter notebook path to import the simulator
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


@dataclass
class CandidateResult:
    """Result from optimization."""
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


class CMAESOptimizer:
    """
    CMA-ES optimizer with triangulation initialization.

    Strategy:
    1. Use triangulation to get a good starting point
    2. Run CMA-ES for global optimization
    3. Optionally polish with L-BFGS-B for final refinement

    The optimizer is tuned for the competition time budget:
    - 1-source: fewer evaluations (simpler landscape)
    - 2-source: more evaluations (multi-modal landscape)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 25,
        sigma0_1src: float = 0.10,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        use_lbfgsb_polish: bool = True,
        polish_max_iter: int = 5,
        polish_max_iter_1src: Optional[int] = None,
        polish_max_iter_2src: Optional[int] = None,
    ):
        """
        Initialize the CMA-ES optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_fevals_1src: Max function evaluations for 1-source problems
            max_fevals_2src: Max function evaluations for 2-source problems
            sigma0_1src: Initial step size for 1-source (smaller = trust init more)
            sigma0_2src: Initial step size for 2-source (larger = explore more)
            use_triangulation: Use triangulation for initialization
            use_lbfgsb_polish: Add L-BFGS-B polishing step after CMA-ES
            polish_max_iter: Max iterations for L-BFGS-B polish (default for both)
            polish_max_iter_1src: Override polish iterations for 1-source (None = use polish_max_iter)
            polish_max_iter_2src: Override polish iterations for 2-source (None = use polish_max_iter)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.use_lbfgsb_polish = use_lbfgsb_polish
        self.polish_max_iter = polish_max_iter
        # Per-source-type polish settings (None means use default)
        self.polish_max_iter_1src = polish_max_iter_1src
        self.polish_max_iter_2src = polish_max_iter_2src

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(
        self,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        """Get parameter bounds as (lower, upper) lists for CMA-ES."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return lb, ub

    def _get_scipy_bounds(
        self,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05
    ) -> List[Tuple[float, float]]:
        """Get bounds in scipy format for L-BFGS-B."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
            bounds.append(q_range)
        return bounds

    def _smart_init(
        self,
        sample: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> np.ndarray:
        """Fallback initialization from hottest sensors."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        # Select well-separated hot sensors
        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[p]) >= 0.25 for p in selected):
                selected.append(idx)

        # Fill if needed
        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        params = []
        max_temp = np.max(avg_temps) + 1e-8
        for idx in selected:
            x, y = sensors[idx]
            q = 0.5 + (avg_temps[idx] / max_temp) * 1.2
            q = np.clip(q, q_range[0], q_range[1])
            params.extend([x, y, q])

        return np.array(params)

    def _get_initial_point(
        self,
        sample: Dict,
        meta: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> np.ndarray:
        """Get initial point using triangulation or fallback."""
        if self.use_triangulation:
            try:
                return triangulation_init(
                    sample, meta, n_sources, q_range, self.Lx, self.Ly
                )
            except Exception:
                pass
        return self._smart_init(sample, n_sources, q_range)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate source parameters using CMA-ES.

        Args:
            sample: Sample dict with n_sources, sensors_xy, Y_noisy, sample_metadata
            meta: Meta dict with dt
            q_range: Valid range for source intensity
            verbose: Print progress

        Returns:
            (best_sources, best_rmse, candidates)
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create solver
        solver = self._create_solver(kappa, bc)

        # Objective function
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

        # Get initialization
        x0 = self._get_initial_point(sample, meta, n_sources, q_range)

        # Set CMA-ES parameters based on problem size
        if n_sources == 1:
            max_fevals = self.max_fevals_1src
            sigma0 = self.sigma0_1src
        else:
            max_fevals = self.max_fevals_2src
            sigma0 = self.sigma0_2src

        # Bounds
        lb, ub = self._get_bounds(n_sources, q_range)

        # CMA-ES options
        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9  # Silent
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        if verbose:
            print(f"  CMA-ES: n_sources={n_sources}, max_fevals={max_fevals}, sigma0={sigma0:.2f}")

        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        n_evals = 0
        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
            n_evals += len(solutions)

        best_params = es.result.xbest
        best_rmse = es.result.fbest

        if verbose:
            print(f"  CMA-ES complete: RMSE={best_rmse:.4f}, evals={n_evals}")

        # Optional L-BFGS-B polish (with per-source-type settings)
        # Determine polish iterations for this problem type
        if n_sources == 1:
            polish_iter = self.polish_max_iter_1src if self.polish_max_iter_1src is not None else self.polish_max_iter
        else:
            polish_iter = self.polish_max_iter_2src if self.polish_max_iter_2src is not None else self.polish_max_iter

        if self.use_lbfgsb_polish and polish_iter > 0:
            from scipy.optimize import minimize

            scipy_bounds = self._get_scipy_bounds(n_sources, q_range)

            try:
                result = minimize(
                    objective,
                    x0=best_params,
                    method='L-BFGS-B',
                    bounds=scipy_bounds,
                    options={'maxiter': polish_iter}
                )
                if result.fun < best_rmse:
                    best_params = result.x
                    best_rmse = result.fun
                    n_evals += result.nfev

                    if verbose:
                        print(f"  L-BFGS-B polish (iter={polish_iter}): RMSE={best_rmse:.4f}")
            except Exception:
                pass
        elif verbose and self.use_lbfgsb_polish:
            print(f"  L-BFGS-B polish: SKIPPED (iter=0 for {n_sources}-source)")

        # Build result
        sources = []
        for i in range(n_sources):
            x, y, q = best_params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        candidate = CandidateResult(
            params=best_params,
            rmse=best_rmse,
            init_type='triangulation' if self.use_triangulation else 'smart',
            n_evals=n_evals
        )

        return sources, best_rmse, [candidate]
