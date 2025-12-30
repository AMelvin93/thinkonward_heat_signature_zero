"""
Intensity-Only Polish Optimizer for Heat Source Identification.

Key insight: Triangulation + CMA-ES give good (x, y) positions.
The polish step mainly refines intensity (q). By fixing positions
and only optimizing intensity, we reduce polish from 3-6 params
to 1-2 params, potentially 3-6x fewer evaluations.

Approach:
1. Triangulation init for good starting point
2. CMA-ES for global optimization (position + intensity)
3. Intensity-only polish: fix (x, y), optimize only q

For 2-source problems, we can optimize both q values:
- Option A: Jointly (2D optimization)
- Option B: Alternating (1D each, iterate)
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cma
from scipy.optimize import minimize_scalar, minimize

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


class IntensityPolishOptimizer:
    """
    CMA-ES optimizer with intensity-only polish.

    Instead of full L-BFGS-B polish (which uses expensive finite differences
    for all 3-6 parameters), this optimizer fixes the (x, y) positions after
    CMA-ES and only optimizes the intensity (q) values.

    Benefits:
    - 1-source: 1 param to optimize instead of 3 (3x fewer evals)
    - 2-source: 2 params to optimize instead of 6 (3x fewer evals)
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
        intensity_polish_method: str = 'bounded',  # 'bounded', 'brent', 'nelder-mead'
        intensity_polish_maxiter: int = 10,
    ):
        """
        Initialize the optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_fevals_1src: Max CMA-ES evaluations for 1-source
            max_fevals_2src: Max CMA-ES evaluations for 2-source
            sigma0_1src: Initial step size for 1-source
            sigma0_2src: Initial step size for 2-source
            use_triangulation: Use triangulation for initialization
            intensity_polish_method: Method for intensity optimization
                - 'bounded': scipy.optimize.minimize_scalar with bounds
                - 'brent': Brent's method (faster but no bounds)
                - 'nelder-mead': Nelder-Mead for 2-source joint optimization
            intensity_polish_maxiter: Max iterations for intensity polish
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
        self.intensity_polish_method = intensity_polish_method
        self.intensity_polish_maxiter = intensity_polish_maxiter

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(
        self,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        """Get parameter bounds for CMA-ES."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return lb, ub

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

    def _intensity_polish_1source(
        self,
        x: float,
        y: float,
        q_init: float,
        objective_fn,
        q_range: Tuple[float, float],
    ) -> Tuple[float, float, int]:
        """
        Polish intensity for 1-source problem.

        Returns: (optimal_q, rmse, n_evals)
        """
        n_evals = 0

        def obj_q(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x, y, q])

        if self.intensity_polish_method == 'bounded':
            result = minimize_scalar(
                obj_q,
                bounds=q_range,
                method='bounded',
                options={'maxiter': self.intensity_polish_maxiter}
            )
            return result.x, result.fun, n_evals

        elif self.intensity_polish_method == 'brent':
            result = minimize_scalar(
                obj_q,
                bracket=(q_range[0], q_init, q_range[1]),
                method='brent',
                options={'maxiter': self.intensity_polish_maxiter}
            )
            return np.clip(result.x, q_range[0], q_range[1]), result.fun, n_evals

        else:
            # Fallback to bounded
            result = minimize_scalar(
                obj_q,
                bounds=q_range,
                method='bounded',
                options={'maxiter': self.intensity_polish_maxiter}
            )
            return result.x, result.fun, n_evals

    def _intensity_polish_2source(
        self,
        positions: List[Tuple[float, float]],
        q_inits: List[float],
        objective_fn,
        q_range: Tuple[float, float],
    ) -> Tuple[List[float], float, int]:
        """
        Polish intensities for 2-source problem.

        Options:
        - Joint optimization (Nelder-Mead on 2D)
        - Alternating optimization (1D each)

        Returns: ([q1, q2], rmse, n_evals)
        """
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        n_evals = 0

        if self.intensity_polish_method == 'nelder-mead':
            # Joint 2D optimization
            def obj_q(q_vals):
                nonlocal n_evals
                n_evals += 1
                q1, q2 = q_vals
                q1 = np.clip(q1, q_range[0], q_range[1])
                q2 = np.clip(q2, q_range[0], q_range[1])
                return objective_fn([x1, y1, q1, x2, y2, q2])

            result = minimize(
                obj_q,
                x0=q_inits,
                method='Nelder-Mead',
                options={'maxiter': self.intensity_polish_maxiter, 'xatol': 0.01}
            )
            q_opt = [np.clip(result.x[0], q_range[0], q_range[1]),
                     np.clip(result.x[1], q_range[0], q_range[1])]
            return q_opt, result.fun, n_evals

        else:
            # Alternating 1D optimization (faster)
            q1, q2 = q_inits
            best_rmse = objective_fn([x1, y1, q1, x2, y2, q2])
            n_evals += 1

            for _ in range(2):  # 2 alternating rounds
                # Optimize q1
                def obj_q1(q):
                    nonlocal n_evals
                    n_evals += 1
                    return objective_fn([x1, y1, q, x2, y2, q2])

                result1 = minimize_scalar(
                    obj_q1,
                    bounds=q_range,
                    method='bounded',
                    options={'maxiter': self.intensity_polish_maxiter // 2}
                )
                q1 = result1.x

                # Optimize q2
                def obj_q2(q):
                    nonlocal n_evals
                    n_evals += 1
                    return objective_fn([x1, y1, q1, x2, y2, q])

                result2 = minimize_scalar(
                    obj_q2,
                    bounds=q_range,
                    method='bounded',
                    options={'maxiter': self.intensity_polish_maxiter // 2}
                )
                q2 = result2.x
                best_rmse = result2.fun

            return [q1, q2], best_rmse, n_evals

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate source parameters using CMA-ES + intensity-only polish.
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

        # Set CMA-ES parameters
        if n_sources == 1:
            max_fevals = self.max_fevals_1src
            sigma0 = self.sigma0_1src
        else:
            max_fevals = self.max_fevals_2src
            sigma0 = self.sigma0_2src

        # Bounds for CMA-ES
        lb, ub = self._get_bounds(n_sources, q_range)

        # CMA-ES options
        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        if verbose:
            print(f"  CMA-ES: n_sources={n_sources}, max_fevals={max_fevals}")

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

        # Intensity-only polish
        if n_sources == 1:
            x, y, q = best_params[0], best_params[1], best_params[2]
            q_opt, polish_rmse, polish_evals = self._intensity_polish_1source(
                x, y, q, objective, q_range
            )
            if polish_rmse < best_rmse:
                best_params = np.array([x, y, q_opt])
                best_rmse = polish_rmse
            n_evals += polish_evals

            if verbose:
                print(f"  Intensity polish: RMSE={best_rmse:.4f}, evals={polish_evals}")

        else:  # 2-source
            positions = [
                (best_params[0], best_params[1]),
                (best_params[3], best_params[4])
            ]
            q_inits = [best_params[2], best_params[5]]

            q_opts, polish_rmse, polish_evals = self._intensity_polish_2source(
                positions, q_inits, objective, q_range
            )

            if polish_rmse < best_rmse:
                best_params = np.array([
                    positions[0][0], positions[0][1], q_opts[0],
                    positions[1][0], positions[1][1], q_opts[1]
                ])
                best_rmse = polish_rmse
            n_evals += polish_evals

            if verbose:
                print(f"  Intensity polish: RMSE={best_rmse:.4f}, evals={polish_evals}")

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
