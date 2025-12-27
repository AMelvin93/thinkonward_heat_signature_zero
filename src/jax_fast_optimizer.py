"""
Fast JAX Optimizer using autodiff gradients with L-BFGS-B.

This combines:
1. JAX JIT-compiled forward simulation (GPU accelerated)
2. JAX autodiff for EXACT gradients (not finite differences)
3. scipy L-BFGS-B for fast convergence

Expected speedup: 10-20x over numerical gradients version.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

try:
    from .jax_simulator import simulate_and_sample, check_gpu
except ImportError:
    from jax_simulator import simulate_and_sample, check_gpu


@dataclass
class CandidateResult:
    """Result from a single optimization run."""
    params: np.ndarray
    rmse: float
    init_type: str


class JAXFastOptimizer:
    """
    Fast optimizer using JAX autodiff + L-BFGS-B.

    Key difference from JAXHybridOptimizer:
    - Uses JAX value_and_grad for EXACT gradients
    - L-BFGS-B uses these gradients directly (no finite differences)
    - ~12x fewer forward passes per gradient evaluation
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_smart_inits: int = 1,
        n_random_inits: int = 2,
        min_candidate_distance: float = 0.15,
        n_max_candidates: int = 3,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_smart_inits = n_smart_inits
        self.n_random_inits = n_random_inits
        self.min_candidate_distance = min_candidate_distance
        self.n_max_candidates = n_max_candidates

        self.device_info = check_gpu()
        self.gpu_available = self.device_info.get('gpu_available', False)

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
            bounds.append(q_range)
        return bounds

    def _analyze_sensor_temperatures(
        self,
        sample: Dict,
        n_sources: int,
        min_separation: float = 0.3,
    ) -> List[Tuple[float, float, float]]:
        """Analyze sensor temperatures to infer likely source locations."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_indices = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_indices:
            if len(selected) >= n_sources:
                break
            is_separated = True
            for prev_idx in selected:
                dist = np.linalg.norm(sensors[idx] - sensors[prev_idx])
                if dist < min_separation:
                    is_separated = False
                    break
            if is_separated:
                selected.append(idx)

        for idx in hot_indices:
            if len(selected) >= n_sources:
                break
            if idx not in selected:
                selected.append(idx)

        sources = []
        max_temp = np.max(avg_temps) + 1e-8
        for s_idx in selected:
            loc = sensors[s_idx]
            temp_ratio = avg_temps[s_idx] / max_temp
            q = 0.5 + temp_ratio * 1.2
            q = np.clip(q, 0.5, 2.0)
            sources.append((loc[0], loc[1], q))

        return sources

    def _smart_init(
        self,
        sample: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
        perturbation: float = 0.1,
    ) -> np.ndarray:
        """Generate smart initial point from sensor analysis."""
        base_sources = self._analyze_sensor_temperatures(sample, n_sources)
        bounds = self._get_bounds(n_sources, q_range)

        params = []
        for i, (x, y, q) in enumerate(base_sources):
            x_new = x + np.random.uniform(-perturbation, perturbation) * self.Lx
            y_new = y + np.random.uniform(-perturbation, perturbation) * self.Ly
            q_new = q + np.random.uniform(-0.2, 0.2)
            x_new = np.clip(x_new, bounds[i*3][0], bounds[i*3][1])
            y_new = np.clip(y_new, bounds[i*3+1][0], bounds[i*3+1][1])
            q_new = np.clip(q_new, q_range[0], q_range[1])
            params.extend([x_new, y_new, q_new])

        return np.array(params)

    def _random_init(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05) -> np.ndarray:
        """Generate random initial parameters."""
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

    def _normalized_distance(
        self,
        params1: np.ndarray,
        params2: np.ndarray,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> float:
        """Compute normalized distance between parameter sets."""
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
        """Filter to keep only distinct candidates."""
        sorted_results = sorted(results, key=lambda r: r.rmse)
        distinct = []
        for result in sorted_results:
            if len(distinct) >= self.n_max_candidates:
                break
            is_distinct = True
            for existing in distinct:
                dist = self._normalized_distance(result.params, existing.params, n_sources, q_range)
                if dist < self.min_candidate_distance:
                    is_distinct = False
                    break
            if is_distinct and result.rmse < float('inf'):
                distinct.append(result)
        return distinct

    def _create_objective_and_grad(
        self,
        n_sources: int,
        sensors_xy: jnp.ndarray,
        Y_observed: jnp.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str,
        T0: float,
    ):
        """
        Create JIT-compiled objective and gradient functions.

        Returns a function that computes BOTH value and gradient in one pass.
        This is the key speedup - no finite differences needed.
        """
        sensors_jax = jnp.array(sensors_xy)
        Y_obs_jax = jnp.array(Y_observed)

        @jit
        def objective(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = simulate_and_sample(
                sources,
                self.nx, self.ny, self.Lx, self.Ly,
                kappa, dt, nt, sensors_jax, bc, T0
            )
            # Use MSE for smoother gradients, convert to RMSE at end
            mse = jnp.mean((Y_pred - Y_obs_jax) ** 2)
            return mse

        # JIT-compiled value_and_grad - computes both in ~1 forward pass
        @jit
        def value_and_gradient(params):
            return value_and_grad(objective)(params)

        return objective, value_and_gradient

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 30,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate source parameters using JAX autodiff + L-BFGS-B.
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        bounds = self._get_bounds(n_sources, q_range)

        # Create JAX autodiff objective
        objective, value_and_gradient = self._create_objective_and_grad(
            n_sources, sensors_xy, Y_observed,
            kappa, dt, nt, bc, T0
        )

        # Warmup JIT compilation
        if verbose:
            print("  Warming up JIT (objective + gradient)...")
        dummy_params = jnp.array(self._random_init(n_sources, q_range))
        _ = value_and_gradient(dummy_params)

        # Wrapper for scipy that uses JAX gradients
        def scipy_objective(params_np):
            params_jax = jnp.array(params_np)
            mse, grad = value_and_gradient(params_jax)
            # Convert MSE to RMSE for reporting, but optimize MSE
            return float(mse), np.array(grad)

        # Generate initial points (fewer needed with better optimizer)
        initial_points = []

        for i in range(self.n_smart_inits):
            perturbation = 0.05 + i * 0.03
            x0 = self._smart_init(sample, n_sources, q_range, perturbation)
            initial_points.append(('smart', x0))

        for _ in range(self.n_random_inits):
            x0 = self._random_init(n_sources, q_range)
            initial_points.append(('random', x0))

        if verbose:
            print(f"  Running {len(initial_points)} L-BFGS-B optimizations with JAX autodiff...")

        # Run optimizations
        candidates = []
        for init_type, x0 in initial_points:
            try:
                result = minimize(
                    scipy_objective,
                    x0=x0,
                    method='L-BFGS-B',
                    jac=True,  # We provide gradients!
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-7, 'gtol': 1e-6},
                )
                # Convert MSE back to RMSE
                rmse = np.sqrt(result.fun)
                candidates.append(CandidateResult(result.x, rmse, init_type))
                if verbose:
                    print(f"    {init_type}: RMSE={rmse:.4f}")
            except Exception as e:
                if verbose:
                    print(f"    {init_type}: failed - {e}")

        # Filter to distinct candidates
        distinct_candidates = self._filter_distinct_candidates(candidates, n_sources, q_range)

        if verbose:
            print(f"  Found {len(distinct_candidates)} distinct candidates")

        # Best result
        if distinct_candidates:
            best = distinct_candidates[0]
            best_sources = []
            for i in range(n_sources):
                x, y, q = best.params[i*3:(i+1)*3]
                best_sources.append((x, y, q))
            return best_sources, best.rmse, distinct_candidates
        else:
            return [], float('inf'), []
