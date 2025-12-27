"""
Batched JAX Optimizer using vmap for parallel finite differences.

Key insight: Autodiff through time-stepping is expensive (stores all states).
Instead, we use vmap to run ALL finite-difference perturbations in parallel on GPU.

For 6 params: Instead of 12 sequential forward passes, we do 1 batched pass.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, vmap

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


class JAXBatchedOptimizer:
    """
    Optimizer using batched finite differences via vmap.

    Instead of autodiff (expensive memory), we:
    1. Create all perturbed parameter vectors
    2. Use vmap to evaluate them ALL in parallel on GPU
    3. Compute gradients from finite differences

    This gives GPU parallelism without autodiff memory overhead.
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

    def _create_batched_objective(
        self,
        n_sources: int,
        sensors_xy: jnp.ndarray,
        Y_observed: jnp.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str,
        T0: float,
    ) -> Tuple[Callable, Callable]:
        """
        Create single-eval and batched objective functions.

        The batched version uses vmap to evaluate multiple param sets in parallel.
        """
        sensors_jax = jnp.array(sensors_xy)
        Y_obs_jax = jnp.array(Y_observed)

        @jit
        def single_objective(params):
            """Evaluate single parameter set."""
            sources = params.reshape(n_sources, 3)
            Y_pred = simulate_and_sample(
                sources,
                self.nx, self.ny, self.Lx, self.Ly,
                kappa, dt, nt, sensors_jax, bc, T0
            )
            mse = jnp.mean((Y_pred - Y_obs_jax) ** 2)
            return mse

        # Batched version - evaluate many param sets at once
        @jit
        def batched_objective(params_batch):
            """Evaluate batch of parameter sets in parallel."""
            return vmap(single_objective)(params_batch)

        return single_objective, batched_objective

    def _batched_finite_diff_gradient(
        self,
        params: jnp.ndarray,
        batched_obj: Callable,
        eps: float = 1e-4,
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute gradient using batched finite differences.

        Creates all perturbation vectors, evaluates them in ONE batched call.
        """
        n_params = len(params)

        # Create perturbation matrix: [params, params+eps*e1, params-eps*e1, ...]
        # Shape: (1 + 2*n_params, n_params)
        perturbations = [params]  # Base point

        for i in range(n_params):
            # Forward perturbation
            p_plus = params.at[i].add(eps)
            perturbations.append(p_plus)
            # Backward perturbation
            p_minus = params.at[i].add(-eps)
            perturbations.append(p_minus)

        # Stack into batch
        params_batch = jnp.stack(perturbations)

        # Single batched forward pass (all evaluations in parallel on GPU)
        losses = batched_obj(params_batch)

        # Extract base loss and compute gradients
        base_loss = losses[0]
        gradients = jnp.zeros(n_params)

        for i in range(n_params):
            f_plus = losses[1 + 2*i]
            f_minus = losses[2 + 2*i]
            gradients = gradients.at[i].set((f_plus - f_minus) / (2 * eps))

        return base_loss, gradients

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 30,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate source parameters using batched finite differences.
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
        n_params = n_sources * 3

        # Create batched objective
        single_obj, batched_obj = self._create_batched_objective(
            n_sources, sensors_xy, Y_observed,
            kappa, dt, nt, bc, T0
        )

        # Warmup JIT
        if verbose:
            print("  Warming up JIT (batched forward)...")
        dummy_params = jnp.array(self._random_init(n_sources, q_range))
        dummy_batch = jnp.stack([dummy_params] * (1 + 2 * n_params))
        _ = batched_obj(dummy_batch)

        # Wrapper for scipy
        def scipy_objective(params_np):
            params_jax = jnp.array(params_np)
            mse, grad = self._batched_finite_diff_gradient(params_jax, batched_obj)
            return float(mse), np.array(grad)

        # Generate initial points
        initial_points = []
        for i in range(self.n_smart_inits):
            perturbation = 0.05 + i * 0.03
            x0 = self._smart_init(sample, n_sources, q_range, perturbation)
            initial_points.append(('smart', x0))

        for _ in range(self.n_random_inits):
            x0 = self._random_init(n_sources, q_range)
            initial_points.append(('random', x0))

        if verbose:
            print(f"  Running {len(initial_points)} L-BFGS-B with batched gradients...")

        # Run optimizations
        candidates = []
        for init_type, x0 in initial_points:
            try:
                result = minimize(
                    scipy_objective,
                    x0=x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-7, 'gtol': 1e-6},
                )
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
