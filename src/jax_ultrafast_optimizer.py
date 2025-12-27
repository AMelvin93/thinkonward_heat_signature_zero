"""
Ultra-Fast JAX Optimizer.

Combines ALL optimizations:
1. Memory-efficient simulator (samples during loop, not after)
2. Batched finite differences via vmap (parallel gradient computation)
3. Minimal initializations (1 smart + 1 random)
4. Fewer iterations (20)
5. JIT caching per nt value (only 3 unique values in dataset)

Target: 400 samples in <1 hour on G4dn.2xlarge = 9s per sample
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
    from .jax_simulator_fast import simulate_and_sample_fast, check_gpu, get_simulate_fn
except ImportError:
    from jax_simulator_fast import simulate_and_sample_fast, check_gpu, get_simulate_fn


@dataclass
class CandidateResult:
    params: np.ndarray
    rmse: float
    init_type: str


class JAXUltraFastOptimizer:
    """
    Ultra-fast optimizer with all optimizations enabled.

    Performance targets:
    - 9s per sample (400 samples in 60 min)
    - GPU-accelerated forward passes
    - Batched gradient computation
    - Minimal restarts
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_smart_inits: int = 1,
        n_random_inits: int = 1,
        n_max_candidates: int = 3,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_smart_inits = n_smart_inits
        self.n_random_inits = n_random_inits
        self.n_max_candidates = n_max_candidates

        self.device_info = check_gpu()
        self.gpu_available = self.device_info.get('gpu_available', False)

        # Cache for batched objective functions
        self._batched_obj_cache = {}

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
    ) -> List[Tuple[float, float, float]]:
        """Quick sensor analysis for smart initialization."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_indices = np.argsort(avg_temps)[::-1]

        # Fast selection - just take hottest separated sensors
        selected = []
        min_separation = 0.25
        for idx in hot_indices:
            if len(selected) >= n_sources:
                break
            is_sep = all(np.linalg.norm(sensors[idx] - sensors[prev]) >= min_separation
                        for prev in selected)
            if is_sep:
                selected.append(idx)

        # Fill remaining if needed
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
            q = 0.5 + temp_ratio * 1.5
            q = np.clip(q, 0.5, 2.0)
            sources.append((loc[0], loc[1], q))

        return sources

    def _smart_init(self, sample: Dict, n_sources: int, q_range: Tuple[float, float]) -> np.ndarray:
        base_sources = self._analyze_sensor_temperatures(sample, n_sources)
        bounds = self._get_bounds(n_sources, q_range)

        params = []
        for i, (x, y, q) in enumerate(base_sources):
            # Small perturbation
            x_new = x + np.random.uniform(-0.05, 0.05) * self.Lx
            y_new = y + np.random.uniform(-0.05, 0.05) * self.Ly
            q_new = q + np.random.uniform(-0.1, 0.1)
            x_new = np.clip(x_new, bounds[i*3][0], bounds[i*3][1])
            y_new = np.clip(y_new, bounds[i*3+1][0], bounds[i*3+1][1])
            q_new = np.clip(q_new, q_range[0], q_range[1])
            params.extend([x_new, y_new, q_new])

        return np.array(params)

    def _random_init(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.1) -> np.ndarray:
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

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
        """Create single and batched objective functions."""
        sensors_jax = jnp.array(sensors_xy)
        Y_obs_jax = jnp.array(Y_observed)

        # Get or create the simulate function for this nt
        sim_fn = get_simulate_fn(nt, self.nx, self.ny, self.Lx, self.Ly, bc)

        @jit
        def single_objective(params):
            sources = params.reshape(n_sources, 3)
            Y_pred = sim_fn(sources, kappa, dt, sensors_jax, T0)
            mse = jnp.mean((Y_pred - Y_obs_jax) ** 2)
            return mse

        @jit
        def batched_objective(params_batch):
            return vmap(single_objective)(params_batch)

        return single_objective, batched_objective

    def _batched_gradient(
        self,
        params: jnp.ndarray,
        batched_obj: Callable,
        eps: float = 1e-4,
    ) -> Tuple[float, jnp.ndarray]:
        """Compute gradient using batched central differences."""
        n_params = len(params)

        # Create perturbation batch: [base, +eps for each, -eps for each]
        perturbations = [params]
        for i in range(n_params):
            perturbations.append(params.at[i].add(eps))
            perturbations.append(params.at[i].add(-eps))

        params_batch = jnp.stack(perturbations)
        losses = batched_obj(params_batch)

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
        max_iter: int = 20,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """Estimate source parameters - ultra-fast version."""
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

        # Create objectives
        single_obj, batched_obj = self._create_batched_objective(
            n_sources, sensors_xy, Y_observed,
            kappa, dt, nt, bc, T0
        )

        # Warmup JIT (critical for first sample of each nt)
        if verbose:
            print(f"  Warming up JIT for nt={nt}...")
        dummy_params = jnp.array(self._random_init(n_sources, q_range))
        dummy_batch = jnp.stack([dummy_params] * (1 + 2 * n_params))
        _ = batched_obj(dummy_batch)

        # Scipy wrapper with batched gradients
        def scipy_objective(params_np):
            params_jax = jnp.array(params_np)
            mse, grad = self._batched_gradient(params_jax, batched_obj)
            return float(mse), np.array(grad)

        # Generate initial points (minimal)
        initial_points = []
        for _ in range(self.n_smart_inits):
            initial_points.append(('smart', self._smart_init(sample, n_sources, q_range)))
        for _ in range(self.n_random_inits):
            initial_points.append(('random', self._random_init(n_sources, q_range)))

        if verbose:
            print(f"  Running {len(initial_points)} L-BFGS-B optimizations...")

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
                    options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-5},
                )
                rmse = np.sqrt(result.fun)
                candidates.append(CandidateResult(result.x, rmse, init_type))
                if verbose:
                    print(f"    {init_type}: RMSE={rmse:.4f}, nit={result.nit}")
            except Exception as e:
                if verbose:
                    print(f"    {init_type}: failed - {e}")

        # Sort by RMSE
        candidates.sort(key=lambda c: c.rmse)

        if verbose:
            print(f"  Best RMSE: {candidates[0].rmse:.4f}" if candidates else "  No valid candidates")

        # Return best
        if candidates:
            best = candidates[0]
            best_sources = []
            for i in range(n_sources):
                x, y, q = best.params[i*3:(i+1)*3]
                best_sources.append((x, y, q))
            return best_sources, best.rmse, candidates[:self.n_max_candidates]
        else:
            return [], float('inf'), []


def precompile_for_dataset(samples: List[Dict], nx: int = 100, ny: int = 50,
                           Lx: float = 2.0, Ly: float = 1.0):
    """
    Pre-compile JIT functions for all unique nt values in dataset.
    Call this once before processing to avoid compilation during timing.
    """
    unique_nts = set()
    unique_bcs = set()
    for sample in samples:
        unique_nts.add(sample['sample_metadata']['nt'])
        unique_bcs.add(sample['sample_metadata']['bc'])

    print(f"Pre-compiling for {len(unique_nts)} unique nt values: {sorted(unique_nts)}")
    print(f"Boundary conditions: {unique_bcs}")

    for nt in unique_nts:
        for bc in unique_bcs:
            fn = get_simulate_fn(nt, nx, ny, Lx, Ly, bc)
            # Warmup call
            dummy_sources = jnp.array([[1.0, 0.5, 1.0]])
            dummy_sensors = jnp.array([[0.5, 0.25], [1.5, 0.75]])
            _ = fn(dummy_sources, 0.01, 0.004, dummy_sensors, 0.0)

    print("Pre-compilation complete!")
