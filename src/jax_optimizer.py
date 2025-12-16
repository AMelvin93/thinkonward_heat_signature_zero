"""
JAX-based Optimizer using automatic differentiation.

This optimizer uses JAX's autodiff capabilities for true gradient-based
optimization, which is much more efficient than finite-difference methods.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import partial

try:
    from .jax_simulator import (
        simulate_and_sample,
        objective_rmse,
        check_gpu,
    )
except ImportError:
    from jax_simulator import (
        simulate_and_sample,
        objective_rmse,
        check_gpu,
    )


class JAXOptimizer:
    """
    Gradient-based optimizer using JAX automatic differentiation.

    Key advantages over scipy.optimize:
    1. True gradients (not finite differences)
    2. GPU acceleration
    3. JIT compilation for speed
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        # Check device
        self.device_info = check_gpu()
        print(f"JAX backend: {self.device_info.get('default_backend', 'unknown')}")
        print(f"Devices: {self.device_info.get('devices', [])}")

    def _create_objective_and_grad(
        self,
        Y_observed: jnp.ndarray,
        sensors_xy: jnp.ndarray,
        kappa: float,
        dt: float,
        nt: int,
        bc: str,
        T0: float,
    ):
        """Create JIT-compiled objective and gradient functions."""

        @jit
        def objective(sources_flat):
            # Reshape flat params to (n_sources, 3)
            sources = sources_flat.reshape(-1, 3)
            return objective_rmse(
                sources, Y_observed,
                self.nx, self.ny, self.Lx, self.Ly,
                kappa, dt, nt, sensors_xy, bc, T0
            )

        @jit
        def objective_and_grad(sources_flat):
            return value_and_grad(objective)(sources_flat)

        return objective, objective_and_grad

    def _clip_to_bounds(
        self,
        params: jnp.ndarray,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05,
    ) -> jnp.ndarray:
        """Clip parameters to valid bounds."""
        params = params.reshape(n_sources, 3)

        # Clip x
        params = params.at[:, 0].set(
            jnp.clip(params[:, 0], margin * self.Lx, (1 - margin) * self.Lx)
        )
        # Clip y
        params = params.at[:, 1].set(
            jnp.clip(params[:, 1], margin * self.Ly, (1 - margin) * self.Ly)
        )
        # Clip q
        params = params.at[:, 2].set(
            jnp.clip(params[:, 2], q_range[0], q_range[1])
        )

        return params.flatten()

    def _random_init(
        self,
        key: jax.random.PRNGKey,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05,
    ) -> jnp.ndarray:
        """Generate random initial parameters."""
        keys = jax.random.split(key, n_sources * 3)

        params = []
        for i in range(n_sources):
            x = jax.random.uniform(keys[i*3], minval=margin*self.Lx, maxval=(1-margin)*self.Lx)
            y = jax.random.uniform(keys[i*3+1], minval=margin*self.Ly, maxval=(1-margin)*self.Ly)
            q = jax.random.uniform(keys[i*3+2], minval=q_range[0], maxval=q_range[1])
            params.extend([x, y, q])

        return jnp.array(params)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        n_restarts: int = 3,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Estimate source parameters using gradient descent.

        Args:
            sample: Sample dict
            meta: Meta dict with dt
            q_range: Valid range for q
            n_restarts: Number of random restarts
            max_iter: Maximum gradient descent iterations
            learning_rate: Step size for gradient descent
            verbose: Print progress

        Returns:
            (estimated_sources, best_rmse)
        """
        # Extract sample info
        n_sources = sample['n_sources']
        sensors_xy = jnp.array(sample['sensors_xy'])
        Y_observed = jnp.array(sample['Y_noisy'])

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create objective and gradient functions
        objective, obj_and_grad = self._create_objective_and_grad(
            Y_observed, sensors_xy, kappa, dt, nt, bc, T0
        )

        best_params = None
        best_rmse = float('inf')

        # Random key for initialization
        key = jax.random.PRNGKey(42)

        for restart in range(n_restarts):
            # Random initialization
            key, subkey = jax.random.split(key)
            params = self._random_init(subkey, n_sources, q_range)

            # Gradient descent with Adam-like momentum
            velocity = jnp.zeros_like(params)
            beta = 0.9

            for iteration in range(max_iter):
                # Compute loss and gradient
                loss, grads = obj_and_grad(params)

                # Update with momentum
                velocity = beta * velocity + (1 - beta) * grads
                params = params - learning_rate * velocity

                # Project to valid bounds
                params = self._clip_to_bounds(params, n_sources, q_range)

                if verbose and iteration % 20 == 0:
                    print(f"  Restart {restart}, Iter {iteration}: RMSE = {loss:.6f}")

            # Final loss
            final_loss = float(objective(params))

            if final_loss < best_rmse:
                best_rmse = final_loss
                best_params = params

            if verbose:
                print(f"Restart {restart} complete: RMSE = {final_loss:.6f}")

        # Convert to list of tuples
        best_params = best_params.reshape(n_sources, 3)
        estimated_sources = [
            (float(best_params[i, 0]), float(best_params[i, 1]), float(best_params[i, 2]))
            for i in range(n_sources)
        ]

        return estimated_sources, best_rmse

    def estimate_sources_adam(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        n_restarts: int = 3,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Estimate sources using Adam optimizer.

        Adam typically converges faster and more reliably than vanilla gradient descent.
        """
        # Extract sample info
        n_sources = sample['n_sources']
        sensors_xy = jnp.array(sample['sensors_xy'])
        Y_observed = jnp.array(sample['Y_noisy'])

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create objective and gradient functions
        objective, obj_and_grad = self._create_objective_and_grad(
            Y_observed, sensors_xy, kappa, dt, nt, bc, T0
        )

        best_params = None
        best_rmse = float('inf')

        key = jax.random.PRNGKey(42)

        # Adam hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for restart in range(n_restarts):
            key, subkey = jax.random.split(key)
            params = self._random_init(subkey, n_sources, q_range)

            # Adam state
            m = jnp.zeros_like(params)  # First moment
            v = jnp.zeros_like(params)  # Second moment

            for t in range(1, max_iter + 1):
                loss, grads = obj_and_grad(params)

                # Update biased moments
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * (grads ** 2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Update params
                params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)

                # Project to bounds
                params = self._clip_to_bounds(params, n_sources, q_range)

                if verbose and t % 50 == 0:
                    print(f"  Restart {restart}, Iter {t}: RMSE = {loss:.6f}")

            final_loss = float(objective(params))

            if final_loss < best_rmse:
                best_rmse = final_loss
                best_params = params

            if verbose:
                print(f"Restart {restart} complete: RMSE = {final_loss:.6f}")

        # Convert to list of tuples
        best_params = best_params.reshape(n_sources, 3)
        estimated_sources = [
            (float(best_params[i, 0]), float(best_params[i, 1]), float(best_params[i, 2]))
            for i in range(n_sources)
        ]

        return estimated_sources, best_rmse


def benchmark_jax_vs_numpy(sample: Dict, meta: Dict, n_iterations: int = 10):
    """
    Benchmark JAX simulator against numpy simulator.
    """
    import time
    import sys
    import os

    # Import numpy simulator
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))
    from simulator import Heat2D

    Lx, Ly = 2.0, 1.0
    nx, ny = 100, 50

    n_sources = sample['n_sources']
    sensors_xy = sample['sensors_xy']
    dt = meta['dt']
    nt = sample['sample_metadata']['nt']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']

    # Create test sources
    sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
    sources_jax = jnp.array([[1.0, 0.5, 1.0]])

    # === Numpy benchmark ===
    solver_np = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    start = time.time()
    for _ in range(n_iterations):
        times_np, Us_np = solver_np.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y_np = np.array([solver_np.sample_sensors(U, sensors_xy) for U in Us_np])
    numpy_time = (time.time() - start) / n_iterations

    # === JAX benchmark (with compilation) ===
    # First call compiles
    _ = simulate_and_sample(
        sources_jax, nx, ny, Lx, Ly, kappa, dt, nt,
        jnp.array(sensors_xy), bc, T0
    )

    # Time subsequent calls
    start = time.time()
    for _ in range(n_iterations):
        Y_jax = simulate_and_sample(
            sources_jax, nx, ny, Lx, Ly, kappa, dt, nt,
            jnp.array(sensors_xy), bc, T0
        )
        Y_jax.block_until_ready()  # Wait for GPU
    jax_time = (time.time() - start) / n_iterations

    return {
        'numpy_time': numpy_time,
        'jax_time': jax_time,
        'speedup': numpy_time / jax_time,
        'n_iterations': n_iterations,
        'nt': nt,
        'grid': f'{nx}x{ny}',
    }
