"""
Seeded CMA-ES wrapper for reproducible optimization.

This module provides helper functions to create reproducibly-seeded CMA-ES
instances. Use these instead of creating cma.CMAEvolutionStrategy directly.

Usage:
    from src.seeded_cmaes import create_seeded_cmaes

    # In your optimizer's _run_single_optimization method:
    for init_idx, (init_params, init_type) in enumerate(initializations):
        cmaes_seed = sample_seed + init_idx * 1000  # Or use SeedManager

        es = create_seeded_cmaes(
            x0=init_params.tolist(),
            sigma0=sigma0,
            bounds=[lb, ub],
            maxfevals=fevals_per_init,
            seed=cmaes_seed,
        )
        # ... run optimization loop ...
"""

import cma
import numpy as np
from typing import List, Optional, Tuple, Any


def create_seeded_cmaes(
    x0: List[float],
    sigma0: float,
    bounds: Optional[List[List[float]]] = None,
    maxfevals: int = 100,
    seed: int = 42,
    verbose: int = -9,
    tolfun: float = 1e-6,
    tolx: float = 1e-6,
    **extra_options
) -> cma.CMAEvolutionStrategy:
    """
    Create a seeded CMA-ES instance with common options.

    Args:
        x0: Initial solution vector.
        sigma0: Initial step size (standard deviation).
        bounds: [lower_bounds, upper_bounds] or None for unbounded.
        maxfevals: Maximum function evaluations.
        seed: Random seed for reproducibility.
        verbose: Verbosity level (-9 for silent).
        tolfun: Tolerance on function value changes.
        tolx: Tolerance on solution changes.
        **extra_options: Additional CMA options.

    Returns:
        Seeded CMAEvolutionStrategy instance.

    Example:
        es = create_seeded_cmaes(
            x0=[0.5, 0.5],
            sigma0=0.2,
            bounds=[[0, 0], [2, 1]],
            maxfevals=50,
            seed=12345,
        )

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
    """
    opts = cma.CMAOptions()
    opts['maxfevals'] = maxfevals
    opts['verbose'] = verbose
    opts['tolfun'] = tolfun
    opts['tolx'] = tolx
    opts['seed'] = seed  # THIS IS THE KEY ADDITION

    if bounds is not None:
        opts['bounds'] = bounds

    # Apply any extra options
    for key, value in extra_options.items():
        opts[key] = value

    return cma.CMAEvolutionStrategy(x0, sigma0, opts)


def derive_cmaes_seed(sample_seed: int, init_idx: int) -> int:
    """
    Derive a CMA-ES seed from sample seed and initialization index.

    Use this when you don't have access to SeedManager but have a sample seed.

    Args:
        sample_seed: The seed for the current sample.
        init_idx: Index of the initialization (0, 1, 2, ...).

    Returns:
        Deterministic seed for CMA-ES.
    """
    # Simple but effective derivation
    return (sample_seed * 1000 + init_idx * 7) % (2**31)


class SeededOptimizationContext:
    """
    Context manager for seeded optimization within a sample.

    Provides both the numpy random state and CMA-ES seeds for a single sample.
    Use this in worker functions to ensure all randomness is seeded.

    Example:
        def process_sample(args):
            idx, sample, meta, config, sample_seed = args

            with SeededOptimizationContext(sample_seed) as ctx:
                # Now np.random.* calls are seeded
                random_init = np.random.random(3)

                # Get CMA-ES seed for first initialization
                cmaes_seed = ctx.get_cmaes_seed(init_idx=0)
                es = create_seeded_cmaes(x0, sigma0, seed=cmaes_seed, ...)
    """

    def __init__(self, sample_seed: int):
        self.sample_seed = sample_seed
        self._old_state = None

    def __enter__(self):
        # Save old state
        self._old_state = np.random.get_state()
        # Set new seed
        np.random.seed(self.sample_seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old state (optional, but clean)
        if self._old_state is not None:
            np.random.set_state(self._old_state)
        return False

    def get_cmaes_seed(self, init_idx: int = 0) -> int:
        """Get CMA-ES seed for a given initialization index."""
        return derive_cmaes_seed(self.sample_seed, init_idx)

    def get_perturbation_seed(self, candidate_idx: int, perturbation_idx: int) -> int:
        """Get seed for perturbation generation."""
        return (self.sample_seed * 10000 +
                candidate_idx * 100 +
                perturbation_idx) % (2**31)


def seed_numpy_for_sample(sample_seed: int):
    """
    Simple function to seed numpy for a sample.

    Call this at the start of each worker function.

    Args:
        sample_seed: Seed for this sample.
    """
    np.random.seed(sample_seed)


# Example showing how to modify an existing optimizer's run method
EXAMPLE_USAGE = """
# Before (not reproducible):
def _run_single_optimization(self, sample, meta, ...):
    for init_params, init_type in initializations:
        opts = cma.CMAOptions()
        opts['maxfevals'] = fevals_per_init
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)
        # ...

# After (reproducible):
def _run_single_optimization(self, sample, meta, ..., sample_seed=None):
    from src.seeded_cmaes import create_seeded_cmaes, derive_cmaes_seed

    for init_idx, (init_params, init_type) in enumerate(initializations):
        cmaes_seed = derive_cmaes_seed(sample_seed, init_idx) if sample_seed else None

        es = create_seeded_cmaes(
            x0=init_params.tolist(),
            sigma0=sigma0,
            bounds=[lb, ub],
            maxfevals=fevals_per_init,
            seed=cmaes_seed if cmaes_seed else 42,
        )
        # ...
"""
