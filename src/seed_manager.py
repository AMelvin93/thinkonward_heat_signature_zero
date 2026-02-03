"""
Seed Manager for Reproducible Experiments.

Provides deterministic seeding for:
- Per-sample reproducibility via derived seeds
- CMA-ES optimizer seeding
- NumPy random state management in parallel workers

Usage:
    # In run.py
    from src.seed_manager import SeedManager

    seed_manager = SeedManager(master_seed=42)

    # For parallel processing
    def process_sample(args):
        idx, sample, meta, config, sample_seed = args
        seed_manager.seed_worker(sample_seed)  # Call at start of worker
        ...

    # Create work items with seeds
    work_items = [
        (i, samples[i], meta, config, seed_manager.get_sample_seed(i))
        for i in range(n_samples)
    ]

    # For CMA-ES
    opts = cma.CMAOptions()
    opts['seed'] = seed_manager.get_cmaes_seed(sample_idx, init_idx)

Note on Reproducibility:
    - Full determinism requires seeding at the start of each worker process
    - CMA-ES may have minor non-determinism in some dimensions due to np.linalg.eigh
    - Always store the master_seed in STATE.json for reproducibility
"""

import numpy as np
from typing import Optional, Dict, Any
import hashlib


class SeedManager:
    """
    Manages seeds for reproducible experiments.

    Generates deterministic sub-seeds from a master seed for:
    - Individual samples (for parallel processing)
    - CMA-ES optimizer instances
    - Perturbation generation

    The same master_seed will always produce the same sequence of sub-seeds.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize seed manager with a master seed.

        Args:
            master_seed: The master seed for all random operations.
                         Store this in STATE.json for reproducibility.
        """
        self.master_seed = master_seed
        self._rng = np.random.default_rng(master_seed)

    def get_sample_seed(self, sample_idx: int) -> int:
        """
        Get a deterministic seed for a specific sample.

        The same (master_seed, sample_idx) pair always produces the same seed.
        This allows reproducible parallel processing where each worker gets
        a deterministic seed based on the sample index.

        Args:
            sample_idx: Index of the sample being processed.

        Returns:
            Deterministic seed for this sample.
        """
        # Use hash-based seed derivation for determinism regardless of order
        hash_input = f"{self.master_seed}:sample:{sample_idx}".encode()
        hash_value = hashlib.sha256(hash_input).digest()
        # Use first 4 bytes as seed (gives 32-bit integer range)
        seed = int.from_bytes(hash_value[:4], 'little') % (2**31)
        return seed

    def get_cmaes_seed(self, sample_idx: int, init_idx: int = 0) -> int:
        """
        Get a deterministic seed for CMA-ES optimizer.

        Different initializations within the same sample get different seeds,
        but the same (master_seed, sample_idx, init_idx) always produces
        the same CMA-ES seed.

        Args:
            sample_idx: Index of the sample being processed.
            init_idx: Index of the initialization (for multi-init strategies).

        Returns:
            Deterministic seed for CMA-ES.
        """
        hash_input = f"{self.master_seed}:cmaes:{sample_idx}:{init_idx}".encode()
        hash_value = hashlib.sha256(hash_input).digest()
        seed = int.from_bytes(hash_value[:4], 'little') % (2**31)
        return seed

    def get_perturbation_seed(self, sample_idx: int, candidate_idx: int,
                               perturbation_idx: int) -> int:
        """
        Get a deterministic seed for perturbation generation.

        Args:
            sample_idx: Index of the sample.
            candidate_idx: Index of the candidate being perturbed.
            perturbation_idx: Index of the perturbation.

        Returns:
            Deterministic seed for perturbation RNG.
        """
        hash_input = (f"{self.master_seed}:perturb:{sample_idx}:"
                      f"{candidate_idx}:{perturbation_idx}").encode()
        hash_value = hashlib.sha256(hash_input).digest()
        seed = int.from_bytes(hash_value[:4], 'little') % (2**31)
        return seed

    @staticmethod
    def seed_worker(seed: int) -> np.random.Generator:
        """
        Seed NumPy's random state at the start of a worker process.

        Call this at the very beginning of each worker function when
        using ProcessPoolExecutor.

        Args:
            seed: The seed for this worker (from get_sample_seed).

        Returns:
            A numpy Generator instance for use in the worker.
            Can also use np.random.* functions as global state is set.
        """
        # Seed the legacy global random state (for compatibility with code
        # that uses np.random.random(), np.random.randn(), etc.)
        np.random.seed(seed)

        # Also return a Generator for modern usage
        return np.random.default_rng(seed)

    @staticmethod
    def create_seeded_rng(seed: int) -> np.random.Generator:
        """
        Create a new seeded random number generator.

        Use this when you need a local RNG instance without affecting
        the global random state.

        Args:
            seed: Seed for the generator.

        Returns:
            A seeded numpy Generator instance.
        """
        return np.random.default_rng(seed)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize seed manager state for storage in STATE.json.

        Returns:
            Dict with seed information for reproducibility.
        """
        return {
            'master_seed': self.master_seed,
            'seed_manager_version': '1.0',
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeedManager':
        """
        Restore seed manager from stored state.

        Args:
            data: Dict from STATE.json containing seed info.

        Returns:
            SeedManager instance with restored state.
        """
        return cls(master_seed=data['master_seed'])


def generate_sample_seeds(master_seed: int, n_samples: int) -> list:
    """
    Pre-generate all sample seeds for a run.

    Useful for storing in results or passing to workers.

    Args:
        master_seed: The master seed for the run.
        n_samples: Number of samples to generate seeds for.

    Returns:
        List of (sample_idx, seed) tuples.
    """
    seed_manager = SeedManager(master_seed)
    return [(i, seed_manager.get_sample_seed(i)) for i in range(n_samples)]


# Convenience functions for quick seeding
def seed_all(seed: int):
    """
    Seed all random sources with a single seed.

    Use this at the start of a script for basic reproducibility.
    For parallel processing, use SeedManager instead.

    Args:
        seed: Seed value.
    """
    np.random.seed(seed)

    # Try to seed Python's random module too
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
