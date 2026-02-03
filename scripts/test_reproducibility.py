"""
Test script to validate reproducibility of seeded experiments.

Run this script twice with the same seed and compare results.
If seeding is working correctly, results should be identical.

Usage:
    # Run twice with same seed
    python scripts/test_reproducibility.py --seed 42 --samples 5
    python scripts/test_reproducibility.py --seed 42 --samples 5

    # Compare the output - should be identical

    # Run with different seed - should differ
    python scripts/test_reproducibility.py --seed 123 --samples 5
"""

import sys
import pickle
import argparse
import json
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.seed_manager import SeedManager


def test_seed_manager_determinism():
    """Test that SeedManager produces consistent seeds."""
    print("\n=== Testing SeedManager Determinism ===")

    # Create two managers with same seed
    sm1 = SeedManager(master_seed=42)
    sm2 = SeedManager(master_seed=42)

    # Generate seeds for 10 samples
    seeds1 = [sm1.get_sample_seed(i) for i in range(10)]
    seeds2 = [sm2.get_sample_seed(i) for i in range(10)]

    print(f"Seeds from manager 1: {seeds1}")
    print(f"Seeds from manager 2: {seeds2}")

    assert seeds1 == seeds2, "FAIL: SeedManager produces inconsistent seeds!"
    print("PASS: SeedManager produces identical seeds")

    # Test CMA-ES seeds
    cmaes_seeds1 = [sm1.get_cmaes_seed(i, 0) for i in range(5)]
    cmaes_seeds2 = [sm2.get_cmaes_seed(i, 0) for i in range(5)]

    assert cmaes_seeds1 == cmaes_seeds2, "FAIL: CMA-ES seeds inconsistent!"
    print("PASS: CMA-ES seeds are identical")

    # Test different init_idx produces different seeds
    init0_seeds = [sm1.get_cmaes_seed(0, i) for i in range(5)]
    print(f"CMA-ES seeds for sample 0, init 0-4: {init0_seeds}")
    assert len(set(init0_seeds)) == 5, "FAIL: Different init_idx should produce different seeds!"
    print("PASS: Different init_idx produces different seeds")


def test_numpy_seeding():
    """Test that numpy seeding produces reproducible random numbers."""
    print("\n=== Testing NumPy Seeding ===")

    sm = SeedManager(master_seed=42)

    # Simulate what happens in a worker
    sample_seed = sm.get_sample_seed(0)

    np.random.seed(sample_seed)
    random1 = [np.random.random() for _ in range(5)]

    np.random.seed(sample_seed)
    random2 = [np.random.random() for _ in range(5)]

    print(f"Sample seed: {sample_seed}")
    print(f"Random sequence 1: {random1}")
    print(f"Random sequence 2: {random2}")

    assert random1 == random2, "FAIL: NumPy seeding not reproducible!"
    print("PASS: NumPy seeding is reproducible")


def test_sample_processing_reproducibility(n_samples: int = 5, seed: int = 42):
    """Test that sample processing produces reproducible results."""
    print(f"\n=== Testing Sample Processing (seed={seed}, n={n_samples}) ===")

    # Load test data
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    if not data_path.exists():
        print(f"SKIP: Test data not found at {data_path}")
        return

    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples'][:n_samples]
    meta = test_data['meta']

    sm = SeedManager(master_seed=seed)

    results = []
    for i, sample in enumerate(samples):
        sample_seed = sm.get_sample_seed(i)
        np.random.seed(sample_seed)

        # Simulate some random operations that would happen in optimizer
        random_init = np.random.random(3)
        perturbation = np.random.randn(3)
        cmaes_seed = sm.get_cmaes_seed(i, 0)

        results.append({
            'idx': i,
            'sample_seed': sample_seed,
            'random_init': random_init.tolist(),
            'perturbation': perturbation.tolist(),
            'cmaes_seed': cmaes_seed,
            'n_sources': sample['n_sources'],
        })

    # Print results in a reproducible format
    print(f"\nResults for seed={seed}:")
    for r in results:
        print(f"  Sample {r['idx']}: seed={r['sample_seed']}, "
              f"init[0]={r['random_init'][0]:.6f}, "
              f"perturb[0]={r['perturbation'][0]:.6f}, "
              f"cmaes_seed={r['cmaes_seed']}")

    # Return hash for comparison
    results_str = json.dumps(results, sort_keys=True)
    results_hash = hash(results_str)
    print(f"\nResults hash: {results_hash}")
    return results_hash


def main():
    parser = argparse.ArgumentParser(description="Test reproducibility of seeded experiments")
    parser.add_argument('--seed', type=int, default=42, help='Master seed')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
    args = parser.parse_args()

    print("=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)
    print(f"Master seed: {args.seed}")
    print(f"Samples: {args.samples}")

    # Run tests
    test_seed_manager_determinism()
    test_numpy_seeding()
    results_hash = test_sample_processing_reproducibility(
        n_samples=args.samples,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"All basic tests PASSED")
    print(f"Results hash: {results_hash}")
    print(f"\nTo verify full reproducibility:")
    print(f"1. Run: python scripts/test_reproducibility.py --seed {args.seed}")
    print(f"2. Run again with same seed")
    print(f"3. Compare the results hash - should be identical")
    print("=" * 60)


if __name__ == '__main__':
    main()
