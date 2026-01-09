#!/usr/bin/env python
"""
Benchmark NumPy (CPU) vs JAX (GPU) forward pass.

This will help decide which approach to optimize:
1. JAX GPU: Fast per-forward-pass but no sample parallelism
2. NumPy CPU: Slower per-forward-pass but 8x sample parallelism on G4dn.2xlarge
"""

import sys
import os
import time
import pickle

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "data", "Heat_Signature_zero-starter_notebook"))
sys.path.insert(0, os.path.join(project_root, "src"))

import numpy as np


def benchmark_numpy():
    """Benchmark NumPy-based simulator."""
    from simulator import Heat2D

    # Load data
    data_path = os.path.join(project_root, "data", "heat-signature-zero-test-data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sample = data['samples'][0]
    meta = data['meta']

    Lx, Ly = 2.0, 1.0
    nx, ny = 100, 50
    dt = meta['dt']
    nt = sample['sample_metadata']['nt']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']
    sensors_xy = sample['sensors_xy']

    print(f"Sample: nt={nt}, kappa={kappa}, bc={bc}")

    # Create solver
    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)
    sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]

    # First pass (warmup)
    print("\nNumPy warmup...")
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

    # Benchmark
    print("NumPy benchmark (10 runs)...")
    numpy_times = []
    for i in range(10):
        start = time.time()
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
        numpy_times.append(time.time() - start)

    return {
        'mean': np.mean(numpy_times),
        'std': np.std(numpy_times),
        'min': min(numpy_times),
        'max': max(numpy_times),
    }


def benchmark_jax():
    """Benchmark JAX-based simulator."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap

        print(f"\nJAX devices: {jax.devices()}")
        print(f"Backend: {jax.default_backend()}")
    except ImportError:
        return None

    # Load data
    data_path = os.path.join(project_root, "data", "heat-signature-zero-test-data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sample = data['samples'][0]
    meta = data['meta']

    Lx, Ly = 2.0, 1.0
    nx, ny = 100, 50
    dt = meta['dt']
    nt = sample['sample_metadata']['nt']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']
    sensors_xy = jnp.array(sample['sensors_xy'])

    # Import JAX simulator
    from jax_simulator_fast import get_simulate_fn

    print(f"\nCreating JAX simulator for nt={nt}...")
    sim_fn = get_simulate_fn(nt, nx, ny, Lx, Ly, bc)

    sources = jnp.array([[1.0, 0.5, 1.0]])

    # First pass (JIT compilation)
    print("JAX JIT compilation...")
    start = time.time()
    Y = sim_fn(sources, kappa, dt, sensors_xy, T0)
    Y.block_until_ready()
    jit_time = time.time() - start
    print(f"  JIT compilation time: {jit_time:.2f}s")

    # Benchmark
    print("JAX benchmark (10 runs)...")
    jax_times = []
    for i in range(10):
        start = time.time()
        Y = sim_fn(sources, kappa, dt, sensors_xy, T0)
        Y.block_until_ready()
        jax_times.append(time.time() - start)

    return {
        'mean': np.mean(jax_times),
        'std': np.std(jax_times),
        'min': min(jax_times),
        'max': max(jax_times),
        'jit_time': jit_time,
    }


def main():
    print("=" * 60)
    print("NUMPY vs JAX FORWARD PASS BENCHMARK")
    print("=" * 60)

    # NumPy benchmark
    print("\n" + "-" * 40)
    print("NUMPY (CPU)")
    print("-" * 40)
    numpy_results = benchmark_numpy()
    print(f"  Mean: {numpy_results['mean']:.3f}s")
    print(f"  Std:  {numpy_results['std']:.3f}s")
    print(f"  Range: {numpy_results['min']:.3f}s - {numpy_results['max']:.3f}s")

    # JAX benchmark
    print("\n" + "-" * 40)
    print("JAX (GPU)")
    print("-" * 40)
    jax_results = benchmark_jax()
    if jax_results:
        print(f"  Mean: {jax_results['mean']:.3f}s")
        print(f"  Std:  {jax_results['std']:.3f}s")
        print(f"  Range: {jax_results['min']:.3f}s - {jax_results['max']:.3f}s")
        print(f"  JIT time: {jax_results['jit_time']:.2f}s")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    numpy_time = numpy_results['mean']
    if jax_results:
        jax_time = jax_results['mean']
        speedup = numpy_time / jax_time

        print(f"\nSingle forward pass:")
        print(f"  NumPy: {numpy_time:.3f}s")
        print(f"  JAX:   {jax_time:.3f}s")
        print(f"  JAX speedup: {speedup:.1f}x")

        # Project optimization time (50 L-BFGS-B iterations, numerical gradients)
        # For 2 sources: 6 params, ~12 forward passes per gradient
        n_iters = 50
        n_fwd_per_iter = 12  # finite differences
        total_fwd = n_iters * n_fwd_per_iter

        numpy_opt_time = numpy_time * total_fwd
        jax_opt_time = jax_time * total_fwd

        print(f"\nProjected optimization (50 iters, numerical grad):")
        print(f"  NumPy: {numpy_opt_time:.1f}s per sample")
        print(f"  JAX:   {jax_opt_time:.1f}s per sample")

        # With parallelism
        n_cpus = 8  # G4dn.2xlarge
        n_samples = 400

        numpy_total_parallel = numpy_opt_time * n_samples / (n_cpus - 1)
        jax_total = jax_opt_time * n_samples  # No sample parallelism

        print(f"\nProjected total time for 400 samples:")
        print(f"  NumPy (7-way parallel): {numpy_total_parallel / 60:.1f} min")
        print(f"  JAX (sequential):       {jax_total / 60:.1f} min")

        target = 60  # minutes
        print(f"\nTarget: {target} min")
        if numpy_total_parallel / 60 < target:
            print(f"  NumPy: [OK]")
        else:
            print(f"  NumPy: Need {numpy_total_parallel / 60 / target:.1f}x speedup")

        if jax_total / 60 < target:
            print(f"  JAX: [OK]")
        else:
            print(f"  JAX: Need {jax_total / 60 / target:.1f}x speedup")


if __name__ == '__main__':
    main()
