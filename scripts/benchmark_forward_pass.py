#!/usr/bin/env python
"""
Benchmark a single forward pass to understand the bottleneck.

This will help us understand:
1. How long a single forward simulation takes
2. Whether GPU is being utilized
3. Where the time is going
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


def check_gpu():
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    return devices


@jit
def thomas_solve(lower, diag, upper, rhs):
    n = diag.shape[0]
    def forward_step(carry, idx):
        c_prev, d_prev = carry
        denom = jnp.maximum(jnp.abs(diag[idx] - lower[idx] * c_prev), 1e-12)
        denom = jnp.sign(diag[idx] - lower[idx] * c_prev + 1e-12) * denom
        c_new = upper[idx] / denom
        d_new = (rhs[idx] - lower[idx] * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    denom0 = jnp.where(jnp.abs(diag[0]) < 1e-12, 1e-12, diag[0])
    c0, d0 = upper[0] / denom0, rhs[0] / denom0
    _, (c_arr, d_arr) = jax.lax.scan(forward_step, (c0, d0), jnp.arange(1, n))
    c_full = jnp.concatenate([jnp.array([c0]), c_arr])
    d_full = jnp.concatenate([jnp.array([d0]), d_arr])

    def backward_step(x_next, idx):
        x_curr = d_full[idx] - c_full[idx] * x_next
        return x_curr, x_curr
    _, x_rev = jax.lax.scan(backward_step, d_full[-1], jnp.arange(n-2, -1, -1))
    return jnp.concatenate([x_rev[::-1], jnp.array([d_full[-1]])])


def create_simulator(nt, bc, nx=100, ny=50):
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    @jit
    def simulate(sources, kappa, dt, sensors_xy, T0):
        rx = kappa * dt / (2.0 * dx * dx)
        ry = kappa * dt / (2.0 * dy * dy)
        r = kappa * dt / 2.0

        main_x = (1 + 2*rx) * jnp.ones(nx)
        lower_x = -rx * jnp.ones(nx)
        upper_x = -rx * jnp.ones(nx)
        main_y = (1 + 2*ry) * jnp.ones(ny)
        lower_y = -ry * jnp.ones(ny)
        upper_y = -ry * jnp.ones(ny)

        if bc == 'dirichlet':
            main_x = main_x.at[0].set(1.0).at[-1].set(1.0)
            lower_x = lower_x.at[0].set(0.0).at[-1].set(0.0)
            upper_x = upper_x.at[0].set(0.0).at[-1].set(0.0)
            main_y = main_y.at[0].set(1.0).at[-1].set(1.0)
            lower_y = lower_y.at[0].set(0.0).at[-1].set(0.0)
            upper_y = upper_y.at[0].set(0.0).at[-1].set(0.0)

        x = jnp.linspace(0, Lx, nx)
        y = jnp.linspace(0, Ly, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        sigma = 2.5 * max(dx, dy)
        S = jnp.zeros((nx, ny))
        def add_source(S, source):
            x0, y0, q = source
            r_sq = (X - x0)**2 + (Y - y0)**2
            G = jnp.exp(-r_sq / (2.0 * sigma**2))
            integral_G = jnp.maximum(jnp.sum(G) * dx * dy, 1e-10)
            return S + G * q / integral_G, None
        S, _ = jax.lax.scan(add_source, S, sources)

        def sample_sensors(U):
            def sample_one(sensor):
                sx, sy = sensor
                fx, fy = sx / dx, sy / dy
                ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
                iy = jnp.clip(jnp.floor(fy).astype(int), 0, ny - 2)
                tx, ty = fx - ix, fy - iy
                return ((1-tx)*(1-ty)*U[ix,iy] + tx*(1-ty)*U[ix+1,iy] +
                        (1-tx)*ty*U[ix,iy+1] + tx*ty*U[ix+1,iy+1])
            return vmap(sample_one)(sensors_xy)

        U = jnp.full((nx, ny), T0)

        def adi_step(U, _):
            S_half = S * dt / 2.0
            if bc == 'dirichlet':
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), constant_values=0.0)
            else:
                U_pad_y = jnp.pad(U, ((0,0),(1,1)), mode='edge')
            Lyy = (U_pad_y[:,2:] - 2*U + U_pad_y[:,:-2]) / (dy**2)
            RHS_x = U + r * Lyy + S_half
            if bc == 'dirichlet':
                RHS_x = RHS_x.at[0,:].set(0.0).at[-1,:].set(0.0)
            U_star = vmap(lambda col: thomas_solve(lower_x, main_x, upper_x, col),
                         in_axes=1, out_axes=1)(RHS_x)

            if bc == 'dirichlet':
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), constant_values=0.0)
            else:
                U_pad_x = jnp.pad(U_star, ((1,1),(0,0)), mode='edge')
            Lxx = (U_pad_x[2:,:] - 2*U_star + U_pad_x[:-2,:]) / (dx**2)
            RHS_y = U_star + r * Lxx + S_half
            if bc == 'dirichlet':
                RHS_y = RHS_y.at[:,0].set(0.0).at[:,-1].set(0.0)
            U_next = vmap(lambda row: thomas_solve(lower_y, main_y, upper_y, row),
                         in_axes=0, out_axes=0)(RHS_y)

            return U_next, sample_sensors(U_next)

        Y0 = sample_sensors(U)
        _, Y_rest = jax.lax.scan(adi_step, U, None, length=nt)
        return jnp.concatenate([Y0[None,:], Y_rest], axis=0)

    return simulate


def main():
    print("=" * 60)
    print("FORWARD PASS BENCHMARK")
    print("=" * 60)

    check_gpu()

    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'heat-signature-zero-test-data.pkl'
    )
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    sample = samples[0]

    nt = sample['sample_metadata']['nt']
    bc = sample['sample_metadata']['bc']
    kappa = sample['sample_metadata']['kappa']
    T0 = sample['sample_metadata']['T0']
    dt = meta['dt']

    print(f"\nSample 0: nt={nt}, bc={bc}")

    sensors_xy = jnp.array(sample['sensors_xy'])
    sources = jnp.array([[1.0, 0.5, 1.0]])  # Single source

    # Create simulator
    print("\nCreating simulator (includes JIT compilation)...")
    start = time.time()
    simulate = create_simulator(nt, bc)
    print(f"  Simulator created: {time.time() - start:.2f}s")

    # First call (JIT compilation)
    print("\nFirst forward pass (JIT compilation)...")
    start = time.time()
    Y_pred = simulate(sources, kappa, dt, sensors_xy, T0)
    Y_pred.block_until_ready()  # Wait for GPU
    first_time = time.time() - start
    print(f"  First pass: {first_time:.2f}s")

    # Subsequent calls (cached)
    print("\nSubsequent forward passes (cached)...")
    times = []
    for i in range(10):
        start = time.time()
        Y_pred = simulate(sources, kappa, dt, sensors_xy, T0)
        Y_pred.block_until_ready()
        times.append(time.time() - start)

    avg_time = np.mean(times)
    print(f"  Avg time (10 runs): {avg_time:.3f}s")
    print(f"  Min/Max: {min(times):.3f}s / {max(times):.3f}s")

    # Extrapolate
    print("\n" + "=" * 60)
    print("EXTRAPOLATION")
    print("=" * 60)
    print(f"Single forward pass: {avg_time:.3f}s")
    print(f"With 50 Adam steps (gradient = 1 fwd + 1 bwd ≈ 2 fwd):")
    print(f"  Estimated time: {avg_time * 50 * 2:.1f}s per sample")
    print(f"  For 400 samples: {avg_time * 50 * 2 * 400 / 60:.1f} min")

    # What we need
    target_per_sample = 9  # seconds
    target_fwd_passes = target_per_sample / avg_time
    print(f"\nTo hit 9s/sample target:")
    print(f"  Max forward passes: {target_fwd_passes:.0f}")
    print(f"  Max Adam steps (with grad): {target_fwd_passes / 2:.0f}")


if __name__ == '__main__':
    main()
