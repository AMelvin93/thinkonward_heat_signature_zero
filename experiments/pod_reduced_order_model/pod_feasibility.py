"""
POD Feasibility Check for Heat Source Identification

Goal: Determine if POD can capture thermal fields accurately with few modes.
If reconstruction error is high, the POD approach is not viable.
"""

import numpy as np
import pickle
from pathlib import Path
import os
import sys

# Add paths for imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))

from simulator import Heat2D

def generate_random_sources(n_samples, n_sources, q_range=(0.5, 2.0), margin=0.1):
    """Generate random source configurations."""
    Lx, Ly = 2.0, 1.0
    sources_list = []
    for _ in range(n_samples):
        sources = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * Lx, (1 - margin) * Lx)
            y = np.random.uniform(margin * Ly, (1 - margin) * Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            sources.append((x, y, q))
        sources_list.append(sources)
    return sources_list

def run_simulation(sources, kappa=0.05, dt=0.004, nt=500, T0=0.0, bc='neumann'):
    """Run a heat simulation and return the final temperature field."""
    solver = Heat2D(Lx=2.0, Ly=1.0, nx=100, ny=50, kappa=kappa, bc=bc)
    sources_dict = [{'x': s[0], 'y': s[1], 'q': s[2]} for s in sources]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources_dict)
    return Us[-1]  # Return final temperature field

def collect_snapshots(n_samples, n_sources):
    """Collect temperature field snapshots for POD analysis."""
    print(f"Collecting {n_samples} snapshots with {n_sources} sources each...")
    sources_list = generate_random_sources(n_samples, n_sources)

    snapshots = []
    for i, sources in enumerate(sources_list):
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_samples}...")
        U_final = run_simulation(sources)
        snapshots.append(U_final.flatten())

    return np.array(snapshots).T  # Shape: (nx*ny, n_samples)

def analyze_pod(snapshot_matrix, max_modes=50):
    """Perform POD analysis and compute reconstruction errors."""
    print(f"\nSnapshot matrix shape: {snapshot_matrix.shape}")

    # Center the data
    mean_snapshot = snapshot_matrix.mean(axis=1, keepdims=True)
    centered = snapshot_matrix - mean_snapshot

    # Compute SVD
    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Analyze singular values
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy

    print("\nSingular value analysis:")
    for n_modes in [1, 2, 5, 10, 20, 50]:
        if n_modes <= len(S):
            print(f"  {n_modes} modes: {cumulative_energy[n_modes-1]*100:.2f}% energy captured")

    # Compute reconstruction errors for different numbers of modes
    print("\nReconstruction errors (normalized RMSE):")
    n_test = min(20, snapshot_matrix.shape[1])  # Test on subset

    for n_modes in [1, 2, 5, 10, 20, 50]:
        if n_modes > len(S):
            continue

        # Reconstruct using n_modes
        U_k = U[:, :n_modes]
        S_k = S[:n_modes]
        Vt_k = Vt[:n_modes, :]

        reconstructed = U_k @ np.diag(S_k) @ Vt_k + mean_snapshot

        # Compute RMSE for each sample
        errors = []
        for i in range(n_test):
            orig = snapshot_matrix[:, i]
            recon = reconstructed[:, i]
            rmse = np.sqrt(np.mean((orig - recon)**2))
            norm_rmse = rmse / (np.max(orig) - np.min(orig) + 1e-10)
            errors.append(norm_rmse)

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"  {n_modes:2d} modes: mean={mean_error*100:.3f}%, max={max_error*100:.3f}%")

    return U, S, Vt, mean_snapshot

def main():
    print("="*60)
    print("POD Feasibility Check for Heat Source Identification")
    print("="*60)

    np.random.seed(42)

    # Test 1: 1-source configurations
    print("\n--- Test 1: 1-source configurations ---")
    snapshots_1src = collect_snapshots(n_samples=100, n_sources=1)
    analyze_pod(snapshots_1src)

    # Test 2: 2-source configurations
    print("\n--- Test 2: 2-source configurations ---")
    snapshots_2src = collect_snapshots(n_samples=100, n_sources=2)
    analyze_pod(snapshots_2src)

    # Test 3: Mixed configurations
    print("\n--- Test 3: Mixed 1+2 source configurations ---")
    mixed = np.hstack([snapshots_1src[:, :50], snapshots_2src[:, :50]])
    analyze_pod(mixed)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nInterpretation:")
    print("- If 5-10 modes capture >95% energy with <5% reconstruction error,")
    print("  POD is viable for this problem.")
    print("- If many modes are needed or errors are high, POD won't help.")

if __name__ == "__main__":
    main()
