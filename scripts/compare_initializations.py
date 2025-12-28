#!/usr/bin/env python
"""
Compare triangulation initialization vs hottest-sensor (smart) initialization.

Measures how good each initialization is by:
1. Computing RMSE of simulated output with initial guess vs observed data
2. Lower initial RMSE = better starting point for optimization
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
from typing import Dict, List, Tuple

from src.triangulation import triangulation_init


# Import simulator
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))
from simulator import Heat2D


def hottest_sensor_init(
    sample: Dict,
    meta: Dict,
    n_sources: int,
    q_range: Tuple[float, float] = (0.5, 2.0),
    min_separation: float = 0.3,
) -> np.ndarray:
    """
    Simple initialization: place sources at hottest sensors.
    This is what the current HybridOptimizer uses.
    """
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

    # Fill if needed
    for idx in hot_indices:
        if len(selected) >= n_sources:
            break
        if idx not in selected:
            selected.append(idx)

    params = []
    max_temp = np.max(avg_temps) + 1e-8
    for idx in selected:
        x, y = sensors[idx]
        q = 0.5 + (avg_temps[idx] / max_temp) * 1.5
        q = np.clip(q, q_range[0], q_range[1])
        params.extend([x, y, q])

    return np.array(params)


def evaluate_init(
    params: np.ndarray,
    sample: Dict,
    meta: Dict,
    Lx: float = 2.0,
    Ly: float = 1.0,
    nx: int = 100,
    ny: int = 50,
) -> float:
    """
    Evaluate an initialization by simulating and computing RMSE.
    """
    sample_meta = sample['sample_metadata']
    Y_obs = sample['Y_noisy']
    sensors_xy = sample['sensors_xy']

    dt = meta['dt']
    nt = sample_meta['nt']
    kappa = sample_meta['kappa']
    bc = sample_meta['bc']
    T0 = sample_meta['T0']

    n_sources = sample['n_sources']

    # Create simulator
    sim = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    # Build sources
    sources = []
    for i in range(n_sources):
        x, y, q = params[i*3:(i+1)*3]
        sources.append({'x': float(x), 'y': float(y), 'q': float(q)})

    # Simulate
    try:
        times, T_history = sim.solve(
            dt=dt,
            nt=nt,
            T0=T0,
            sources=sources,
            store_every=1,
        )

        # Sample at sensor locations
        Y_sim = np.zeros((len(T_history), len(sensors_xy)))
        for t in range(len(T_history)):
            Y_sim[t, :] = sim.sample_sensors(T_history[t], sensors_xy)

        # Compute RMSE
        rmse = np.sqrt(np.mean((Y_sim - Y_obs) ** 2))
        return rmse

    except Exception as e:
        print(f"Simulation error: {e}")
        return float('inf')


def main():
    # Load data
    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    q_range = tuple(meta['q_range'])

    n_samples = min(20, len(samples))  # Test on first 20 samples

    print("Comparing initialization methods...")
    print("=" * 70)
    print(f"{'Sample':<12} {'N_src':>5} {'Hottest RMSE':>14} {'Triang RMSE':>14} {'Winner':>10}")
    print("-" * 70)

    hottest_rmses = []
    triang_rmses = []
    wins = {'hottest': 0, 'triangulation': 0, 'tie': 0}

    for i, sample in enumerate(samples[:n_samples]):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        # Get initializations
        hottest_params = hottest_sensor_init(sample, meta, n_sources, q_range)
        triang_params = triangulation_init(sample, meta, n_sources, q_range)

        # Evaluate
        hottest_rmse = evaluate_init(hottest_params, sample, meta)
        triang_rmse = evaluate_init(triang_params, sample, meta)

        hottest_rmses.append(hottest_rmse)
        triang_rmses.append(triang_rmse)

        # Determine winner
        if abs(hottest_rmse - triang_rmse) < 0.01:
            winner = "tie"
        elif triang_rmse < hottest_rmse:
            winner = "triang"
            wins['triangulation'] += 1
        else:
            winner = "hottest"
            wins['hottest'] += 1

        if winner == 'tie':
            wins['tie'] += 1

        print(f"{sample_id:<12} {n_sources:>5} {hottest_rmse:>14.4f} {triang_rmse:>14.4f} {winner:>10}")

    print("-" * 70)
    print(f"{'AVERAGE':<12} {'':<5} {np.mean(hottest_rmses):>14.4f} {np.mean(triang_rmses):>14.4f}")
    print(f"{'STD':<12} {'':<5} {np.std(hottest_rmses):>14.4f} {np.std(triang_rmses):>14.4f}")
    print("=" * 70)

    print(f"\nWins: Hottest={wins['hottest']}, Triangulation={wins['triangulation']}, Tie={wins['tie']}")

    improvement = (np.mean(hottest_rmses) - np.mean(triang_rmses)) / np.mean(hottest_rmses) * 100
    print(f"Average improvement from triangulation: {improvement:.1f}%")

    if improvement > 0:
        print("\n>>> Triangulation provides better starting points on average!")
    else:
        print("\n>>> Hottest sensor approach is still better on average.")


if __name__ == "__main__":
    main()
