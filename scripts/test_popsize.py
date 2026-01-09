#!/usr/bin/env python
"""Quick test of CMA-ES population size effect on score."""

import sys
import os
import time
import pickle
from pathlib import Path
from copy import deepcopy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import numpy as np
import cma
from joblib import Parallel, delayed

from simulator import Heat2D
from src.triangulation import triangulation_init


# Domain params
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50

# Competition params
N_MAX = 3
LAMBDA = 0.3


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range):
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()
    denom = np.dot(Y_unit_flat, Y_unit_flat)
    if denom < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denom
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range):
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    Y1_flat, Y2_flat = Y1.flatten(), Y2.flatten()
    Y_obs_flat = Y_observed.flatten()
    A = np.array([
        [np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
        [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]
    ])
    b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])
    try:
        q1, q2 = np.linalg.solve(A + 1e-6 * np.eye(2), b)
    except:
        q1, q2 = 1.0, 1.0
    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])
    Y_pred = q1 * Y1 + q2 * Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return (q1, q2), Y_pred, rmse


def process_sample(sample, meta, max_fevals, popsize=None):
    n_sources = sample['n_sources']
    sensors_xy = np.array(sample['sensors_xy'])
    Y_observed = sample['Y_noisy']

    dt = meta['dt']
    nt = sample['sample_metadata']['nt']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']
    q_range = tuple(meta['q_range'])

    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    # Get triangulation init
    try:
        full_init = triangulation_init(sample, meta, n_sources, q_range, Lx, Ly)
        positions = []
        for i in range(n_sources):
            positions.extend([full_init[i*3], full_init[i*3 + 1]])
        init_params = np.array(positions)
    except:
        # Smart init fallback
        readings = sample['Y_noisy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1][:n_sources]
        params = []
        for idx in hot_idx:
            x, y = sensors_xy[idx]
            params.extend([x, y])
        init_params = np.array(params)

    # Bounds
    margin = 0.05
    lb, ub = [], []
    for _ in range(n_sources):
        lb.extend([margin * Lx, margin * Ly])
        ub.extend([(1 - margin) * Lx, (1 - margin) * Ly])

    sigma0 = 0.15 if n_sources == 1 else 0.20

    # Objective
    if n_sources == 1:
        def objective(xy_params):
            x, y = xy_params
            q, _, rmse = compute_optimal_intensity_1src(
                x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            return rmse
    else:
        def objective(xy_params):
            x1, y1, x2, y2 = xy_params
            (q1, q2), _, rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
            return rmse

    # CMA-ES options
    opts = {
        'maxfevals': max_fevals,
        'bounds': [lb, ub],
        'verbose': -9,
        'tolfun': 1e-6,
        'tolx': 1e-6,
    }
    if popsize is not None:
        opts['popsize'] = popsize

    start = time.time()
    es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        fitness = [objective(s) for s in solutions]
        es.tell(solutions, fitness)
    elapsed = time.time() - start

    best_pos = np.array(es.result.xbest)
    if n_sources == 1:
        x, y = best_pos
        q, _, rmse = compute_optimal_intensity_1src(
            x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
    else:
        x1, y1, x2, y2 = best_pos
        (q1, q2), _, rmse = compute_optimal_intensity_2src(
            x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)

    return {
        'sample_id': sample['sample_id'],
        'n_sources': n_sources,
        'rmse': rmse,
        'time': elapsed,
        'popsize_used': es.opts['popsize'],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--popsize', type=int, default=None)
    parser.add_argument('--fevals-1src', type=int, default=12)
    parser.add_argument('--fevals-2src', type=int, default=23)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    data_path = project_root / "data" / "heat-signature-zero-test-data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = list(data['samples'])
    meta = data['meta']

    if args.max_samples:
        samples = samples[:args.max_samples]

    if args.shuffle:
        np.random.seed(42)
        np.random.shuffle(samples)

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print("=" * 60)
    print("CMA-ES POPULATION SIZE TEST")
    print("=" * 60)
    print(f"Samples: {n_samples} ({n_1src} 1-src, {n_2src} 2-src)")
    print(f"Workers: {args.workers}")
    print(f"Popsize: {args.popsize if args.popsize else 'default'}")
    print(f"1-src fevals: {args.fevals_1src}, 2-src fevals: {args.fevals_2src}")
    print("=" * 60)

    start = time.time()
    results = Parallel(n_jobs=args.workers, verbose=0)(
        delayed(process_sample)(
            s, meta,
            args.fevals_1src if s['n_sources'] == 1 else args.fevals_2src,
            args.popsize
        )
        for s in samples
    )
    total_time = time.time() - start

    rmses = [r['rmse'] for r in results]
    rmse_1src = [r['rmse'] for r in results if r['n_sources'] == 1]
    rmse_2src = [r['rmse'] for r in results if r['n_sources'] == 2]

    # Estimate score (simplified)
    scores = [1.0 / (1.0 + r) + LAMBDA * (1/3) for r in rmses]  # Assume 1 candidate
    avg_score = np.mean(scores)

    proj_400 = (total_time / n_samples) * 400 / 60

    print(f"\nResults:")
    print(f"  RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  1-src RMSE: {np.mean(rmse_1src):.4f}")
    print(f"  2-src RMSE: {np.mean(rmse_2src):.4f}")
    print(f"  Approx Score: {avg_score:.4f}")
    print(f"  Projected (400): {proj_400:.1f} min")
    print(f"  Actual popsize: {results[0]['popsize_used']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
