"""
Higher Population Density Experiment

Test whether increasing CMA-ES population size (2x default) improves candidate diversity.

Usage:
    python run.py [--workers N] [--popsize-mult M]
"""

import os
import sys
import pickle
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cma
import mlflow
from scipy.optimize import minimize

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.triangulation import triangulation_init

sys.path.insert(0, str(project_root / 'data' / 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D

N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


def normalize_sources(sources):
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1, sources2):
    norm1 = normalize_sources(sources1)
    norm2 = normalize_sources(sources2)
    n = len(sources1)
    if n != len(sources2):
        return float('inf')
    if n == 1:
        return np.linalg.norm(norm1[0] - norm2[0])
    from itertools import permutations
    min_total = float('inf')
    for perm in permutations(range(n)):
        total = sum(np.linalg.norm(norm1[i] - norm2[j])**2 for i, j in enumerate(perm))
        min_total = min(min_total, np.sqrt(total / n))
    return min_total


def filter_dissimilar(candidates, tau=TAU, n_max=N_MAX):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[1])
    kept = [candidates[0]]
    for cand in candidates[1:]:
        if all(candidate_distance(cand[0], k[0]) >= tau for k in kept):
            kept.append(cand)
            if len(kept) >= n_max:
                break
    return kept


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range=(0.5, 2.0)):
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    n_steps = len(Y_unit)
    Y_obs_trunc = Y_observed[:n_steps]
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range=(0.5, 2.0)):
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    n_steps = len(Y1)
    Y_obs_trunc = Y_observed[:n_steps]
    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
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
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    return (q1, q2), Y_pred, rmse


def process_single_sample(args):
    """Process a single sample with higher population CMA-ES."""
    idx, sample, meta, config = args

    Lx, Ly = 2.0, 1.0
    nx_fine, ny_fine = 100, 50
    nx_coarse, ny_coarse = 50, 25

    max_fevals_1src = config.get('max_fevals_1src', 20)
    max_fevals_2src = config.get('max_fevals_2src', 36)
    sigma0_1src = config.get('sigma0_1src', 0.15)
    sigma0_2src = config.get('sigma0_2src', 0.20)
    timestep_fraction = config.get('timestep_fraction', 0.25)
    refine_maxiter = config.get('refine_maxiter', 3)
    refine_top_n = config.get('refine_top_n', 2)
    candidate_pool_size = config.get('candidate_pool_size', 10)
    popsize_mult = config.get('popsize_mult', 2)  # Population multiplier
    q_range = (0.5, 2.0)

    n_sources = sample['n_sources']
    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    nt_full = sample['sample_metadata']['nt']
    dt = meta['dt']
    T0 = sample['sample_metadata']['T0']
    sensors_xy = np.array(sample['sensors_xy'])
    Y_observed = sample['Y_noisy']

    nt_reduced = max(10, int(nt_full * timestep_fraction))

    solver_coarse = Heat2D(Lx, Ly, nx_coarse, ny_coarse, kappa, bc=bc)
    solver_fine = Heat2D(Lx, Ly, nx_fine, ny_fine, kappa, bc=bc)

    start = time.time()
    n_sims = [0]

    try:
        # Build initializations
        initializations = []

        # Triangulation init
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, Lx, Ly)
            positions = [full_init[i*3:i*3+2] for i in range(n_sources)]
            tri_init = np.array([p for pos in positions for p in pos])
            initializations.append((tri_init, 'triangulation'))
        except:
            pass

        # Smart init
        readings = sample['Y_noisy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]
        selected = []
        for i in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors_xy[i] - sensors_xy[p]) >= 0.25 for p in selected):
                selected.append(i)
        while len(selected) < n_sources:
            for i in hot_idx:
                if i not in selected:
                    selected.append(i)
                    break
        smart_init = np.array([sensors_xy[i][j] for i in selected for j in range(2)])
        initializations.append((smart_init, 'smart'))

        # Get position bounds
        margin = 0.05
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * Lx, margin * Ly])
            ub.extend([(1 - margin) * Lx, (1 - margin) * Ly])

        # CMA-ES objective
        if n_sources == 1:
            def objective(xy):
                x, y = xy
                n_sims[0] += 1
                q, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy):
                x1, y1, x2, y2 = xy
                n_sims[0] += 2
                (q1, q2), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

        max_fevals = max_fevals_1src if n_sources == 1 else max_fevals_2src
        sigma0 = sigma0_1src if n_sources == 1 else sigma0_2src
        fevals_per_init = max(5, max_fevals // len(initializations))

        # Calculate population size (2x default)
        dim = 2 * n_sources
        default_popsize = 4 + int(3 * np.log(dim))
        popsize = popsize_mult * default_popsize

        all_solutions = []

        for init_params, init_type in initializations:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6
            opts['popsize'] = popsize  # KEY CHANGE: Set higher population

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)
                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit, init_type))

        all_solutions.sort(key=lambda x: x[1])

        # Refine top N
        refined = []
        for pos, rmse, init_type in all_solutions[:refine_top_n]:
            if refine_maxiter > 0:
                result = minimize(objective, pos, method='Nelder-Mead',
                                options={'maxiter': refine_maxiter, 'xatol': 0.01, 'fatol': 0.001})
                if result.fun < rmse:
                    refined.append((result.x, result.fun, 'refined'))
                else:
                    refined.append((pos, rmse, init_type))
            else:
                refined.append((pos, rmse, init_type))

        for pos, rmse, init_type in all_solutions[refine_top_n:candidate_pool_size]:
            refined.append((pos, rmse, init_type))

        # Final evaluation on fine grid
        candidates_raw = []
        for pos, rmse_coarse, init_type in refined:
            if n_sources == 1:
                x, y = pos
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2
                sources = [(float(x1), float(y1), float(q1)), (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, final_rmse))

        # Dissimilarity filter
        filtered = filter_dissimilar(candidates_raw)

        elapsed = time.time() - start
        best_rmse = min(f[1] for f in filtered) if filtered else float('inf')

        return {
            'idx': idx,
            'candidates': [f[0] for f in filtered],
            'best_rmse': best_rmse,
            'n_sources': n_sources,
            'n_candidates': len(filtered),
            'n_sims': n_sims[0],
            'elapsed': elapsed,
            'success': True,
            'popsize_used': popsize,
        }

    except Exception as e:
        import traceback
        return {
            'idx': idx,
            'candidates': [],
            'best_rmse': float('inf'),
            'n_sources': sample.get('n_sources', 0),
            'n_candidates': 0,
            'n_sims': n_sims[0],
            'elapsed': time.time() - start,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def calculate_sample_score(rmse, n_candidates, lambda_=0.3, n_max=3):
    if n_candidates == 0:
        return 0.0
    return 1.0 / (1.0 + rmse) + lambda_ * (n_candidates / n_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--popsize-mult', type=int, default=2, help='Population size multiplier')
    parser.add_argument('--no-mlflow', action='store_true')
    args = parser.parse_args()

    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    n_samples = len(samples)
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = n_samples - n_1src

    print(f"\n{'='*60}")
    print(f"HIGHER POPULATION DENSITY EXPERIMENT")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} ({n_1src} 1-source, {n_2src} 2-source)")
    print(f"Workers: {args.workers}")
    print(f"Population multiplier: {args.popsize_mult}x")
    print(f"{'='*60}")

    config = {
        'max_fevals_1src': 20,
        'max_fevals_2src': 36,
        'timestep_fraction': 0.25,
        'refine_maxiter': 3,
        'refine_top_n': 2,
        'candidate_pool_size': 10,
        'popsize_mult': args.popsize_mult,
    }

    start_time = time.time()
    results = []

    work_items = [(i, samples[i], meta, config) for i in range(n_samples)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['success']:
                print(f"  Sample {result['idx']:3d}: RMSE={result['best_rmse']:.4f}, "
                      f"n_cand={result['n_candidates']}, "
                      f"popsize={result.get('popsize_used', 'N/A')}, "
                      f"time={result['elapsed']:.1f}s")
            else:
                print(f"  Sample {result['idx']:3d}: FAILED - {result.get('error', 'unknown')}")

    total_time = time.time() - start_time

    sample_scores = [calculate_sample_score(r['best_rmse'], r['n_candidates']) for r in results]
    score = np.mean(sample_scores)

    rmses = [r['best_rmse'] for r in results if r['success']]
    rmses_1src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 1]
    rmses_2src = [r['best_rmse'] for r in results if r['success'] and r['n_sources'] == 2]

    projected_400 = (total_time / n_samples) * 400 / 60

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Score: {score:.4f}")
    print(f"RMSE mean: {np.mean(rmses):.4f}")
    print(f"RMSE mean 1-src: {np.mean(rmses_1src):.4f}")
    print(f"RMSE mean 2-src: {np.mean(rmses_2src):.4f}")
    print(f"Projected 400-sample time: {projected_400:.1f} min")
    print(f"{'='*70}\n")

    summary = {
        'popsize_mult': args.popsize_mult,
        'score': score,
        'projected_400_min': projected_400,
        'rmse_mean': np.mean(rmses),
        'rmse_mean_1src': np.mean(rmses_1src),
        'rmse_mean_2src': np.mean(rmses_2src),
    }

    if not args.no_mlflow:
        mlflow.set_tracking_uri(str(project_root / 'mlruns'))
        mlflow.set_experiment('heat-signature-zero')

        run_name = f"higher_population_density_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param('experiment_name', 'higher_population_density')
            mlflow.log_param('experiment_id', 'EXP_HIGHER_POPULATION_DENSITY_001')
            mlflow.log_param('worker', 'W1')
            mlflow.log_param('popsize_mult', args.popsize_mult)
            mlflow.log_param('n_workers', args.workers)
            mlflow.log_param('platform', 'wsl')

            mlflow.log_metric('score', score)
            mlflow.log_metric('rmse_mean', np.mean(rmses))
            mlflow.log_metric('rmse_mean_1src', np.mean(rmses_1src))
            mlflow.log_metric('rmse_mean_2src', np.mean(rmses_2src))
            mlflow.log_metric('projected_400_min', projected_400)

            mlflow_run_id = run.info.run_id
            print(f"MLflow run ID: {mlflow_run_id}")
            summary['mlflow_run_id'] = mlflow_run_id

    state_path = Path(__file__).parent / 'STATE.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)

        state['tuning_runs'].append({
            'run': len(state['tuning_runs']) + 1,
            'config': {'popsize_mult': args.popsize_mult, 'n_workers': args.workers},
            'results': summary
        })

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    return summary


if __name__ == '__main__':
    main()
