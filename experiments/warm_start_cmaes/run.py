"""
Run script for WS-CMA-ES experiment with MLflow logging.

Tests warm start CMA-ES using CyberAgentAILab's cmaes library.
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime

import numpy as np
import mlflow

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import WarmStartCMAESOptimizer

# Configuration
N_WORKERS = os.cpu_count()  # Use all CPUs for prototyping
DATA_PATH = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')


def run_experiment(
    n_probing_starts=3,
    probing_fevals=5,
    main_fevals_1src=15,
    main_fevals_2src=25,
    ws_gamma=0.1,
    ws_alpha=0.1,
    sigma0_1src=0.18,
    sigma0_2src=0.22,
    early_fraction=0.3,
    run_name_suffix="",
    verbose=True
):
    """Run the WS-CMA-ES experiment."""

    # Load test data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']
    n_samples = len(samples)

    if verbose:
        print(f"[W1] Running WS-CMA-ES experiment with {n_samples} samples, {N_WORKERS} workers")
        print(f"Config: probing_starts={n_probing_starts}, probing_fevals={probing_fevals}")
        print(f"        main_fevals={main_fevals_1src}/{main_fevals_2src}")
        print(f"        ws_gamma={ws_gamma}, ws_alpha={ws_alpha}")

    # Create optimizer
    optimizer = WarmStartCMAESOptimizer(
        n_probing_starts=n_probing_starts,
        probing_fevals=probing_fevals,
        main_fevals_1src=main_fevals_1src,
        main_fevals_2src=main_fevals_2src,
        ws_gamma=ws_gamma,
        ws_alpha=ws_alpha,
        sigma0_1src=sigma0_1src,
        sigma0_2src=sigma0_2src,
        early_fraction=early_fraction,
    )

    # Process samples
    from joblib import Parallel, delayed

    def process_sample(idx, sample):
        try:
            candidate_sources, best_rmse, results, n_sims = optimizer.estimate_sources(
                sample, meta, q_range=(0.5, 2.0), verbose=False
            )
            return idx, candidate_sources, best_rmse, n_sims, None
        except Exception as e:
            import traceback
            return idx, None, float('inf'), 0, str(e) + '\n' + traceback.format_exc()

    start_time = time.time()

    results_list = Parallel(n_jobs=N_WORKERS, verbose=10 if verbose else 0)(
        delayed(process_sample)(idx, sample)
        for idx, sample in enumerate(samples)
    )

    total_time = time.time() - start_time

    # Process results
    predictions = {}
    rmses = []
    errors = []
    n_sims_total = 0

    for idx, candidate_sources, best_rmse, n_sims, error in results_list:
        n_sims_total += n_sims
        if error:
            errors.append((idx, error))
            # Fallback prediction
            sample = samples[idx]
            sensors = sample['sensors_xy']
            n_src = sample['n_sources']
            avg_temps = np.mean(sample['Y_noisy'], axis=0)
            hot_idx = np.argsort(avg_temps)[::-1][:n_src]
            fallback_sources = [(float(sensors[i][0]), float(sensors[i][1]), 1.0) for i in hot_idx]
            predictions[idx] = [fallback_sources]
            rmses.append(1.0)  # Penalty for error
        else:
            predictions[idx] = candidate_sources if candidate_sources else [[(1.0, 0.5, 1.0)]]
            rmses.append(best_rmse)

    # Calculate metrics
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)
    rmse_max = np.max(rmses)

    # Count samples by type
    n_1src = sum(1 for s in samples if s['n_sources'] == 1)
    n_2src = len(samples) - n_1src
    rmses_1src = [rmses[i] for i, s in enumerate(samples) if s['n_sources'] == 1]
    rmses_2src = [rmses[i] for i, s in enumerate(samples) if s['n_sources'] == 2]
    rmse_mean_1src = np.mean(rmses_1src) if rmses_1src else 0
    rmse_mean_2src = np.mean(rmses_2src) if rmses_2src else 0

    # Calculate projected time for 400 samples
    time_per_sample = total_time / n_samples
    projected_400 = (total_time / n_samples) * 400 / 60  # minutes

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: WS-CMA-ES Experiment")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
        print(f"Projected 400 samples: {projected_400:.1f} min")
        print(f"RMSE mean: {rmse_mean:.4f} (+/- {rmse_std:.4f})")
        print(f"RMSE max: {rmse_max:.4f}")
        print(f"RMSE 1-src ({n_1src} samples): {rmse_mean_1src:.4f}")
        print(f"RMSE 2-src ({n_2src} samples): {rmse_mean_2src:.4f}")
        print(f"Total simulations: {n_sims_total}")
        print(f"Errors: {len(errors)}")
        if errors:
            print(f"Error details: {errors[0][1][:200]}...")
        print(f"{'='*60}")

    return {
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'rmse_max': rmse_max,
        'rmse_mean_1src': rmse_mean_1src,
        'rmse_mean_2src': rmse_mean_2src,
        'total_time_sec': total_time,
        'projected_400_min': projected_400,
        'n_sims_total': n_sims_total,
        'n_errors': len(errors),
        'predictions': predictions,
        'in_budget': projected_400 <= 60,
    }


def main():
    # Set up MLflow
    mlflow.set_tracking_uri(os.path.join(_project_root, 'mlruns'))
    mlflow.set_experiment("heat-signature-zero")

    # Configuration - Run 2: Reduce probing, increase main optimization
    config = {
        'n_probing_starts': 2,  # Reduced from 3
        'probing_fevals': 3,     # Reduced from 5
        'main_fevals_1src': 20,  # Increased from 15
        'main_fevals_2src': 32,  # Increased from 25
        'ws_gamma': 0.1,
        'ws_alpha': 0.1,
        'sigma0_1src': 0.18,
        'sigma0_2src': 0.22,
        'early_fraction': 0.3,
    }

    run_name = f"ws_cmaes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("experiment_id", "EXP_WS_CMAES_001")
        mlflow.log_param("worker", "W1")
        mlflow.log_param("optimizer", "WarmStartCMAES")
        mlflow.log_param("platform", "wsl")
        mlflow.log_param("n_workers", N_WORKERS)

        for key, val in config.items():
            mlflow.log_param(key, val)

        # Run experiment
        results = run_experiment(**config)

        # Log metrics
        mlflow.log_metric("projected_400_samples_min", results['projected_400_min'])
        mlflow.log_metric("rmse_mean", results['rmse_mean'])
        mlflow.log_metric("rmse_std", results['rmse_std'])
        mlflow.log_metric("rmse_max", results['rmse_max'])
        mlflow.log_metric("rmse_mean_1src", results['rmse_mean_1src'])
        mlflow.log_metric("rmse_mean_2src", results['rmse_mean_2src'])
        mlflow.log_metric("total_time_sec", results['total_time_sec'])
        mlflow.log_metric("n_sims_total", results['n_sims_total'])
        mlflow.log_metric("n_errors", results['n_errors'])
        mlflow.log_metric("in_budget", 1 if results['in_budget'] else 0)

        mlflow_run_id = run.info.run_id

        print(f"\nMLflow run ID: {mlflow_run_id}")
        print(f"In budget: {results['in_budget']}")

        # Update STATE.json
        state_path = os.path.join(os.path.dirname(__file__), 'STATE.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        tuning_run = {
            'run': len(state['tuning_runs']) + 1,
            'config': config,
            'rmse_mean': results['rmse_mean'],
            'time_min': results['projected_400_min'],
            'in_budget': results['in_budget'],
            'mlflow_id': mlflow_run_id,
            'rmse_mean_1src': results['rmse_mean_1src'],
            'rmse_mean_2src': results['rmse_mean_2src'],
            'timestamp': datetime.now().isoformat(),
        }
        state['tuning_runs'].append(tuning_run)

        # Update best in budget
        if results['in_budget']:
            if state['best_in_budget'] is None or results['rmse_mean'] < state['best_in_budget'].get('rmse_mean', float('inf')):
                state['best_in_budget'] = {
                    'run': tuning_run['run'],
                    'rmse_mean': results['rmse_mean'],
                    'time_min': results['projected_400_min'],
                    'config': config,
                }

        # Update best overall
        if state['best_overall'] is None or results['rmse_mean'] < state.get('best_overall', {}).get('rmse_mean', float('inf')):
            state['best_overall'] = {
                'run': tuning_run['run'],
                'rmse_mean': results['rmse_mean'],
                'time_min': results['projected_400_min'],
                'rmse_mean_1src': results['rmse_mean_1src'],
                'rmse_mean_2src': results['rmse_mean_2src'],
            }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"STATE.json updated with run {tuning_run['run']}")

    return results


if __name__ == '__main__':
    main()
