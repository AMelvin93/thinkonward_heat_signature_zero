"""
Run script for Perturbation + Verification experiment - Run 2.

PIVOT: Run 1 with 40% timestep was ~190 min projected (3.2x over budget).
This run uses 25% timestep fraction to fit within budget.
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime

import numpy as np

# Add project root
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from optimizer import PerturbationPlusVerificationOptimizer

# Add utils
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from utils import score_submission


def run_experiment(config, run_name="Run"):
    """Run experiment with given config."""
    # Load data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']
    n_samples = len(samples)

    print(f"\n{'='*60}")
    print(f"Perturbation + Verification Experiment - {run_name}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")

    # Create optimizer
    optimizer = PerturbationPlusVerificationOptimizer(**config)

    # Run on all samples
    all_predictions = []
    total_time = 0
    n_1src = 0
    n_2src = 0

    for i, sample in enumerate(samples):
        start_time = time.time()

        candidates, best_rmse, results, n_sims = optimizer.estimate_sources(
            sample, meta, verbose=(i == 0)
        )

        sample_time = time.time() - start_time
        total_time += sample_time

        if sample['n_sources'] == 1:
            n_1src += 1
        else:
            n_2src += 1

        all_predictions.append({
            'candidates': candidates,
            'best_rmse': best_rmse,
            'n_sims': n_sims
        })

        if (i + 1) % 10 == 0:
            elapsed = total_time
            projected = (elapsed / (i + 1)) * 400 / 60
            print(f"  Sample {i+1}/{n_samples}: RMSE={best_rmse:.4f}, "
                  f"n_cand={len(candidates)}, time={sample_time:.1f}s, "
                  f"projected_400={projected:.1f}min")

    # Prepare prediction dataset for scoring
    pred_dataset = {'samples': []}
    for i, (sample, pred) in enumerate(zip(samples, all_predictions)):
        pred_sample = {
            'sample_id': i,
            'candidates': pred['candidates']
        }
        pred_dataset['samples'].append(pred_sample)

    # Compute score
    score = score_submission(
        gt_dataset=test_data,
        pred_dataset=pred_dataset,
        N_max=3,
        lambda_=0.3,
        tau=0.2,
        scale_factors=(2.0, 1.0, 2.0),
        forward_loss='rmse',
        solver_kwargs={'Lx': 2.0, 'Ly': 1.0, 'nx': 100, 'ny': 50}
    )

    # Compute timing
    projected_400_min = (total_time / n_samples) * 400 / 60

    # Results
    results = {
        'run_name': run_name,
        'score': float(score),
        'total_time_sec': total_time,
        'total_time_min': total_time / 60,
        'projected_400_min': projected_400_min,
        'n_samples': n_samples,
        'n_1src': n_1src,
        'n_2src': n_2src,
        'config': config,
        'in_budget': projected_400_min <= 60,
        'timestamp': datetime.now().isoformat()
    }

    return results


def main():
    # Run 2: PIVOT from Run 1 - use 25% timestep to fit budget
    # Run 1 was ~190 min projected with 40% timestep
    # 25% timestep should bring this to ~50-55 min

    config = {
        # Perturbation (same as best perturbed_local_restart)
        "enable_perturbation": True,
        "perturb_top_n": 1,
        "n_perturbations": 2,
        "perturbation_scale": 0.05,
        "perturb_nm_iters": 3,
        # Verification (same as best solution_verification_pass)
        "enable_verification": True,
        "gradient_eps": 0.02,
        "gradient_threshold": 0.1,
        "step_size": 0.05,
        # Base CMA-ES params
        "max_fevals_1src": 20,
        "max_fevals_2src": 36,
        "timestep_fraction": 0.25,  # PIVOT: 25% instead of 40%
        "refine_maxiter": 8,
        "refine_top_n": 2,
        "sigma0_1src": 0.18,  # Higher sigma from best
        "sigma0_2src": 0.22,
    }

    print(f"\n{'#'*60}")
    print("RUN 2 - PIVOT: 25% timestep fraction")
    print("Reason: Run 1 @ 40% timestep was ~190 min projected (3.2x over budget)")
    print(f"{'#'*60}")

    results = run_experiment(config, "Run 2 (25% timestep)")

    print(f"\n--- Run 2 Results ---")
    print(f"Score: {results['score']:.4f}")
    print(f"Time: {results['total_time_min']:.1f} min")
    print(f"Projected 400: {results['projected_400_min']:.1f} min")
    print(f"In budget: {results['in_budget']}")

    # Save state
    state_path = os.path.join(os.path.dirname(__file__), 'STATE.json')

    # Load existing state if present
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
    except:
        state = {
            'experiment': 'perturbation_plus_verification',
            'all_results': [],
        }

    # Add Run 1 result (killed early, estimated)
    run1_result = {
        'run_name': 'Run 1 (40% timestep)',
        'score': None,
        'projected_400_min': 190.6,
        'in_budget': False,
        'note': 'KILLED EARLY - projected 190.6 min (3.2x over budget)',
        'config': {
            'timestep_fraction': 0.4,
            'enable_perturbation': True,
            'enable_verification': True,
            'sigma0_1src': 0.18,
            'sigma0_2src': 0.22,
        }
    }

    # Check if Run 1 already exists
    run1_exists = any(r.get('run_name') == 'Run 1 (40% timestep)' for r in state.get('all_results', []))
    if not run1_exists:
        state.setdefault('all_results', []).append(run1_result)

    state['all_results'].append(results)
    state['run_count'] = len(state['all_results'])

    # Find best results
    state['best_in_budget'] = None
    state['best_overall'] = None
    for r in state['all_results']:
        if r.get('score') is not None:
            if r.get('in_budget'):
                if state['best_in_budget'] is None or r['score'] > state['best_in_budget']['score']:
                    state['best_in_budget'] = r
            if state['best_overall'] is None or r['score'] > state['best_overall']['score']:
                state['best_overall'] = r

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    print(f"\nState saved to {state_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY SO FAR")
    print(f"{'='*60}")

    for r in state['all_results']:
        score_str = f"{r['score']:.4f}" if r.get('score') else "N/A"
        time_str = f"{r['projected_400_min']:.1f}" if r.get('projected_400_min') else "N/A"
        in_budget_str = "YES" if r.get('in_budget') else "NO"
        print(f"  {r.get('run_name', 'Unknown')}: Score={score_str}, Proj={time_str} min, In-budget={in_budget_str}")

    if state['best_in_budget']:
        print(f"\nBest in-budget: {state['best_in_budget'].get('run_name')} "
              f"Score={state['best_in_budget']['score']:.4f} @ {state['best_in_budget']['projected_400_min']:.1f} min")


if __name__ == '__main__':
    main()
