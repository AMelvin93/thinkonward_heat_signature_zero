"""
Second NM Polish Round Optimizer

Extends the TabuBasinHoppingOptimizer with an additional NM polish round
on the final best candidate. This is efficient because it only polishes
the winner, not all candidates.

Based on successful nm4_perturb1 config (1.1585 @ 54.9 min with 5 min to spare).
"""

import os
import sys
import importlib.util

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Import the base optimizer using importlib to avoid circular import
base_optimizer_path = os.path.join(_project_root, 'experiments', 'hopping_with_tabu_memory', 'optimizer.py')
spec = importlib.util.spec_from_file_location("tabu_optimizer", base_optimizer_path)
tabu_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tabu_module)

TabuBasinHoppingOptimizer = tabu_module.TabuBasinHoppingOptimizer
compute_optimal_intensity_1src = tabu_module.compute_optimal_intensity_1src
compute_optimal_intensity_2src = tabu_module.compute_optimal_intensity_2src
filter_dissimilar = tabu_module.filter_dissimilar
TAU = tabu_module.TAU

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


class SecondPolishOptimizer(TabuBasinHoppingOptimizer):
    """
    Adds a second NM polish round on the best candidate after standard processing.

    Rationale: After CMA-ES + NM + perturbation, the best candidate is often
    close but not exactly at the local minimum. An additional focused polish
    can squeeze out extra accuracy with minimal overhead.
    """

    def __init__(
        self,
        second_polish_iterations: int = 4,  # Additional NM iterations on best
        **kwargs
    ):
        super().__init__(**kwargs)
        self.second_polish_iterations = second_polish_iterations

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        """Run standard optimization + second polish round on best candidate."""
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        # Create objective function for second polish (uses reduced timesteps)
        n_sims_extra = [0]
        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims_extra[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims_extra[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

        # Run standard optimization (CMA-ES + NM + perturbation)
        candidates, best_rmse, results, n_sims = super().estimate_sources(
            sample, meta, q_range, verbose
        )

        if not candidates:
            return candidates, best_rmse, results, n_sims

        # Find the best candidate
        best_idx = 0
        best_candidate_rmse = results[0].rmse
        for i, r in enumerate(results):
            if r.rmse < best_candidate_rmse:
                best_candidate_rmse = r.rmse
                best_idx = i

        # Extract position parameters from best candidate
        best_sources = candidates[best_idx]
        if n_sources == 1:
            pos_params = np.array([best_sources[0][0], best_sources[0][1]])
        else:
            pos_params = np.array([
                best_sources[0][0], best_sources[0][1],
                best_sources[1][0], best_sources[1][1]
            ])

        # Second polish round on best candidate only
        if self.second_polish_iterations > 0:
            result = minimize(
                objective_coarse,
                pos_params,
                method='Nelder-Mead',
                options={
                    'maxiter': self.second_polish_iterations,
                    'xatol': 0.005,  # Tighter tolerance for fine-tuning
                    'fatol': 0.0005,
                }
            )

            polished_pos = result.x

            # Evaluate on fine grid with full timesteps
            if n_sources == 1:
                x, y = polished_pos
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims += 1 + n_sims_extra[0]
                polished_sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = polished_pos
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims += 2 + n_sims_extra[0]
                polished_sources = [
                    (float(x1), float(y1), float(q1)),
                    (float(x2), float(y2), float(q2))
                ]

            # If polished result is better, update best candidate
            if final_rmse < best_rmse:
                candidates[best_idx] = polished_sources
                best_rmse = final_rmse
                results[best_idx].rmse = final_rmse
                results[best_idx].init_type = 'second_polish'

        return candidates, best_rmse, results, n_sims
