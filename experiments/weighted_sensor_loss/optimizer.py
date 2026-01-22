"""
Weighted Sensor Loss Optimizer - Weights Sensors by Informativeness

Hypothesis: Not all sensors are equally informative. Sensors with higher temperature
variance are likely closer to heat sources and provide stronger signal. Weighting
sensors by informativeness may improve CMA-ES convergence.

Weighting strategies:
1. variance: Weight by temporal variance of sensor readings
2. mean_temp: Weight by mean temperature (higher = closer to source)
3. gradient: Weight by temperature gradient magnitude
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
import cma
from scipy.optimize import minimize

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


@dataclass
class CandidateResult:
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


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


def compute_sensor_weights(Y_observed, strategy='variance', normalize=True):
    """
    Compute weights for each sensor based on observed data.

    Args:
        Y_observed: Shape (n_timesteps, n_sensors)
        strategy: 'variance', 'mean_temp', 'gradient', or 'uniform'
        normalize: If True, weights sum to number of sensors

    Returns:
        weights: Shape (n_sensors,)
    """
    n_timesteps, n_sensors = Y_observed.shape

    if strategy == 'uniform':
        weights = np.ones(n_sensors)

    elif strategy == 'variance':
        # Higher variance = more informative
        weights = np.var(Y_observed, axis=0)
        # Avoid zero weights
        weights = weights + 1e-6

    elif strategy == 'mean_temp':
        # Higher mean temperature = closer to source
        weights = np.mean(Y_observed, axis=0)
        weights = weights - np.min(weights) + 0.1  # Shift to positive

    elif strategy == 'gradient':
        # Higher gradient magnitude = more informative
        gradients = np.diff(Y_observed, axis=0)
        weights = np.mean(np.abs(gradients), axis=0)
        weights = weights + 1e-6

    elif strategy == 'sqrt_variance':
        # Sqrt of variance (less extreme weighting)
        weights = np.sqrt(np.var(Y_observed, axis=0) + 1e-6)

    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    if normalize:
        # Normalize so weights sum to n_sensors (equivalent to uniform on average)
        weights = weights * (n_sensors / np.sum(weights))

    return weights


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    """Run simulation and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_weighted_rmse(Y_pred, Y_obs, weights):
    """
    Compute weighted RMSE across sensors.

    Args:
        Y_pred: Shape (n_timesteps, n_sensors)
        Y_obs: Shape (n_timesteps, n_sensors)
        weights: Shape (n_sensors,)

    Returns:
        weighted_rmse: Scalar
    """
    # Squared errors per timestep per sensor
    sq_errors = (Y_pred - Y_obs) ** 2

    # Weight each sensor's errors
    # Shape: (n_timesteps, n_sensors) * (n_sensors,) -> (n_timesteps, n_sensors)
    weighted_sq_errors = sq_errors * weights

    # Mean over all timesteps and sensors
    weighted_mse = np.mean(weighted_sq_errors)

    return np.sqrt(weighted_mse)


def compute_optimal_intensity_1src_weighted(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                            weights, q_range=(0.5, 2.0)):
    """Compute optimal intensity for 1-source using weighted loss."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)

    n_steps = len(Y_unit)
    Y_obs_trunc = Y_observed[:n_steps]

    # For optimal q, we want to minimize sum_t sum_s w_s * (q*Y_unit - Y_obs)^2
    # Derivative: 2 * sum_t sum_s w_s * (q*Y_unit - Y_obs) * Y_unit = 0
    # q_opt = (sum_t sum_s w_s * Y_unit * Y_obs) / (sum_t sum_s w_s * Y_unit^2)

    # Weight matrix matching Y_unit shape
    w_matrix = np.tile(weights, (n_steps, 1))  # (n_steps, n_sensors)

    numerator = np.sum(w_matrix * Y_unit * Y_obs_trunc)
    denominator = np.sum(w_matrix * Y_unit ** 2)

    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = numerator / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])

    Y_pred = q_optimal * Y_unit
    rmse = compute_weighted_rmse(Y_pred, Y_obs_trunc, weights)

    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src_weighted(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0,
                                            sensors_xy, weights, q_range=(0.5, 2.0)):
    """Compute optimal intensities for 2-source using weighted loss."""
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)

    n_steps = len(Y1)
    Y_obs_trunc = Y_observed[:n_steps]

    # Weight matrix
    w_matrix = np.tile(weights, (n_steps, 1))  # (n_steps, n_sensors)

    # Weighted least squares: minimize sum_t sum_s w_s * (q1*Y1 + q2*Y2 - Y_obs)^2
    # Normal equations with weights
    A = np.array([
        [np.sum(w_matrix * Y1 ** 2), np.sum(w_matrix * Y1 * Y2)],
        [np.sum(w_matrix * Y1 * Y2), np.sum(w_matrix * Y2 ** 2)]
    ])
    b = np.array([
        np.sum(w_matrix * Y1 * Y_obs_trunc),
        np.sum(w_matrix * Y2 * Y_obs_trunc)
    ])

    try:
        q1, q2 = np.linalg.solve(A + 1e-6 * np.eye(2), b)
    except:
        q1, q2 = 1.0, 1.0

    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])

    Y_pred = q1 * Y1 + q2 * Y2
    rmse = compute_weighted_rmse(Y_pred, Y_obs_trunc, weights)

    return (q1, q2), Y_pred, rmse


# Also need unweighted versions for final evaluation (scoring uses unweighted RMSE)
def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensity for 1-source (unweighted)."""
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


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensities for 2-source (unweighted)."""
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


class WeightedSensorLossOptimizer:
    """
    Optimizer that uses weighted sensor loss for CMA-ES evaluation.

    Key idea: Weight sensors by their informativeness (variance, mean temp, etc.)
    to help CMA-ES converge to better solutions.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        fevals_1src: int = 20,
        fevals_2src: int = 36,
        sigma_1src: float = 0.18,
        sigma_2src: float = 0.22,
        candidate_pool_size: int = 10,
        refine_maxiter: int = 8,
        refine_top_n: int = 3,
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
        timestep_fraction: float = 0.40,
        weighting_strategy: str = 'variance',
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.fevals_1src = fevals_1src
        self.fevals_2src = fevals_2src
        self.sigma_1src = sigma_1src
        self.sigma_2src = sigma_2src
        self.candidate_pool_size = candidate_pool_size
        self.refine_maxiter = refine_maxiter
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.timestep_fraction = timestep_fraction
        self.weighting_strategy = weighting_strategy

    def _get_smart_inits(self, readings, sensors, n_sources):
        """Get initialization points using smart heuristics."""
        inits = []

        # Hottest sensor initialization
        avg_temps = np.mean(readings, axis=0)
        hottest_idx = np.argmax(avg_temps)
        hot_sensor = sensors[hottest_idx]
        inits.append((hot_sensor.copy(), 'hottest'))

        if n_sources == 2:
            # Triangulation for 2-source
            try:
                tri_init = triangulation_init(readings, sensors, n_sources=2)
                if tri_init is not None:
                    inits.append((tri_init.flatten()[:4], 'triangulation'))
            except:
                pass

            # Two hottest sensors
            sorted_idx = np.argsort(avg_temps)[::-1]
            two_hot = sensors[sorted_idx[:2]].flatten()
            inits.append((two_hot, 'two_hot'))

            # Weighted centroid + hottest
            weights = avg_temps - avg_temps.min() + 0.1
            centroid = np.average(sensors, axis=0, weights=weights)
            combined = np.concatenate([hot_sensor, centroid])
            inits.append((combined, 'centroid_hot'))

        return inits

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=True):
        """Estimate heat sources using weighted sensor loss for CMA-ES."""
        n_sources = sample['n_sources']
        Y_observed = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])

        # Compute sensor weights from observed data
        weights = compute_sensor_weights(Y_observed, strategy=self.weighting_strategy)

        # Simulation parameters
        dt = meta['dt']
        nt_full = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']

        # Create solvers
        solver_coarse = Heat2D(
            Lx=self.Lx, Ly=self.Ly, nx=self.nx_coarse, ny=self.ny_coarse,
            kappa=kappa, bc=bc
        )
        solver_fine = Heat2D(
            Lx=self.Lx, Ly=self.Ly, nx=100, ny=50, kappa=kappa, bc=bc
        )

        # Reduced timesteps for CMA-ES
        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        # Get smart initializations
        inits = self._get_smart_inits(Y_observed, sensors_xy, n_sources)

        # Bounds
        if n_sources == 1:
            bounds = [[0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05]]
            sigma = self.sigma_1src
            fevals = self.fevals_1src
        else:
            bounds = [[0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05],
                      [0.05, self.Lx - 0.05], [0.05, self.Ly - 0.05]]
            sigma = self.sigma_2src
            fevals = self.fevals_2src

        # Define objective using WEIGHTED RMSE for CMA-ES
        n_sims = 0

        if n_sources == 1:
            def objective_weighted(xy_params):
                nonlocal n_sims
                n_sims += 1
                x, y = xy_params
                q, Y_pred, rmse = compute_optimal_intensity_1src_weighted(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy,
                    weights, q_range)
                return rmse
        else:
            def objective_weighted(xy_params):
                nonlocal n_sims
                n_sims += 2
                x1, y1, x2, y2 = xy_params
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src_weighted(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0,
                    sensors_xy, weights, q_range)
                return rmse

        # Run CMA-ES with weighted loss
        all_solutions = []

        for init_params, init_type in inits:
            opts = {
                'bounds': list(zip(*bounds)),
                'maxfevals': fevals,
                'verbose': -9,
                'seed': 42
            }

            try:
                es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma, opts)
                es.optimize(objective_weighted)
                result = es.result
                all_solutions.append((result.xbest, result.fbest, init_type))
            except Exception as e:
                if verbose:
                    print(f"CMA-ES failed for {init_type}: {e}")

        # Sort by weighted RMSE
        all_solutions.sort(key=lambda x: x[1])

        # NM refinement (using UNWEIGHTED RMSE for proper scoring)
        if n_sources == 1:
            def objective_nm(xy_params):
                nonlocal n_sims
                n_sims += 1
                x, y = xy_params
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_nm(xy_params):
                nonlocal n_sims
                n_sims += 2
                x1, y1, x2, y2 = xy_params
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0,
                    sensors_xy, q_range)
                return rmse

        # Refine top N solutions with NM using UNWEIGHTED loss
        refined_solutions = []
        for i, (pos_params, rmse_weighted, init_type) in enumerate(all_solutions[:self.refine_top_n]):
            bounds_nm = [(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
            try:
                result = minimize(
                    objective_nm, pos_params, method='Nelder-Mead',
                    options={'maxiter': self.refine_maxiter, 'xatol': 0.01, 'fatol': 0.001}
                )
                if result.fun < rmse_weighted:
                    refined_solutions.append((result.x, result.fun, init_type))
                else:
                    refined_solutions.append((pos_params, rmse_weighted, init_type))
            except:
                refined_solutions.append((pos_params, rmse_weighted, init_type))

        # Add remaining solutions unrefined
        for pos_params, rmse_weighted, init_type in all_solutions[self.refine_top_n:self.candidate_pool_size]:
            refined_solutions.append((pos_params, rmse_weighted, init_type))

        # Final evaluation with UNWEIGHTED RMSE (for proper scoring)
        candidates_raw = []
        for pos_params, _, init_type in refined_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                sources = [(x, y, q)]
                full_params = np.array([x, y, q])
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0,
                    sensors_xy, q_range)
                sources = [(x1, y1, q1), (x2, y2, q2)]
                full_params = np.array([x1, y1, q1, x2, y2, q2])

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        # Filter for diverse candidates
        candidates_for_filter = [(c[0], c[2]) for c in candidates_raw]
        filtered = filter_dissimilar(candidates_for_filter, tau=TAU, n_max=N_MAX)

        # Fallback if no candidates pass threshold
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            # Run more CMA-ES with higher sigma as fallback
            fallback_sigma = sigma * 1.5
            fallback_fevals = fevals * 2

            for init_params, init_type in inits:
                opts = {
                    'bounds': list(zip(*bounds)),
                    'maxfevals': fallback_fevals,
                    'verbose': -9,
                    'seed': 123
                }

                try:
                    es = cma.CMAEvolutionStrategy(init_params.tolist(), fallback_sigma, opts)
                    es.optimize(objective_weighted)
                    result = es.result

                    # Evaluate with unweighted
                    if n_sources == 1:
                        x, y = result.xbest
                        q, _, final_rmse = compute_optimal_intensity_1src(
                            x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                        sources = [(x, y, q)]
                        full_params = np.array([x, y, q])
                    else:
                        x1, y1, x2, y2 = result.xbest
                        (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                            x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0,
                            sensors_xy, q_range)
                        sources = [(x1, y1, q1), (x2, y2, q2)]
                        full_params = np.array([x1, y1, q1, x2, y2, q2])

                    candidates_raw.append((sources, full_params, final_rmse, init_type + '_fallback'))
                except:
                    pass

            candidates_for_filter = [(c[0], c[2]) for c in candidates_raw]
            filtered = filter_dissimilar(candidates_for_filter, tau=TAU, n_max=N_MAX)

        # Build final results
        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[1], rmse=c[2], init_type=c[3],
                n_evals=fevals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims
