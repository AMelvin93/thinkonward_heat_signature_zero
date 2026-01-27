"""
Extended Kalman Filter Optimizer for Inverse Heat Source Problem

This experiment tests whether EKF can work better than optimization-based approaches
by treating source parameters as hidden state and temperature as observations.

Key differences from LM/optimization:
- Sequential processing of timesteps
- Maintains uncertainty estimates (covariance)
- Natural handling of measurement noise

Concerns:
- For static sources, state transition is identity
- Observation function is highly nonlinear (PDE-based)
- EKF is still fundamentally local (linearizes around current estimate)
- Jacobian computation is expensive (requires multiple simulations)
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
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


class ExtendedKalmanFilterOptimizer:
    """
    Extended Kalman Filter for heat source identification.

    State vector: [x1, y1, q1, ...] for n sources
    Observation: Temperature at sensors at each timestep
    State transition: Identity (static sources)
    Observation model: PDE simulation + sensor sampling
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_timesteps_ekf: int = 20,  # Number of timesteps to process with EKF
        process_noise_std: float = 0.01,  # Process noise for state evolution
        measurement_noise_std: float = 0.1,  # Measurement noise (from problem)
        jacobian_eps: float = 0.01,  # Step for numerical Jacobian
        n_candidates: int = N_MAX,
        nm_polish_iters: int = 8,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_timesteps_ekf = n_timesteps_ekf
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.jacobian_eps = jacobian_eps
        self.n_candidates = min(n_candidates, N_MAX)
        self.nm_polish_iters = nm_polish_iters

    def _create_solver(self, kappa, bc):
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources, q_range, margin=0.05):
        """Get bounds for state vector [x1, y1, q1, ...]"""
        lb = []
        ub = []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return np.array(lb), np.array(ub)

    def _clip_state(self, state, lb, ub):
        """Project state to feasible region."""
        return np.clip(state, lb, ub)

    def _simulate_observation(self, state, solver, dt, nt, T0, sensors_xy, n_sources):
        """
        Simulate temperature observations given state vector.
        state: [x1, y1, q1, ...] for n_sources
        Returns: Temperature at sensors for all timesteps [nt, n_sensors]
        """
        sources = []
        for i in range(n_sources):
            x = state[i*3]
            y = state[i*3 + 1]
            q = state[i*3 + 2]
            sources.append({'x': x, 'y': y, 'q': q})

        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y_sim = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
        return Y_sim

    def _compute_jacobian(self, state, solver, dt, nt, T0, sensors_xy, n_sources, timestep_idx, lb, ub):
        """
        Compute Jacobian of observation function at given timestep using finite differences.
        Returns: H matrix [n_sensors x state_dim]
        """
        state_dim = len(state)
        n_sensors = len(sensors_xy)

        # Get base observation at this timestep
        Y_base = self._simulate_observation(state, solver, dt, nt, T0, sensors_xy, n_sources)
        z_base = Y_base[timestep_idx]

        H = np.zeros((n_sensors, state_dim))

        for i in range(state_dim):
            state_plus = state.copy()
            state_plus[i] += self.jacobian_eps
            state_plus = self._clip_state(state_plus, lb, ub)

            Y_plus = self._simulate_observation(state_plus, solver, dt, nt, T0, sensors_xy, n_sources)
            z_plus = Y_plus[timestep_idx]

            H[:, i] = (z_plus - z_base) / self.jacobian_eps

        return H

    def _smart_init(self, sample, n_sources, q_range):
        """Initialize state at hottest sensors with mid-range intensity."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[p]) >= 0.25 for p in selected):
                selected.append(idx)
        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        state = []
        q_init = (q_range[0] + q_range[1]) / 2
        for idx in selected:
            x, y = sensors[idx]
            state.extend([x, y, q_init])
        return np.array(state)

    def _triangulation_init(self, sample, meta, n_sources, q_range):
        """Initialize using triangulation."""
        try:
            return triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
        except:
            return None

    def _run_ekf(self, init_state, Y_observed, solver, dt, nt_full, T0, sensors_xy,
                 n_sources, q_range, lb, ub, init_type):
        """
        Run Extended Kalman Filter to estimate source parameters.
        """
        state_dim = len(init_state)
        n_sensors = len(sensors_xy)
        n_sims = [0]

        # Select subset of timesteps for EKF
        n_timesteps = min(self.n_timesteps_ekf, nt_full)
        timestep_indices = np.linspace(0, nt_full - 1, n_timesteps, dtype=int)

        # Initialize state and covariance
        x = init_state.copy()

        # Initial covariance - larger for positions, smaller for intensity
        P = np.eye(state_dim)
        for i in range(n_sources):
            P[i*3, i*3] = 0.1  # x uncertainty
            P[i*3+1, i*3+1] = 0.05  # y uncertainty
            P[i*3+2, i*3+2] = 0.25  # q uncertainty

        # Process noise covariance (for static sources, this adds small drift)
        Q = (self.process_noise_std ** 2) * np.eye(state_dim)

        # Measurement noise covariance
        R = (self.measurement_noise_std ** 2) * np.eye(n_sensors)

        # Process each selected timestep
        for t_idx in timestep_indices:
            # Predict step (for static sources, state stays same)
            # x_pred = x (identity transition)
            P_pred = P + Q

            # Compute observation Jacobian H
            H = self._compute_jacobian(x, solver, dt, nt_full, T0, sensors_xy,
                                        n_sources, t_idx, lb, ub)
            n_sims[0] += (state_dim + 1)  # Base + perturbations

            # Predicted observation
            Y_pred = self._simulate_observation(x, solver, dt, nt_full, T0, sensors_xy, n_sources)
            n_sims[0] += 1
            z_pred = Y_pred[t_idx]

            # Actual observation
            z_obs = Y_observed[t_idx]

            # Innovation
            innovation = z_obs - z_pred

            # Innovation covariance
            S = H @ P_pred @ H.T + R

            # Kalman gain
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Singular matrix - use pseudo-inverse
                K = P_pred @ H.T @ np.linalg.pinv(S)

            # State update
            x = x + K @ innovation
            x = self._clip_state(x, lb, ub)

            # Covariance update
            P = (np.eye(state_dim) - K @ H) @ P_pred

        # Compute final RMSE
        Y_final = self._simulate_observation(x, solver, dt, nt_full, T0, sensors_xy, n_sources)
        n_sims[0] += 1
        rmse = np.sqrt(np.mean((Y_final[:nt_full] - Y_observed[:nt_full]) ** 2))

        return {
            'state': x,
            'rmse': rmse,
            'covariance': P,
            'n_sims': n_sims[0],
            'init_type': init_type,
        }

    def _nm_polish(self, state, Y_observed, solver, dt, nt, T0, sensors_xy, n_sources, lb, ub):
        """Apply Nelder-Mead polish to refine solution."""
        n_sims = [0]

        def objective(s):
            n_sims[0] += 1
            s_clipped = self._clip_state(s, lb, ub)
            Y_sim = self._simulate_observation(s_clipped, solver, dt, nt, T0, sensors_xy, n_sources)
            return np.sqrt(np.mean((Y_sim[:nt] - Y_observed[:nt]) ** 2))

        result = minimize(
            objective,
            state,
            method='Nelder-Mead',
            options={
                'maxiter': self.nm_polish_iters,
                'xatol': 0.01,
                'fatol': 0.001,
            }
        )

        return self._clip_state(result.x, lb, ub), result.fun, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        """
        Estimate heat source parameters using Extended Kalman Filter.
        """
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        solver = self._create_solver(kappa, bc)
        lb, ub = self._get_bounds(n_sources, q_range)

        total_sims = [0]

        # Generate initializations
        initializations = []

        # Triangulation init
        tri_init = self._triangulation_init(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # Smart init (hottest sensor)
        smart_init = self._smart_init(sample, n_sources, q_range)
        initializations.append((smart_init, 'smart'))

        # Run EKF from each initialization
        ekf_results = []
        for init_state, init_type in initializations:
            result = self._run_ekf(
                init_state, Y_observed, solver, dt, nt_full, T0, sensors_xy,
                n_sources, q_range, lb, ub, init_type
            )
            ekf_results.append(result)
            total_sims[0] += result['n_sims']

        # Sort by RMSE
        ekf_results.sort(key=lambda x: x['rmse'])

        # Polish best results
        polished_results = []
        for result in ekf_results[:2]:  # Polish top 2
            if self.nm_polish_iters > 0:
                polished_state, polished_rmse, n_polish_sims = self._nm_polish(
                    result['state'], Y_observed, solver, dt, nt_full, T0, sensors_xy,
                    n_sources, lb, ub
                )
                total_sims[0] += n_polish_sims

                if polished_rmse < result['rmse']:
                    polished_results.append({
                        'state': polished_state,
                        'rmse': polished_rmse,
                        'init_type': f"{result['init_type']}_polished",
                    })
                else:
                    polished_results.append({
                        'state': result['state'],
                        'rmse': result['rmse'],
                        'init_type': result['init_type'],
                    })
            else:
                polished_results.append({
                    'state': result['state'],
                    'rmse': result['rmse'],
                    'init_type': result['init_type'],
                })

        # Add remaining results
        for result in ekf_results[2:]:
            polished_results.append({
                'state': result['state'],
                'rmse': result['rmse'],
                'init_type': result['init_type'],
            })

        # Convert to candidates format
        candidates_raw = []
        for result in polished_results:
            state = result['state']
            sources = []
            for i in range(n_sources):
                x = float(state[i*3])
                y = float(state[i*3 + 1])
                q = float(state[i*3 + 2])
                sources.append((x, y, q))

            candidates_raw.append((sources, state, result['rmse'], result['init_type']))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

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
                n_evals=total_sims[0] // len(final_candidates) if final_candidates else total_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, total_sims[0]
