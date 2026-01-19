"""
Warm Start CMA-ES Optimizer (WS-CMA-ES).

Uses CyberAgentAILab's cmaes library with get_warm_start_mgd() for
transferring learned distributions between optimization runs.

Approach 1 (Intra-sample): Multiple short probing runs -> warm start -> main run
Approach 2 (Inter-sample): Pre-compute template from initial samples -> use for rest
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import permutations

import numpy as np
from cmaes import CMA, get_warm_start_mgd

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


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0), early_fraction=1.0):
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    n_early = max(1, int(len(Y_unit) * early_fraction))
    rmse_early = np.sqrt(np.mean((q_optimal * Y_unit[:n_early] - Y_observed[:n_early]) ** 2))
    rmse_full = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return q_optimal, Y_pred, rmse_early, rmse_full


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0), early_fraction=1.0):
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
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
    n_early = max(1, int(len(Y1) * early_fraction))
    rmse_early = np.sqrt(np.mean((q1 * Y1[:n_early] + q2 * Y2[:n_early] - Y_observed[:n_early]) ** 2))
    rmse_full = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
    return (q1, q2), Y_pred, rmse_early, rmse_full


class WarmStartCMAESOptimizer:
    """
    WS-CMA-ES optimizer using CyberAgentAILab's cmaes library.

    Strategy: Intra-sample warm start
    1. Run N short "probing" CMA-ES runs from different initializations
    2. Collect (params, fitness) pairs from all probing runs
    3. Use get_warm_start_mgd() to compute warm start distribution
    4. Run main CMA-ES optimization from warm start
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        # Probing phase config
        n_probing_starts: int = 3,
        probing_fevals: int = 5,
        # Main phase config
        main_fevals_1src: int = 15,
        main_fevals_2src: int = 25,
        # Warm start hyperparameters
        ws_gamma: float = 0.1,  # Weight for prior mean
        ws_alpha: float = 0.1,  # Weight for prior covariance
        # CMA-ES settings
        sigma0_1src: float = 0.18,
        sigma0_2src: float = 0.22,
        use_triangulation: bool = True,
        early_fraction: float = 0.3,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_probing_starts = n_probing_starts
        self.probing_fevals = probing_fevals
        self.main_fevals_1src = main_fevals_1src
        self.main_fevals_2src = main_fevals_2src
        self.ws_gamma = ws_gamma
        self.ws_alpha = ws_alpha
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.early_fraction = early_fraction

    def _create_solver(self, kappa, bc):
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources, margin=0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return np.array(lb), np.array(ub)

    def _smart_init_positions(self, sample, n_sources):
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
        params = []
        for idx in selected:
            x, y = sensors[idx]
            params.extend([x, y])
        return np.array(params)

    def _triangulation_init_positions(self, sample, meta, n_sources, q_range):
        if not self.use_triangulation:
            return None
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except:
            return None

    def _random_init_positions(self, n_sources, lb, ub):
        """Generate random initialization within bounds."""
        return np.random.uniform(lb, ub)

    def _run_probing_cmaes(self, objective, init_params, n_sources, lb, ub, sigma0):
        """Run short CMA-ES probing run and collect solutions."""
        dim = len(init_params)
        solutions = []

        optimizer = CMA(
            mean=init_params,
            sigma=sigma0,
            bounds=np.column_stack([lb, ub]),
            population_size=4 + int(3 * np.log(dim)),
        )

        n_evals = 0
        while n_evals < self.probing_fevals:
            x_list = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                x_list.append(x)

            fitness_list = []
            for x in x_list:
                f = objective(x)
                n_evals += 1
                fitness_list.append(f)
                solutions.append((x.copy(), f))
                if n_evals >= self.probing_fevals:
                    break

            if len(x_list) == len(fitness_list):
                optimizer.tell([(x, f) for x, f in zip(x_list, fitness_list)])

        return solutions, n_evals

    def _run_warm_start_cmaes(self, objective, warm_mean, warm_sigma, warm_cov,
                              max_fevals, lb, ub):
        """Run CMA-ES with warm start distribution."""
        dim = len(warm_mean)
        all_solutions = []

        optimizer = CMA(
            mean=warm_mean,
            sigma=warm_sigma,
            cov=warm_cov,
            bounds=np.column_stack([lb, ub]),
            population_size=4 + int(3 * np.log(dim)),
        )

        n_evals = 0
        while n_evals < max_fevals:
            x_list = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                x_list.append(x)

            fitness_list = []
            for x in x_list:
                f = objective(x)
                n_evals += 1
                fitness_list.append(f)
                all_solutions.append((x.copy(), f))
                if n_evals >= max_fevals:
                    break

            if len(x_list) == len(fitness_list):
                optimizer.tell([(x, f) for x, f in zip(x_list, fitness_list)])

        return all_solutions, n_evals

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)
        lb, ub = self._get_position_bounds(n_sources)
        n_sims = [0]
        early_frac = self.early_fraction

        # Define objective function
        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse_early, rmse_full = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=early_frac)
                return rmse_early
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse_early, rmse_full = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=early_frac)
                return rmse_early

        # Get initializations for probing phase
        probing_inits = []

        # Triangulation init
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            probing_inits.append(('triangulation', tri_init))

        # Smart init
        smart_init = self._smart_init_positions(sample, n_sources)
        probing_inits.append(('smart', smart_init))

        # Random inits
        for i in range(self.n_probing_starts - len(probing_inits)):
            rand_init = self._random_init_positions(n_sources, lb, ub)
            probing_inits.append((f'random_{i}', rand_init))

        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        max_fevals = self.main_fevals_1src if n_sources == 1 else self.main_fevals_2src

        # ===== PROBING PHASE =====
        all_probing_solutions = []

        for init_name, init_params in probing_inits:
            solutions, _ = self._run_probing_cmaes(
                objective, init_params, n_sources, lb, ub, sigma0)
            all_probing_solutions.extend(solutions)

        if verbose:
            best_probe = min(s[1] for s in all_probing_solutions)
            print(f"  [Probing] {len(all_probing_solutions)} solutions, best: {best_probe:.4f}")

        # ===== WARM START COMPUTATION =====
        try:
            warm_mean, warm_sigma, warm_cov = get_warm_start_mgd(
                all_probing_solutions,
                gamma=self.ws_gamma,
                alpha=self.ws_alpha
            )
        except Exception as e:
            # Fallback to best probing solution if warm start fails
            if verbose:
                print(f"  [Warm Start] Failed: {e}, using best probing solution")
            best_sol = min(all_probing_solutions, key=lambda x: x[1])
            warm_mean = best_sol[0]
            warm_sigma = sigma0 * 0.5
            warm_cov = None

        # ===== MAIN OPTIMIZATION PHASE =====
        if warm_cov is not None:
            main_solutions, _ = self._run_warm_start_cmaes(
                objective, warm_mean, warm_sigma, warm_cov,
                max_fevals, lb, ub)
        else:
            # No covariance, use standard CMA-ES
            main_solutions, _ = self._run_probing_cmaes(
                objective, warm_mean, n_sources, lb, ub, warm_sigma)

        # Combine all solutions
        all_solutions = all_probing_solutions + main_solutions
        all_solutions.sort(key=lambda x: x[1])

        if verbose:
            print(f"  [Main] {len(main_solutions)} solutions, best: {all_solutions[0][1]:.4f}")

        # ===== FINAL EVALUATION =====
        # Take top candidates and do full RMSE evaluation
        top_n = min(5, len(all_solutions))
        candidates_raw = []

        for i in range(top_n):
            pos_params = all_solutions[i][0]

            if n_sources == 1:
                x, y = pos_params
                q, _, rmse_early, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=1.0)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, rmse_early, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                    early_fraction=1.0)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, 'ws_cmaes'))

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
                n_evals=n_sims[0] // len(final_candidates) if final_candidates else n_sims[0]
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims[0]
