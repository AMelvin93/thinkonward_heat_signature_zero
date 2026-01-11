"""
IPOP-CMA-ES: Restart CMA-ES with Increasing Population.

Key Innovation: Use restart strategy to escape local minima on 2-source problems.

Research Basis:
    - IPOP-CMA-ES restarts with 2x population size when converged
    - BIPOP-CMA-ES alternates between small (local) and large (global) restarts
    - This helps find global optimum on multimodal functions

For our problem:
    - 2-source optimization is 4D and may have multiple local minima
    - Standard CMA-ES with limited fevals may get stuck
    - Restarts with increasing population can explore more globally

Reference: Hansen 2005 "A Restart CMA Evolution Strategy With Increasing Population Size"
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add simulator path
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


# Competition parameters
N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


@dataclass
class CandidateResult:
    """Result from optimization."""
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    """Normalize source parameters using scale factors."""
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
    """Compute minimum distance between two candidate sets."""
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


def filter_dissimilar(candidates: List[Tuple], tau: float = TAU, n_max: int = N_MAX) -> List[Tuple]:
    """Filter to keep only dissimilar candidates."""
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


def simulate_unit_source(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                         T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    """Simulate a source with q=1.0 and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    """Compute optimal intensity for 1-source analytically."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()

    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        numerator = np.dot(Y_unit_flat, Y_obs_flat)
        q_optimal = numerator / denominator

    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], np.ndarray, float]:
    """Compute optimal intensities for 2-source analytically."""
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
        A_reg = A + 1e-6 * np.eye(2)
        q1, q2 = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        q1, q2 = 1.0, 1.0

    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])

    Y_pred = q1 * Y1 + q2 * Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    return (q1, q2), Y_pred, rmse


class IPOPCMAESOptimizer:
    """
    IPOP-CMA-ES: CMA-ES with restart strategy using increasing population.

    Key Strategy:
        - For 1-source: Standard CMA-ES (2D, usually finds optimum)
        - For 2-source: IPOP restart strategy (4D, may have local minima)

    IPOP Strategy:
        1. Run CMA-ES with initial population size
        2. If converged (not improved), restart with 2x population
        3. Each restart starts from a new random point
        4. Collect best solutions from all runs
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 12,
        max_fevals_2src: int = 24,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        n_restarts_2src: int = 2,  # Number of restarts for 2-source
        popsize_inc: float = 2.0,  # Population size increase factor
        n_candidates: int = N_MAX,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.n_restarts_2src = n_restarts_2src
        self.popsize_inc = popsize_inc
        self.n_candidates = min(n_candidates, N_MAX)

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, sample: Dict, n_sources: int) -> np.ndarray:
        """Get position-only initialization from hottest sensors."""
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

    def _onset_init_positions(self, sample: Dict, meta: Dict,
                              n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get position-only initialization from onset triangulation."""
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _random_init(self, n_sources: int, seed: int = None) -> np.ndarray:
        """Generate random position initialization within bounds."""
        if seed is not None:
            np.random.seed(seed)

        lb, ub = self._get_position_bounds(n_sources)
        positions = []
        for i in range(n_sources):
            x = np.random.uniform(lb[i*2], ub[i*2])
            y = np.random.uniform(lb[i*2+1], ub[i*2+1])
            positions.extend([x, y])
        return np.array(positions)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources with IPOP-CMA-ES restart strategy.
        """
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        n_sims = [0]

        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

            return self._optimize_1src(
                sample, meta, objective, solver, dt, nt, T0, sensors_xy,
                Y_observed, q_range, n_sims, verbose
            )
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

            return self._optimize_2src_ipop(
                sample, meta, objective, solver, dt, nt, T0, sensors_xy,
                Y_observed, q_range, n_sims, verbose
            )

    def _optimize_1src(
        self, sample, meta, objective, solver, dt, nt, T0, sensors_xy,
        Y_observed, q_range, n_sims, verbose
    ):
        """Standard CMA-ES for 1-source (simpler landscape)."""
        n_sources = 1
        lb, ub = self._get_position_bounds(n_sources)

        # Get initializations
        onset_init = self._onset_init_positions(sample, meta, n_sources, q_range)
        smart_init = self._smart_init_positions(sample, n_sources)

        # Evaluate both and pick best
        inits = []
        if onset_init is not None:
            rmse = objective(onset_init)
            inits.append((onset_init, rmse, 'onset'))

        rmse = objective(smart_init)
        inits.append((smart_init, rmse, 'smart'))

        # Pick best init
        inits.sort(key=lambda x: x[1])
        best_init, init_rmse, init_type = inits[0]

        # Run CMA-ES from best init
        fevals_remaining = self.max_fevals_1src - len(inits)

        opts = cma.CMAOptions()
        opts['maxfevals'] = fevals_remaining
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(best_init.tolist(), self.sigma0_1src, opts)

        all_solutions = []

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit, init_type))

        # Add initial evaluations
        for init_params, rmse, itype in inits:
            all_solutions.append((init_params, rmse, itype))

        # Get diverse candidates
        return self._finalize_candidates(
            all_solutions, solver, dt, nt, T0, sensors_xy,
            Y_observed, q_range, n_sources, n_sims[0]
        )

    def _optimize_2src_ipop(
        self, sample, meta, objective, solver, dt, nt, T0, sensors_xy,
        Y_observed, q_range, n_sims, verbose
    ):
        """IPOP-CMA-ES with restarts for 2-source problems."""
        n_sources = 2
        lb, ub = self._get_position_bounds(n_sources)

        # Get initializations
        onset_init = self._onset_init_positions(sample, meta, n_sources, q_range)
        smart_init = self._smart_init_positions(sample, n_sources)

        all_solutions = []

        # Distribute fevals across runs
        # Main run gets most fevals, restarts share the rest
        n_runs = 1 + self.n_restarts_2src
        fevals_per_run = self.max_fevals_2src // n_runs

        # Run 0: Best initialization
        inits = []
        if onset_init is not None:
            rmse = objective(onset_init)
            inits.append((onset_init, rmse, 'onset'))

        rmse = objective(smart_init)
        inits.append((smart_init, rmse, 'smart'))

        inits.sort(key=lambda x: x[1])
        best_init, init_rmse, init_type = inits[0]

        # Track global best for restart decisions
        global_best_rmse = init_rmse
        global_best_solution = best_init

        # Initial population size
        popsize = 6  # CMA-ES default for 4D

        for run_idx in range(n_runs):
            if run_idx == 0:
                # First run: use best init
                init_params = best_init
                run_type = init_type
            else:
                # Restart: increase population and start from new point
                popsize = int(popsize * self.popsize_inc)

                # Alternate between: perturbed best, random, opposite corner
                if run_idx % 3 == 1:
                    # Perturb best solution found so far
                    perturbation = np.random.randn(4) * self.sigma0_2src * 0.5
                    init_params = np.clip(
                        global_best_solution + perturbation,
                        lb, ub
                    )
                    run_type = 'restart_perturb'
                elif run_idx % 3 == 2:
                    # Random restart
                    init_params = self._random_init(n_sources, seed=run_idx*42)
                    run_type = 'restart_random'
                else:
                    # Opposite region of domain
                    init_params = np.array([
                        self.Lx - global_best_solution[0],
                        global_best_solution[1],  # Keep y same
                        self.Lx - global_best_solution[2],
                        global_best_solution[3]
                    ])
                    init_params = np.clip(init_params, lb, ub)
                    run_type = 'restart_opposite'

            # CMA-ES run with current population size
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_run
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6
            opts['popsize'] = popsize

            es = cma.CMAEvolutionStrategy(init_params.tolist(), self.sigma0_2src, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit, run_type))

                    # Track global best
                    if fit < global_best_rmse:
                        global_best_rmse = fit
                        global_best_solution = np.array(sol)

            if verbose:
                print(f"  Run {run_idx} ({run_type}): best RMSE = {global_best_rmse:.4f}, popsize = {popsize}")

        # Add initial evaluations
        for init_params, rmse, itype in inits:
            all_solutions.append((init_params, rmse, itype))

        # Get diverse candidates
        return self._finalize_candidates(
            all_solutions, solver, dt, nt, T0, sensors_xy,
            Y_observed, q_range, n_sources, n_sims[0]
        )

    def _finalize_candidates(
        self, all_solutions, solver, dt, nt, T0, sensors_xy,
        Y_observed, q_range, n_sources, n_evals
    ):
        """Convert position solutions to full candidates with intensities."""
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:15]  # Consider top 15

        candidates_raw = []
        for pos_params, rmse, init_type in top_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

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
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_evals // len(final_candidates) if final_candidates else n_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_evals
