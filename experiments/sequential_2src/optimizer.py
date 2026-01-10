"""
Sequential 2-Source Estimation for Heat Source Identification.

Key Innovation: Instead of optimizing both sources simultaneously (4D),
we optimize them sequentially:
1. Find dominant source first (2D optimization)
2. Subtract its contribution (using heat equation linearity)
3. Find second source in residual (2D optimization)

Physics Insight:
    - Heat equation is LINEAR: T_total = T_source1 + T_source2
    - We can subtract out a source's contribution: T_residual = T_obs - T_source1
    - The residual looks like a single-source problem
    - 2D optimization converges faster than 4D with same fevals

Expected Benefits:
    - Better convergence for 2-source problems
    - More efficient use of feval budget
    - Reduced "inversion problem" (sources swapping)
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


def compute_optimal_intensity(Y_unit: np.ndarray, Y_observed: np.ndarray,
                               q_range: Tuple[float, float] = (0.5, 2.0)) -> float:
    """Compute optimal intensity analytically."""
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()

    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        return 1.0

    numerator = np.dot(Y_unit_flat, Y_obs_flat)
    q_optimal = numerator / denominator

    return float(np.clip(q_optimal, q_range[0], q_range[1]))


class Sequential2SourceOptimizer:
    """
    CMA-ES optimizer using sequential 2-source estimation.

    For 2-source problems:
    1. Find dominant source first (2D)
    2. Subtract its contribution
    3. Find second source in residual (2D)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 22,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        early_fraction: float = 0.3,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.early_fraction = early_fraction

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int = 1, margin: float = 0.05):
        """Get bounds for position parameters."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, sample: Dict, n_sources: int) -> np.ndarray:
        """Get position initialization from hottest sensors."""
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

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get triangulation init."""
        if not self.use_triangulation:
            return None
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _optimize_single_source(
        self,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        init_params: np.ndarray,
        max_fevals: int,
        sigma0: float,
    ) -> Tuple[np.ndarray, float, int]:
        """Optimize a single source position (2D)."""
        n_sims = 0
        early_frac = self.early_fraction
        n_early = max(1, int(len(Y_observed) * early_frac))

        def objective(xy_params):
            nonlocal n_sims
            x, y = xy_params
            n_sims += 1
            Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
            q = compute_optimal_intensity(Y_unit, Y_observed, q_range)
            Y_pred = q * Y_unit
            rmse_early = np.sqrt(np.mean((Y_pred[:n_early] - Y_observed[:n_early]) ** 2))
            return rmse_early

        lb, ub = self._get_position_bounds(1)

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)
        best_solution = init_params
        best_rmse = float('inf')

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                if fit < best_rmse:
                    best_rmse = fit
                    best_solution = np.array(sol)

        return best_solution, best_rmse, n_sims

    def _estimate_1source(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float],
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """Estimate 1-source problem (standard approach)."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        # Build initialization pool
        initializations = []

        tri_init = self._triangulation_init_positions(sample, meta, 1, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        smart_init = self._smart_init_positions(sample, 1)
        initializations.append((smart_init, 'smart'))

        # CMA-ES optimization
        fevals_per_init = max(5, self.max_fevals_1src // len(initializations))

        all_solutions = []
        total_sims = 0

        for init_params, init_type in initializations:
            best_pos, best_rmse, n_sims = self._optimize_single_source(
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                init_params, fevals_per_init, self.sigma0_1src
            )
            total_sims += n_sims

            # Get full RMSE
            Y_unit = simulate_unit_source(best_pos[0], best_pos[1], solver, dt, nt, T0, sensors_xy)
            total_sims += 1
            q = compute_optimal_intensity(Y_unit, Y_observed, q_range)
            Y_pred = q * Y_unit
            full_rmse = float(np.sqrt(np.mean((Y_pred - Y_observed) ** 2)))

            x, y = best_pos
            sources = [(float(x), float(y), float(q))]
            full_params = np.array([x, y, q])

            all_solutions.append((sources, full_params, full_rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in all_solutions], tau=TAU)

        final_candidates = []
        for sources, rmse in filtered:
            for c in all_solutions:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # Build results
        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=total_sims // len(final_candidates) if final_candidates else total_sims
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, total_sims

    def _estimate_2source_sequential(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float],
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate 2-source problem using sequential approach.

        Strategy:
        1. Find first source (dominant) using 1-source optimization
        2. Subtract its contribution from observations
        3. Find second source in the residual
        4. Joint refinement (optional)
        """
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)
        total_sims = 0

        # Allocate fevals: 50% for first source, 50% for second
        fevals_first = self.max_fevals_2src // 2
        fevals_second = self.max_fevals_2src - fevals_first

        # Build initialization pool for first source
        initializations = []

        tri_init = self._triangulation_init_positions(sample, meta, 2, q_range)
        if tri_init is not None:
            # Use first source from triangulation
            initializations.append((tri_init[:2], 'triangulation'))

        smart_init = self._smart_init_positions(sample, 2)
        initializations.append((smart_init[:2], 'smart'))

        all_solutions = []

        for init_params, init_type in initializations:
            # Stage 1: Find first source
            pos1, _, n_sims = self._optimize_single_source(
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                init_params, fevals_first // len(initializations), self.sigma0_2src
            )
            total_sims += n_sims

            # Compute source 1's contribution
            Y1_unit = simulate_unit_source(pos1[0], pos1[1], solver, dt, nt, T0, sensors_xy)
            total_sims += 1
            q1 = compute_optimal_intensity(Y1_unit, Y_observed, q_range)
            Y1_contribution = q1 * Y1_unit

            # Stage 2: Subtract and find residual source
            Y_residual = Y_observed - Y1_contribution

            # Initialize second source (use far position from first)
            if tri_init is not None and init_type == 'triangulation':
                init2 = tri_init[2:4]  # Second source from triangulation
            else:
                # Use furthest hot sensor from first source
                readings = sample['Y_noisy']
                sensors = sample['sensors_xy']
                avg_temps = np.mean(readings, axis=0)

                # Weight by temperature and distance from first source
                weights = []
                for i, (sx, sy) in enumerate(sensors):
                    dist = np.sqrt((sx - pos1[0])**2 + (sy - pos1[1])**2)
                    weight = avg_temps[i] * (dist + 0.1)
                    weights.append(weight)

                best_idx = np.argmax(weights)
                init2 = np.array(sensors[best_idx])

            pos2, _, n_sims = self._optimize_single_source(
                Y_residual, solver, dt, nt, T0, sensors_xy, q_range,
                init2, fevals_second // len(initializations), self.sigma0_2src
            )
            total_sims += n_sims

            # Final joint intensity computation
            Y2_unit = simulate_unit_source(pos2[0], pos2[1], solver, dt, nt, T0, sensors_xy)
            total_sims += 1

            # Solve 2x2 system for optimal intensities
            Y1_flat = Y1_unit.flatten()
            Y2_flat = Y2_unit.flatten()
            Y_obs_flat = Y_observed.flatten()

            A = np.array([
                [np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
                [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]
            ])
            b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])

            try:
                A_reg = A + 1e-6 * np.eye(2)
                q1_opt, q2_opt = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                q1_opt, q2_opt = 1.0, 1.0

            q1_opt = float(np.clip(q1_opt, q_range[0], q_range[1]))
            q2_opt = float(np.clip(q2_opt, q_range[0], q_range[1]))

            # Compute full RMSE
            Y_pred = q1_opt * Y1_unit + q2_opt * Y2_unit
            full_rmse = float(np.sqrt(np.mean((Y_pred - Y_observed) ** 2)))

            sources = [(float(pos1[0]), float(pos1[1]), q1_opt),
                      (float(pos2[0]), float(pos2[1]), q2_opt)]
            full_params = np.array([pos1[0], pos1[1], q1_opt, pos2[0], pos2[1], q2_opt])

            all_solutions.append((sources, full_params, full_rmse, init_type))

        # Also try joint optimization as alternative
        # (for diversity)
        if len(initializations) >= 2:
            # Try joint optimization with smart init
            joint_init = smart_init[:4]
            joint_solutions = self._joint_2source_optimization(
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                joint_init, fevals_first, self.sigma0_2src
            )
            total_sims += joint_solutions[1]
            all_solutions.append((joint_solutions[0], joint_solutions[2],
                                  joint_solutions[3], 'joint'))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in all_solutions], tau=TAU)

        final_candidates = []
        for sources, rmse in filtered:
            for c in all_solutions:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # Build results
        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=total_sims // len(final_candidates) if final_candidates else total_sims
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, total_sims

    def _joint_2source_optimization(
        self,
        Y_observed: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        init_params: np.ndarray,
        max_fevals: int,
        sigma0: float,
    ) -> Tuple[List[Tuple], int, np.ndarray, float]:
        """Joint 4D optimization (for comparison/diversity)."""
        n_sims = 0
        early_frac = self.early_fraction
        n_early = max(1, int(len(Y_observed) * early_frac))

        def objective(xy_params):
            nonlocal n_sims
            x1, y1, x2, y2 = xy_params
            n_sims += 2

            Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
            Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)

            # Optimal intensities
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
            rmse_early = np.sqrt(np.mean((Y_pred[:n_early] - Y_observed[:n_early]) ** 2))
            return rmse_early

        lb, ub = self._get_position_bounds(2)

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)
        best_solution = init_params
        best_rmse = float('inf')

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                if fit < best_rmse:
                    best_rmse = fit
                    best_solution = np.array(sol)

        # Get final solution with full RMSE
        x1, y1, x2, y2 = best_solution
        Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
        Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
        n_sims += 2

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

        q1 = float(np.clip(q1, q_range[0], q_range[1]))
        q2 = float(np.clip(q2, q_range[0], q_range[1]))

        Y_pred = q1 * Y1 + q2 * Y2
        full_rmse = float(np.sqrt(np.mean((Y_pred - Y_observed) ** 2)))

        sources = [(float(x1), float(y1), q1), (float(x2), float(y2), q2)]
        full_params = np.array([x1, y1, q1, x2, y2, q2])

        return sources, n_sims, full_params, full_rmse

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources using sequential 2-source estimation.
        """
        n_sources = sample['n_sources']

        if n_sources == 1:
            return self._estimate_1source(sample, meta, q_range)
        else:
            return self._estimate_2source_sequential(sample, meta, q_range)
