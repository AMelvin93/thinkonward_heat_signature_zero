"""
A15: Sequential Source Estimation for 2-Source Problems.

Key Innovation: Decompose 4D optimization into two 2D optimizations.

Physics Insight:
    The heat equation is LINEAR: T_total = T_source1 + T_source2
    Therefore: T_source2 = T_total - T_source1

Algorithm for 2-source:
    1. Find dominant source using 1-source optimization
    2. Simulate this source to get Y_source1
    3. Compute residual: Y_residual = Y_observed - Y_source1
    4. Find second source from Y_residual using 1-source optimization
    5. Optional: Fine-tune both sources together

Advantages:
    - Decomposes 4D problem into two 2D problems
    - 2D problems are easier to solve (fewer local minima)
    - Each source gets full optimization budget
    - Exploits linearity of heat equation
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
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
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


class SequentialEstimationOptimizer:
    """
    Sequential Source Estimation for 2-Source Problems.

    Key Innovation: Decompose 4D optimization into two 2D optimizations
    by exploiting the linearity of the heat equation.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 10,
        max_fevals_2src: int = 18,  # Total for 2-source = per_source*2 + refinement
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.15,  # Smaller since each source is 2D
        n_candidates: int = N_MAX,
        use_refinement: bool = True,
        refinement_fevals: int = 6,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.n_candidates = min(n_candidates, N_MAX)
        self.use_refinement = use_refinement
        self.refinement_fevals = refinement_fevals

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds_1src(self, margin: float = 0.05):
        lb = [margin * self.Lx, margin * self.Ly]
        ub = [(1 - margin) * self.Lx, (1 - margin) * self.Ly]
        return lb, ub

    def _get_position_bounds_2src(self, margin: float = 0.05):
        lb = [margin * self.Lx, margin * self.Ly] * 2
        ub = [(1 - margin) * self.Lx, (1 - margin) * self.Ly] * 2
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

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _optimize_1src(
        self,
        Y_target: np.ndarray,
        solver: Heat2D,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        q_range: Tuple[float, float],
        init_positions: List[np.ndarray],
        max_fevals: int,
        sigma0: float,
    ) -> Tuple[float, float, float, float, int]:
        """
        Optimize a single source to match Y_target.
        Returns: (x, y, q, rmse, n_evals)
        """
        n_evals = [0]

        def objective(xy):
            x, y = xy
            n_evals[0] += 1
            q, Y_pred, rmse = compute_optimal_intensity_1src(
                x, y, Y_target, solver, dt, nt, T0, sensors_xy, q_range)
            return rmse

        lb, ub = self._get_position_bounds_1src()
        fevals_per_init = max(3, max_fevals // len(init_positions))

        best_x, best_y, best_q, best_rmse = 0, 0, 1.0, float('inf')

        for init_xy in init_positions:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_xy.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

            result_xy = es.result.xbest
            q, Y_pred, rmse = compute_optimal_intensity_1src(
                result_xy[0], result_xy[1], Y_target, solver, dt, nt, T0, sensors_xy, q_range)

            if rmse < best_rmse:
                best_rmse = rmse
                best_x, best_y = result_xy
                best_q = q

        return best_x, best_y, best_q, best_rmse, n_evals[0]

    def _sequential_2src(
        self,
        sample: Dict,
        meta: Dict,
        solver: Heat2D,
        q_range: Tuple[float, float],
    ) -> Tuple[List[Tuple], float, int]:
        """
        Sequential estimation for 2-source problems.

        1. Find dominant source (1D optimization)
        2. Subtract its contribution
        3. Find second source from residual
        4. Optionally refine together
        """
        Y_observed = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        total_evals = [0]

        # Get init positions for 1-source
        smart_init = self._smart_init_positions(sample, 1)[:2]
        tri_init = self._triangulation_init_positions(sample, meta, 1, q_range)
        init_positions = [smart_init]
        if tri_init is not None:
            init_positions.append(tri_init[:2])

        # Fevals budget: half for first source, half for second, rest for refinement
        fevals_phase1 = (self.max_fevals_2src - self.refinement_fevals) // 2
        fevals_phase2 = fevals_phase1

        # Phase 1: Find first (dominant) source
        x1, y1, q1, rmse1, n1 = self._optimize_1src(
            Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
            init_positions, fevals_phase1, self.sigma0_2src
        )
        total_evals[0] += n1

        # Compute contribution of first source
        Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
        Y_source1 = q1 * Y1

        # Phase 2: Compute residual and find second source
        Y_residual = Y_observed - Y_source1

        # Use hottest residual sensor as init for second source
        residual_temps = np.mean(np.abs(Y_residual), axis=0)
        hot_idx = np.argmax(residual_temps)
        residual_init = np.array([sensors_xy[hot_idx, 0], sensors_xy[hot_idx, 1]])

        x2, y2, q2, rmse2, n2 = self._optimize_1src(
            Y_residual, solver, dt, nt, T0, sensors_xy, q_range,
            [residual_init], fevals_phase2, self.sigma0_2src
        )
        total_evals[0] += n2

        # Phase 3: Optional joint refinement
        if self.use_refinement and self.refinement_fevals > 0:
            n_refine = [0]

            def joint_objective(xy4):
                x1r, y1r, x2r, y2r = xy4
                n_refine[0] += 2
                (q1r, q2r), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1r, y1r, x2r, y2r, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

            lb, ub = self._get_position_bounds_2src()
            init_4d = [x1, y1, x2, y2]

            opts = cma.CMAOptions()
            opts['maxfevals'] = self.refinement_fevals
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9

            es = cma.CMAEvolutionStrategy(init_4d, 0.05, opts)  # Small sigma for refinement

            while not es.stop():
                solutions = es.ask()
                fitness = [joint_objective(s) for s in solutions]
                es.tell(solutions, fitness)

            total_evals[0] += n_refine[0]

            result_4d = es.result.xbest
            x1, y1, x2, y2 = result_4d
            (q1, q2), Y_pred, final_rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
        else:
            # Compute joint RMSE without refinement
            (q1, q2), Y_pred, final_rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)

        sources = [(float(x1), float(y1), float(q1)), (float(x2), float(y2), float(q2))]
        return sources, final_rmse, total_evals[0]

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources with sequential estimation for 2-source problems.
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

        if n_sources == 1:
            # Standard 1-source optimization
            smart_init = self._smart_init_positions(sample, 1)[:2]
            tri_init = self._triangulation_init_positions(sample, meta, 1, q_range)
            init_positions = [smart_init]
            if tri_init is not None:
                init_positions.append(tri_init[:2])

            x, y, q, rmse, n_evals = self._optimize_1src(
                Y_observed, solver, dt, nt, T0, sensors_xy, q_range,
                init_positions, self.max_fevals_1src, self.sigma0_1src
            )

            sources = [(float(x), float(y), float(q))]
            results = [CandidateResult(
                params=np.array([x, y, q]),
                rmse=rmse,
                init_type='sequential',
                n_evals=n_evals
            )]

            return [sources], rmse, results, n_evals

        else:
            # Sequential 2-source optimization
            sources, rmse, n_evals = self._sequential_2src(sample, meta, solver, q_range)

            params_flat = []
            for s in sources:
                params_flat.extend([s[0], s[1], s[2]])
            results = [CandidateResult(
                params=np.array(params_flat),
                rmse=rmse,
                init_type='sequential',
                n_evals=n_evals
            )]

            return [sources], rmse, results, n_evals
