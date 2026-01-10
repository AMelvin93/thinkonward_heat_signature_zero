"""
Sensor Subset Diversity Optimizer for Heat Source Identification.

Key Innovation: Use different sensor subsets to generate naturally diverse candidates.

Physics Insight (from 2025 SIAM paper):
    - Two boundary measurement points can uniquely determine a heat source
    - Our samples have 8-12 sensors (4-6x more than needed!)
    - Different sensor subsets should give different but valid solutions
    - This creates NATURAL diversity without artificial perturbation

Benefits:
1. Each candidate uses different information → different solution
2. All candidates should be high quality (each uses sufficient info)
3. Natural diversity (not forced perturbation)
4. Better captures diversity bonus (0.3 points)
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
    sensor_subset: str = "all"


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


def create_sensor_subsets(sensors_xy: np.ndarray, n_subsets: int = 3) -> List[Tuple[np.ndarray, str]]:
    """
    Create diverse sensor subsets for multi-candidate generation.

    Strategy:
    - Subset 1: All sensors (baseline)
    - Subset 2: Left half of sensors (x < median_x)
    - Subset 3: Right half of sensors (x >= median_x)

    For larger sensor counts, also consider:
    - Top/bottom split
    - Alternating sensors
    """
    n_sensors = len(sensors_xy)
    indices = np.arange(n_sensors)

    subsets = []

    # Subset 1: All sensors
    subsets.append((indices.copy(), "all"))

    if n_sensors >= 4:
        # Subset 2: Left half (by x coordinate)
        median_x = np.median(sensors_xy[:, 0])
        left_mask = sensors_xy[:, 0] < median_x
        left_indices = indices[left_mask]
        if len(left_indices) >= 3:  # Need at least 3 for meaningful optimization
            subsets.append((left_indices, "left"))

        # Subset 3: Right half (by x coordinate)
        right_indices = indices[~left_mask]
        if len(right_indices) >= 3:
            subsets.append((right_indices, "right"))

    if n_sensors >= 6 and len(subsets) < n_subsets:
        # Subset 4: Top half (by y coordinate)
        median_y = np.median(sensors_xy[:, 1])
        top_mask = sensors_xy[:, 1] >= median_y
        top_indices = indices[top_mask]
        if len(top_indices) >= 3:
            subsets.append((top_indices, "top"))

        # Subset 5: Bottom half
        bottom_indices = indices[~top_mask]
        if len(bottom_indices) >= 3:
            subsets.append((bottom_indices, "bottom"))

    if n_sensors >= 6 and len(subsets) < n_subsets:
        # Subset: Alternating sensors (odd indices)
        odd_indices = indices[1::2]
        if len(odd_indices) >= 3:
            subsets.append((odd_indices, "odd"))

    return subsets[:n_subsets]


class SensorSubsetDiversityOptimizer:
    """
    Optimizer that uses different sensor subsets to generate diverse candidates.

    Key Innovation: Run optimization on different sensor subsets, each giving
    a different but valid solution. Natural diversity from different information.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 20,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_subsets: int = 3,  # Number of sensor subsets to use
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
        self.n_subsets = n_subsets

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        """Get bounds for position parameters."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, Y_subset: np.ndarray, sensors_subset: np.ndarray,
                              n_sources: int) -> np.ndarray:
        """Get position initialization from hottest sensors in subset."""
        avg_temps = np.mean(Y_subset, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors_subset[idx] - sensors_subset[p]) >= 0.25
                   for p in selected):
                selected.append(idx)

        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        params = []
        for idx in selected:
            x, y = sensors_subset[idx]
            params.extend([x, y])

        return np.array(params)

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       sensor_indices: np.ndarray,
                                       n_sources: int,
                                       q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get triangulation init using specific sensor subset."""
        if not self.use_triangulation:
            return None
        try:
            # Create modified sample with subset of sensors
            subset_sample = sample.copy()
            subset_sample['sensors_xy'] = [sample['sensors_xy'][i] for i in sensor_indices]
            subset_sample['Y_noisy'] = sample['Y_noisy'][:, sensor_indices]

            full_init = triangulation_init(subset_sample, meta, n_sources, q_range,
                                           self.Lx, self.Ly)
            # Extract positions only
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _optimize_on_subset(
        self,
        Y_subset: np.ndarray,
        sensors_subset: np.ndarray,
        Y_full: np.ndarray,
        sensors_full: np.ndarray,
        sample: Dict,
        meta: Dict,
        solver: Heat2D,
        n_sources: int,
        dt: float,
        nt: int,
        T0: float,
        q_range: Tuple[float, float],
        sensor_indices: np.ndarray,
        subset_name: str,
    ) -> List[Tuple]:
        """
        Optimize using a subset of sensors, but evaluate RMSE on ALL sensors.

        This ensures:
        1. Different subsets find different solutions (diversity)
        2. Final RMSE is comparable across candidates (uses all sensors)
        """
        n_sims = [0]

        # Objective uses SUBSET for optimization guidance
        if n_sources == 1:
            def objective_subset(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                # Compute RMSE on subset for optimization
                q, _, rmse = compute_optimal_intensity_1src(
                    x, y, Y_subset, solver, dt, nt, T0, sensors_subset, q_range)
                return rmse
        else:
            def objective_subset(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), _, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_subset, solver, dt, nt, T0, sensors_subset, q_range)
                return rmse

        # Initialize from subset sensors
        smart_init = self._smart_init_positions(Y_subset, sensors_subset, n_sources)
        tri_init = self._triangulation_init_positions(sample, meta, sensor_indices,
                                                       n_sources, q_range)

        # Pick best init
        init_options = [(smart_init, 'smart')]
        if tri_init is not None:
            init_options.append((tri_init, 'triangulation'))

        best_init = None
        best_init_rmse = float('inf')
        best_init_type = 'smart'

        for init_params, init_type in init_options:
            try:
                rmse = objective_subset(init_params)
                if rmse < best_init_rmse:
                    best_init_rmse = rmse
                    best_init = init_params
                    best_init_type = init_type
            except:
                continue

        if best_init is None:
            best_init = smart_init

        # Run CMA-ES with subset objective
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_position_bounds(n_sources)

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        es = cma.CMAEvolutionStrategy(best_init.tolist(), sigma0, opts)

        all_solutions = []
        while not es.stop():
            solutions = es.ask()
            fitness = [objective_subset(s) for s in solutions]
            es.tell(solutions, fitness)

            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit))

        # Get best solution from subset optimization
        all_solutions.sort(key=lambda x: x[1])
        best_pos = all_solutions[0][0]

        # NOW evaluate on FULL sensor set for final RMSE
        if n_sources == 1:
            x, y = best_pos
            q, _, final_rmse = compute_optimal_intensity_1src(
                x, y, Y_full, solver, dt, nt, T0, sensors_full, q_range)
            full_params = np.array([x, y, q])
            sources = [(float(x), float(y), float(q))]
        else:
            x1, y1, x2, y2 = best_pos
            (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                x1, y1, x2, y2, Y_full, solver, dt, nt, T0, sensors_full, q_range)
            full_params = np.array([x1, y1, q1, x2, y2, q2])
            sources = [(float(x1), float(y1), float(q1)),
                      (float(x2), float(y2), float(q2))]

        return [(sources, full_params, final_rmse, f"{best_init_type}_{subset_name}",
                 n_sims[0])]

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], int]:
        """
        Estimate sources using sensor subset diversity.

        Strategy:
        1. Create N sensor subsets (e.g., all, left, right)
        2. Run CMA-ES on each subset independently
        3. Each subset gives a candidate with different perspective
        4. Filter for dissimilarity and return top candidates
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

        # Create sensor subsets
        subsets = create_sensor_subsets(sensors_xy, n_subsets=self.n_subsets)

        if verbose:
            print(f"  Sensor subsets: {[s[1] for s in subsets]}")
            print(f"  Subset sizes: {[len(s[0]) for s in subsets]}")

        # Optimize on each subset
        all_candidates = []
        total_sims = 0

        for sensor_indices, subset_name in subsets:
            sensors_subset = sensors_xy[sensor_indices]
            Y_subset = Y_observed[:, sensor_indices]

            candidates = self._optimize_on_subset(
                Y_subset=Y_subset,
                sensors_subset=sensors_subset,
                Y_full=Y_observed,
                sensors_full=sensors_xy,
                sample=sample,
                meta=meta,
                solver=solver,
                n_sources=n_sources,
                dt=dt,
                nt=nt,
                T0=T0,
                q_range=q_range,
                sensor_indices=sensor_indices,
                subset_name=subset_name,
            )

            for c in candidates:
                all_candidates.append(c)
                total_sims += c[4]  # n_sims

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in all_candidates], tau=TAU)

        # Match filtered back to full candidates
        final_candidates = []
        for sources, rmse in filtered:
            for c in all_candidates:
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
                n_evals=c[4],
                sensor_subset=c[3].split('_')[-1] if '_' in c[3] else 'all'
            )
            for c in final_candidates
        ]

        if verbose:
            print(f"  Candidates: {len(final_candidates)}, best RMSE: {best_rmse:.4f}")
            for r in results:
                print(f"    - {r.init_type}: RMSE={r.rmse:.4f}")

        return candidate_sources, best_rmse, results, total_sims
