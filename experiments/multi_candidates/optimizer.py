"""
Multiple Candidates Optimizer for Heat Source Identification.

Key insight: The competition scoring formula rewards diversity:
  P = (1/N_valid) * Σ(1/(1+RMSE)) + λ * (N_valid/N_max)

Math comparison:
- 3 candidates @ RMSE=0.5: score = 0.667 + 0.3 = 0.967
- 1 candidate @ RMSE=0.3: score = 0.769 + 0.1 = 0.869

Strategy:
1. Run CMA-ES and collect population solutions (not just best)
2. Select top N candidates by RMSE
3. Apply dissimilarity filtering (τ = 0.2)
4. Optionally polish each candidate
5. Return up to N_max = 3 valid candidates
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma
from scipy.optimize import minimize_scalar

# Add project root to path for imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add the starter notebook path to import the simulator
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


# Competition parameters
N_MAX = 3  # Maximum candidates per sample
LAMBDA = 0.3  # Diversity weight
TAU = 0.2  # Dissimilarity threshold
SCALE_FACTORS = (2.0, 1.0, 2.0)  # (Lx, Ly, q_max) for normalization


@dataclass
class CandidateResult:
    """Result from optimization."""
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    """Normalize source parameters using scale factors."""
    normalized = []
    for x, y, q in sources:
        normalized.append([
            x / SCALE_FACTORS[0],  # x / Lx
            y / SCALE_FACTORS[1],  # y / Ly
            q / SCALE_FACTORS[2],  # q / q_max
        ])
    return np.array(normalized)


def source_distance(src1: np.ndarray, src2: np.ndarray) -> float:
    """Compute Euclidean distance between two normalized sources."""
    return np.linalg.norm(src1 - src2)


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
    """
    Compute minimum distance between two candidate source sets.

    For 2-source problems, considers both permutations.
    """
    norm1 = normalize_sources(sources1)
    norm2 = normalize_sources(sources2)

    n = len(sources1)
    if n != len(sources2):
        return float('inf')  # Different number of sources

    if n == 1:
        return source_distance(norm1[0], norm2[0])

    # For 2-source: try both permutations
    # Distance = max of min-paired distances
    min_total = float('inf')
    for perm in permutations(range(n)):
        total = 0
        for i, j in enumerate(perm):
            total += source_distance(norm1[i], norm2[j]) ** 2
        total = np.sqrt(total / n)  # RMS of distances
        min_total = min(min_total, total)

    return min_total


def filter_dissimilar(candidates: List[Tuple], tau: float = TAU) -> List[Tuple]:
    """
    Filter candidates to keep only dissimilar ones.

    Uses greedy selection: keep first, then add only if dissimilar to all kept.
    """
    if not candidates:
        return []

    kept = [candidates[0]]

    for cand in candidates[1:]:
        is_similar = False
        for kept_cand in kept:
            dist = candidate_distance(cand[0], kept_cand[0])
            if dist < tau:
                is_similar = True
                break
        if not is_similar:
            kept.append(cand)
            if len(kept) >= N_MAX:
                break

    return kept


class MultiCandidateOptimizer:
    """
    CMA-ES optimizer that returns multiple diverse candidates.

    Instead of returning only the best solution, this optimizer:
    1. Collects all solutions evaluated during CMA-ES
    2. Selects top candidates by RMSE
    3. Filters to keep only dissimilar candidates
    4. Returns up to N_max valid candidates
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 25,
        max_fevals_2src: int = 45,
        sigma0_1src: float = 0.10,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        intensity_polish: bool = True,
        intensity_polish_maxiter: int = 5,
        candidate_pool_size: int = 10,
    ):
        """
        Initialize the optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_fevals_1src: Max CMA-ES evaluations for 1-source
            max_fevals_2src: Max CMA-ES evaluations for 2-source
            sigma0_1src: Initial step size for 1-source
            sigma0_2src: Initial step size for 2-source
            use_triangulation: Use triangulation for initialization
            n_candidates: Target number of candidates to return
            intensity_polish: Apply intensity-only polish to candidates
            intensity_polish_maxiter: Max iterations for intensity polish
            candidate_pool_size: Number of top solutions to consider
        """
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
        self.intensity_polish = intensity_polish
        self.intensity_polish_maxiter = intensity_polish_maxiter
        self.candidate_pool_size = candidate_pool_size

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(
        self,
        n_sources: int,
        q_range: Tuple[float, float],
        margin: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        """Get parameter bounds for CMA-ES."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return lb, ub

    def _smart_init(
        self,
        sample: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> np.ndarray:
        """Fallback initialization from hottest sensors."""
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
        max_temp = np.max(avg_temps) + 1e-8
        for idx in selected:
            x, y = sensors[idx]
            q = 0.5 + (avg_temps[idx] / max_temp) * 1.2
            q = np.clip(q, q_range[0], q_range[1])
            params.extend([x, y, q])

        return np.array(params)

    def _get_initial_point(
        self,
        sample: Dict,
        meta: Dict,
        n_sources: int,
        q_range: Tuple[float, float],
    ) -> np.ndarray:
        """Get initial point using triangulation or fallback."""
        if self.use_triangulation:
            try:
                return triangulation_init(
                    sample, meta, n_sources, q_range, self.Lx, self.Ly
                )
            except Exception:
                pass
        return self._smart_init(sample, n_sources, q_range)

    def _intensity_polish_1source(
        self,
        x: float,
        y: float,
        q_init: float,
        objective_fn,
        q_range: Tuple[float, float],
    ) -> Tuple[float, float, int]:
        """Polish intensity for 1-source problem."""
        n_evals = 0

        def obj_q(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x, y, q])

        result = minimize_scalar(
            obj_q,
            bounds=q_range,
            method='bounded',
            options={'maxiter': self.intensity_polish_maxiter}
        )
        return result.x, result.fun, n_evals

    def _intensity_polish_2source(
        self,
        positions: List[Tuple[float, float]],
        q_inits: List[float],
        objective_fn,
        q_range: Tuple[float, float],
    ) -> Tuple[List[float], float, int]:
        """Polish intensities for 2-source problem (alternating 1D)."""
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        q1, q2 = q_inits
        n_evals = 0

        # Single round of alternating optimization
        def obj_q1(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x1, y1, q, x2, y2, q2])

        result1 = minimize_scalar(
            obj_q1,
            bounds=q_range,
            method='bounded',
            options={'maxiter': self.intensity_polish_maxiter // 2}
        )
        q1 = result1.x

        def obj_q2(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x1, y1, q1, x2, y2, q])

        result2 = minimize_scalar(
            obj_q2,
            bounds=q_range,
            method='bounded',
            options={'maxiter': self.intensity_polish_maxiter // 2}
        )
        q2 = result2.x
        best_rmse = result2.fun

        return [q1, q2], best_rmse, n_evals

    def _polish_candidate(
        self,
        params: np.ndarray,
        n_sources: int,
        objective_fn,
        q_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, float, int]:
        """Apply intensity-only polish to a candidate."""
        if n_sources == 1:
            x, y, q = params[0], params[1], params[2]
            q_opt, rmse, n_evals = self._intensity_polish_1source(
                x, y, q, objective_fn, q_range
            )
            return np.array([x, y, q_opt]), rmse, n_evals
        else:
            positions = [(params[0], params[1]), (params[3], params[4])]
            q_inits = [params[2], params[5]]
            q_opts, rmse, n_evals = self._intensity_polish_2source(
                positions, q_inits, objective_fn, q_range
            )
            return np.array([
                positions[0][0], positions[0][1], q_opts[0],
                positions[1][0], positions[1][1], q_opts[1]
            ]), rmse, n_evals

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple[float, float, float]]], float, List[CandidateResult]]:
        """
        Estimate source parameters using CMA-ES with multiple candidates.

        Returns:
            candidates: List of candidate solutions, each is a list of (x, y, q) tuples
            best_rmse: RMSE of best candidate
            results: List of CandidateResult objects
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        # Extract metadata
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        # Create solver
        solver = self._create_solver(kappa, bc)

        # Objective function
        def objective(params):
            sources = []
            for i in range(n_sources):
                sources.append({
                    'x': params[i * 3],
                    'y': params[i * 3 + 1],
                    'q': params[i * 3 + 2]
                })
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            return np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

        # Get initialization
        x0 = self._get_initial_point(sample, meta, n_sources, q_range)

        # Set CMA-ES parameters
        if n_sources == 1:
            max_fevals = self.max_fevals_1src
            sigma0 = self.sigma0_1src
        else:
            max_fevals = self.max_fevals_2src
            sigma0 = self.sigma0_2src

        # Bounds for CMA-ES
        lb, ub = self._get_bounds(n_sources, q_range)

        # CMA-ES options
        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6
        opts['tolx'] = 1e-6

        if verbose:
            print(f"  CMA-ES: n_sources={n_sources}, max_fevals={max_fevals}")

        # Run CMA-ES and collect all evaluated solutions
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        n_evals = 0
        all_solutions = []  # (params, rmse) pairs

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
            n_evals += len(solutions)

            # Collect all solutions with their fitness
            for sol, fit in zip(solutions, fitness):
                all_solutions.append((np.array(sol), fit))

        # Sort by RMSE and take top candidates
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        if verbose:
            print(f"  CMA-ES complete: {n_evals} evals, top RMSE={top_solutions[0][1]:.4f}")

        # Convert to candidate format for filtering
        candidates_raw = []
        for params, rmse in top_solutions:
            sources = []
            for i in range(n_sources):
                x, y, q = params[i*3:(i+1)*3]
                sources.append((float(x), float(y), float(q)))
            candidates_raw.append((sources, params, rmse))

        # Apply dissimilarity filtering
        filtered = filter_dissimilar(candidates_raw, tau=TAU)

        if verbose:
            print(f"  Filtered to {len(filtered)} dissimilar candidates")

        # Polish each candidate if enabled
        final_candidates = []
        total_polish_evals = 0

        for sources, params, rmse in filtered:
            if self.intensity_polish:
                polished_params, polished_rmse, polish_evals = self._polish_candidate(
                    params, n_sources, objective, q_range
                )
                total_polish_evals += polish_evals

                # Use polished if better
                if polished_rmse < rmse:
                    params = polished_params
                    rmse = polished_rmse
                    sources = []
                    for i in range(n_sources):
                        x, y, q = params[i*3:(i+1)*3]
                        sources.append((float(x), float(y), float(q)))

            final_candidates.append((sources, rmse))

        n_evals += total_polish_evals

        if verbose:
            print(f"  Polish complete: +{total_polish_evals} evals")

        # Build results
        candidate_sources = [cand[0] for cand in final_candidates]
        candidate_rmses = [cand[1] for cand in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=np.array([p for src in sources for p in src]),
                rmse=rmse,
                init_type='triangulation' if self.use_triangulation else 'smart',
                n_evals=n_evals // len(final_candidates) if final_candidates else n_evals
            )
            for sources, rmse in final_candidates
        ]

        return candidate_sources, best_rmse, results
