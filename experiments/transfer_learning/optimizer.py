"""
Transfer Learning Optimizer for Heat Source Identification.

Key Innovation: Uses solutions from similar previously-solved samples
as additional initializations, demonstrating learning at inference.

This addresses the evaluation criterion:
"Evidence of learning at inference or adaptive refinement"

How it works:
1. Process samples in batches to maintain parallelism
2. After each batch, store (features, best_solution) pairs
3. For subsequent batches, find similar samples and use their solutions
4. This shows the optimizer LEARNS and IMPROVES as it processes more samples
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma
from scipy.optimize import minimize_scalar

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


def extract_features(sample: Dict) -> np.ndarray:
    """
    Extract features for similarity matching.

    Features are chosen to capture thermal characteristics:
    - Temperature statistics (max, mean, std)
    - Physical parameters (kappa)
    - Problem structure (n_sensors)

    Fast: O(n) in data size, ~0.1ms per sample.
    """
    Y = sample['Y_noisy']

    # Normalize features to similar scales
    return np.array([
        np.max(Y) / 10.0,           # Peak temp (normalized)
        np.mean(Y) / 5.0,           # Mean temp (normalized)
        np.std(Y) / 2.0,            # Temp variance (normalized)
        sample['sample_metadata']['kappa'] * 10,  # Diffusivity (scaled up)
        len(sample['sensors_xy']) / 10.0,         # Sensor count (normalized)
    ])


def find_similar_solutions(
    features: np.ndarray,
    history: List[Tuple[np.ndarray, np.ndarray]],
    k: int = 2
) -> List[np.ndarray]:
    """
    Find k most similar solutions from history.

    Uses Euclidean distance on feature vectors.
    Returns solutions (not features) for use as initializations.
    """
    if not history:
        return []

    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


class TransferLearningOptimizer:
    """
    CMA-ES optimizer with transfer learning from similar samples.

    Innovation: Uses solutions from previously-solved similar samples
    as additional initializations. This demonstrates:
    - Learning at inference (optimizer improves over time)
    - Adaptive refinement (uses past experience)
    - Generalizable pattern (works for any simulation-driven problem)
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 20,
        max_fevals_2src: int = 40,
        sigma0_1src: float = 0.10,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        intensity_polish: bool = True,
        intensity_polish_maxiter: int = 5,
        candidate_pool_size: int = 10,
        k_similar: int = 2,
    ):
        """
        Initialize the transfer learning optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_fevals_1src: Max CMA-ES evaluations for 1-source
            max_fevals_2src: Max CMA-ES evaluations for 2-source
            sigma0_1src: Initial step size for 1-source
            sigma0_2src: Initial step size for 2-source
            use_triangulation: Use physics-based triangulation init
            n_candidates: Target candidates per sample
            intensity_polish: Apply intensity-only polish
            intensity_polish_maxiter: Polish iterations
            candidate_pool_size: Solutions to consider for filtering
            k_similar: Number of similar samples to transfer from
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
        self.k_similar = k_similar

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05):
        """Get parameter bounds."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])
        return lb, ub

    def _smart_init(self, sample: Dict, n_sources: int, q_range: Tuple[float, float]) -> np.ndarray:
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

    def _triangulation_init(self, sample: Dict, meta: Dict, n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get physics-based triangulation initialization."""
        if not self.use_triangulation:
            return None
        try:
            return triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
        except Exception:
            return None

    def _intensity_polish_1source(self, x, y, q_init, objective_fn, q_range):
        """Polish intensity for 1-source."""
        n_evals = 0
        def obj_q(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x, y, q])

        result = minimize_scalar(obj_q, bounds=q_range, method='bounded',
                                 options={'maxiter': self.intensity_polish_maxiter})
        return result.x, result.fun, n_evals

    def _intensity_polish_2source(self, positions, q_inits, objective_fn, q_range):
        """Polish intensities for 2-source (alternating)."""
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        q1, q2 = q_inits
        n_evals = 0

        def obj_q1(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x1, y1, q, x2, y2, q2])

        result1 = minimize_scalar(obj_q1, bounds=q_range, method='bounded',
                                  options={'maxiter': self.intensity_polish_maxiter // 2})
        q1 = result1.x

        def obj_q2(q):
            nonlocal n_evals
            n_evals += 1
            return objective_fn([x1, y1, q1, x2, y2, q])

        result2 = minimize_scalar(obj_q2, bounds=q_range, method='bounded',
                                  options={'maxiter': self.intensity_polish_maxiter // 2})
        q2 = result2.x

        return [q1, q2], result2.fun, n_evals

    def _polish_candidate(self, params, n_sources, objective_fn, q_range):
        """Apply intensity-only polish to a candidate."""
        if n_sources == 1:
            x, y, q = params[0], params[1], params[2]
            q_opt, rmse, n_evals = self._intensity_polish_1source(x, y, q, objective_fn, q_range)
            return np.array([x, y, q_opt]), rmse, n_evals
        else:
            positions = [(params[0], params[1]), (params[3], params[4])]
            q_inits = [params[2], params[5]]
            q_opts, rmse, n_evals = self._intensity_polish_2source(positions, q_inits, objective_fn, q_range)
            return np.array([positions[0][0], positions[0][1], q_opts[0],
                            positions[1][0], positions[1][1], q_opts[1]]), rmse, n_evals

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """
        Estimate sources with transfer learning.

        Args:
            sample: Sample data
            meta: Dataset metadata
            q_range: Intensity bounds
            history_1src: History for 1-source samples [(features, solution), ...]
            history_2src: History for 2-source samples [(features, solution), ...]
            verbose: Print debug info

        Returns:
            candidates: List of candidate solutions
            best_rmse: Best RMSE achieved
            results: List of CandidateResult objects
            features: Extracted features for this sample
            best_params: Best solution parameters (for history)
            n_transferred: Number of transferred initializations used
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        # Objective function
        def objective(params):
            sources = [{'x': params[i*3], 'y': params[i*3+1], 'q': params[i*3+2]}
                      for i in range(n_sources)]
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            return np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

        # === FEATURE EXTRACTION ===
        features = extract_features(sample)

        # === TRANSFER LEARNING: Find similar solutions ===
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []

        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        if verbose:
            print(f"  Transfer learning: {n_transferred} similar solutions from history of {len(history)}")

        # === BUILD INITIALIZATION POOL ===
        initializations = []

        # 1. Triangulation (physics-informed)
        tri_init = self._triangulation_init(sample, meta, n_sources, q_range)
        if tri_init is not None:
            initializations.append((tri_init, 'triangulation'))

        # 2. Hottest sensor fallback
        smart_init = self._smart_init(sample, n_sources, q_range)
        initializations.append((smart_init, 'smart'))

        # 3. TRANSFER LEARNING: Add similar solutions
        for i, sol in enumerate(similar_solutions):
            # Validate solution is within bounds
            lb, ub = self._get_bounds(n_sources, q_range)
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        if verbose:
            print(f"  Total initializations: {len(initializations)} ({n_transferred} from transfer)")

        # === CMA-ES OPTIMIZATION ===
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_bounds(n_sources, q_range)

        # Distribute fevals across initializations
        fevals_per_init = max(5, max_fevals // len(initializations))

        all_solutions = []
        n_evals = 0

        for init_params, init_type in initializations:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)
                n_evals += len(solutions)

                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit, init_type))

        # === CANDIDATE SELECTION ===
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        # Convert to candidate format
        candidates_raw = []
        for params, rmse, init_type in top_solutions:
            sources = [(float(params[i*3]), float(params[i*3+1]), float(params[i*3+2]))
                      for i in range(n_sources)]
            candidates_raw.append((sources, params, rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        # Match filtered back to full candidates
        filtered_full = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    filtered_full.append(c)
                    break

        # === INTENSITY POLISH ===
        final_candidates = []
        total_polish_evals = 0

        for sources, params, rmse, init_type in filtered_full:
            if self.intensity_polish:
                polished_params, polished_rmse, polish_evals = self._polish_candidate(
                    params, n_sources, objective, q_range)
                total_polish_evals += polish_evals

                if polished_rmse < rmse:
                    params = polished_params
                    rmse = polished_rmse
                    sources = [(float(params[i*3]), float(params[i*3+1]), float(params[i*3+2]))
                              for i in range(n_sources)]

            final_candidates.append((sources, params, rmse, init_type))

        n_evals += total_polish_evals

        # === BUILD RESULTS ===
        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        # Best params for history
        if final_candidates:
            best_idx = np.argmin([c[2] for c in final_candidates])
            best_params = final_candidates[best_idx][1]
        else:
            best_params = smart_init

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_evals // len(final_candidates) if final_candidates else n_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, features, best_params, n_transferred
