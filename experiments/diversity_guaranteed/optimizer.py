"""
A12: Guaranteed Diversity Generator for Heat Source Identification.

Key Innovation: Instead of hoping for diverse candidates from single optimization,
we GUARANTEE diversity by:
1. Generating structurally different initializations (left-right split, top-bottom split)
2. Running dedicated CMA-ES from each initialization
3. Producing candidates that explore different local minima

Hypothesis: The scoring formula rewards 3 diverse candidates (0.3 points = 23% of max).
By ensuring diversity through structural variation in starting points, we should
maximize the diversity bonus while maintaining good accuracy.
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


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    """Enhanced feature extraction for similarity matching."""
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    basic = [
        np.max(Y) / 10.0,
        np.mean(Y) / 5.0,
        np.std(Y) / 2.0,
        kappa * 10,
        len(sensors) / 10.0,
    ]

    max_temps_per_sensor = Y.max(axis=0)
    weights = max_temps_per_sensor / (max_temps_per_sensor.sum() + 1e-8)
    centroid_x = np.average(sensors[:, 0], weights=weights) / 2.0
    centroid_y = np.average(sensors[:, 1], weights=weights)
    spatial_spread = np.sqrt(
        np.average((sensors[:, 0] / 2.0 - centroid_x)**2 +
                   (sensors[:, 1] - centroid_y)**2, weights=weights)
    )
    spatial = [centroid_x, centroid_y, spatial_spread]

    onset_times = []
    for i in range(Y.shape[1]):
        signal = Y[:, i]
        threshold = 0.1 * (signal.max() + 1e-8)
        onset_idx = np.argmax(signal > threshold)
        onset_times.append(onset_idx)
    onset_times = np.array(onset_times)
    temporal = [np.mean(onset_times) / 100.0, np.std(onset_times) / 50.0]

    if Y.shape[1] > 1:
        try:
            corr_matrix = np.corrcoef(Y.T)
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix[triu_indices]
            correlations = correlations[~np.isnan(correlations)]
            avg_corr = np.mean(correlations) if len(correlations) > 0 else 0.5
        except:
            avg_corr = 0.5
    else:
        avg_corr = 1.0

    return np.array(basic + spatial + temporal + [avg_corr])


def find_similar_solutions(
    features: np.ndarray,
    history: List[Tuple[np.ndarray, np.ndarray]],
    k: int = 1
) -> List[np.ndarray]:
    """Find k most similar solutions from history."""
    if not history or k == 0:
        return []

    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


class DiversityGuaranteedOptimizer:
    """
    Optimizer that GUARANTEES diverse candidates through structural variation.

    Key Innovation:
    - For 2-source: Generate 3 structurally different starting points
      1. Triangulation-based (current best method)
      2. Left-Right split (assume horizontal separation)
      3. Top-Bottom split (assume vertical separation)
    - Run dedicated CMA-ES from each, ensuring exploration of different local minima
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 12,
        max_fevals_2src: int = 23,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        k_similar: int = 1,
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
        """Get initialization from hottest sensors."""
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
        q_mean = np.mean(q_range)
        for idx in selected:
            x, y = sensors[idx]
            params.extend([x, y, q_mean])

        return np.array(params)

    def _generate_diverse_2src_inits(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate 3 STRUCTURALLY DIFFERENT initializations for 2-source.

        This is the key innovation - we ensure diversity by construction:
        1. Triangulation/Smart: Best physics-based estimate
        2. Left-Right Split: Assume sources are horizontally separated
        3. Top-Bottom Split: Assume sources are vertically separated
        """
        inits = []
        sensors = np.array(sample['sensors_xy'])
        readings = sample['Y_noisy']
        avg_temps = np.mean(readings, axis=0)
        q_mean = np.mean(q_range)

        # 1. Triangulation or Smart init (best physics-based)
        try:
            tri_params = triangulation_init(sample, meta, 2, q_range, self.Lx, self.Ly)
            inits.append((tri_params, 'triangulation'))
        except:
            smart_params = self._smart_init(sample, 2, q_range)
            inits.append((smart_params, 'smart'))

        # 2. Left-Right Split
        # Find hottest sensor in left half and right half
        left_mask = sensors[:, 0] < self.Lx / 2
        right_mask = ~left_mask

        left_temps = avg_temps.copy()
        left_temps[~left_mask] = -np.inf
        right_temps = avg_temps.copy()
        right_temps[~right_mask] = -np.inf

        if np.any(left_mask) and np.any(right_mask):
            left_idx = np.argmax(left_temps)
            right_idx = np.argmax(right_temps)

            x1, y1 = sensors[left_idx]
            x2, y2 = sensors[right_idx]

            # Adjust positions slightly inward to avoid boundaries
            x1 = np.clip(x1, 0.1, 0.9)
            x2 = np.clip(x2, 1.1, 1.9)
            y1 = np.clip(y1, 0.1, 0.9)
            y2 = np.clip(y2, 0.1, 0.9)

            lr_params = np.array([x1, y1, q_mean, x2, y2, q_mean])
            inits.append((lr_params, 'left_right'))

        # 3. Top-Bottom Split
        # Find hottest sensor in top half and bottom half
        top_mask = sensors[:, 1] > self.Ly / 2
        bottom_mask = ~top_mask

        top_temps = avg_temps.copy()
        top_temps[~top_mask] = -np.inf
        bottom_temps = avg_temps.copy()
        bottom_temps[~bottom_mask] = -np.inf

        if np.any(top_mask) and np.any(bottom_mask):
            top_idx = np.argmax(top_temps)
            bottom_idx = np.argmax(bottom_temps)

            x1, y1 = sensors[top_idx]
            x2, y2 = sensors[bottom_idx]

            # Adjust positions slightly inward
            x1 = np.clip(x1, 0.1, 1.9)
            x2 = np.clip(x2, 0.1, 1.9)
            y1 = np.clip(y1, 0.55, 0.95)
            y2 = np.clip(y2, 0.05, 0.45)

            tb_params = np.array([x1, y1, q_mean, x2, y2, q_mean])
            inits.append((tb_params, 'top_bottom'))

        # 4. Transfer learning (if available)
        # Will be added by caller

        # Ensure we have at least 3 inits
        if len(inits) < 3:
            # Add perturbed versions of triangulation
            base = inits[0][0].copy()
            for i in range(3 - len(inits)):
                perturbed = base.copy()
                perturbed[0] += np.random.uniform(-0.3, 0.3)  # x1
                perturbed[3] += np.random.uniform(-0.3, 0.3)  # x2
                perturbed[0] = np.clip(perturbed[0], 0.1, 1.9)
                perturbed[3] = np.clip(perturbed[3], 0.1, 1.9)
                inits.append((perturbed, f'perturbed_{i}'))

        return inits

    def _generate_diverse_1src_inits(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, str]]:
        """Generate diverse initializations for 1-source."""
        inits = []
        sensors = np.array(sample['sensors_xy'])
        readings = sample['Y_noisy']
        avg_temps = np.mean(readings, axis=0)
        q_mean = np.mean(q_range)

        # 1. Triangulation
        try:
            tri_params = triangulation_init(sample, meta, 1, q_range, self.Lx, self.Ly)
            inits.append((tri_params, 'triangulation'))
        except:
            pass

        # 2. Hottest sensor
        hot_idx = np.argmax(avg_temps)
        x, y = sensors[hot_idx]
        smart_params = np.array([x, y, q_mean])
        inits.append((smart_params, 'smart'))

        # 3. Second hottest sensor (for diversity)
        temps_copy = avg_temps.copy()
        temps_copy[hot_idx] = -np.inf
        second_hot_idx = np.argmax(temps_copy)
        x2, y2 = sensors[second_hot_idx]
        second_params = np.array([x2, y2, q_mean])
        inits.append((second_params, 'second_hot'))

        return inits

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """Estimate sources with guaranteed diversity."""
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        # Feature extraction
        features = extract_enhanced_features(sample, meta)

        # Transfer learning
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

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

        # Generate structurally diverse initializations
        if n_sources == 1:
            initializations = self._generate_diverse_1src_inits(sample, meta, q_range)
        else:
            initializations = self._generate_diverse_2src_inits(sample, meta, q_range)

        # Add transfer learning candidates
        lb, ub = self._get_bounds(n_sources, q_range)
        for i, sol in enumerate(similar_solutions):
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        if verbose:
            print(f"  Generated {len(initializations)} diverse inits")

        # Allocate fevals across initializations
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        n_inits = min(len(initializations), 3)  # Run at most 3 inits
        fevals_per_init = max(5, max_fevals // n_inits)

        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        # Run CMA-ES from each initialization
        all_candidates = []  # List of (sources, rmse, init_type, params)
        total_evals = 0

        for init_params, init_type in initializations[:n_inits]:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)
                total_evals += len(solutions)

                # Collect all evaluated solutions for candidate pool
                for sol, fit in zip(solutions, fitness):
                    sources = []
                    for i in range(n_sources):
                        sources.append((float(sol[i*3]), float(sol[i*3+1]), float(sol[i*3+2])))
                    all_candidates.append((sources, fit, init_type, np.array(sol)))

        # Sort by RMSE and filter for diversity
        all_candidates.sort(key=lambda x: x[1])

        # Filter for dissimilar candidates
        filtered = filter_dissimilar([(c[0], c[1]) for c in all_candidates], tau=TAU, n_max=self.n_candidates)

        if verbose:
            print(f"  After filtering: {len(filtered)} candidates")

        # Build final results
        final_candidates = []
        for sources, rmse in filtered:
            # Find matching candidate to get init_type and params
            for c in all_candidates:
                if c[0] == sources and abs(c[1] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[1] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        # Best params for history
        if final_candidates:
            best_idx = np.argmin([c[1] for c in final_candidates])
            best_params = final_candidates[best_idx][3]
        else:
            best_params = initializations[0][0]

        results = [
            CandidateResult(
                params=c[3],
                rmse=c[1],
                init_type=c[2],
                n_evals=total_evals // len(final_candidates) if final_candidates else total_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, features, best_params, n_transferred
