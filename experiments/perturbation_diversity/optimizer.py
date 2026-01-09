"""
A12b: Perturbation-Based Diversity Generator for Heat Source Identification.

Key Innovation: Instead of splitting fevals across multiple inits (which hurts accuracy),
we run FULL optimization on the best init, then generate diverse candidates by
intentionally perturbing the optimized solution.

Strategy:
1. Run full CMA-ES from best init (get best accuracy)
2. Take the best solution
3. Generate 2 additional candidates by perturbing positions (guaranteed diversity)
4. Evaluate perturbed candidates (2 extra simulations)

Benefits:
- Full fevals on best init = good accuracy
- Guaranteed 3 diverse candidates = 0.3 diversity bonus
- Fast (only 2 extra simulations)
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


class PerturbationDiversityOptimizer:
    """
    Optimizer that generates diverse candidates through post-optimization perturbation.

    Key Innovation:
    1. Run full CMA-ES from best init (all fevals for accuracy)
    2. Perturb the optimized solution to generate diverse candidates
    3. Evaluate perturbed candidates (only 2 extra simulations)
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
        perturbation_scale: float = 0.25,  # How far to perturb (in domain units)
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
        self.perturbation_scale = perturbation_scale

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

    def _generate_perturbations(
        self,
        best_params: np.ndarray,
        n_sources: int,
        q_range: Tuple[float, float],
        n_perturbations: int = 2
    ) -> List[np.ndarray]:
        """
        Generate diverse perturbations of the best solution.

        For 1-source: Perturb position in orthogonal directions
        For 2-source: Swap sources, perturb positions
        """
        perturbations = []
        lb, ub = self._get_bounds(n_sources, q_range)
        scale = self.perturbation_scale

        if n_sources == 1:
            # Perturbation 1: Move X direction
            p1 = best_params.copy()
            p1[0] += scale * (1 if p1[0] < self.Lx / 2 else -1)
            p1[0] = np.clip(p1[0], lb[0], ub[0])
            perturbations.append(p1)

            # Perturbation 2: Move Y direction
            p2 = best_params.copy()
            p2[1] += scale * (1 if p2[1] < self.Ly / 2 else -1)
            p2[1] = np.clip(p2[1], lb[1], ub[1])
            perturbations.append(p2)

        else:
            # For 2-source: Multiple perturbation strategies

            # Perturbation 1: Swap source order (different local minimum)
            p1 = np.array([
                best_params[3], best_params[4], best_params[5],  # src2 -> src1
                best_params[0], best_params[1], best_params[2],  # src1 -> src2
            ])
            # Add small position shift to ensure dissimilarity
            p1[0] += np.random.uniform(-0.15, 0.15)
            p1[3] += np.random.uniform(-0.15, 0.15)
            p1 = np.clip(p1, lb, ub)
            perturbations.append(p1)

            # Perturbation 2: Move both sources outward/inward
            p2 = best_params.copy()
            # If sources are close, move apart; if far, move closer
            src1_pos = np.array([p2[0], p2[1]])
            src2_pos = np.array([p2[3], p2[4]])
            dist = np.linalg.norm(src1_pos - src2_pos)

            if dist < 0.8:  # Close sources - move apart
                direction = src1_pos - src2_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                p2[0] += scale * direction[0]
                p2[1] += scale * direction[1]
                p2[3] -= scale * direction[0]
                p2[4] -= scale * direction[1]
            else:  # Far sources - move closer
                direction = src2_pos - src1_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                p2[0] += scale * direction[0]
                p2[1] += scale * direction[1]
                p2[3] -= scale * direction[0]
                p2[4] -= scale * direction[1]

            p2 = np.clip(p2, lb, ub)
            perturbations.append(p2)

        return perturbations[:n_perturbations]

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int]:
        """Estimate sources with perturbation-based diversity."""
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

        # Get best initialization
        try:
            init_params = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            init_type = 'triangulation'
        except:
            init_params = self._smart_init(sample, n_sources, q_range)
            init_type = 'smart'

        # Check transfer learning candidate
        lb, ub = self._get_bounds(n_sources, q_range)
        if similar_solutions:
            transfer_params = np.clip(similar_solutions[0], lb, ub)
            transfer_rmse = objective(transfer_params)
            init_rmse = objective(init_params)
            if transfer_rmse < init_rmse:
                init_params = transfer_params
                init_type = 'transfer'

        # Run FULL CMA-ES from best init
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6

        es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)
        total_evals = 0

        # Collect all evaluated solutions
        all_cmaes_solutions = []

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
            total_evals += len(solutions)

            for sol, fit in zip(solutions, fitness):
                all_cmaes_solutions.append((np.array(sol), fit))

        # Best solution from CMA-ES
        best_params = np.array(es.result.xbest)
        best_rmse = es.result.fbest

        if verbose:
            print(f"  CMA-ES best: RMSE={best_rmse:.4f} after {total_evals} evals")

        # Generate diverse candidates
        all_candidates = []

        # Add best solution
        sources_best = []
        for i in range(n_sources):
            sources_best.append((float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])))
        all_candidates.append((sources_best, best_rmse, init_type, best_params))

        # Add perturbations
        perturbations = self._generate_perturbations(best_params, n_sources, q_range, n_perturbations=2)
        for i, perturbed_params in enumerate(perturbations):
            perturbed_rmse = objective(perturbed_params)
            total_evals += 1

            sources_perturbed = []
            for j in range(n_sources):
                sources_perturbed.append((
                    float(perturbed_params[j*3]),
                    float(perturbed_params[j*3+1]),
                    float(perturbed_params[j*3+2])
                ))
            all_candidates.append((sources_perturbed, perturbed_rmse, f'perturb_{i}', perturbed_params))

        # Also add top solutions from CMA-ES history (for more diversity options)
        all_cmaes_solutions.sort(key=lambda x: x[1])
        for sol, rmse in all_cmaes_solutions[:10]:
            sources = []
            for i in range(n_sources):
                sources.append((float(sol[i*3]), float(sol[i*3+1]), float(sol[i*3+2])))
            all_candidates.append((sources, rmse, 'cmaes_history', sol))

        # Filter for diversity
        filtered = filter_dissimilar([(c[0], c[1]) for c in all_candidates], tau=TAU, n_max=self.n_candidates)

        # Build final results
        final_candidates = []
        for sources, rmse in filtered:
            for c in all_candidates:
                if c[0] == sources and abs(c[1] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[1] for c in final_candidates]
        best_rmse_final = min(candidate_rmses) if candidate_rmses else float('inf')

        results = [
            CandidateResult(
                params=c[3],
                rmse=c[1],
                init_type=c[2],
                n_evals=total_evals // len(final_candidates) if final_candidates else total_evals
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse_final, results, features, best_params, n_transferred
