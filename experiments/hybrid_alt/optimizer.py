"""
A11: Hybrid Alternating + Standard CMA-ES Optimizer.

Key Insight from A10: Alternating optimization improves 2-source RMSE (0.22 vs 0.27-0.30)
but loses diversity score. This hybrid approach combines both:

For 2-source problems:
1. Primary candidate: Alternating optimization from best init (best accuracy)
2. Secondary candidate: Standard 4D CMA-ES from second init (diverse solution)
3. Tertiary: Third init evaluated (fast diversity)

This gives 3 diverse candidates while leveraging alternating optimization's accuracy.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations

import numpy as np
import cma

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


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    basic = [np.max(Y)/10.0, np.mean(Y)/5.0, np.std(Y)/2.0, kappa*10, len(sensors)/10.0]

    max_temps = Y.max(axis=0)
    weights = max_temps / (max_temps.sum() + 1e-8)
    cx = np.average(sensors[:, 0], weights=weights) / 2.0
    cy = np.average(sensors[:, 1], weights=weights)
    spread = np.sqrt(np.average((sensors[:, 0]/2.0 - cx)**2 + (sensors[:, 1] - cy)**2, weights=weights))
    spatial = [cx, cy, spread]

    onset_times = [np.argmax(Y[:, i] > 0.1*(Y[:, i].max() + 1e-8)) for i in range(Y.shape[1])]
    temporal = [np.mean(onset_times)/100.0, np.std(onset_times)/50.0]

    try:
        corr = np.corrcoef(Y.T)
        avg_corr = np.nanmean(corr[np.triu_indices_from(corr, k=1)])
    except:
        avg_corr = 0.5

    return np.array(basic + spatial + temporal + [avg_corr])


def find_similar_solutions(features, history, k=1):
    if not history or k == 0:
        return []
    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


def simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy):
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    return np.array([solver.sample_sensors(U, sensors_xy) for U in Us])


def compute_optimal_intensity_1src(x, y, Y_obs, solver, dt, nt, T0, sensors_xy, q_range):
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat, Y_obs_flat = Y_unit.flatten(), Y_obs.flatten()
    denom = np.dot(Y_unit_flat, Y_unit_flat)
    q = np.dot(Y_unit_flat, Y_obs_flat) / denom if denom > 1e-10 else 1.0
    q = np.clip(q, q_range[0], q_range[1])
    Y_pred = q * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_obs)**2))
    return q, Y_pred, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_obs, solver, dt, nt, T0, sensors_xy, q_range):
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    Y1_flat, Y2_flat, Y_obs_flat = Y1.flatten(), Y2.flatten(), Y_obs.flatten()
    A = np.array([[np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
                  [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]])
    b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])
    try:
        q1, q2 = np.linalg.solve(A + 1e-6*np.eye(2), b)
    except:
        q1, q2 = 1.0, 1.0
    q1, q2 = np.clip(q1, q_range[0], q_range[1]), np.clip(q2, q_range[0], q_range[1])
    Y_pred = q1*Y1 + q2*Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_obs)**2))
    return (q1, q2), Y_pred, rmse


class HybridAltOptimizer:
    """
    Hybrid optimizer combining alternating optimization (for accuracy) with
    standard CMA-ES (for diversity).
    """

    def __init__(
        self,
        Lx=2.0, Ly=1.0, nx=100, ny=50,
        max_fevals_1src=12,
        max_fevals_2src=23,
        sigma0_1src=0.15,
        sigma0_2src=0.20,
        use_triangulation=True,
        n_candidates=N_MAX,
        k_similar=1,
        n_alt_rounds=2,
    ):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.k_similar = k_similar
        self.n_alt_rounds = n_alt_rounds

    def _create_solver(self, kappa, bc):
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_bounds(self, n_sources, margin=0.05):
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1-margin) * self.Lx, (1-margin) * self.Ly])
        return lb, ub

    def _smart_init(self, sample, n_sources):
        readings, sensors = sample['Y_noisy'], sample['sensors_xy']
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
        return np.array([p for idx in selected for p in sensors[idx]])

    def _tri_init(self, sample, meta, n_sources, q_range):
        if not self.use_triangulation:
            return None
        try:
            full = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            return np.array([full[i*3+j] for i in range(n_sources) for j in range(2)])
        except:
            return None

    def _evaluate_init(self, params, n_sources, Y_obs, solver, dt, nt, T0, sensors, q_range):
        if n_sources == 1:
            _, _, rmse = compute_optimal_intensity_1src(params[0], params[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
        else:
            _, _, rmse = compute_optimal_intensity_2src(params[0], params[1], params[2], params[3], Y_obs, solver, dt, nt, T0, sensors, q_range)
        return rmse

    def _optimize_1src_cmaes(self, init, Y_obs, solver, dt, nt, T0, sensors, q_range, max_fevals):
        def obj(p):
            _, _, rmse = compute_optimal_intensity_1src(p[0], p[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
            return rmse
        lb, ub = self._get_bounds(1)
        opts = {'maxfevals': max_fevals, 'bounds': [lb, ub], 'verbose': -9}
        es = cma.CMAEvolutionStrategy(init.tolist(), self.sigma0_1src, opts)
        n_evals = 0
        while not es.stop():
            sols = es.ask()
            fits = [obj(s) for s in sols]
            n_evals += len(fits)
            es.tell(sols, fits)
        return np.array(es.result.xbest), es.result.fbest, n_evals

    def _optimize_2src_cmaes(self, init, Y_obs, solver, dt, nt, T0, sensors, q_range, max_fevals):
        """Standard 4D CMA-ES for 2-source."""
        def obj(p):
            _, _, rmse = compute_optimal_intensity_2src(p[0], p[1], p[2], p[3], Y_obs, solver, dt, nt, T0, sensors, q_range)
            return rmse
        lb, ub = self._get_bounds(2)
        opts = {'maxfevals': max_fevals, 'bounds': [lb, ub], 'verbose': -9}
        es = cma.CMAEvolutionStrategy(init.tolist(), self.sigma0_2src, opts)
        n_evals = 0
        while not es.stop():
            sols = es.ask()
            fits = [obj(s) for s in sols]
            n_evals += len(fits)
            es.tell(sols, fits)
        return np.array(es.result.xbest), es.result.fbest, n_evals

    def _optimize_2src_alternating(self, init, Y_obs, solver, dt, nt, T0, sensors, q_range, max_fevals):
        """Alternating 2D CMA-ES for 2-source."""
        x1, y1, x2, y2 = init
        s1, s2 = np.array([x1, y1]), np.array([x2, y2])
        fevals_per_sub = max(max_fevals // (2 * self.n_alt_rounds), 4)
        total_evals = 0
        lb1, ub1 = self._get_bounds(1)

        for _ in range(self.n_alt_rounds):
            # Optimize source 1
            def obj1(p):
                _, _, rmse = compute_optimal_intensity_2src(p[0], p[1], s2[0], s2[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
                return rmse
            opts = {'maxfevals': fevals_per_sub, 'bounds': [lb1, ub1], 'verbose': -9}
            es = cma.CMAEvolutionStrategy(s1.tolist(), 0.15, opts)
            while not es.stop():
                sols = es.ask()
                fits = [obj1(s) for s in sols]
                total_evals += len(fits)
                es.tell(sols, fits)
            s1 = np.array(es.result.xbest)

            # Optimize source 2
            def obj2(p):
                _, _, rmse = compute_optimal_intensity_2src(s1[0], s1[1], p[0], p[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
                return rmse
            es = cma.CMAEvolutionStrategy(s2.tolist(), 0.15, opts)
            while not es.stop():
                sols = es.ask()
                fits = [obj2(s) for s in sols]
                total_evals += len(fits)
                es.tell(sols, fits)
            s2 = np.array(es.result.xbest)

        best_params = np.array([s1[0], s1[1], s2[0], s2[1]])
        _, _, best_rmse = compute_optimal_intensity_2src(s1[0], s1[1], s2[0], s2[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
        return best_params, best_rmse, total_evals

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), history_1src=None, history_2src=None, verbose=False):
        n_sources = sample['n_sources']
        sensors = np.array(sample['sensors_xy'])
        Y_obs = sample['Y_noisy']
        dt, nt = meta['dt'], sample['sample_metadata']['nt']
        kappa, bc, T0 = sample['sample_metadata']['kappa'], sample['sample_metadata']['bc'], sample['sample_metadata']['T0']
        solver = self._create_solver(kappa, bc)

        features = extract_enhanced_features(sample, meta)
        history = history_1src if n_sources == 1 else history_2src
        similar_sols = find_similar_solutions(features, history or [], k=self.k_similar)
        n_transferred = len(similar_sols)

        # Build inits
        inits = []
        tri = self._tri_init(sample, meta, n_sources, q_range)
        if tri is not None:
            inits.append((tri, 'triangulation'))
        inits.append((self._smart_init(sample, n_sources), 'smart'))
        for i, sol in enumerate(similar_sols):
            pos = np.array([sol[j*3+k] for j in range(n_sources) for k in range(2)])
            inits.append((pos, f'transfer_{i}'))

        # Evaluate and sort inits
        init_results = [(p, t, self._evaluate_init(p, n_sources, Y_obs, solver, dt, nt, T0, sensors, q_range)) for p, t in inits]
        init_results.sort(key=lambda x: x[2])

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        all_results = []

        if n_sources == 1:
            # 1-source: standard CMA-ES from best init, add others as candidates
            for i, (init_p, init_t, init_r) in enumerate(init_results[:self.n_candidates]):
                if i == 0:
                    params, rmse, n_evals = self._optimize_1src_cmaes(init_p, Y_obs, solver, dt, nt, T0, sensors, q_range, max_fevals)
                else:
                    params, rmse, n_evals = init_p.copy(), init_r, 0
                q, _, rmse = compute_optimal_intensity_1src(params[0], params[1], Y_obs, solver, dt, nt, T0, sensors, q_range)
                full = np.array([params[0], params[1], q])
                all_results.append(CandidateResult(params=full, rmse=rmse, init_type=init_t, n_evals=n_evals))
        else:
            # 2-source: alternating from best init, standard CMA-ES from second, third as-is
            fevals_primary = int(max_fevals * 0.6)
            fevals_secondary = max_fevals - fevals_primary

            for i, (init_p, init_t, init_r) in enumerate(init_results[:self.n_candidates]):
                if i == 0:
                    # Primary: alternating optimization
                    params, rmse, n_evals = self._optimize_2src_alternating(init_p, Y_obs, solver, dt, nt, T0, sensors, q_range, fevals_primary)
                elif i == 1:
                    # Secondary: standard 4D CMA-ES
                    params, rmse, n_evals = self._optimize_2src_cmaes(init_p, Y_obs, solver, dt, nt, T0, sensors, q_range, fevals_secondary)
                else:
                    # Tertiary: just evaluate
                    params, rmse, n_evals = init_p.copy(), init_r, 0

                (q1, q2), _, rmse = compute_optimal_intensity_2src(params[0], params[1], params[2], params[3], Y_obs, solver, dt, nt, T0, sensors, q_range)
                full = np.array([params[0], params[1], q1, params[2], params[3], q2])
                all_results.append(CandidateResult(params=full, rmse=rmse, init_type=init_t, n_evals=n_evals))

        all_results.sort(key=lambda x: x.rmse)

        # Build candidates for filtering
        candidate_list = []
        for r in all_results:
            if n_sources == 1:
                srcs = [(r.params[0], r.params[1], r.params[2])]
            else:
                srcs = [(r.params[0], r.params[1], r.params[2]), (r.params[3], r.params[4], r.params[5])]
            candidate_list.append((srcs, r.rmse, r))

        filtered = filter_dissimilar(candidate_list, TAU, self.n_candidates)
        candidates = [f[0] for f in filtered]
        best_rmse = filtered[0][1] if filtered else float('inf')
        results = [f[2] for f in filtered]

        best = all_results[0] if all_results else None
        best_positions = best.params if best else np.zeros(n_sources * 3)

        return candidates, best_rmse, results, features, best_positions, n_transferred
