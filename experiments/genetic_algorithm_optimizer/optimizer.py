"""
Genetic Algorithm Optimizer for Heat Source Identification

Tests whether a simpler evolutionary algorithm (GA) can match CMA-ES performance.
CMA-ES has sophisticated covariance adaptation - GA uses simpler selection/crossover/mutation.

Key features:
- Real-valued encoding (position coordinates)
- Tournament selection
- Blend crossover (BLX-alpha)
- Gaussian mutation
- Elitism (preserve best solutions)
- 40% timesteps for evaluation (temporal fidelity)
- NM polish for final refinement
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

import numpy as np
from scipy.optimize import minimize

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
    """Run simulation and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(x, y, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensity for 1-source."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    n_steps = len(Y_unit)
    Y_obs_trunc = Y_observed[:n_steps]

    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator
    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])

    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy,
                                    q_range=(0.5, 2.0)):
    """Compute optimal intensities for 2-source."""
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)
    n_steps = len(Y1)
    Y_obs_trunc = Y_observed[:n_steps]

    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
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
    rmse = np.sqrt(np.mean((Y_pred - Y_obs_trunc) ** 2))
    return (q1, q2), Y_pred, rmse


class SimpleGA:
    """Simple real-valued Genetic Algorithm."""
    
    def __init__(self, objective, bounds, pop_size=15, mutation_rate=0.3, 
                 crossover_rate=0.7, elite_size=2, max_generations=10):
        self.objective = objective
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.n_dim = len(bounds)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.all_solutions = []
        
    def _init_population(self, seed_individual=None):
        """Initialize population with optional seed."""
        pop = []
        if seed_individual is not None:
            pop.append(np.clip(seed_individual, self.lb, self.ub))
        
        # Fill rest with random individuals
        while len(pop) < self.pop_size:
            ind = self.lb + np.random.random(self.n_dim) * (self.ub - self.lb)
            pop.append(ind)
        return pop
    
    def _evaluate(self, population):
        """Evaluate fitness of all individuals."""
        fitness = []
        for ind in population:
            f = self.objective(ind)
            fitness.append(f)
            self.all_solutions.append((ind.copy(), f))
        return fitness
    
    def _tournament_select(self, population, fitness, tournament_size=3):
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            contestants = np.random.choice(len(population), size=tournament_size, replace=False)
            winner = contestants[np.argmin([fitness[i] for i in contestants])]
            selected.append(population[winner].copy())
        return selected
    
    def _blx_crossover(self, parent1, parent2, alpha=0.5):
        """BLX-alpha crossover for real-valued encoding."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros(self.n_dim)
        child2 = np.zeros(self.n_dim)
        
        for i in range(self.n_dim):
            d = abs(parent1[i] - parent2[i])
            lower = min(parent1[i], parent2[i]) - alpha * d
            upper = max(parent1[i], parent2[i]) + alpha * d
            child1[i] = lower + np.random.random() * (upper - lower)
            child2[i] = lower + np.random.random() * (upper - lower)
        
        child1 = np.clip(child1, self.lb, self.ub)
        child2 = np.clip(child2, self.lb, self.ub)
        return child1, child2
    
    def _mutate(self, individual, sigma_frac=0.1):
        """Gaussian mutation."""
        mutant = individual.copy()
        for i in range(self.n_dim):
            if np.random.random() < self.mutation_rate:
                sigma = sigma_frac * (self.ub[i] - self.lb[i])
                mutant[i] += np.random.normal(0, sigma)
        return np.clip(mutant, self.lb, self.ub)
    
    def run(self, seed_individual=None, max_evals=None):
        """Run the GA."""
        population = self._init_population(seed_individual)
        fitness = self._evaluate(population)
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for gen in range(self.max_generations):
            if max_evals and len(self.all_solutions) >= max_evals:
                break
                
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness)
            elites = [population[i].copy() for i in sorted_indices[:self.elite_size]]
            
            # Selection
            selected = self._tournament_select(population, fitness)
            
            # Crossover and mutation
            new_population = elites.copy()
            while len(new_population) < self.pop_size:
                p1, p2 = np.random.choice(len(selected), size=2, replace=False)
                c1, c2 = self._blx_crossover(selected[p1], selected[p2])
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
            
            population = new_population[:self.pop_size]
            fitness = self._evaluate(population)
            
            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
        
        return best_individual, best_fitness, self.all_solutions


class GAOptimizer:
    """Genetic Algorithm Optimizer with temporal fidelity and NM polish."""

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx_fine: int = 100,
        ny_fine: int = 50,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        max_fevals_1src: int = 20,
        max_fevals_2src: int = 36,
        use_triangulation: bool = True,
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        refine_top_n: int = 2,
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
        timestep_fraction: float = 0.40,
        final_polish_maxiter: int = 8,
        # GA-specific parameters
        population_size: int = 15,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        max_generations: int = 10,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx_fine = nx_fine
        self.ny_fine = ny_fine
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.use_triangulation = use_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.refine_top_n = refine_top_n
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src
        self.timestep_fraction = timestep_fraction
        self.final_polish_maxiter = final_polish_maxiter
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations

    def _create_solver(self, kappa, bc, coarse=False):
        if coarse:
            return Heat2D(self.Lx, self.Ly, self.nx_coarse, self.ny_coarse, kappa, bc=bc)
        return Heat2D(self.Lx, self.Ly, self.nx_fine, self.ny_fine, kappa, bc=bc)

    def _get_position_bounds(self, n_sources, margin=0.05):
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
        return bounds

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

    def _weighted_centroid_init(self, sample, n_sources):
        readings = sample['Y_noisy']
        sensors = np.array(sample['sensors_xy'])
        max_temps = np.max(readings, axis=0)
        weights = max_temps / (max_temps.sum() + 1e-8)
        centroid = np.average(sensors, axis=0, weights=weights)

        if n_sources == 1:
            return np.array([centroid[0], centroid[1]])
        else:
            spread = np.sqrt(np.average(
                (sensors[:, 0] - centroid[0])**2 + (sensors[:, 1] - centroid[1])**2,
                weights=weights
            ))
            offset = max(0.1, spread * 0.3)
            return np.array([
                centroid[0] - offset, centroid[1],
                centroid[0] + offset, centroid[1]
            ])

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

    def _run_single_optimization(self, sample, meta, q_range, solver_coarse, solver_fine,
                                  initializations, n_sources, nt_reduced, nt_full):
        """Run GA optimization with final polish."""
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        T0 = sample['sample_metadata']['T0']

        n_sims = [0]

        if n_sources == 1:
            def objective_coarse(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

            def objective_fine_full(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective_coarse(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_coarse, dt, nt_reduced, T0, sensors_xy, q_range)
                return rmse

            def objective_fine_full(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                return rmse

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        bounds = self._get_position_bounds(n_sources)
        fevals_per_init = max(5, max_fevals // len(initializations))

        all_solutions = []

        # Run GA for each initialization
        for init_params, init_type in initializations:
            ga = SimpleGA(
                objective_coarse,
                bounds,
                pop_size=self.population_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                max_generations=self.max_generations,
            )
            best_ind, best_fit, solutions = ga.run(
                seed_individual=init_params,
                max_evals=fevals_per_init
            )
            
            for sol, fit in solutions:
                all_solutions.append((sol, fit, init_type))

        # Sort by fitness
        all_solutions.sort(key=lambda x: x[1])

        # Evaluate top candidates on FINE grid with FULL timesteps
        candidates_raw = []
        for pos_params, rmse_coarse, init_type in all_solutions[:self.candidate_pool_size]:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 1
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                n_sims[0] += 2
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type, pos_params))

        # FINAL POLISH: NM refinement on best candidate using FULL timesteps
        if self.final_polish_maxiter > 0 and candidates_raw:
            best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][2])
            best_pos_params = candidates_raw[best_idx][4]
            best_rmse = candidates_raw[best_idx][2]

            result = minimize(
                objective_fine_full,
                best_pos_params,
                method='Nelder-Mead',
                options={'maxiter': self.final_polish_maxiter, 'xatol': 0.005, 'fatol': 0.0005}
            )

            if result.fun < best_rmse:
                if n_sources == 1:
                    x, y = result.x
                    q, _, final_rmse = compute_optimal_intensity_1src(
                        x, y, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 1
                    full_params = np.array([x, y, q])
                    sources = [(float(x), float(y), float(q))]
                else:
                    x1, y1, x2, y2 = result.x
                    (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                        x1, y1, x2, y2, Y_observed, solver_fine, dt, nt_full, T0, sensors_xy, q_range)
                    n_sims[0] += 2
                    full_params = np.array([x1, y1, q1, x2, y2, q2])
                    sources = [(float(x1), float(y1), float(q1)),
                              (float(x2), float(y2), float(q2))]

                candidates_raw[best_idx] = (sources, full_params, final_rmse, 'polished', result.x)

        # Remove pos_params from output
        candidates_raw = [(c[0], c[1], c[2], c[3]) for c in candidates_raw]

        return candidates_raw, n_sims[0]

    def estimate_sources(self, sample, meta, q_range=(0.5, 2.0), verbose=False):
        n_sources = sample['n_sources']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        nt_full = sample['sample_metadata']['nt']

        solver_coarse = self._create_solver(kappa, bc, coarse=True)
        solver_fine = self._create_solver(kappa, bc, coarse=False)

        nt_reduced = max(10, int(nt_full * self.timestep_fraction))

        # Primary initializations
        primary_inits = []
        tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
        if tri_init is not None:
            primary_inits.append((tri_init, 'triangulation'))
        smart_init = self._smart_init_positions(sample, n_sources)
        primary_inits.append((smart_init, 'smart'))

        # Run GA optimization with final polish
        candidates_raw, n_sims = self._run_single_optimization(
            sample, meta, q_range, solver_coarse, solver_fine,
            primary_inits, n_sources, nt_reduced, nt_full
        )

        # Check if result is bad
        best_rmse_initial = min(c[2] for c in candidates_raw) if candidates_raw else float('inf')
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src

        if best_rmse_initial > threshold:
            # Fallback
            fallback_inits = []
            centroid_init = self._weighted_centroid_init(sample, n_sources)
            fallback_inits.append((centroid_init, 'centroid'))

            fallback_candidates, fallback_sims = self._run_single_optimization(
                sample, meta, q_range, solver_coarse, solver_fine,
                fallback_inits, n_sources, nt_reduced, nt_full
            )
            n_sims += fallback_sims
            candidates_raw.extend(fallback_candidates)

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
                n_evals=n_sims // len(final_candidates) if final_candidates else n_sims
            )
            for c in final_candidates
        ]

        return candidate_sources, best_rmse, results, n_sims
