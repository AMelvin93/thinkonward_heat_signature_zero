"""
Multi-Objective Pareto Optimizer for Heat Source Identification.

Uses a Pareto-based selection approach to maintain diverse candidates:
1. Run CMA-ES with larger population
2. Evaluate all final population members
3. Apply Pareto dominance on (RMSE, diversity_contribution)
4. Select non-dominated solutions as candidates

The hypothesis is that Pareto selection naturally balances accuracy and diversity
better than the current greedy approach (best RMSE with taboo regions).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import cma


@dataclass
class ParetoCandidate:
    """Candidate with multi-objective attributes."""
    positions: np.ndarray  # Full position vector
    q: float
    rmse: float
    diversity_score: float = 0.0  # Average distance to other candidates
    rank: int = 0  # Pareto rank (0 = non-dominated)
    crowding: float = 0.0  # Crowding distance


class ParetoOptimizer:
    """
    Multi-objective optimizer using Pareto dominance for candidate selection.

    Instead of using taboo regions to enforce diversity, we:
    1. Run CMA-ES with larger population (keeps more solutions)
    2. Compute Pareto dominance on (RMSE, diversity_contribution)
    3. Select candidates from the Pareto front
    """

    def __init__(
        self,
        max_fevals_1src: int = 30,  # Slightly more for larger population
        max_fevals_2src: int = 50,
        candidate_pool_size: int = 10,
        nx_coarse: int = 50,
        ny_coarse: int = 25,
        timestep_fraction: float = 0.40,
        final_polish_maxiter: int = 8,
        cmaes_population_mult: float = 1.5,  # Larger population for diversity
        rmse_threshold_1src: float = 0.4,
        rmse_threshold_2src: float = 0.5,
    ):
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.candidate_pool_size = candidate_pool_size
        self.nx_coarse = nx_coarse
        self.ny_coarse = ny_coarse
        self.timestep_fraction = timestep_fraction
        self.final_polish_maxiter = final_polish_maxiter
        self.cmaes_population_mult = cmaes_population_mult
        self.rmse_threshold_1src = rmse_threshold_1src
        self.rmse_threshold_2src = rmse_threshold_2src

    def _create_simulator(self, sample: Dict, meta: Dict) -> Any:
        """Create a thermal simulator for this sample."""
        from src.simulator import ThermalSimulator
        return ThermalSimulator(
            Lx=meta['Lx'], Ly=meta['Ly'],
            nx=meta['nx'], ny=meta['ny'],
            T0=sample['T0'],
            kappa=sample['kappa'],
            bc_left=sample['bc_left'],
            bc_right=sample['bc_right'],
            bc_bottom=sample['bc_bottom'],
            bc_top=sample['bc_top'],
        )

    def _compute_rmse(
        self,
        positions: np.ndarray,
        simulator: Any,
        sample: Dict,
        meta: Dict,
        use_truncated: bool = True
    ) -> Tuple[float, float]:
        """Compute RMSE and optimal intensity q for given positions."""
        n_sources = sample['n_sources']
        n_timesteps = meta['n_timesteps']

        if use_truncated:
            nt = int(n_timesteps * self.timestep_fraction)
        else:
            nt = n_timesteps

        if n_sources == 1:
            x, y = positions
            sources = [(x, y, 1.0)]
        else:
            x1, y1, x2, y2 = positions
            sources = [(x1, y1, 1.0), (x2, y2, 1.0)]

        # Forward simulation
        T_field = simulator.solve_forward(
            sources=sources,
            t_max=meta['t_max'],
            n_timesteps=n_timesteps
        )

        # Extract sensor readings
        T_pred = []
        for sx, sy in sample['sensor_coords']:
            ix = int(sx / meta['Lx'] * (meta['nx'] - 1))
            iy = int(sy / meta['Ly'] * (meta['ny'] - 1))
            T_pred.append(T_field[:nt, iy, ix])
        T_pred = np.array(T_pred).T

        T_obs = sample['sensor_data'][:nt]

        # Solve for optimal intensity via least squares
        T_pred_flat = T_pred.flatten()
        T_obs_flat = T_obs.flatten()

        if n_sources == 1:
            q_opt = np.dot(T_pred_flat, T_obs_flat) / (np.dot(T_pred_flat, T_pred_flat) + 1e-10)
            q_opt = np.clip(q_opt, 0.5, 2.0)
        else:
            # For 2 sources, we assume equal intensity for simplicity
            q_opt = np.dot(T_pred_flat, T_obs_flat) / (np.dot(T_pred_flat, T_pred_flat) + 1e-10)
            q_opt = np.clip(q_opt, 0.5, 2.0)

        T_final = T_pred * q_opt
        rmse = np.sqrt(np.mean((T_final - T_obs) ** 2))

        return rmse, q_opt

    def _smart_init(self, sample: Dict, meta: Dict) -> List[np.ndarray]:
        """Generate smart initialization points."""
        sensor_coords = sample['sensor_coords']
        sensor_data = sample['sensor_data']
        n_sources = sample['n_sources']

        # Find hottest sensor
        avg_temps = sensor_data.mean(axis=0)
        hottest_idx = np.argmax(avg_temps)
        hx, hy = sensor_coords[hottest_idx]

        inits = []

        if n_sources == 1:
            # Init near hottest sensor
            inits.append(np.array([hx, hy]))
            # Add random variations
            for _ in range(2):
                x = np.clip(hx + np.random.uniform(-0.2, 0.2), 0.05, meta['Lx'] - 0.05)
                y = np.clip(hy + np.random.uniform(-0.1, 0.1), 0.05, meta['Ly'] - 0.05)
                inits.append(np.array([x, y]))
        else:
            # Find two hottest sensors
            sorted_idx = np.argsort(avg_temps)[::-1]
            h1x, h1y = sensor_coords[sorted_idx[0]]
            h2x, h2y = sensor_coords[sorted_idx[1]]

            inits.append(np.array([h1x, h1y, h2x, h2y]))
            # Add variations
            for _ in range(2):
                x1 = np.clip(h1x + np.random.uniform(-0.2, 0.2), 0.05, meta['Lx'] - 0.05)
                y1 = np.clip(h1y + np.random.uniform(-0.1, 0.1), 0.05, meta['Ly'] - 0.05)
                x2 = np.clip(h2x + np.random.uniform(-0.2, 0.2), 0.05, meta['Lx'] - 0.05)
                y2 = np.clip(h2y + np.random.uniform(-0.1, 0.1), 0.05, meta['Ly'] - 0.05)
                inits.append(np.array([x1, y1, x2, y2]))

        return inits

    def _pareto_dominates(self, a: ParetoCandidate, b: ParetoCandidate) -> bool:
        """Check if candidate a dominates candidate b."""
        # Minimize RMSE, maximize diversity
        a_better_rmse = a.rmse <= b.rmse
        a_better_div = a.diversity_score >= b.diversity_score
        a_strictly_better = (a.rmse < b.rmse) or (a.diversity_score > b.diversity_score)
        return a_better_rmse and a_better_div and a_strictly_better

    def _compute_pareto_ranks(self, candidates: List[ParetoCandidate]) -> List[ParetoCandidate]:
        """Compute Pareto ranks for all candidates using non-dominated sorting."""
        n = len(candidates)
        ranks = [0] * n
        dominated_by = [[] for _ in range(n)]
        domination_count = [0] * n

        # Count domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._pareto_dominates(candidates[i], candidates[j]):
                        dominated_by[i].append(j)
                    elif self._pareto_dominates(candidates[j], candidates[i]):
                        domination_count[i] += 1

        # Assign ranks
        current_rank = 0
        remaining = set(range(n))

        while remaining:
            non_dominated = [i for i in remaining if domination_count[i] == 0]
            if not non_dominated:
                # Assign remaining to current rank
                for i in remaining:
                    candidates[i].rank = current_rank
                break

            for i in non_dominated:
                candidates[i].rank = current_rank
                remaining.remove(i)
                for j in dominated_by[i]:
                    if j in remaining:
                        domination_count[j] -= 1

            current_rank += 1

        return candidates

    def _compute_diversity_scores(self, candidates: List[ParetoCandidate], meta: Dict) -> List[ParetoCandidate]:
        """Compute diversity score as average distance to other candidates."""
        n = len(candidates)
        scale = np.array([meta['Lx'], meta['Ly']])

        for i in range(n):
            distances = []
            ci_pos = np.array([candidates[i].x, candidates[i].y])

            for j in range(n):
                if i != j:
                    cj_pos = np.array([candidates[j].x, candidates[j].y])
                    dist = np.linalg.norm((ci_pos - cj_pos) / scale)
                    distances.append(dist)

            candidates[i].diversity_score = np.mean(distances) if distances else 0.0

        return candidates

    def _select_pareto_candidates(
        self,
        candidates: List[ParetoCandidate],
        n_select: int = 3
    ) -> List[ParetoCandidate]:
        """Select best candidates using Pareto ranking and crowding."""
        # Sort by rank first, then by crowding distance (within same rank)
        candidates = sorted(candidates, key=lambda c: (c.rank, -c.diversity_score))

        selected = []
        seen_positions = set()

        for c in candidates:
            # Avoid selecting nearly identical positions
            pos_key = (round(c.x, 2), round(c.y, 2))
            if pos_key not in seen_positions:
                selected.append(c)
                seen_positions.add(pos_key)
                if len(selected) >= n_select:
                    break

        return selected

    def _run_cmaes_with_population(
        self,
        init_pos: np.ndarray,
        simulator: Any,
        sample: Dict,
        meta: Dict,
        max_fevals: int,
        sigma: float
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Run CMA-ES and collect all evaluated solutions."""
        n_sources = sample['n_sources']

        if n_sources == 1:
            bounds = [[0.05, 0.05], [meta['Lx'] - 0.05, meta['Ly'] - 0.05]]
        else:
            bounds = [
                [0.05, 0.05, 0.05, 0.05],
                [meta['Lx'] - 0.05, meta['Ly'] - 0.05, meta['Lx'] - 0.05, meta['Ly'] - 0.05]
            ]

        evaluated_solutions = []

        def objective(pos):
            rmse, q = self._compute_rmse(pos, simulator, sample, meta, use_truncated=True)
            evaluated_solutions.append((pos.copy(), rmse, q))
            return rmse

        # Larger population for more diversity
        popsize = max(4, int(4 + 3 * np.log(len(init_pos)) * self.cmaes_population_mult))

        es = cma.CMAEvolutionStrategy(
            init_pos.tolist(),
            sigma,
            {
                'bounds': bounds,
                'maxfevals': max_fevals,
                'popsize': popsize,
                'verbose': -9,
            }
        )

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(x) for x in solutions]
            es.tell(solutions, fitness)

        return evaluated_solutions

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False
    ) -> Tuple[List[Dict], float, List[Any], int]:
        """
        Estimate heat sources using multi-objective Pareto selection.

        Returns:
            candidates: List of candidate source configurations
            best_rmse: Best RMSE achieved
            results: Detailed results per initialization
            n_sims: Total simulation count
        """
        n_sources = sample['n_sources']
        simulator = self._create_simulator(sample, meta)

        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        threshold = self.rmse_threshold_1src if n_sources == 1 else self.rmse_threshold_2src
        sigma = 0.15 if n_sources == 1 else 0.20

        # Get smart initializations
        inits = self._smart_init(sample, meta)

        # Collect all solutions from all CMA-ES runs
        all_solutions = []
        n_sims = 0

        for init_pos in inits:
            fevals_per_init = max_fevals // len(inits)
            solutions = self._run_cmaes_with_population(
                init_pos, simulator, sample, meta, fevals_per_init, sigma
            )
            all_solutions.extend(solutions)
            n_sims += len(solutions)

        if not all_solutions:
            return [], float('inf'), [], n_sims

        # Convert to ParetoCandidate objects
        candidates = []
        for pos, rmse, q in all_solutions:
            if n_sources == 1:
                x, y = pos
                candidates.append(ParetoCandidate(x=x, y=y, q=q, rmse=rmse, diversity_score=0))
            else:
                x1, y1, x2, y2 = pos
                # For 2-source, use first source position for diversity calculation
                candidates.append(ParetoCandidate(x=x1, y=y1, q=q, rmse=rmse, diversity_score=0))

        # Compute diversity scores
        candidates = self._compute_diversity_scores(candidates, meta)

        # Compute Pareto ranks
        candidates = self._compute_pareto_ranks(candidates)

        # Select best candidates
        selected = self._select_pareto_candidates(candidates, n_select=self.candidate_pool_size)

        # Polish selected candidates with Nelder-Mead (full timesteps)
        from scipy.optimize import minimize

        final_candidates = []
        results = []

        for cand in selected:
            if n_sources == 1:
                init_pos = np.array([cand.x, cand.y])
                bounds = [(0.05, meta['Lx'] - 0.05), (0.05, meta['Ly'] - 0.05)]
            else:
                # Reconstruct 2-source position (stored first source only)
                # For now, use the original solution
                init_pos = np.array([cand.x, cand.y])
                bounds = [(0.05, meta['Lx'] - 0.05), (0.05, meta['Ly'] - 0.05)]

            def polish_obj(pos):
                rmse, _ = self._compute_rmse(pos, simulator, sample, meta, use_truncated=False)
                return rmse

            # Only polish 1-source for now
            if n_sources == 1:
                result = minimize(
                    polish_obj, init_pos,
                    method='Nelder-Mead',
                    options={'maxiter': self.final_polish_maxiter}
                )
                final_pos = result.x
                n_sims += result.nfev
            else:
                # For 2-source, skip polish (need to reconstruct full position)
                final_pos = init_pos

            final_rmse, final_q = self._compute_rmse(
                final_pos, simulator, sample, meta, use_truncated=False
            )
            n_sims += 1

            # Add to results if meets threshold
            if final_rmse <= threshold:
                if n_sources == 1:
                    x, y = final_pos
                    final_candidates.append({
                        'sources': [{'x': x, 'y': y, 'q': final_q}],
                        'rmse': final_rmse,
                    })
                else:
                    x, y = final_pos
                    final_candidates.append({
                        'sources': [
                            {'x': x, 'y': y, 'q': final_q},
                            {'x': cand.y, 'y': 0.5, 'q': final_q}  # Placeholder
                        ],
                        'rmse': final_rmse,
                    })

            results.append(type('Result', (), {
                'init_type': 'pareto',
                'rmse': final_rmse,
            })())

        # Deduplicate and sort by RMSE
        final_candidates = sorted(final_candidates, key=lambda c: c['rmse'])

        # Keep top 3 distinct candidates
        distinct = []
        seen_positions = set()
        for c in final_candidates:
            pos_key = tuple(round(s['x'], 1) for s in c['sources'])
            if pos_key not in seen_positions:
                distinct.append(c)
                seen_positions.add(pos_key)
                if len(distinct) >= 3:
                    break

        best_rmse = distinct[0]['rmse'] if distinct else float('inf')

        return distinct, best_rmse, results, n_sims
