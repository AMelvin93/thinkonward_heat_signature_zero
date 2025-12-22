"""
Tabu Search Optimizer for Heat Source Identification.

Tabu Search is a metaheuristic that uses memory structures (tabu list)
to escape local optima by preventing the search from revisiting recent solutions.

Key features:
- Memory-guided search avoids cycling
- Accepts worse moves to escape local optima
- Aspiration criteria allows exceptional moves
- Multiple restarts for diverse candidates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

# Add simulator path
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "Heat_Signature_zero-starter_notebook"))

from simulator import Heat2D


@dataclass
class TabuEntry:
    """Entry in the tabu list."""
    params: np.ndarray
    remaining_tenure: int


@dataclass
class SearchResult:
    """Results from a single tabu search run."""
    best_params: np.ndarray
    best_cost: float
    all_candidates: List[Tuple[np.ndarray, float]]
    history: List[float]


class TabuSearchOptimizer:
    """
    Tabu Search for inverse heat source identification.

    Actively uses the thermal simulator during inference to iteratively
    refine heat source parameter estimates.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_iterations: int = 100,
        tabu_tenure: int = 10,
        tabu_radius: float = 0.05,
        n_neighbors: int = 20,
        initial_step: float = 0.15,
        step_decay: float = 0.98,
        n_restarts: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Tabu Search optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_iterations: Maximum iterations per search
            tabu_tenure: How long a solution stays tabu
            tabu_radius: Minimum distance for tabu proximity check (normalized)
            n_neighbors: Number of neighbors to generate per iteration
            initial_step: Initial perturbation magnitude (fraction of range)
            step_decay: Multiplicative decay for step size
            n_restarts: Number of independent searches
            seed: Random seed for reproducibility
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_radius = tabu_radius
        self.n_neighbors = n_neighbors
        self.initial_step = initial_step
        self.step_decay = step_decay
        self.n_restarts = n_restarts
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(
            self.Lx,
            self.Ly,
            self.nx,
            self.ny,
            kappa,
            bc=bc,
        )

    def _simulate_sources(
        self,
        sources: List[Tuple[float, float, float]],
        sample: Dict,
        meta: Dict,
    ) -> np.ndarray:
        """
        Run the thermal simulator with given source parameters.

        Args:
            sources: List of (x, y, q) tuples for each source
            sample: Sample dict with sensor locations and metadata
            meta: Dataset metadata

        Returns:
            Simulated temperature readings at sensor locations (nt+1, n_sensors)
        """
        # Get simulation parameters
        sample_meta = sample['sample_metadata']
        kappa = sample_meta.get('kappa', meta.get('kappa', 0.1))
        bc = sample_meta.get('bc', meta.get('bc', 'dirichlet'))
        dt = sample_meta.get('dt', meta.get('dt', 0.004))
        nt = sample_meta.get('nt', meta.get('nt', 400))
        T0 = sample_meta.get('T0', meta.get('T0', 0.0))

        # Create solver
        solver = self._create_solver(kappa, bc)

        # Set up sources as list of dicts
        source_list = [{'x': s[0], 'y': s[1], 'q': s[2]} for s in sources]

        # Run simulation
        sensors_xy = sample['sensors_xy']
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=source_list)

        # Sample at sensor locations
        Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

        return Y_pred

    def _objective(
        self,
        params: np.ndarray,
        n_sources: int,
        sample: Dict,
        meta: Dict,
    ) -> float:
        """
        Compute RMSE between simulated and observed temperatures.

        This is where we ACTIVELY USE THE SIMULATOR during inference!
        """
        try:
            # Convert flat params to source list
            sources = []
            for i in range(n_sources):
                x, y, q = params[i*3:(i+1)*3]
                sources.append((x, y, q))

            # Run simulator
            simulated = self._simulate_sources(sources, sample, meta)
            observed = sample['Y_noisy']  # Noisy sensor readings

            # Compute RMSE across all timesteps
            rmse = np.sqrt(np.mean((simulated - observed) ** 2))
            return rmse

        except Exception as e:
            # Return high cost for invalid parameters
            return float('inf')

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float]) -> np.ndarray:
        """Get parameter bounds for n sources."""
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (0.05, self.Lx - 0.05),  # x with margin
                (0.05, self.Ly - 0.05),  # y with margin
                q_range,                  # intensity
            ])
        return np.array(bounds)

    def _generate_initial_solution(
        self,
        bounds: np.ndarray,
        sample: Dict,
        use_smart_init: bool = True,
    ) -> np.ndarray:
        """Generate initial solution, optionally using sensor data."""
        n_params = len(bounds)
        n_sources = n_params // 3

        if use_smart_init and n_sources == 1:
            # Smart initialization: start near hottest sensor
            readings = sample['Y_noisy']  # Shape: (timesteps, n_sensors)
            sensors = sample['sensors_xy']

            # Find sensor with highest average temperature (average over timesteps)
            avg_temps = np.mean(readings, axis=0)  # Shape: (n_sensors,)
            hottest_idx = np.argmax(avg_temps)
            hottest_loc = sensors[hottest_idx]

            # Add some noise to avoid exact sensor location
            x = np.clip(
                hottest_loc[0] + np.random.uniform(-0.1, 0.1),
                bounds[0, 0], bounds[0, 1]
            )
            y = np.clip(
                hottest_loc[1] + np.random.uniform(-0.1, 0.1),
                bounds[1, 0], bounds[1, 1]
            )
            q = np.random.uniform(bounds[2, 0], bounds[2, 1])

            return np.array([x, y, q])

        # Random initialization
        return np.array([
            np.random.uniform(low, high)
            for low, high in bounds
        ])

    def _generate_neighbors(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
        step_size: float,
    ) -> List[np.ndarray]:
        """
        Generate neighborhood of candidate solutions.

        Uses multiple perturbation strategies:
        1. Single-dimension perturbations
        2. Multi-dimension perturbations
        """
        neighbors = []
        n_params = len(current)
        param_ranges = bounds[:, 1] - bounds[:, 0]

        # Strategy 1: Perturb each dimension individually (±)
        for dim in range(n_params):
            for direction in [-1, 1]:
                neighbor = current.copy()
                delta = direction * step_size * param_ranges[dim]
                neighbor[dim] = np.clip(
                    neighbor[dim] + delta,
                    bounds[dim, 0],
                    bounds[dim, 1]
                )
                neighbors.append(neighbor)

        # Strategy 2: Random multi-dimension perturbations
        n_random = max(1, self.n_neighbors - len(neighbors))
        for _ in range(n_random):
            neighbor = current.copy()
            # Perturb random subset of dimensions
            n_dims_to_perturb = np.random.randint(1, n_params + 1)
            dims = np.random.choice(n_params, n_dims_to_perturb, replace=False)
            for dim in dims:
                delta = np.random.uniform(-1, 1) * step_size * param_ranges[dim]
                neighbor[dim] = np.clip(
                    neighbor[dim] + delta,
                    bounds[dim, 0],
                    bounds[dim, 1]
                )
            neighbors.append(neighbor)

        return neighbors

    def _is_tabu(
        self,
        candidate: np.ndarray,
        tabu_list: List[TabuEntry],
        bounds: np.ndarray,
    ) -> bool:
        """Check if candidate is within tabu radius of any tabu entry."""
        param_ranges = bounds[:, 1] - bounds[:, 0]
        normalized_candidate = (candidate - bounds[:, 0]) / param_ranges

        for entry in tabu_list:
            normalized_tabu = (entry.params - bounds[:, 0]) / param_ranges
            distance = np.linalg.norm(normalized_candidate - normalized_tabu)
            if distance < self.tabu_radius:
                return True
        return False

    def _update_tabu_list(
        self,
        tabu_list: List[TabuEntry],
        new_entry: np.ndarray,
    ) -> List[TabuEntry]:
        """Update tabu list: decrement tenures, remove expired, add new."""
        updated_list = []
        for entry in tabu_list:
            entry.remaining_tenure -= 1
            if entry.remaining_tenure > 0:
                updated_list.append(entry)

        updated_list.append(TabuEntry(
            params=new_entry.copy(),
            remaining_tenure=self.tabu_tenure
        ))

        return updated_list

    def _single_search(
        self,
        sample: Dict,
        meta: Dict,
        bounds: np.ndarray,
        n_sources: int,
        initial_solution: Optional[np.ndarray] = None,
        use_smart_init: bool = True,
    ) -> SearchResult:
        """Run a single tabu search from one starting point."""

        # Initialize
        if initial_solution is None:
            current = self._generate_initial_solution(bounds, sample, use_smart_init)
        else:
            current = initial_solution.copy()

        current_cost = self._objective(current, n_sources, sample, meta)

        best_ever = current.copy()
        best_ever_cost = current_cost

        tabu_list: List[TabuEntry] = []
        history = [current_cost]
        good_candidates = [(current.copy(), current_cost)]

        step_size = self.initial_step
        iterations_without_improvement = 0

        for iteration in range(self.max_iterations):
            # Generate neighborhood
            neighbors = self._generate_neighbors(current, bounds, step_size)

            # Evaluate all neighbors (THIS USES THE SIMULATOR!)
            neighbor_costs = []
            for neighbor in neighbors:
                cost = self._objective(neighbor, n_sources, sample, meta)
                neighbor_costs.append((neighbor, cost))

            # Sort by cost (best first)
            neighbor_costs.sort(key=lambda x: x[1])

            # Find best admissible (non-tabu or aspiration) neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor, cost in neighbor_costs:
                is_tabu = self._is_tabu(neighbor, tabu_list, bounds)

                # Aspiration criteria: accept if better than best ever
                if is_tabu and cost >= best_ever_cost:
                    continue

                best_neighbor = neighbor
                best_neighbor_cost = cost
                break

            # If all neighbors are tabu, pick the least bad one
            if best_neighbor is None:
                best_neighbor, best_neighbor_cost = neighbor_costs[0]

            # Move to best neighbor (even if worse - key feature!)
            current = best_neighbor
            current_cost = best_neighbor_cost

            # Update tabu list
            tabu_list = self._update_tabu_list(tabu_list, current)

            # Update best ever
            if current_cost < best_ever_cost:
                best_ever = current.copy()
                best_ever_cost = current_cost
                iterations_without_improvement = 0
                good_candidates.append((current.copy(), current_cost))
            else:
                iterations_without_improvement += 1

            # Adaptive step size decay
            step_size *= self.step_decay

            # Intensification: if stuck, reduce step size more
            if iterations_without_improvement > 10:
                step_size *= 0.5
                iterations_without_improvement = 0

            history.append(current_cost)

            # Early stopping if very good solution found
            if best_ever_cost < 1e-6:
                break

        # Filter distinct candidates
        distinct_candidates = self._filter_distinct_candidates(good_candidates, bounds)

        return SearchResult(
            best_params=best_ever,
            best_cost=best_ever_cost,
            all_candidates=distinct_candidates,
            history=history
        )

    def _filter_distinct_candidates(
        self,
        candidates: List[Tuple[np.ndarray, float]],
        bounds: np.ndarray,
        min_distance: float = 0.1,
    ) -> List[Tuple[np.ndarray, float]]:
        """Filter candidates to keep only sufficiently distinct ones."""
        if not candidates:
            return []

        param_ranges = bounds[:, 1] - bounds[:, 0]
        sorted_candidates = sorted(candidates, key=lambda x: x[1])

        distinct = [sorted_candidates[0]]

        for params, cost in sorted_candidates[1:]:
            normalized = (params - bounds[:, 0]) / param_ranges

            is_distinct = True
            for existing_params, _ in distinct:
                existing_normalized = (existing_params - bounds[:, 0]) / param_ranges
                if np.linalg.norm(normalized - existing_normalized) < min_distance:
                    is_distinct = False
                    break

            if is_distinct:
                distinct.append((params, cost))

        return distinct

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        use_smart_init: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Estimate heat source parameters using Tabu Search.

        Args:
            sample: Sample dictionary with sensor data
            meta: Dataset metadata
            q_range: (min, max) bounds for source intensity
            use_smart_init: Use sensor data for smart initialization
            verbose: Print progress

        Returns:
            Tuple of (estimated_sources, best_rmse)
            where estimated_sources is list of (x, y, q) tuples
        """
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []

        for restart in range(self.n_restarts):
            # Alternate between smart and random initialization
            use_smart = use_smart_init and (restart == 0)

            result = self._single_search(
                sample, meta, bounds, n_sources,
                use_smart_init=use_smart
            )
            all_results.append(result)

            if verbose:
                print(f"  Restart {restart + 1}/{self.n_restarts}: "
                      f"RMSE = {result.best_cost:.6f}")

        # Find overall best
        best_result = min(all_results, key=lambda r: r.best_cost)

        # Convert to source list
        estimated_sources = []
        for i in range(n_sources):
            x, y, q = best_result.best_params[i*3:(i+1)*3]
            estimated_sources.append((x, y, q))

        return estimated_sources, best_result.best_cost

    def estimate_sources_with_candidates(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        use_smart_init: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[Tuple[np.ndarray, float]]]:
        """
        Estimate sources and return diverse candidates.

        Returns:
            Tuple of (best_sources, best_rmse, all_distinct_candidates)
        """
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []
        all_candidates = []

        for restart in range(self.n_restarts):
            use_smart = use_smart_init and (restart == 0)

            result = self._single_search(
                sample, meta, bounds, n_sources,
                use_smart_init=use_smart
            )
            all_results.append(result)
            all_candidates.extend(result.all_candidates)

            if verbose:
                print(f"  Restart {restart + 1}/{self.n_restarts}: "
                      f"RMSE = {result.best_cost:.6f}")

        # Find overall best
        best_result = min(all_results, key=lambda r: r.best_cost)

        # Filter distinct across all restarts
        distinct_candidates = self._filter_distinct_candidates(all_candidates, bounds)

        # Convert best to source list
        estimated_sources = []
        for i in range(n_sources):
            x, y, q = best_result.best_params[i*3:(i+1)*3]
            estimated_sources.append((x, y, q))

        return estimated_sources, best_result.best_cost, distinct_candidates
