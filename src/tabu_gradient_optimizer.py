"""
Gradient-Informed Tabu Search Optimizer for Heat Source Identification.

This variant enhances the standard Tabu Search by learning gradient information
from recent evaluations and using it to bias the neighborhood generation
toward promising descent directions.

Key learning mechanisms:
- Estimates gradient from recent (params, cost) evaluations
- Biases neighborhood generation toward descent direction
- Balances gradient-informed moves with exploratory moves
- Adapts based on search progress
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
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
class Evaluation:
    """Record of a single objective evaluation."""
    params: np.ndarray
    cost: float


@dataclass
class SearchResult:
    """Results from a single tabu search run."""
    best_params: np.ndarray
    best_cost: float
    all_candidates: List[Tuple[np.ndarray, float]]
    history: List[float]
    gradient_usage: float  # Fraction of moves that used gradient info


class GradientInformedTabuOptimizer:
    """
    Gradient-Informed Tabu Search for heat source identification.

    This optimizer learns from its own evaluations during inference:
    1. Maintains a buffer of recent (params, cost) evaluations
    2. Estimates local gradient from this buffer
    3. Generates neighbors biased toward descent direction
    4. Balances exploitation (gradient) with exploration (random)
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
        gradient_buffer_size: int = 15,
        gradient_exploitation_ratio: float = 0.6,
        min_gradient_samples: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Gradient-Informed Tabu Search optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_iterations: Maximum iterations per search
            tabu_tenure: How long a solution stays tabu
            tabu_radius: Minimum distance for tabu proximity check
            n_neighbors: Number of neighbors to generate per iteration
            initial_step: Initial perturbation magnitude (fraction of range)
            step_decay: Multiplicative decay for step size
            n_restarts: Number of independent searches
            gradient_buffer_size: Number of recent evaluations to use for gradient
            gradient_exploitation_ratio: Fraction of neighbors using gradient info
            min_gradient_samples: Minimum evaluations before using gradient
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
        self.gradient_buffer_size = gradient_buffer_size
        self.gradient_exploitation_ratio = gradient_exploitation_ratio
        self.min_gradient_samples = min_gradient_samples
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
        """Run the thermal simulator with given source parameters."""
        sample_meta = sample['sample_metadata']
        kappa = sample_meta.get('kappa', meta.get('kappa', 0.1))
        bc = sample_meta.get('bc', meta.get('bc', 'dirichlet'))
        dt = sample_meta.get('dt', meta.get('dt', 0.004))
        nt = sample_meta.get('nt', meta.get('nt', 400))
        T0 = sample_meta.get('T0', meta.get('T0', 0.0))

        solver = self._create_solver(kappa, bc)
        source_list = [{'x': s[0], 'y': s[1], 'q': s[2]} for s in sources]
        sensors_xy = sample['sensors_xy']
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=source_list)
        Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

        return Y_pred

    def _objective(
        self,
        params: np.ndarray,
        n_sources: int,
        sample: Dict,
        meta: Dict,
    ) -> float:
        """Compute RMSE between simulated and observed temperatures."""
        try:
            sources = []
            for i in range(n_sources):
                x, y, q = params[i*3:(i+1)*3]
                sources.append((x, y, q))

            simulated = self._simulate_sources(sources, sample, meta)
            observed = sample['Y_noisy']
            rmse = np.sqrt(np.mean((simulated - observed) ** 2))
            return rmse

        except Exception:
            return float('inf')

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float]) -> np.ndarray:
        """Get parameter bounds for n sources."""
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (0.05, self.Lx - 0.05),
                (0.05, self.Ly - 0.05),
                q_range,
            ])
        return np.array(bounds)

    def _estimate_gradient(
        self,
        current: np.ndarray,
        eval_buffer: deque,
        bounds: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate gradient from recent evaluations using weighted least squares.

        This is the core learning mechanism - we learn the local gradient
        direction from our own evaluations during the search.
        """
        if len(eval_buffer) < self.min_gradient_samples:
            return None

        # Get recent evaluations
        evals = list(eval_buffer)
        X = np.array([e.params for e in evals])
        y = np.array([e.cost for e in evals])

        # Normalize parameters
        param_ranges = bounds[:, 1] - bounds[:, 0]
        X_norm = (X - bounds[:, 0]) / param_ranges
        current_norm = (current - bounds[:, 0]) / param_ranges

        # Weight by distance from current (closer = more relevant)
        distances = np.linalg.norm(X_norm - current_norm, axis=1)
        weights = np.exp(-distances * 5)  # Exponential decay
        weights /= weights.sum()

        # Compute weighted mean
        X_centered = X_norm - current_norm
        y_centered = y - np.average(y, weights=weights)

        # Estimate gradient via weighted linear regression
        # gradient ≈ (X^T W X)^-1 X^T W y
        try:
            W = np.diag(weights)
            XtWX = X_centered.T @ W @ X_centered
            XtWy = X_centered.T @ W @ y_centered

            # Add regularization for stability
            reg = 1e-6 * np.eye(len(current))
            gradient_norm = np.linalg.solve(XtWX + reg, XtWy)

            # Convert back to original scale
            gradient = gradient_norm / param_ranges

            # Check for valid gradient
            if np.any(np.isnan(gradient)) or np.linalg.norm(gradient) < 1e-10:
                return None

            return gradient

        except np.linalg.LinAlgError:
            return None

    def _generate_gradient_informed_neighbors(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
        step_size: float,
        gradient: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate neighbors with mix of gradient-informed and exploratory moves.

        Returns:
            Tuple of (neighbors list, number of gradient-informed moves)
        """
        neighbors = []
        param_ranges = bounds[:, 1] - bounds[:, 0]
        n_gradient_moves = 0

        if gradient is not None:
            # Compute descent direction
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                descent_dir = -gradient / grad_norm

                # Number of gradient-informed neighbors
                n_gradient = int(self.n_neighbors * self.gradient_exploitation_ratio)
                n_gradient_moves = n_gradient

                # Generate gradient-informed neighbors
                for _ in range(n_gradient):
                    # Step along descent direction with some variation
                    scale = np.random.uniform(0.3, 1.5)
                    noise = np.random.randn(len(current)) * 0.2

                    step = (descent_dir + noise) * step_size * param_ranges * scale
                    neighbor = current + step
                    neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
                    neighbors.append(neighbor)

        # Fill remaining with exploratory moves
        n_exploratory = self.n_neighbors - len(neighbors)

        # Strategy 1: Single-dimension perturbations
        n_single_dim = min(n_exploratory // 2, len(current) * 2)
        dims = np.random.choice(len(current), n_single_dim, replace=True)
        for dim in dims:
            neighbor = current.copy()
            delta = np.random.uniform(-1, 1) * step_size * param_ranges[dim]
            neighbor[dim] = np.clip(
                neighbor[dim] + delta,
                bounds[dim, 0],
                bounds[dim, 1]
            )
            neighbors.append(neighbor)

        # Strategy 2: Random multi-dimension perturbations
        while len(neighbors) < self.n_neighbors:
            neighbor = current.copy()
            n_dims = np.random.randint(1, len(current) + 1)
            dims = np.random.choice(len(current), n_dims, replace=False)
            for dim in dims:
                delta = np.random.uniform(-1, 1) * step_size * param_ranges[dim]
                neighbor[dim] = np.clip(
                    neighbor[dim] + delta,
                    bounds[dim, 0],
                    bounds[dim, 1]
                )
            neighbors.append(neighbor)

        return neighbors, n_gradient_moves

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
            readings = sample['Y_noisy']
            sensors = sample['sensors_xy']
            avg_temps = np.mean(readings, axis=0)
            hottest_idx = np.argmax(avg_temps)
            hottest_loc = sensors[hottest_idx]

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

        elif use_smart_init and n_sources == 2:
            # Smart init for 2 sources: find 2 hottest well-separated sensors
            readings = sample['Y_noisy']
            sensors = sample['sensors_xy']
            avg_temps = np.mean(readings, axis=0)

            # Find hottest sensor for source 1
            hot_indices = np.argsort(avg_temps)[::-1]
            idx1 = hot_indices[0]
            loc1 = sensors[idx1]

            # Find second hottest that's far enough from first
            idx2 = hot_indices[1]
            for idx in hot_indices[1:]:
                dist = np.linalg.norm(sensors[idx] - loc1)
                if dist > 0.3:  # Minimum separation
                    idx2 = idx
                    break
            loc2 = sensors[idx2]

            params = []
            for loc in [loc1, loc2]:
                x = np.clip(
                    loc[0] + np.random.uniform(-0.1, 0.1),
                    bounds[0, 0], bounds[0, 1]
                )
                y = np.clip(
                    loc[1] + np.random.uniform(-0.1, 0.1),
                    bounds[1, 0], bounds[1, 1]
                )
                q = np.random.uniform(bounds[2, 0], bounds[2, 1])
                params.extend([x, y, q])

            return np.array(params)

        # Random initialization
        return np.array([
            np.random.uniform(low, high)
            for low, high in bounds
        ])

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
        """Run a single gradient-informed tabu search."""

        # Initialize
        if initial_solution is None:
            current = self._generate_initial_solution(bounds, sample, use_smart_init)
        else:
            current = initial_solution.copy()

        current_cost = self._objective(current, n_sources, sample, meta)

        # Evaluation buffer for gradient learning
        eval_buffer: deque = deque(maxlen=self.gradient_buffer_size)
        eval_buffer.append(Evaluation(current.copy(), current_cost))

        best_ever = current.copy()
        best_ever_cost = current_cost

        tabu_list: List[TabuEntry] = []
        history = [current_cost]
        good_candidates = [(current.copy(), current_cost)]

        step_size = self.initial_step
        iterations_without_improvement = 0
        total_gradient_moves = 0
        total_moves = 0

        for iteration in range(self.max_iterations):
            # Learn gradient from recent evaluations
            gradient = self._estimate_gradient(current, eval_buffer, bounds)

            # Generate neighbors (mix of gradient-informed and exploratory)
            neighbors, n_gradient_moves = self._generate_gradient_informed_neighbors(
                current, bounds, step_size, gradient
            )
            total_gradient_moves += n_gradient_moves
            total_moves += len(neighbors)

            # Evaluate all neighbors
            neighbor_costs = []
            for neighbor in neighbors:
                cost = self._objective(neighbor, n_sources, sample, meta)
                neighbor_costs.append((neighbor, cost))

                # Add to evaluation buffer for gradient learning
                eval_buffer.append(Evaluation(neighbor.copy(), cost))

            # Sort by cost
            neighbor_costs.sort(key=lambda x: x[1])

            # Find best admissible neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor, cost in neighbor_costs:
                is_tabu = self._is_tabu(neighbor, tabu_list, bounds)

                # Aspiration criteria
                if is_tabu and cost >= best_ever_cost:
                    continue

                best_neighbor = neighbor
                best_neighbor_cost = cost
                break

            if best_neighbor is None:
                best_neighbor, best_neighbor_cost = neighbor_costs[0]

            # Move to best neighbor
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

            # Adaptive step size
            step_size *= self.step_decay

            if iterations_without_improvement > 10:
                step_size *= 0.5
                iterations_without_improvement = 0

            history.append(current_cost)

            if best_ever_cost < 1e-6:
                break

        # Filter distinct candidates
        distinct_candidates = self._filter_distinct_candidates(good_candidates, bounds)

        gradient_usage = total_gradient_moves / max(total_moves, 1)

        return SearchResult(
            best_params=best_ever,
            best_cost=best_ever_cost,
            all_candidates=distinct_candidates,
            history=history,
            gradient_usage=gradient_usage,
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
        """Estimate heat source parameters using Gradient-Informed Tabu Search."""
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []

        for restart in range(self.n_restarts):
            use_smart = use_smart_init and (restart == 0)

            result = self._single_search(
                sample, meta, bounds, n_sources,
                use_smart_init=use_smart
            )
            all_results.append(result)

            if verbose:
                print(f"  Restart {restart + 1}/{self.n_restarts}: "
                      f"RMSE = {result.best_cost:.6f}, "
                      f"Gradient usage = {result.gradient_usage:.1%}")

        best_result = min(all_results, key=lambda r: r.best_cost)

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
    ) -> Tuple[List[Tuple[float, float, float]], float, List[Tuple[np.ndarray, float]], float]:
        """
        Estimate sources and return diverse candidates.

        Returns:
            Tuple of (best_sources, best_rmse, all_distinct_candidates, avg_gradient_usage)
        """
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []
        all_candidates = []
        total_gradient_usage = 0

        for restart in range(self.n_restarts):
            use_smart = use_smart_init and (restart == 0)

            result = self._single_search(
                sample, meta, bounds, n_sources,
                use_smart_init=use_smart
            )
            all_results.append(result)
            all_candidates.extend(result.all_candidates)
            total_gradient_usage += result.gradient_usage

            if verbose:
                print(f"  Restart {restart + 1}/{self.n_restarts}: "
                      f"RMSE = {result.best_cost:.6f}, "
                      f"Gradient usage = {result.gradient_usage:.1%}")

        best_result = min(all_results, key=lambda r: r.best_cost)
        distinct_candidates = self._filter_distinct_candidates(all_candidates, bounds)
        avg_gradient_usage = total_gradient_usage / self.n_restarts

        estimated_sources = []
        for i in range(n_sources):
            x, y, q = best_result.best_params[i*3:(i+1)*3]
            estimated_sources.append((x, y, q))

        return estimated_sources, best_result.best_cost, distinct_candidates, avg_gradient_usage
