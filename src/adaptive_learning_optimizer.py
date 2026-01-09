"""
Adaptive Learning Tabu Search Optimizer for Heat Source Identification.

This optimizer implements learning-based mechanisms that adapt during inference:
1. Multi-source smart initialization from sensor observations
2. Gradient-informed neighborhood generation (learns from evaluations)
3. Adaptive step size based on landscape roughness learning
4. L-BFGS-B polishing for efficient local refinement
5. Adaptive tabu tenure based on search dynamics

Key philosophy: The optimizer learns from each simulator call during inference,
not from pre-computed strategies or grid search.
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
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
    iteration: int


@dataclass
class SearchMetrics:
    """Metrics tracked during search for learning."""
    gradient_moves_used: int = 0
    total_moves: int = 0
    improvements: int = 0
    iterations_stuck: int = 0
    polish_improvement: float = 0.0
    final_step_size: float = 0.0
    adaptive_tenure_changes: int = 0


@dataclass
class SearchResult:
    """Results from a single search run."""
    best_params: np.ndarray
    best_cost: float
    all_candidates: List[Tuple[np.ndarray, float]]
    history: List[float]
    metrics: SearchMetrics


class AdaptiveLearningOptimizer:
    """
    Adaptive Learning Tabu Search for heat source identification.

    This optimizer learns from its own evaluations during inference:
    1. Observational learning: Uses sensor data to infer likely source locations
    2. Gradient learning: Estimates descent direction from recent evaluations
    3. Landscape learning: Adapts step size based on objective roughness
    4. Curvature learning: Uses L-BFGS-B to learn local curvature for polishing
    5. Search dynamics learning: Adapts tabu tenure based on exploration/exploitation
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_iterations: int = 25,
        base_tabu_tenure: int = 8,
        tabu_radius: float = 0.05,
        n_neighbors: int = 16,
        initial_step: float = 0.15,
        min_step: float = 0.02,
        n_restarts: int = 3,
        # Gradient learning parameters
        gradient_buffer_size: int = 20,
        gradient_exploitation_ratio: float = 0.5,
        min_gradient_samples: int = 4,
        # Adaptive learning parameters
        enable_adaptive_step: bool = True,
        enable_adaptive_tenure: bool = True,
        enable_lbfgs_polish: bool = True,
        polish_max_iter: int = 15,
        # Separation for multi-source
        min_source_separation: float = 0.25,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Adaptive Learning optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            max_iterations: Maximum iterations per search
            base_tabu_tenure: Base tenure for tabu list (adapts during search)
            tabu_radius: Minimum distance for tabu proximity check
            n_neighbors: Number of neighbors to generate per iteration
            initial_step: Initial perturbation magnitude (fraction of range)
            min_step: Minimum step size to prevent over-refinement
            n_restarts: Number of independent searches
            gradient_buffer_size: Number of recent evaluations for gradient
            gradient_exploitation_ratio: Fraction of gradient-informed moves
            min_gradient_samples: Minimum evaluations before using gradient
            enable_adaptive_step: Enable landscape-based step adaptation
            enable_adaptive_tenure: Enable search-dynamics-based tenure adaptation
            enable_lbfgs_polish: Enable L-BFGS-B local refinement
            polish_max_iter: Maximum iterations for L-BFGS-B polishing
            min_source_separation: Minimum distance between sources in smart init
            seed: Random seed
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_iterations = max_iterations
        self.base_tabu_tenure = base_tabu_tenure
        self.tabu_radius = tabu_radius
        self.n_neighbors = n_neighbors
        self.initial_step = initial_step
        self.min_step = min_step
        self.n_restarts = n_restarts
        self.gradient_buffer_size = gradient_buffer_size
        self.gradient_exploitation_ratio = gradient_exploitation_ratio
        self.min_gradient_samples = min_gradient_samples
        self.enable_adaptive_step = enable_adaptive_step
        self.enable_adaptive_tenure = enable_adaptive_tenure
        self.enable_lbfgs_polish = enable_lbfgs_polish
        self.polish_max_iter = polish_max_iter
        self.min_source_separation = min_source_separation
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

    # =========================================================================
    # LEARNING MECHANISM 1: Observational Learning (Smart Initialization)
    # =========================================================================

    def _smart_init_from_observations(
        self,
        bounds: np.ndarray,
        sample: Dict,
        n_sources: int,
    ) -> np.ndarray:
        """
        Learn initial source locations from sensor temperature observations.

        For single source: Start near the hottest sensor.
        For multiple sources: Find well-separated hot regions.
        """
        readings = sample['Y_noisy']  # Shape: (timesteps, n_sensors)
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)

        # Sort sensors by average temperature (hottest first)
        hot_indices = np.argsort(avg_temps)[::-1]

        # Select well-separated hot sensors
        selected_sensors = []
        for idx in hot_indices:
            if len(selected_sensors) >= n_sources:
                break

            # Check separation from already selected
            is_separated = True
            for prev_idx in selected_sensors:
                dist = np.linalg.norm(sensors[idx] - sensors[prev_idx])
                if dist < self.min_source_separation:
                    is_separated = False
                    break

            if is_separated:
                selected_sensors.append(idx)

        # If we couldn't find enough separated sensors, fill with remaining hot ones
        for idx in hot_indices:
            if len(selected_sensors) >= n_sources:
                break
            if idx not in selected_sensors:
                selected_sensors.append(idx)

        # Build initial parameters
        params = []
        for i, s_idx in enumerate(selected_sensors):
            loc = sensors[s_idx]

            # Add small noise to avoid exact sensor location
            x = np.clip(
                loc[0] + np.random.uniform(-0.08, 0.08),
                bounds[i*3][0], bounds[i*3][1]
            )
            y = np.clip(
                loc[1] + np.random.uniform(-0.08, 0.08),
                bounds[i*3+1][0], bounds[i*3+1][1]
            )

            # Estimate intensity from relative temperature
            temp_ratio = avg_temps[s_idx] / (np.max(avg_temps) + 1e-8)
            q_range = (bounds[i*3+2][0], bounds[i*3+2][1])
            q = q_range[0] + temp_ratio * (q_range[1] - q_range[0]) * 0.8
            q = np.clip(q, q_range[0], q_range[1])

            params.extend([x, y, q])

        return np.array(params)

    def _random_init(self, bounds: np.ndarray) -> np.ndarray:
        """Generate random initial solution."""
        return np.array([
            np.random.uniform(low, high)
            for low, high in bounds
        ])

    # =========================================================================
    # LEARNING MECHANISM 2: Gradient Learning
    # =========================================================================

    def _estimate_gradient(
        self,
        current: np.ndarray,
        eval_buffer: deque,
        bounds: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate gradient from recent evaluations using weighted least squares.

        This learns the local descent direction from the search history.
        """
        if len(eval_buffer) < self.min_gradient_samples:
            return None

        evals = list(eval_buffer)
        X = np.array([e.params for e in evals])
        y = np.array([e.cost for e in evals])

        # Normalize parameters
        param_ranges = bounds[:, 1] - bounds[:, 0]
        X_norm = (X - bounds[:, 0]) / param_ranges
        current_norm = (current - bounds[:, 0]) / param_ranges

        # Weight by distance and recency
        distances = np.linalg.norm(X_norm - current_norm, axis=1)
        recency = np.linspace(0.5, 1.0, len(evals))
        weights = np.exp(-distances * 3) * recency
        weights /= weights.sum() + 1e-10

        X_centered = X_norm - current_norm
        y_centered = y - np.average(y, weights=weights)

        try:
            W = np.diag(weights)
            XtWX = X_centered.T @ W @ X_centered
            XtWy = X_centered.T @ W @ y_centered

            # Regularization for stability
            reg = 1e-5 * np.eye(len(current))
            gradient_norm = np.linalg.solve(XtWX + reg, XtWy)

            # Convert back to original scale
            gradient = gradient_norm / param_ranges

            if np.any(np.isnan(gradient)) or np.linalg.norm(gradient) < 1e-10:
                return None

            return gradient

        except np.linalg.LinAlgError:
            return None

    # =========================================================================
    # LEARNING MECHANISM 3: Landscape Learning (Adaptive Step Size)
    # =========================================================================

    def _compute_adaptive_step(
        self,
        current_step: float,
        eval_buffer: deque,
        iterations_improving: int,
        iterations_stuck: int,
    ) -> float:
        """
        Learn appropriate step size from objective landscape and search progress.

        - Flat regions (low variance) -> increase step to escape
        - Improving trend -> decrease step for precision
        - Stuck (no progress) -> increase step for exploration
        """
        if not self.enable_adaptive_step or len(eval_buffer) < 5:
            return current_step * 0.95  # Default decay

        recent_costs = [e.cost for e in list(eval_buffer)[-8:]]
        cost_variance = np.var(recent_costs)
        cost_trend = recent_costs[-1] - recent_costs[0] if len(recent_costs) > 1 else 0

        new_step = current_step

        if cost_variance < 0.0005:
            # Very flat region - take bigger steps to escape plateau
            new_step = min(current_step * 1.4, self.initial_step * 1.5)
        elif cost_trend < -0.01 and iterations_improving > 3:
            # Consistently improving - smaller, more precise steps
            new_step = current_step * 0.85
        elif iterations_stuck > 6:
            # Stuck - need more exploration
            new_step = min(current_step * 1.3, self.initial_step)
        else:
            # Normal decay
            new_step = current_step * 0.93

        # Clamp to reasonable range
        return np.clip(new_step, self.min_step, self.initial_step * 2)

    # =========================================================================
    # LEARNING MECHANISM 4: Search Dynamics Learning (Adaptive Tenure)
    # =========================================================================

    def _compute_adaptive_tenure(
        self,
        base_tenure: int,
        iterations_improving: int,
        iterations_stuck: int,
    ) -> int:
        """
        Learn appropriate tabu tenure from search dynamics.

        - Improving -> shorter tenure (exploit current region)
        - Stuck -> longer tenure (force exploration)
        """
        if not self.enable_adaptive_tenure:
            return base_tenure

        if iterations_improving > 5:
            # Exploiting good region - shorter memory
            return max(4, base_tenure - 2)
        elif iterations_stuck > 8:
            # Need more exploration - longer memory
            return min(15, base_tenure + 3)

        return base_tenure

    # =========================================================================
    # LEARNING MECHANISM 5: Curvature Learning (L-BFGS-B Polish)
    # =========================================================================

    def _polish_with_lbfgs(
        self,
        params: np.ndarray,
        n_sources: int,
        sample: Dict,
        meta: Dict,
        bounds: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Use L-BFGS-B to polish the solution.

        L-BFGS-B approximates the Hessian from function evaluations,
        effectively learning the local curvature for efficient refinement.
        """
        if not self.enable_lbfgs_polish:
            cost = self._objective(params, n_sources, sample, meta)
            return params, cost

        def obj(x):
            return self._objective(x, n_sources, sample, meta)

        try:
            result = minimize(
                obj,
                params,
                method='L-BFGS-B',
                bounds=[(b[0], b[1]) for b in bounds],
                options={
                    'maxiter': self.polish_max_iter,
                    'ftol': 1e-7,
                    'gtol': 1e-6,
                }
            )
            return result.x, result.fun
        except Exception:
            cost = self._objective(params, n_sources, sample, meta)
            return params, cost

    # =========================================================================
    # Neighborhood Generation
    # =========================================================================

    def _generate_neighbors(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
        step_size: float,
        gradient: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """Generate neighbors with gradient-informed and exploratory moves."""
        neighbors = []
        param_ranges = bounds[:, 1] - bounds[:, 0]
        n_gradient_moves = 0

        # Gradient-informed moves
        if gradient is not None:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                descent_dir = -gradient / grad_norm
                n_gradient = int(self.n_neighbors * self.gradient_exploitation_ratio)
                n_gradient_moves = n_gradient

                for _ in range(n_gradient):
                    scale = np.random.uniform(0.4, 1.4)
                    noise = np.random.randn(len(current)) * 0.15
                    step = (descent_dir + noise) * step_size * param_ranges * scale
                    neighbor = np.clip(current + step, bounds[:, 0], bounds[:, 1])
                    neighbors.append(neighbor)

        # Exploratory moves
        n_exploratory = self.n_neighbors - len(neighbors)

        # Single-dimension perturbations
        n_single = min(n_exploratory // 2, len(current) * 2)
        for _ in range(n_single):
            dim = np.random.randint(len(current))
            neighbor = current.copy()
            delta = np.random.uniform(-1, 1) * step_size * param_ranges[dim]
            neighbor[dim] = np.clip(neighbor[dim] + delta, bounds[dim, 0], bounds[dim, 1])
            neighbors.append(neighbor)

        # Multi-dimension perturbations
        while len(neighbors) < self.n_neighbors:
            neighbor = current.copy()
            n_dims = np.random.randint(1, len(current) + 1)
            dims = np.random.choice(len(current), n_dims, replace=False)
            for dim in dims:
                delta = np.random.uniform(-1, 1) * step_size * param_ranges[dim]
                neighbor[dim] = np.clip(neighbor[dim] + delta, bounds[dim, 0], bounds[dim, 1])
            neighbors.append(neighbor)

        return neighbors, n_gradient_moves

    # =========================================================================
    # Tabu Management
    # =========================================================================

    def _is_tabu(
        self,
        candidate: np.ndarray,
        tabu_list: List[TabuEntry],
        bounds: np.ndarray,
    ) -> bool:
        """Check if candidate is within tabu radius of any tabu entry."""
        param_ranges = bounds[:, 1] - bounds[:, 0]
        normalized = (candidate - bounds[:, 0]) / param_ranges

        for entry in tabu_list:
            normalized_tabu = (entry.params - bounds[:, 0]) / param_ranges
            if np.linalg.norm(normalized - normalized_tabu) < self.tabu_radius:
                return True
        return False

    def _update_tabu_list(
        self,
        tabu_list: List[TabuEntry],
        new_entry: np.ndarray,
        tenure: int,
    ) -> List[TabuEntry]:
        """Update tabu list with adaptive tenure."""
        updated = [e for e in tabu_list if e.remaining_tenure > 1]
        for e in updated:
            e.remaining_tenure -= 1

        updated.append(TabuEntry(params=new_entry.copy(), remaining_tenure=tenure))
        return updated

    # =========================================================================
    # Single Search Run
    # =========================================================================

    def _single_search(
        self,
        sample: Dict,
        meta: Dict,
        bounds: np.ndarray,
        n_sources: int,
        use_smart_init: bool,
    ) -> SearchResult:
        """Run a single adaptive learning search."""

        # Initialize (observational learning)
        if use_smart_init:
            current = self._smart_init_from_observations(bounds, sample, n_sources)
        else:
            current = self._random_init(bounds)

        current_cost = self._objective(current, n_sources, sample, meta)

        # Evaluation buffer for gradient learning
        eval_buffer: deque = deque(maxlen=self.gradient_buffer_size)
        eval_buffer.append(Evaluation(current.copy(), current_cost, 0))

        best_ever = current.copy()
        best_ever_cost = current_cost
        best_pre_polish = best_ever_cost

        tabu_list: List[TabuEntry] = []
        history = [current_cost]
        good_candidates = [(current.copy(), current_cost)]

        step_size = self.initial_step
        current_tenure = self.base_tabu_tenure
        iterations_improving = 0
        iterations_stuck = 0

        metrics = SearchMetrics()

        for iteration in range(self.max_iterations):
            # Learn gradient from evaluations
            gradient = self._estimate_gradient(current, eval_buffer, bounds)

            # Generate neighbors (gradient-informed + exploratory)
            neighbors, n_grad = self._generate_neighbors(
                current, bounds, step_size, gradient
            )
            metrics.gradient_moves_used += n_grad
            metrics.total_moves += len(neighbors)

            # Evaluate neighbors
            neighbor_costs = []
            for neighbor in neighbors:
                cost = self._objective(neighbor, n_sources, sample, meta)
                neighbor_costs.append((neighbor, cost))
                eval_buffer.append(Evaluation(neighbor.copy(), cost, iteration))

            neighbor_costs.sort(key=lambda x: x[1])

            # Find best admissible neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor, cost in neighbor_costs:
                is_tabu = self._is_tabu(neighbor, tabu_list, bounds)
                # Aspiration: accept if better than best ever
                if is_tabu and cost >= best_ever_cost:
                    continue
                best_neighbor = neighbor
                best_neighbor_cost = cost
                break

            if best_neighbor is None:
                best_neighbor, best_neighbor_cost = neighbor_costs[0]

            # Move
            current = best_neighbor
            current_cost = best_neighbor_cost

            # Update tabu list with adaptive tenure
            current_tenure = self._compute_adaptive_tenure(
                self.base_tabu_tenure, iterations_improving, iterations_stuck
            )
            if current_tenure != self.base_tabu_tenure:
                metrics.adaptive_tenure_changes += 1
            tabu_list = self._update_tabu_list(tabu_list, current, current_tenure)

            # Track improvement
            if current_cost < best_ever_cost:
                best_ever = current.copy()
                best_ever_cost = current_cost
                iterations_improving += 1
                iterations_stuck = 0
                metrics.improvements += 1
                good_candidates.append((current.copy(), current_cost))
            else:
                iterations_stuck += 1
                iterations_improving = 0

            # Adaptive step size (landscape learning)
            step_size = self._compute_adaptive_step(
                step_size, eval_buffer, iterations_improving, iterations_stuck
            )

            history.append(current_cost)

            # Early termination
            if best_ever_cost < 1e-6:
                break

        metrics.final_step_size = step_size

        # L-BFGS-B polishing (curvature learning)
        best_pre_polish = best_ever_cost
        polished_params, polished_cost = self._polish_with_lbfgs(
            best_ever, n_sources, sample, meta, bounds
        )

        if polished_cost < best_ever_cost:
            best_ever = polished_params
            best_ever_cost = polished_cost
            metrics.polish_improvement = best_pre_polish - polished_cost
            good_candidates.append((polished_params.copy(), polished_cost))

        # Filter distinct candidates
        distinct = self._filter_distinct_candidates(good_candidates, bounds)

        return SearchResult(
            best_params=best_ever,
            best_cost=best_ever_cost,
            all_candidates=distinct,
            history=history,
            metrics=metrics,
        )

    def _filter_distinct_candidates(
        self,
        candidates: List[Tuple[np.ndarray, float]],
        bounds: np.ndarray,
        min_distance: float = 0.1,
    ) -> List[Tuple[np.ndarray, float]]:
        """Filter to keep only distinct candidates."""
        if not candidates:
            return []

        param_ranges = bounds[:, 1] - bounds[:, 0]
        sorted_cands = sorted(candidates, key=lambda x: x[1])

        distinct = [sorted_cands[0]]
        for params, cost in sorted_cands[1:]:
            normalized = (params - bounds[:, 0]) / param_ranges
            is_distinct = all(
                np.linalg.norm(normalized - (ep - bounds[:, 0]) / param_ranges) >= min_distance
                for ep, _ in distinct
            )
            if is_distinct:
                distinct.append((params, cost))

        return distinct

    # =========================================================================
    # Public API
    # =========================================================================

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        use_smart_init: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """Estimate heat source parameters."""
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []
        for restart in range(self.n_restarts):
            use_smart = use_smart_init and (restart == 0)
            result = self._single_search(sample, meta, bounds, n_sources, use_smart)
            all_results.append(result)

            if verbose:
                grad_ratio = result.metrics.gradient_moves_used / max(result.metrics.total_moves, 1)
                print(f"  Restart {restart+1}/{self.n_restarts}: "
                      f"RMSE={result.best_cost:.6f}, grad={grad_ratio:.0%}, "
                      f"polish_gain={result.metrics.polish_improvement:.6f}")

        best = min(all_results, key=lambda r: r.best_cost)
        sources = []
        for i in range(n_sources):
            x, y, q = best.best_params[i*3:(i+1)*3]
            sources.append((x, y, q))

        return sources, best.best_cost

    def estimate_sources_with_metrics(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        use_smart_init: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[Tuple[np.ndarray, float]], Dict]:
        """
        Estimate sources with full metrics for tracking.

        Returns:
            (sources, rmse, candidates, metrics_dict)
        """
        n_sources = sample['n_sources']
        bounds = self._get_bounds(n_sources, q_range)

        all_results = []
        all_candidates = []
        aggregated_metrics = {
            'gradient_moves_used': 0,
            'total_moves': 0,
            'improvements': 0,
            'polish_improvements': [],
            'final_step_sizes': [],
            'adaptive_tenure_changes': 0,
        }

        for restart in range(self.n_restarts):
            use_smart = use_smart_init and (restart == 0)
            result = self._single_search(sample, meta, bounds, n_sources, use_smart)
            all_results.append(result)
            all_candidates.extend(result.all_candidates)

            # Aggregate metrics
            m = result.metrics
            aggregated_metrics['gradient_moves_used'] += m.gradient_moves_used
            aggregated_metrics['total_moves'] += m.total_moves
            aggregated_metrics['improvements'] += m.improvements
            aggregated_metrics['polish_improvements'].append(m.polish_improvement)
            aggregated_metrics['final_step_sizes'].append(m.final_step_size)
            aggregated_metrics['adaptive_tenure_changes'] += m.adaptive_tenure_changes

            if verbose:
                grad_ratio = m.gradient_moves_used / max(m.total_moves, 1)
                print(f"  Restart {restart+1}/{self.n_restarts}: "
                      f"RMSE={result.best_cost:.6f}, grad={grad_ratio:.0%}")

        best = min(all_results, key=lambda r: r.best_cost)
        distinct = self._filter_distinct_candidates(all_candidates, bounds)

        # Finalize metrics
        total_moves = aggregated_metrics['total_moves']
        metrics = {
            'gradient_usage': aggregated_metrics['gradient_moves_used'] / max(total_moves, 1),
            'total_improvements': aggregated_metrics['improvements'],
            'avg_polish_improvement': np.mean(aggregated_metrics['polish_improvements']),
            'avg_final_step_size': np.mean(aggregated_metrics['final_step_sizes']),
            'tenure_adaptations': aggregated_metrics['adaptive_tenure_changes'],
            'n_candidates': len(distinct),
        }

        sources = []
        for i in range(n_sources):
            x, y, q = best.best_params[i*3:(i+1)*3]
            sources.append((x, y, q))

        return sources, best.best_cost, distinct, metrics
