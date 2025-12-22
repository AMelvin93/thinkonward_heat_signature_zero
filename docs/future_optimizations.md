# Future Optimization Options

This document captures optimization strategies for the Tabu Search approach that learn and adapt during inference. These are aligned with competition requirements (no grid search, learns by inference).

---

## 1. Adaptive Step Size from Landscape Learning

**Status:** Not implemented
**Priority:** Medium
**Estimated Impact:** Better convergence, fewer wasted evaluations

### Concept
Learn the "roughness" of the objective function from recent evaluations and adjust step size accordingly:
- Flat region (low variance) → Take bigger steps to escape plateau
- Steep region (improving) → Take smaller steps to refine
- Diverging (getting worse) → Try different scale

### Implementation

```python
def adaptive_step_size(self, history: List[float], current_step: float) -> float:
    """Learn appropriate step size from recent progress."""
    if len(history) < 5:
        return current_step

    recent_costs = history[-10:]
    cost_variance = np.var(recent_costs)
    cost_trend = recent_costs[-1] - recent_costs[0]  # Negative = improving

    if cost_variance < 0.001:  # Flat region - plateau detected
        # Take bigger steps to escape
        return min(current_step * 1.5, 0.3)

    elif cost_trend < -0.01:  # Clearly improving
        # Refine with smaller steps (exploitation)
        return max(current_step * 0.85, 0.02)

    elif cost_trend > 0.01:  # Getting worse
        # Try a different scale (shake things up)
        return current_step * np.random.uniform(0.7, 1.3)

    else:  # Stable
        return current_step * 0.95  # Gradual decay
```

### Integration Point
Call in `_single_search()` before generating neighbors:
```python
step_size = self.adaptive_step_size(history, step_size)
neighbors = self._generate_neighbors(current, bounds, step_size)
```

---

## 2. Elite Memory with Intensification

**Status:** Not implemented
**Priority:** Medium
**Estimated Impact:** Better exploitation of promising regions

### Concept
Maintain a memory of the K best solutions found during search. Periodically:
1. Analyze what elite solutions have in common
2. Focus search around the elite centroid
3. Use elite variance to determine search radius

### Implementation

```python
@dataclass
class EliteMemory:
    """Track best K solutions found during search."""
    max_size: int = 10
    solutions: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    def add(self, params: np.ndarray, cost: float):
        self.solutions.append((params.copy(), cost))
        self.solutions.sort(key=lambda x: x[1])  # Sort by cost
        self.solutions = self.solutions[:self.max_size]  # Keep top K

    def get_centroid(self) -> np.ndarray:
        if not self.solutions:
            return None
        params = np.array([s[0] for s in self.solutions])
        return np.mean(params, axis=0)

    def get_spread(self) -> np.ndarray:
        if len(self.solutions) < 2:
            return None
        params = np.array([s[0] for s in self.solutions])
        return np.std(params, axis=0)


def intensification_phase(
    self,
    elite: EliteMemory,
    bounds: np.ndarray,
    sample: Dict,
    meta: Dict,
    n_samples: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Learn from elite solutions to find even better ones.
    Called periodically during search (e.g., every 10 iterations).
    """
    centroid = elite.get_centroid()
    spread = elite.get_spread()

    if centroid is None or spread is None:
        return None, float('inf')

    # Sample around the learned promising region
    best_params, best_cost = None, float('inf')

    for _ in range(n_samples):
        # Sample near elite centroid with learned spread
        new_params = centroid + spread * np.random.randn(len(centroid)) * 0.5
        new_params = np.clip(new_params, bounds[:, 0], bounds[:, 1])

        cost = self._objective(new_params, n_sources, sample, meta)

        if cost < best_cost:
            best_cost = cost
            best_params = new_params

    return best_params, best_cost
```

### Integration Point
Call periodically in main search loop:
```python
if iteration % 10 == 0 and len(elite.solutions) >= 5:
    intensified, cost = self.intensification_phase(elite, bounds, sample, meta)
    if cost < best_ever_cost:
        best_ever = intensified
        best_ever_cost = cost
```

---

## 3. Sequential Source Learning (Multi-Source)

**Status:** Not implemented
**Priority:** High (60% of test samples are 2-source)
**Estimated Impact:** Significant improvement for multi-source problems

### Concept
For multi-source problems, learn sources sequentially:
1. Find the dominant source first (single-source search)
2. Compute residual (what's unexplained by source 1)
3. Search for source 2 in the residual
4. Jointly refine both sources together

This decomposes a hard 6D problem (2 sources × 3 params) into two easier 3D problems.

### Implementation

```python
def sequential_source_estimation(
    self,
    sample: Dict,
    meta: Dict,
    q_range: Tuple[float, float],
) -> Tuple[List[Tuple], float]:
    """Learn sources sequentially for multi-source problems."""
    n_sources = sample['n_sources']

    if n_sources == 1:
        return self.estimate_sources(sample, meta, q_range)

    # Step 1: Find best single-source approximation
    # This learns the dominant heat source
    single_sample = sample.copy()
    single_sample['n_sources'] = 1

    source_1, rmse_1 = self._single_source_search(single_sample, meta, q_range)
    learned_source_1 = source_1[0]  # (x, y, q)

    # Step 2: Compute residual - what source 1 doesn't explain
    simulated_1 = self._simulate_single_source(learned_source_1, sample, meta)
    observed = sample['Y_noisy']
    residual = observed - simulated_1  # What's left to explain

    # Step 3: Search for source 2 using residual as target
    # Create modified sample with residual as observations
    residual_sample = sample.copy()
    residual_sample['Y_noisy'] = residual
    residual_sample['n_sources'] = 1

    source_2, rmse_2 = self._single_source_search(residual_sample, meta, q_range)
    learned_source_2 = source_2[0]

    # Step 4: Joint refinement using learned initialization
    # Now we have good starting points for both sources
    initial_guess = np.array([
        learned_source_1[0], learned_source_1[1], learned_source_1[2],
        learned_source_2[0], learned_source_2[1], learned_source_2[2],
    ])

    refined_params, final_rmse = self._refine_jointly(
        initial_guess, sample, meta, q_range
    )

    # Convert to source list
    sources = [
        (refined_params[0], refined_params[1], refined_params[2]),
        (refined_params[3], refined_params[4], refined_params[5]),
    ]

    return sources, final_rmse


def _refine_jointly(
    self,
    initial: np.ndarray,
    sample: Dict,
    meta: Dict,
    q_range: Tuple[float, float],
    max_iter: int = 20,
) -> Tuple[np.ndarray, float]:
    """Joint refinement of all sources using learned initialization."""
    # Use L-BFGS-B for efficient local refinement
    from scipy.optimize import minimize

    n_sources = sample['n_sources']
    bounds = self._get_bounds(n_sources, q_range)

    def objective(params):
        return self._objective(params, n_sources, sample, meta)

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter}
    )

    return result.x, result.fun
```

### Integration Point
Replace `estimate_sources()` for multi-source samples:
```python
def estimate_sources(self, sample, meta, q_range):
    if sample['n_sources'] > 1:
        return self.sequential_source_estimation(sample, meta, q_range)
    else:
        return self._standard_search(sample, meta, q_range)
```

---

## 4. Adaptive Tabu Tenure

**Status:** Not implemented
**Priority:** Low
**Estimated Impact:** Modest improvement in exploration/exploitation balance

### Concept
Dynamically adjust tabu tenure based on search progress:
- Improving → Shorter tenure (exploit current region)
- Stuck → Longer tenure (force exploration of new regions)

### Implementation

```python
def adaptive_tenure(
    self,
    current_tenure: int,
    iterations_without_improvement: int,
    just_improved: bool,
    min_tenure: int = 3,
    max_tenure: int = 20,
) -> int:
    """Learn appropriate tabu tenure from search dynamics."""

    if just_improved:
        # Getting better - allow revisiting nearby solutions
        return max(min_tenure, current_tenure - 1)

    elif iterations_without_improvement > 5:
        # Stuck - force exploration by extending tabu
        return min(max_tenure, current_tenure + 2)

    elif iterations_without_improvement > 10:
        # Very stuck - aggressive exploration
        return max_tenure

    return current_tenure  # No change
```

### Integration Point
Update in main search loop after each iteration:
```python
tabu_tenure = self.adaptive_tenure(
    tabu_tenure,
    iterations_without_improvement,
    just_improved=(current_cost < best_ever_cost)
)
```

---

## Implementation Priority

| Optimization | Priority | Complexity | Expected Impact |
|-------------|----------|------------|-----------------|
| Gradient-Informed Neighborhood | **Implemented** | Medium | High |
| Sequential Source Learning | High | Medium | High (for 2-src) |
| Adaptive Step Size | Medium | Low | Medium |
| Elite Memory | Medium | Medium | Medium |
| Adaptive Tenure | Low | Low | Low |

---

## Testing Strategy

For each optimization:
1. Create a new experiment folder (e.g., `experiments/tabu_sequential/`)
2. Run on subset of test data (n=10 samples)
3. Compare against baseline tabu_search
4. Log metrics to MLflow for comparison
5. If promising, run full evaluation

```bash
# Quick comparison
uv run python scripts/run_experiment.py -e tabu_baseline -n 10
uv run python scripts/run_experiment.py -e tabu_gradient -n 10
uv run python scripts/run_experiment.py -e tabu_sequential -n 10

# View in MLflow
uv run mlflow ui --port 5000
```
