# tabu_search

Tabu Search metaheuristic for heat source identification.

## Approach

Tabu Search is a memory-guided local search that:
- **Escapes local optima** by maintaining a "tabu list" of recently visited solutions
- **Accepts worse moves** strategically to explore new regions
- **Uses aspiration criteria** to override tabu status for exceptional solutions
- **Generates diverse candidates** through multiple independent restarts

### Why Tabu Search fits this challenge:

| Requirement | How Tabu Search Meets It |
|-------------|-------------------------|
| Uses simulator during inference | Every neighbor evaluation calls the thermal simulator |
| Not brute-force | Intelligent, memory-guided search |
| Escapes local optima | Tabu list prevents cycling; accepts worse moves |
| Generates diverse candidates | Multiple restarts with different initializations |

### Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      TABU SEARCH LOOP                       │
├─────────────────────────────────────────────────────────────┤
│   Current Solution ──► Generate Neighbors ──► Evaluate All  │
│         ▲                                          │        │
│         │                              Filter out Tabu moves│
│         │                                          │        │
│         └────────────── Pick Best Non-Tabu Neighbor         │
│                              Update Tabu List               │
└─────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tabu_tenure` | 10 | How many iterations a solution stays forbidden |
| `tabu_radius` | 0.05 | Normalized distance for proximity check |
| `n_neighbors` | 20 | Candidates generated per iteration |
| `initial_step` | 0.15 | Starting perturbation size |
| `step_decay` | 0.98 | How fast step size shrinks |
| `n_restarts` | 5 | Independent searches for diversity |

## Usage

```bash
# Run full experiment
python scripts/run_experiment.py --experiment tabu_search

# Run on subset for testing
python scripts/run_experiment.py --experiment tabu_search --n-samples 5

# Quick test with fewer iterations
python scripts/run_experiment.py --experiment tabu_search --n-samples 2
```

## Tuning Guide

### More Exploration (stuck in local optima)
```yaml
tabu:
  tabu_tenure: 15      # Longer memory
  tabu_radius: 0.08    # Larger forbidden zone
  n_neighbors: 30      # More candidates
  initial_step: 0.2    # Bigger jumps
```

### More Exploitation (close but not converging)
```yaml
tabu:
  tabu_tenure: 5       # Shorter memory
  tabu_radius: 0.03    # Smaller forbidden zone
  step_decay: 0.95     # Faster refinement
```

### Faster Runtime
```yaml
tabu:
  max_iterations: 30   # Fewer iterations
  n_neighbors: 15      # Fewer candidates
  n_restarts: 3        # Fewer restarts
```

## Results

| Metric | Value |
|--------|-------|
| RMSE (mean) | TBD |
| RMSE (std) | TBD |
| Avg candidates/sample | TBD |

## Notes

- Created: 2024-12-20
- Smart initialization places first guess near hottest sensor
- Simulator calls per sample: ~n_restarts × max_iterations × n_neighbors
