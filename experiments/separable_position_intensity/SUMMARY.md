# Experiment Summary: separable_position_intensity

## Status: ABORTED (Already Implemented)

## Experiment ID: EXP_SEPARABLE_INTENSITY_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Given fixed positions, optimal intensity may have closed-form solution. This reduces dimensionality from 3D/6D to 2D/4D.

## Why Aborted

This is **ALREADY implemented in the baseline**. The current optimizer:

1. **CMA-ES optimizes positions only** (2D for 1-source, 4D for 2-source)
2. **Intensity is computed analytically** using least-squares:

```python
def compute_optimal_intensity_1src(x, y, Y_observed, ...):
    Y_unit = simulate_unit_source(x, y, ...)  # Simulate with q=1
    # Analytical optimal intensity via dot product
    q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / np.dot(Y_unit_flat, Y_unit_flat)
    return np.clip(q_optimal, q_range[0], q_range[1])
```

The baseline has been using separable position-intensity optimization since the beginning. This experiment proposes functionality that already exists.

## Prior Implementation
- Location: `experiments/early_timestep_filtering/optimizer_with_polish.py` lines 81-127
- Functions: `compute_optimal_intensity_1src()`, `compute_optimal_intensity_2src()`

## Recommendation
**This experiment should be REMOVED from the queue.** It duplicates existing functionality and does not propose any new approach.
