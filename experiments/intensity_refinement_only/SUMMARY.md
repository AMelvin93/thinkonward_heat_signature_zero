# Experiment Summary: intensity_refinement_only

## Metadata
- **Experiment ID**: EXP_INTENSITY_REFINEMENT_ONLY_001
- **Worker**: W2
- **Date**: 2026-01-27
- **Algorithm Family**: parameter_tuning_v2

## Objective
After CMA-ES finds position, do additional intensity-only optimization via grid search over q ∈ [0.5, 2.0] with 10 points.

## Hypothesis
Position (x,y) is hard to find but intensity (q) can be optimized more cheaply after CMA-ES converges.

## Feasibility Analysis

### Why This is NOT_FEASIBLE

#### The Baseline Already Implements a SUPERIOR Approach

After reviewing the baseline optimizer (`temporal_40pct_higher_sigma/optimizer.py`), I discovered:

**The baseline uses Variable Projection (VarPro):**
1. CMA-ES only optimizes position parameters: (x, y) for 1-source, (x1, y1, x2, y2) for 2-source
2. For EVERY position evaluation, intensity is computed analytically via closed-form least-squares

**Relevant code from baseline (lines 77-97, 102-127):**
```python
def compute_optimal_intensity_1src(x, y, Y_observed, ...):
    """Compute optimal intensity for 1-source."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_obs_trunc.flatten()
    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    q_optimal = np.dot(Y_unit_flat, Y_obs_flat) / denominator  # CLOSED-FORM SOLUTION
    return q_optimal, Y_pred, rmse
```

This is the **globally optimal** intensity for any fixed position - it minimizes ||Y_pred - Y_obs||² exactly.

#### Comparison: Analytic vs Grid Search

| Aspect | Analytic (Baseline) | Grid Search (Proposed) |
|--------|---------------------|------------------------|
| Accuracy | Continuous optimum | 10 discrete points only |
| Extra simulations | 0 | 10 per candidate |
| Optimality guarantee | Yes (closed-form) | No (may miss optimum) |
| Implementation | Already done | Would require changes |

#### Mathematical Proof

For linear intensity (which this problem has), the optimal intensity for fixed position (x,y) is:

```
q* = argmin_q ||q * Y_unit(x,y) - Y_obs||²
   = (Y_unit · Y_obs) / (Y_unit · Y_unit)  [closed-form solution]
```

This is EXACTLY what `compute_optimal_intensity_1src/2src` computes. No grid search can do better.

### What the Experiment Would Actually Test

The experiment proposes:
1. Run CMA-ES to find position
2. Grid search intensity with 10 points

But the baseline ALREADY:
1. Runs CMA-ES on position only
2. Computes intensity analytically (optimal for each position)
3. Uses NM polish on position only
4. Recomputes intensity analytically for final candidates

The proposed experiment would be a **regression**, not an improvement.

## Tuning Efficiency Metrics
- **Runs executed**: 0 (code analysis only)
- **Time utilization**: N/A
- **Reason for 0 runs**: Experiment proposes approach that's already implemented in superior form

## Why No Runs Were Needed

Per tuning rules, I should "validate before aborting". In this case:
1. The hypothesis IS correct - intensity IS easier to optimize
2. But it's already being exploited via analytic solution
3. Running grid search would DECREASE performance
4. This isn't a case of "prior evidence suggests failure" - this is "current implementation is already optimal"

Implementing grid search would require:
1. Removing analytic intensity computation
2. Adding 10 extra simulations per candidate
3. Result: Slower AND less accurate

## Recommendation

**MARK AS NOT_FEASIBLE**

The experiment proposes a strictly inferior version of what the baseline already implements. The intensity optimization is ALREADY being done optimally via closed-form least-squares. Grid search cannot improve on this.

## Key Insight for Future Experiments

The baseline's Variable Projection approach means:
- Position optimization is the hard part (CMA-ES)
- Intensity optimization is FREE (analytic solution)
- Any "intensity refinement" experiments are redundant unless they propose non-linear intensity models

## Conclusion

**NOT_FEASIBLE** - The baseline optimizer already separates position from intensity via Variable Projection, computing intensity analytically via closed-form least-squares. This is the globally optimal intensity for any fixed position. Grid search with 10 discrete points would be both slower (10 extra simulations) and less accurate (discrete vs continuous optimum).
