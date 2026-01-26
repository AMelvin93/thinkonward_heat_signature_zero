# Experiment Summary: intensity_prior_from_peak

## Metadata
- **Experiment ID**: EXP_INTENSITY_PRIOR_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: physics_init_v2

## Status: ABORTED (Based on Misunderstanding of Current Approach)

## Objective
Use physics to bound intensity search. The hypothesis was that T_max at sensor ∝ q/r² could be used to estimate feasible (q, r) combinations and narrow the search space.

## Why Aborted

This experiment is based on a **fundamental misunderstanding** of the current optimization approach. **The intensity q is NOT searched** - it is computed analytically.

### Current Approach (Variable Projection)

The baseline already uses Variable Projection implicitly, as documented in `experiments/variable_projection_separable/SUMMARY.md`:

```python
# For each CMA-ES candidate position (x, y):
# 1-source case:
q_optimal = dot(Y_unit, Y_obs) / dot(Y_unit, Y_unit)  # Closed-form solution

# 2-source case:
# Solve linear system: A * [q1, q2] = b
# where A[i,j] = dot(Yi, Yj), b[i] = dot(Yi, Y_obs)
```

This is a **closed-form analytical solution**, not a search. The optimization only searches over positions (x, y), and for each position, the optimal intensity q is computed exactly using least squares.

### The q_range Bounds Are Already Used

The q_range (0.5, 2.0) is used only for **clipping** the analytically computed q values:
```python
q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
```

There is no "intensity search" to narrow. The physics relationship T_max ∝ q/r² cannot help because:
1. q is already computed optimally given positions
2. The positions (related to r) are what CMA-ES searches, not q
3. Bounding q more tightly would only hurt accuracy by clipping good solutions

## Prior Evidence

### variable_projection_separable (EXP_SEPARABLE_VP_001)
- **Key Finding**: "The baseline already uses Variable Projection implicitly"
- **Quote**: "CMA-ES explores globally, VP provides optimal q for each position"
- **Conclusion**: The separable structure is already exploited optimally

### physics_informed_init (EXP_PHYSICS_INIT_001)
- **Result**: FAILED (-0.0046 score, +2.3 min)
- **Finding**: "Temperature gradients at sensors don't accurately point to source locations"
- **Conclusion**: "The initialization family should be marked as EXHAUSTED"

## Technical Explanation

The experiment proposal misunderstands the optimization structure:

| What | Baseline Approach | Proposed (Incorrect) |
|------|-------------------|---------------------|
| **Position (x,y)** | Searched by CMA-ES | Not addressed |
| **Intensity (q)** | Computed analytically | Proposed to "bound search" |

Since q is computed analytically, there is no search to bound. Any physics-based bounds would only:
1. Add computational overhead (gradient computation, thermal analysis)
2. Potentially clip good solutions if bounds are too tight
3. Have no effect if bounds are looser than current (0.5, 2.0)

## Recommendations

1. **Do NOT pursue intensity bounding approaches** - q is not searched
2. **physics_init_v2 family can be marked EXHAUSTED** - prior evidence conclusive
3. **Focus on position optimization** - that's where search happens
4. **Consider problem formulation** before proposing physics-based improvements

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: variable_projection_separable, physics_informed_init

## Conclusion

This experiment is based on a misunderstanding. The intensity q is computed analytically using Variable Projection, not searched. The physics relationship T_max ∝ q/r² cannot narrow a search that doesn't exist. The current approach already exploits the separable structure optimally.
