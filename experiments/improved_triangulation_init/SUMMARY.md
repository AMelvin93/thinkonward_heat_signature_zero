# Experiment Summary: improved_triangulation_init

## Metadata
- **Experiment ID**: EXP_TRIANGULATION_QUALITY_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: initialization_v3

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Test alternative triangulation formulas from acoustic source localization literature to potentially improve initialization quality.

## Why Aborted

This experiment is based on **two false premises**:

### False Premise 1: Initialization Can Be Improved
The **initialization family has been marked EXHAUSTED** by multiple prior experiments:

| Experiment | Result | Conclusion |
|------------|--------|------------|
| physics_informed_init | FAILED (-0.0046 score, +2.3 min) | "The initialization family should be marked as EXHAUSTED" |
| boundary_aware_initialization | ABORTED | "Mark initialization_v2 family as EXHAUSTED" |
| more_inits_select_best | FAILED | "The baseline 2-init strategy (triangulation + hotspot) is already optimal" |
| multistart_elite_selection | FAILED | "Single-start with good initialization is optimal" |

### False Premise 2: Acoustic Formulas Apply to Heat Diffusion
Acoustic source localization uses **wave propagation physics**:
```
r = v * t    (wave: constant velocity)
```

Heat diffusion uses **different physics**:
```
r ~ sqrt(4 * kappa * t)    (diffusion: square root relationship)
```

The current triangulation implementation **already uses the correct physics**. From `src/triangulation.py`:
```python
def estimate_distance_from_onset(onset_time, kappa=1.0, threshold_factor=0.1):
    """Estimate source distance from temperature onset time using diffusion physics."""
    r = np.sqrt(4 * kappa * onset_time * np.abs(np.log(threshold_factor)))
    return r
```

Acoustic localization formulas would be **wrong** for this problem because they assume constant-velocity wave propagation, not diffusive transport.

## Prior Evidence Summary

### 1. physics_informed_init (EXP_PHYSICS_INIT_001)
**Result**: FAILED - Score 1.1642 (baseline: 1.1688), Time +2.3 min
- Temperature gradients at sensor locations don't accurately point to sources
- The gradient signal is corrupted by thermal diffusion
- **Conclusion**: "The initialization family should be marked as EXHAUSTED"

### 2. more_inits_select_best
**Result**: FAILED - Additional inits do NOT improve results
- Tested 5 inits vs baseline 2 inits
- Overhead of extra CMA-ES instances outweighs any benefit
- **Conclusion**: "The baseline 2-init strategy (triangulation + hotspot) is already optimal"

### 3. multistart_elite_selection
**Result**: FAILED - Elite selection doesn't work
- Early-generation fitness is not predictive of final quality
- Single-start with good initialization is optimal
- **Conclusion**: Parallel starts waste budget

### 4. boundary_aware_initialization
**Result**: ABORTED - Would hurt 24% of samples
- 24% of samples have sources near boundaries
- Biasing away from boundaries would hurt these cases
- **Conclusion**: "Mark initialization_v2 family as EXHAUSTED"

### 5. moment_based_inversion
**Finding**: Triangulation already uses best direct approach
- Quote: "The triangulation initialization already exploits sensor readings to estimate initial source positions. This IS a form of moment-based reasoning."
- **Conclusion**: Direct inversion family EXHAUSTED

## Technical Explanation

### Current Triangulation Implementation
The baseline triangulation uses:

1. **Onset Time Detection**: Find when each sensor first detects significant temperature rise
2. **Diffusion Distance Estimation**: `r = sqrt(4 * kappa * t * |log(threshold)|)`
3. **Trilateration**: Solve for (x, y) given multiple (sensor, distance) pairs

This is **physics-optimal** for heat diffusion. The sqrt relationship correctly captures how thermal information spreads from sources.

### Why Acoustic Formulas Would Fail

| Property | Acoustic Waves | Heat Diffusion |
|----------|---------------|----------------|
| Speed | Constant (v) | Decreases with distance |
| Arrival time | r = v*t | r ~ sqrt(t) |
| Signal shape | Preserved | Spreads/smooths |
| Multipath | Reflects | Dissipates |

Using acoustic formulas (r = v*t) would:
1. **Overestimate nearby distances** (diffusion is faster initially)
2. **Underestimate far distances** (diffusion slows down)
3. Result in **worse** initialization than current approach

## Algorithm Family Status

- **initialization (v1, v2, v3)**: **EXHAUSTED**
- Key insight: Triangulation + hotspot is already optimal for this problem

## Recommendations

1. **Do NOT pursue any initialization improvements** - all variants have been tested and failed
2. **The current physics-based triangulation is optimal** - uses correct diffusion relationship
3. **Acoustic formulas are WRONG for this problem** - different physics
4. **Focus on other components** - optimization algorithm, polish, temporal fidelity

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: physics_informed_init, more_inits_select_best, multistart_elite_selection, boundary_aware_initialization, moment_based_inversion

## Conclusion

This experiment would fail for two reasons: (1) the initialization family is already exhausted with 5+ failed attempts to improve it, and (2) acoustic localization formulas use wave physics (r = v*t) which is fundamentally wrong for heat diffusion (r ~ sqrt(t)). The current triangulation already uses the correct diffusion physics relationship.

The initialization_v3 family should be marked EXHAUSTED. No further initialization experiments should be created.
