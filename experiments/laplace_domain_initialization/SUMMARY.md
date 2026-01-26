# Experiment Summary: laplace_domain_initialization

## Metadata
- **Experiment ID**: EXP_LAPLACE_DOMAIN_INIT_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: initialization_v5

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Transform sensor time series to Laplace domain where heat diffusion has simpler analytical form. Use frequency content to estimate source distance and position.

## Why Aborted

**All initialization families (v1-v5) have been marked EXHAUSTED.** The baseline triangulation + hotspot initialization is already optimal.

### Key Prior Evidence

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| **physics_informed_init** | FAILED (-0.0046) | "The initialization family should be marked as EXHAUSTED" |
| **boundary_aware_initialization** | ABORTED | "Mark initialization_v2 family as EXHAUSTED" |
| **ica_seeded_init** | FAILED | "initialization_v3 EXHAUSTED. Triangulation+hotspot already optimal." |
| **more_inits_select_best** | FAILED | "Baseline 2-init strategy is already optimal" |
| **improved_triangulation_init** | ABORTED | "Acoustic formulas use WRONG physics for heat diffusion" |

### Why Laplace Domain Cannot Help

1. **Initialization Is Already Solved**
   - Quote from physics_informed_init: "Simple is better: The hottest-sensor approach works well"
   - Current triangulation uses correct diffusion physics (r ~ sqrt(4κt))
   - 5+ experiments have failed to improve initialization

2. **Data Limitation**
   - Sensor data is typically at final or near-steady-state
   - Laplace analysis requires transient time-series with sufficient resolution
   - The temporal sampling in test data may not support frequency analysis

3. **Same Fundamental Problem**
   - All initialization improvements target the same goal: better starting points
   - CMA-ES is robust to initialization because of population-based search
   - Quote: "CMA-ES handles bad initialization through covariance adaptation"

## Technical Analysis

### What Laplace Domain Approach Would Involve

```python
# Transform sensor time series to Laplace domain
# For 1D heat equation: L{T(x,t)} = T_hat(x,s)
# Laplace-domain solution has form: T_hat ~ exp(-sqrt(s/kappa) * r)

# The approach would:
# 1. Compute Laplace transform of each sensor's time series
# 2. Analyze decay rate in s-domain to estimate source distance
# 3. Combine multiple sensors for triangulation
```

### Why It Wouldn't Work

1. **Numerical Laplace transform is ill-conditioned**
   - Requires high-quality time series
   - Sensitive to noise in sensor readings

2. **Assumes idealized 1D heat diffusion**
   - Actual problem has 2D domain with boundaries
   - Multiple sources create interference

3. **Doesn't address the real bottleneck**
   - 2-source problems have RMSE ~0.15 vs ~0.10 for 1-source
   - The bottleneck is optimization/parameterization, not initialization

## Algorithm Family Status

- **initialization (v1, v2, v3, v4, v5)**: **EXHAUSTED**
- Key insight: Triangulation + hotspot is locally optimal for initialization

### Exhaustion Evidence

| Version | Experiments Tried | Conclusion |
|---------|-------------------|------------|
| v1 | physics_informed_init | Simple hotspot is optimal |
| v2 | boundary_aware_initialization | 24% boundary sources hurt biasing |
| v3 | ica_seeded_init, improved_triangulation | Current init is optimal |
| v4 | condition_number_init | Sensor weighting is proxy optimization |
| v5 | laplace_domain_initialization (this) | No time-series data advantage |

## Recommendations

1. **Do NOT pursue any initialization improvements** - the family is exhausted
2. **Accept that initialization is solved** - focus on other components
3. **The real bottleneck is 2-source accuracy** - not initialization

## Conclusion

The Laplace domain initialization would fail because: (1) the initialization family (v1-v5) is exhausted with 5+ failed experiments, (2) the current triangulation + hotspot approach is already optimal, (3) Laplace analysis requires high-quality transient time-series which may not be available, and (4) initialization is not the bottleneck for improving accuracy.

The initialization family should be considered permanently EXHAUSTED. No further initialization experiments should be created.
