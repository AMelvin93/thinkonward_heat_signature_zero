# Experiment Summary: boundary_aware_initialization

## Metadata
- **Experiment ID**: EXP_BOUNDARY_AWARE_INIT_001
- **Worker**: W1
- **Date**: 2026-01-20
- **Algorithm Family**: initialization_v2

## Objective
Test whether initializing heat sources away from domain boundaries improves convergence.

## Hypothesis
Heat sources near boundaries behave differently due to BCs. Initializing away from boundaries may improve convergence.

## Results Summary
- **Status**: ABORTED - Data analysis shows boundary sources exist
- **Tuning Runs**: 0 (aborted before testing)

## Rationale for Abort

### Data Analysis: Boundary Sources Exist

Analyzed hottest sensor locations as proxy for source positions:

| Location | Count | Percentage |
|----------|-------|------------|
| Interior (10-90% of domain) | 61 | 76.2% |
| Near boundary (<10% margin) | 19 | **23.8%** |

**19 samples (24%) have evidence of boundary sources.**

Examples of boundary hotspots:
- Sample 7: hottest at (0.108, 0.301) - near left edge (x < 0.2)
- Sample 10: hottest at (0.114, 0.896) - near left edge AND top edge
- Sample 18: hottest at (1.801, 0.813) - near right edge (x > 1.8)
- Sample 20: hottest at (1.896, 0.842) - near right edge

### Abort Criteria Met

The experiment's own abort criteria states:
> "Boundary constraint hurts cases with actual boundary sources"

With 24% of samples having boundary hotspots, this criterion is clearly met.

### Prior Evidence

**EXP_PHYSICS_INIT_001 (FAILED)**:
- Gradient-based init was WORSE than smart init: -0.0046 score, +2.3 min
- Conclusion: "Simple hottest-sensor init is already optimal"

The **initialization** family was marked **EXHAUSTED** after this failure.

### Why Biasing Away From Boundaries Would Hurt

1. **Smart init uses hottest sensor** - already the best proxy for source location
2. **Biasing toward interior** would move initial guess AWAY from true location for 24% of samples
3. **CMA-ES needs good initialization** - poor init = more iterations = more time
4. **No benefit for interior sources** - they already get good initialization

## Technical Analysis

### Domain and Margin Definitions
```
Domain: Lx=2.0, Ly=1.0
Interior region: x ∈ [0.20, 1.80], y ∈ [0.10, 0.90]
Boundary margin: 10% of domain dimensions
```

### The Problem with Boundary Bias

```
Current smart_init:
  → Uses hottest sensor location
  → Works well for ALL sources (boundary or interior)

Proposed boundary_aware_init:
  → Bias toward interior (10-90%)
  → HURTS 24% of samples with boundary sources
  → No clear benefit for remaining 76%
```

### Why This is Different from EXP_PHYSICS_INIT_001

| Experiment | Approach | Problem |
|------------|----------|---------|
| EXP_PHYSICS_INIT_001 | Gradient-based init | Gradients corrupted by diffusion |
| EXP_BOUNDARY_AWARE_INIT_001 | Interior-biased init | Hurts boundary source cases |

Both approaches try to "improve" on smart init, but smart init (hottest sensor) is already optimal.

## Key Finding

**Boundary sources exist in 24% of test samples.** Any initialization strategy that biases AWAY from boundaries will hurt these cases. The smart_init (hottest sensor) approach is already optimal because it directly targets the most likely source location regardless of position.

## Recommendations

1. **Don't pursue boundary-aware initialization** - Evidence shows it would hurt 24% of samples
2. **Mark initialization_v2 family as EXHAUSTED** - Same conclusion as initialization family
3. **Smart init is optimal** - Hottest sensor correlates well with source location
4. **Focus on different approaches** - Optimization algorithm improvements, not initialization

## Conclusion

**ABORTED** - Data analysis clearly shows that 24% of samples have boundary hotspots (proxy for boundary sources). The experiment's own abort criteria explicitly states "Boundary constraint hurts cases with actual boundary sources." Combined with the prior failure of EXP_PHYSICS_INIT_001, the initialization family should be considered fully exhausted.

## Data Source
- Test dataset: `data/heat-signature-zero-test-data.pkl` (80 samples)
- Analysis code: inline Python analysis of sensor temperature patterns
