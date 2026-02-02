# Variance Reduction Test

## Status: OVER BUDGET (concept validated but impractical)

## Hypothesis
Running optimization 2 times per sample and taking the best result might reduce variance and improve score.

## Results

| Config | Score | Time (min) | vs Baseline | Budget Status |
|--------|-------|------------|-------------|---------------|
| 1 run | 1.1325 | 46.6 | -0.0105 | IN BUDGET |
| **2 runs** | **1.1438** | 85.8 | **+0.0008** | **OVER BUDGET** |

**Improvement from 2 runs**: +0.0113 score, +39.2 min time

## Key Findings

1. **Multi-run works**: 2 runs per sample gives +0.0113 improvement
2. **Not practical**: Time nearly doubles (46.6 → 85.8 min)
3. **Cannot fit in budget**: 60 min limit prevents this approach
4. **Variance confirmed**: Single run variance explains score differences

## Analysis

The improvement from 2 runs (+0.0113) divided by extra time (+39.2 min) gives:
- ~0.00029 score per minute invested

This is much less efficient than single-run optimization which uses the time budget for more thorough search.

## Why 2 Runs Help

Each optimization run explores different random paths through the search space. Taking the best of 2:
- Reduces likelihood of getting stuck in bad local optima
- Essentially uses multiple starts for each sample

## Practical Alternatives

Instead of multi-run per sample, consider:
1. **More perturbations**: Already tested, diminishing returns after 2-4
2. **More CMA-ES fevals**: Helps but adds time
3. **Better initialization**: Already using triangulation + smart init
4. **Accept variance**: Current approach is near optimal given budget

## Conclusion

**Multi-run is not practical** within the 60-minute budget. The score improvement (+0.0113) is not worth the time cost (nearly 2x).

The observed run-to-run variance (~0.01) is an inherent limitation of the stochastic optimization approach.

---
**Worker**: W1
**Date**: 2026-02-02
