# Iterated Local Search (ILS) Experiment

## Status: MARGINAL_IMPROVEMENT (but over budget)

## Hypothesis
Iterated Local Search (ILS) can escape local optima by iteratively perturbing and running local search. This is a classic approach from optimization literature that may improve our perturbation strategy.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | ils_3iter_4nm | 1.1609 | 71.1 | -11 min OVER |
| 2 | ils_2iter_3nm | 1.1534 | 64.7 | -5 min OVER |
| 3 | no_ils_baseline | 1.1477 | 56.9 | +3 min UNDER |

**Baseline**: 1.1464 @ 51.2 min

## Key Findings

### ILS Improves Accuracy But Adds Overhead
- ILS with 3 iterations improves score by +0.0132 (+1.1%) over baseline
- But adds 14 minutes of overhead (71 vs 57 min)
- Lighter ILS (2 iter) improves by +0.0057 (+0.5%) with 8 min overhead

### Time-Accuracy Tradeoff
| Config | Score | Time | Score/min improvement |
|--------|-------|------|----------------------|
| No ILS | 1.1477 | 57 min | baseline |
| ILS 2 iter | 1.1534 | 65 min | +0.0007/min |
| ILS 3 iter | 1.1609 | 71 min | +0.0009/min |

### Baseline Achieved Good Results
The baseline without ILS (1.1477 @ 57 min) is slightly better than the historical baseline (1.1464 @ 51.2 min). This may be due to:
- 40% temporal fidelity (vs possibly lower in historical)
- Different sigma values

## Root Cause Analysis

The ILS overhead comes from:
1. **Multiple NM iterations per ILS step**: Each ILS iteration runs 3-4 NM iterations
2. **Repeated on coarse grid**: ILS operates on coarse grid to save time, but still adds overhead
3. **Diminishing returns**: Later ILS iterations rarely find improvement

## Tuning Efficiency

- **Runs executed**: 3
- **Time utilization**: All configs ran to completion
- **Budget compliance**: Only baseline finished within budget
- **Parameter space explored**: max_ils_iterations=[2,3], ils_nm_iters=[3,4]

## Conclusion

**ILS provides marginal accuracy improvement but cannot fit within the 60-minute budget.**

Key insights:
1. ILS can escape local optima and improve accuracy (+1% best case)
2. Each ILS iteration adds ~3-4 min overhead
3. The overhead cannot be justified within the strict time budget
4. Standard perturbation (without iterative refinement) is more budget-appropriate

## What Would Have Been Tried With More Time

If budget were ~90 min:
- Try ILS with 4-5 iterations
- Test different perturbation scales (0.03, 0.07, 0.10)
- Apply ILS to top-2 candidates instead of just top-1

## Recommendations

1. **Mark ILS family as BUDGET_INCOMPATIBLE** - works but too slow
2. **Keep baseline perturbation approach** - better time/accuracy tradeoff
3. **Potential future work**: Very lightweight ILS (1 iter, 2 NM steps) might fit budget
4. **Note**: The no-ILS baseline achieved 1.1477, which is slightly better than historical baseline
