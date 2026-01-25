# Experiment Summary: more_inits_select_best

## Metadata
- **Experiment ID**: EXP_MORE_INITS_SELECT_BEST_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: candidate_selection

## Objective
Run more CMA-ES initializations (6-8 instead of 2) to increase chance of finding global optimum. Select best 3 by RMSE before dissimilarity filtering to improve accuracy component of score.

## Hypothesis
More initializations increases chance of finding global optimum. Selecting best 3 by RMSE improves accuracy component of score.

## Results Summary
- **Best In-Budget Score**: NONE (all runs over budget)
- **Best Overall Score**: 1.1590 @ 66.9 min (6 inits)
- **Baseline Comparison**: ALL RUNS WORSE than baseline 1.1688 @ 58.4 min
- **Status**: FAILED

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Delta vs Baseline |
|-----|--------|-------|------------|-----------|-------------------|
| 1 | 6 inits, 20/36 fevals | 1.1590 | 66.9 | NO | -0.0098 / +8.5 min |
| 2 | 6 inits, 12/20 fevals | 1.1499 | 71.0 | NO | -0.0189 / +12.6 min |
| 3 | 4 inits, 20/36 fevals | 1.1447 | 69.8 | NO | -0.0241 / +11.4 min |

## Key Findings

### What Didn't Work
1. **More initializations add overhead** - Each additional CMA-ES instance requires setup time
2. **More fine-grid evaluations needed** - With more inits, we evaluate more candidates on fine grid
3. **Reducing fevals per init hurts accuracy** - Fewer fevals = worse convergence, lower scores
4. **Overhead dominated** - Time increase (~10-12 min) outweighs any accuracy benefit

### Why More Inits Fails
The baseline uses 2 smart initializations:
1. Triangulation-based (physics-informed)
2. Hottest-sensor (data-driven)

These 2 inits are already **very effective** at finding good starting points. Adding random inits:
- Doesn't find better solutions (physics-based inits already near optimal)
- Adds computational overhead (more CMA-ES runs, more evaluations)
- Requires reducing fevals per init to stay in budget (hurts convergence)

### Critical Insight
**The problem is well-conditioned for the baseline initialization strategies.**

CMA-ES with triangulation+hotspot init already converges to near-optimal solutions. Additional random inits just explore the same landscape without finding better solutions.

## Recommendations for Future Experiments

1. **DO NOT increase number of initializations** - 2 smart inits is optimal
2. **Initialization quality matters more than quantity** - Smart inits >> random inits
3. **candidate_selection family is EXHAUSTED** - no improvement via init strategy
4. **Focus on other aspects** - NM polish, sigma, fevals per init are fixed

## Comparison to Prior Results

| Configuration | Score | Time | Notes |
|---------------|-------|------|-------|
| Baseline (2 inits) | **1.1688** | **58.4 min** | BEST |
| 6 inits (default fevals) | 1.1590 | 66.9 min | Over budget |
| 6 inits (reduced fevals) | 1.1499 | 71.0 min | Even worse |
| 4 inits (default fevals) | 1.1447 | 69.8 min | Worst |

## Conclusion

**FAILED**: More initializations do NOT improve results. The overhead of additional CMA-ES instances and fine-grid evaluations outweighs any accuracy benefit. The baseline 2-init strategy (triangulation + hotspot) is already optimal.
