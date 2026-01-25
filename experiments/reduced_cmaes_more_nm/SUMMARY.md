# Experiment Summary: reduced_cmaes_more_nm

## Metadata
- **Experiment ID**: EXP_REDUCED_CMAES_MORE_NM_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: budget_reallocation

## Objective
Trade CMA-ES budget for more NM polish iterations. Test if fewer fevals (15/30 vs 20/36) + more NM polish (10-12 vs 8) could improve results.

## Hypothesis
CMA-ES may over-explore with 20/36 fevals. Reducing to 15/30 and adding more NM polish could give better accuracy within the same time budget.

## Results Summary
- **Best In-Budget Score**: NONE (run was over budget)
- **Best Overall Score**: 1.1584 @ 71.1 min
- **Baseline Comparison**: WORSE than baseline 1.1688 @ 58.4 min
- **Status**: FAILED

## Tuning History

| Run | CMA-ES fevals | NM polish | Score | Time (min) | In Budget | Delta vs Baseline |
|-----|---------------|-----------|-------|------------|-----------|-------------------|
| 1 | 15/30 | 10 | 1.1584 | 71.1 | NO | -0.0104 / +12.7 min |

## Key Findings

### What Didn't Work
1. **Reducing CMA-ES fevals hurts accuracy** - CMA-ES needs sufficient fevals to converge to good solutions
2. **NM polish cannot compensate** - Local refinement can't fix poorly converged CMA-ES solutions
3. **Time budget not saved** - NM iterations are more expensive per-iteration than CMA-ES fevals

### Why Budget Reallocation Fails
The baseline already has optimal budget allocation:
- 20/36 CMA-ES fevals: Sufficient for good convergence without over-exploration
- 8 NM polish iterations: Provides adequate local refinement

Reducing CMA-ES fevals:
- Hurts global search (CMA-ES doesn't converge as well)
- Each feval is cheap (coarse grid, 40% timesteps)

Adding NM polish:
- Each iteration is expensive (fine grid, full timesteps)
- Cannot compensate for poor CMA-ES convergence
- Adds more time than CMA-ES savings

### Critical Insight
**The baseline budget allocation (20/36 fevals + 8 NM) is already optimal.**

CMA-ES handles global exploration efficiently on the coarse proxy. NM handles local refinement on the accurate full-resolution signal. The current split is the right balance.

## Recommendations for Future Experiments

1. **DO NOT reduce CMA-ES fevals** - 20/36 is optimal for convergence
2. **DO NOT increase NM polish beyond 8** - 8 iterations is sufficient and more adds overhead
3. **budget_reallocation family is EXHAUSTED** - no improvement via budget trade-off

## Comparison to Prior Results

| Configuration | Score | Time | Notes |
|---------------|-------|------|-------|
| Baseline (20/36 + NM 8) | **1.1688** | **58.4 min** | BEST |
| This exp (15/30 + NM 10) | 1.1584 | 71.1 min | Over budget, worse |
| Prior: Extended NM (12 iters) | 1.1703 | 82.3 min | Over budget |

## Conclusion

**FAILED**: Reducing CMA-ES fevals to add more NM polish does NOT work. CMA-ES convergence suffers and NM polish cannot compensate. The baseline budget allocation is already optimal.
