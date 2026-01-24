# Experiment: Confidence-Based Early Exit

**Experiment ID**: EXP_EARLY_EXIT_THRESHOLD_001
**Worker**: W1
**Status**: ABORTED
**Date**: 2026-01-24

## Hypothesis
Exit CMA-ES early when "confidence" in the solution is high, saving time for samples that converge quickly.

## Why This Was Aborted

### Prior Experiments Already Proved This Doesn't Work

| Experiment | Approach | Result |
|------------|----------|--------|
| EXP_ADAPTIVE_BUDGET_001 | Exit based on sigma stagnation | FAILED: 1.1143 @ 56.2 min |
| EXP_CMAES_EARLY_STOPPING_001 | Exit based on fitness improvement | FAILED: Early stopping NEVER triggered |

### Key Findings from Prior Work

1. **Early stopping NEVER triggered**
   - EXP_CMAES_EARLY_STOPPING_001 found that CMA-ES continues improving >1% throughout its full budget
   - There's no point where "confidence is high enough" to stop early

2. **Early termination hurts accuracy**
   - EXP_ADAPTIVE_BUDGET_001 showed that terminating early reduces final accuracy
   - CMA-ES needs full budget to properly adapt the covariance matrix

3. **Budget allocation is already optimal**
   - Fixed budget approach with full evaluations is proven optimal
   - Baseline achieves 1.1688 @ 58.4 min with fixed budgets

### Why "Confidence-Based" Is Not Different

"Confidence" in a solution could be measured by:
- Sigma (step size) convergence → Already tested in ADAPTIVE_BUDGET
- Fitness improvement rate → Already tested in EARLY_STOPPING
- Population variance → Same as sigma-based

All confidence metrics ultimately measure the same thing: how much CMA-ES is still exploring. And we know CMA-ES continues exploring productively until its budget is exhausted.

## Theoretical Analysis

CMA-ES is designed for expensive optimization with limited budgets (10-100 evaluations). Its covariance adaptation mechanism:
1. **Requires multiple generations** to learn the fitness landscape
2. **Uses ALL evaluations** to build the covariance model
3. **Cannot be "confident" early** because early generations are still exploring

Stopping early means stopping before the covariance matrix is well-adapted, which always hurts accuracy.

## Conclusion

**ABORTED** - This is the same idea as two prior experiments that already failed. "Confidence-based" early exit is fundamentally incompatible with CMA-ES's covariance adaptation mechanism.

**Recommendation**: The early_exit family should be marked as EXHAUSTED. CMA-ES cannot be safely stopped early.
